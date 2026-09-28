#!/usr/bin/env python3
"""Minimal reproduction: composite loss + SoftAdapt vs. pure MSE.
Having uv installed is beneficial but not required;  The bootstrap creates
.venv_neptuna_repro/ in the current working directory, installs the dependencies into
it, and re-executes this script inside it. 
Usage
-----
    python3 experiment/pipeline_test.py
Requires pipeline_test_utils.py alongside it (environment bootstrap and helpers).

The dataset 2D_SDBA_32train_8test (~1.4 GB) is expected at the root of the repo; if it is
not there it is downloaded automatically from
https://huggingface.co/datasets/FluidVerse/2D_SDBA_small_sample
Ideally run on a CUDA GPU; CPU also works, but is much slower.

Experiment summary
-----------------
    Dataset  2D_SDBA_32train_8test
             32 train / 8 test trajectories, 128^2, 101 frames
             Channels->4 (density, pressure, velocityX, velocityY)
    Model    FFNO latent=64 / 6 layers / 20 modes -> 2.18 M params
             (the full benchmark uses 50 M)
    Task     4 input snapshots -> predict the next; autoregressive rollout at test time
    Configs  mse (baseline, 1 run) vs. composite_softadapt at 3 initial loss-weight
             vectors: [1,1,1,1], [1,2,0.5,1], [3,1,2,1] over
             (MSE, H1SemiNorm, SSIM, InterfaceRMSE)
    Seed     one seed shared by every run, so the composite runs differ ONLY in their
             initial component weights
"""

import multiprocessing
import os
import sys

# Environment bootstrap and small helpers live in pipeline_test_utils, which is
# deliberately stdlib-only at import time so it can run before the stack it installs.
from pipeline_test_utils import (
    bootstrap,
    describe_h5,
    ensure_dataset,
    ensure_sqlite3,
    find_repo_root,
    report_device,
    resolve_device,
)

# Creates/reuses the venv, installs the requirements, and re-executes THIS file inside
# it. Returns immediately if the current interpreter already has everything.
bootstrap(__file__)


# =============================================================================
# 1. What is trained on, and what is measured
#
# These are two different lists, and keeping them distinct is what makes the
# result meaningful.
#
# 1a. Training-loss components (the toggle being demonstrated)
#
#   mse                  MSE                                        fixed weight 1.0
#   composite_softadapt  MSE + H1SemiNorm + SSIM + InterfaceRMSE     SoftAdapt, adaptive
#
# H1SemiNorm penalises gradient error (sharp features), SSIM penalises structural
# error, InterfaceRMSE masks the error to the droplet interface via the density
# field (density_range [350, 550], valid here since density spans 1.2 -> 1000).
# SoftAdapt is configured with the repo's default parameters,
# config/train_strategy_config/weighting_strategies/SoftAdapt/soft_adapt_default.yaml
# (temperature_T 1.0), rebalancing the four terms from their relative rates of
# descent.
#
# Everything else is held fixed between the two runs - same model, same data, same
# optimiser and LR schedule, same seeds, and the same validation loss
# (RMSE) and same metric_for_best_model, so checkpoint selection cannot favour the
# composite run.
#
# 1b. Reported inference metrics on the test dataset
#
# Computed by the repo's existing rollout evaluator using
# config/infer_config/infer_loss.yaml, on the 8 held-out test trajectories,
# autoregressively from the initial condition.
#
#   IRMSE (InterfaceRMSE)          error inside the interface region
#                                  
#   MLW (MultilevelWaveletLoss)    multi-scale spectral error
#                                  
#   SSIM (reported as 1 - SSIM)    structural similarity          
#   VRMSE                          pointwise, variance-normalised 
#
# SIGN CONVENTION: all four are reported as losses, LOWER IS BETTER, including SSIM
# (the repo returns 1 - SSIM). The table prints both the raw loss and plain SSIM.
#
# =============================================================================


# =============================================================================
# 2. Settings
# =============================================================================
REPO_ROOT = find_repo_root(__file__)                 # or: pathlib.Path("/path/to/cfd_bench")
DATA_DIR = "2D_SDBA_32train_8test"      # relative to REPO_ROOT; read-only

DEVICE_PREFERENCE = "auto"     # "auto" | "cuda" | "cpu"
GPU_ID = 0                     # which CUDA GPU (`nvidia-smi` to pick a free one).
                               # Ignored unless the resolved device is cuda.

# One seed for every run: the composite variants are meant to differ ONLY in their
# initial loss weights.
SEED = 0
NUM_TRAIN_EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 5e-4           # higher than the repo default (5e-5):
EVAL_EVERY_STEPS = 250         # validation + checkpoint cadence
N_INFER_ROLLOUTS = 6           # max for sequence_info [4, 1, 10] over 101 frames 

DATALOADER_WORKERS = min(12, os.cpu_count() or 2)

if multiprocessing.get_start_method(allow_none=False) != "fork" and DATALOADER_WORKERS:
    print(f"note: multiprocessing start method is "
          f"{multiprocessing.get_start_method()!r}, not 'fork' -> forcing "
          f"DATALOADER_WORKERS {DATALOADER_WORKERS} -> 0 (data loading runs in-process).")
    DATALOADER_WORKERS = 0

RUN_ROOT = REPO_ROOT / "pipeline_test"      # the ONLY directory this file writes to
RESULTS = RUN_ROOT / "results"

assert DEVICE_PREFERENCE in ("auto", "cuda", "cpu"), \
    f"DEVICE_PREFERENCE must be auto/cuda/cpu, got {DEVICE_PREFERENCE!r}"

if DEVICE_PREFERENCE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""       # genuinely hide the GPUs
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
RESULTS.mkdir(parents=True, exist_ok=True)

# Downloads the mini set from Hugging Face into REPO_ROOT on first run; a no-op once
# train.h5 and test.h5 are there.
ensure_dataset(REPO_ROOT, DATA_DIR)

ensure_sqlite3()

print("repo:", REPO_ROOT)
print("data:", REPO_ROOT / DATA_DIR)
print("dev :", f"DEVICE_PREFERENCE={DEVICE_PREFERENCE!r}",
      f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r})")


# =============================================================================
# 3. Environment check and device selection
# =============================================================================
import csv
import gc
import json
import statistics
import time

import numpy as np
import torch

print("python :", sys.version.split()[0])
print("torch  :", torch.__version__)

try:
    import ptwt, pywt                      # required by MultilevelWaveletLoss
    print("ptwt   : ok")
except ImportError as e:
    raise SystemExit(f"ptwt missing -> MLW cannot be computed: {e}")

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from utils.config_utils import prepare_config
from bench.run import run
print("bench.run : imported ok")


DEVICE = resolve_device(DEVICE_PREFERENCE)
report_device(DEVICE, DEVICE_PREFERENCE)


# =============================================================================
# 4. Verify the mini dataset 
# =============================================================================
tr = describe_h5(f"{DATA_DIR}/train.h5")
te = describe_h5(f"{DATA_DIR}/test.h5")

for tag, d in (("train", tr), ("test", te)):
    print(f"{tag:5s}: {len(d['groups']):2d} trajectories | {d['shape'][0]} frames "
          f"| {d['shape'][2]}x{d['shape'][3]} | {d['dtype']} | {d['size_gb']:.2f} GB")
    print(f"       channels: {d['chans']}")
    print(f"       Mach: {d['mach']}  density range: {d['dens'][0]:.1f} .. {d['dens'][1]:.1f}")

assert tr["chans"] == te["chans"], "train/test channel sets differ"
assert tr["shape"][2:] == te["shape"][2:], "train/test spatial resolution differs"
assert not (set(tr["groups"]) & set(te["groups"])), "train/test trajectory leakage"

CHANNELS = tr["chans"]
GRID = list(tr["shape"][2:])
print(f"\nOK - consistent, no leakage. Total footprint "
      f"{tr['size_gb'] + te['size_gb']:.2f} GB, channels used: {CHANNELS}")

# The interface metric thresholds density in [350, 550];
assert tr["dens"][0] < 350 < 550 < tr["dens"][1], \
    "InterfaceRMSE density_range [350,550] lies outside the data range"
print("InterfaceRMSE density_range [350, 550] is inside the data range -> valid interface mask")


# =============================================================================
# 5. Normalization statistics
# Computed once, from train.h5 only 
# =============================================================================
from utils.compute_stats import compute_statistics_parallel

STATS_JSON = RUN_ROOT / "norm_stats.json"
LOG_CHANNELS = ["density"]

if STATS_JSON.exists():
    NORM_STATS = json.loads(STATS_JSON.read_text())
    print(f"loaded cached stats from {STATS_JSON}")
else:
    t0 = time.perf_counter()
    stats, channel_names, _ = compute_statistics_parallel(
        h5_paths=[f"{DATA_DIR}/train.h5"],
        residual_config=None, filter_groups=None, filter_frames=None,
        frame_stride=1, on_fly_stats=True, num_workers=4,
        log_transform_channels=LOG_CHANNELS,
    )
    NORM_STATS = {c: {k: float(v) for k, v in s.items()} for c, s in dict(stats).items()}
    STATS_JSON.write_text(json.dumps(NORM_STATS, indent=2))
    print(f"computed in {time.perf_counter() - t0:.1f}s -> cached at {STATS_JSON}")

for c, s in NORM_STATS.items():
    print(f"  {c:26s} mean={s['mean']:12.4f}  std={s['std']:12.4f}")
assert "log_density" in NORM_STATS, "log-transformed density stats missing"


# =============================================================================
# 6. Build the configs 
# =============================================================================

# Model: FFNO scaled down from the 50M benchmark config to ~2.2M.
MODEL_CFG = {
    "model_name": "FFNO",
    "latent_channels": 64,    # ffno_10M uses 96
    "n_fno_layers": 6,        # ffno_10M uses 8
    "fno_modes": 20,
    "factor": 4,
    "n_ff_layers": 2,
    "layer_norm": True,
    "norm_layer_eps": 1e-5,
    "norm": "layer",
}

# The composite loss, in a fixed component order. The initial weight of each is what
# the three composite variants vary; SoftAdapt then adapts from there (see the module
# docstring for how long the initial values actually matter).
COMPOSITE_ORDER = [                       # (type, config_file, explicit name)
    ("MSE",           None,                       "MSE_train"),
    ("H1SemiNorm",    None,                       None),
    ("SSIM",          None,                       None),
    ("InterfaceRMSE", "interface_rmse_default",   None),
]

# Initial component weights over (MSE, H1SemiNorm, SSIM, InterfaceRMSE).
COMPOSITE_WEIGHT_INITS = {
    "composite_softadapt":    [1.0, 1.0, 1.0, 1.0],     # uniform (the original run)
    "composite_softadapt_wB": [1.0, 2.0, 0.5, 1.0],     # up-weight gradients, down-weight structure
    "composite_softadapt_wC": [3.0, 1.0, 2.0, 1.0],     # up-weight pointwise and structure
}

SOFTADAPT = {
    "enabled": True,
    "type": "SoftAdapt",            # params from weighting_strategies/SoftAdapt/soft_adapt_default.yaml
    "weight_sub_components": False,
    "weight_per_channel": False,
    "loss_history_interval": 10,
}


def composite_components(weights):
    assert len(weights) == len(COMPOSITE_ORDER), \
        f"need {len(COMPOSITE_ORDER)} weights for {[t for t, _, _ in COMPOSITE_ORDER]}"
    comps = []
    for (typ, cfg_file, name), w in zip(COMPOSITE_ORDER, weights):
        c = {"type": typ, "weight": float(w)}
        if name:
            c["name"] = name
        if cfg_file:
            c["config_file"] = cfg_file
        comps.append(c)
    return comps


# The full run matrix: one MSE baseline plus one composite run per weight vector.
RUNS = {
    "mse": {
        "components": [{"type": "MSE", "weight": 1.0, "name": "MSE_train"}],
        "weighting": {"enabled": False},
        "weights": None,
    },
}
for _label, _w in COMPOSITE_WEIGHT_INITS.items():
    RUNS[_label] = {"components": composite_components(_w),
                    "weighting": dict(SOFTADAPT),
                    "weights": _w}


def train_strategy(run_name):
    return {
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "batch_eval_metrics": True,
        "num_epochs_between_eval": 1,
        # identical checkpoint-selection criterion for every run
        "metric_for_best_model": [{"type": "RMSE", "weight": 1.0}],
        "curriculum": [{
            "name": "block_1",
            "start_epoch": 0,
            "train_loss": {
                "components": RUNS[run_name]["components"],
                "train_loss_weighting_strategy": RUNS[run_name]["weighting"],
            },
            # identical validation loss for every run
            "validation_loss": {"components": [{"type": "RMSE", "weight": 1.0}]},
        }],
    }


def build_cfg(loss_name, seed, out_dir):
    """Compose from the repo's config groups, then patch in memory. Writes nothing."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base="1.3"):
        cfg = compose(config_name="defaults", overrides=[
            "data_config=fluids/2D_Shock_Droplet_in_Air_SSOO/2d_sdba_ssoo_data_default",
            "model_config=FFNO/ffno_10M",
            "train_strategy_config=default_train_strategy",
        ])
    OmegaConf.set_struct(cfg, False)

    d = cfg.data_config
    d.dataset_name = "2D_SDBA_mini_128"
    d.dataset_directory_path = DATA_DIR
    d.seed = seed                         
    d.grid_resolution = None               # auto-detected from the h5 file
    d.sequence_info = [4, 1, 10]           # 4 inputs -> 1 output, stride 10 (as in the full benchmark)
    d.coord_features = True                # appends normalized x/y grids -> 4*4 + 2 = 18 input channels
    d.log_transform_channels = LOG_CHANNELS
    d.data_normalization_strategy = "z_normalization"
    d.data_normalization_stats = OmegaConf.create(NORM_STATS)
    d.filter_features.filter_in_channels = CHANNELS
    d.filter_features.filter_out_channels = CHANNELS

    cfg.model_config = OmegaConf.create(MODEL_CFG)
    cfg.train_strategy_config = OmegaConf.create(train_strategy(loss_name))

    t = cfg.train_config
    t.per_device_train_batch_size = BATCH_SIZE
    t.per_device_eval_batch_size = BATCH_SIZE
    t.dataloader_num_workers = DATALOADER_WORKERS
    t.eval_steps = t.save_steps = EVAL_EVERY_STEPS
    t.logging_steps = max(EVAL_EVERY_STEPS // 5, 10)
    t.plot_after_epoch = [NUM_TRAIN_EPOCHS]        # skip per-eval plotting
    # tf32 is a CUDA-only fast path; 
    t.mix_precision_config.tf32 = (DEVICE == "cuda")
    t.use_cpu = (DEVICE == "cpu")
    cfg.scheduler_config.lr = LEARNING_RATE

    i = cfg.infer_config
    i.do_infer = True                      # rollout evaluation straight after training
    i.n_infer_rollouts = N_INFER_ROLLOUTS
    i.infer_from_ic = True
    i.infer_from_random_timestep = False
    i.n_infer_plot_examples = 1

    cfg.output_log_config.logging.output_dir = str(out_dir)
    cfg.output_log_config.logging.wandb = False
    return prepare_config(cfg)

_stripped = []
for _n in RUNS:
    _c = json.loads(json.dumps(train_strategy(_n)))
    _c["curriculum"][0]["train_loss"] = "<REMOVED>"
    _stripped.append(_c)
assert all(_c == _stripped[0] for _c in _stripped), \
    "the run configs differ outside the train_loss block"
print(f"verified: all {len(RUNS)} training-strategy configs differ ONLY in "
      "curriculum[0].train_loss")

# And that the composite runs differ from each other only in the component weights.
_comp = [RUNS[n]["components"] for n in COMPOSITE_WEIGHT_INITS]
assert all([{k: v for k, v in c.items() if k != "weight"} for c in _comp[0]] ==
           [{k: v for k, v in c.items() if k != "weight"} for c in other] for other in _comp), \
    "the composite runs differ by more than their initial weights"
print("verified: the composite runs differ ONLY in their initial component weights")
for _n, _w in COMPOSITE_WEIGHT_INITS.items():
    print(f"    {_n:26s} {_w}")

probe_cfg = build_cfg("mse", SEED, RUN_ROOT / "_config_probe")
print("config composed in memory and prepared OK (no files written to config/)")


# =============================================================================
# 7. Model size 
# =============================================================================
from utils.load_model import fetch_model

_model = fetch_model(probe_cfg.model_config, probe_cfg.data_config)
n_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)

print(f"FFNO(latent={MODEL_CFG['latent_channels']}, layers={MODEL_CFG['n_fno_layers']}, "
      f"modes={MODEL_CFG['fno_modes']}) -> {n_params/1e6:.2f} M trainable parameters")
del _model
gc.collect()

# Window arithmetic.
n_in, n_out, stride = probe_cfg.data_config.sequence_info
T = tr["shape"][0]
span = (n_in + n_out - 1) * stride + 1          # frames spanned by one sample window
n_win = T - span + 1                            # sliding windows per trajectory
n_train = n_win * len(tr["groups"])
max_roll = (T - 1) // stride - (n_in + n_out - 1)   # max autoregressive rollout steps

print(f"\nwindow spans {span}/{T} frames -> {n_win} windows per trajectory")
print(f"samples: {n_train} in train.h5 (~{int(n_train*0.8)} train / {int(n_train*0.2)} val "
      f"after the 0.2 eval split), {n_win*len(te['groups'])} in test.h5")
print(f"steps/epoch ~ {int(n_train*0.8)//BATCH_SIZE}, "
      f"total ~ {int(int(n_train*0.8)//BATCH_SIZE*NUM_TRAIN_EPOCHS)} optimizer steps per run")
assert N_INFER_ROLLOUTS <= max_roll, f"n_infer_rollouts must be <= {max_roll} for this window config"
print(f"max autoregressive rollout steps = {max_roll}; using {N_INFER_ROLLOUTS}")


# =============================================================================
# 8. Run the experiment - 1 MSE baseline + 3 composite weight initialisations
# Completed runs are skipped
# =============================================================================

def run_dir(run_name):
    return RUN_ROOT / "runs" / f"{run_name}_seed{SEED}"


def results_csv(run_name):
    return run_dir(run_name) / "direct_inference" / "overall_results.csv"


wall0 = time.perf_counter()
failures = []

for run_name in RUNS:
        if results_csv(run_name).exists():
            print(f"[skip] {run_name} - already has results", flush=True)
            continue
        print(f"\n{'='*72}\n[run ] {run_name}  seed={SEED}"
              f"{'  weights=' + str(RUNS[run_name]['weights']) if RUNS[run_name]['weights'] else ''}"
              f"\n{'='*72}", flush=True)
        t0 = time.perf_counter()
        try:
            run(build_cfg(run_name, SEED, run_dir(run_name)))
            ok = results_csv(run_name).exists()
        except Exception as e:
            ok = False
            failures.append((run_name, repr(e)))
            print(f"    EXCEPTION: {e!r}")
        finally:
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        print(f"[{'done' if ok else 'FAIL'}] {run_name} "
              f"in {(time.perf_counter()-t0)/60:.1f} min", flush=True)

print(f"\ntotal wall time: {(time.perf_counter() - wall0)/60:.1f} min "
      f"for {len(RUNS)} runs")
if failures:
    print("\nfailed runs:")
    for f in failures:
        print("  ", f)


# =============================================================================
# 9. Aggregate
#
# One MSE baseline value and three composite runs. The composite spread is
# across INITIAL LOSS WEIGHTS
#
# Metric keys in overall_results.csv are prefixed by rollout mode; we read the
# ic_start_ block (rollout from the initial condition).
# =============================================================================
PREFIX = "ic_start_"      # autoregressive rollout starting from the initial condition
HEADLINE = {
    "IRMSE": "InterfaceRMSE",
    "MLW": "MultilevelWaveletLoss",
    "SSIM": "SSIM",
    "VRMSE": "VRMSE",
}


def read_metrics(run_name):
    path = results_csv(run_name)
    if not path.exists():
        return None
    raw = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) == 2 and row[0] != "Metric":
                try:
                    raw[row[0]] = float(row[1])
                except ValueError:
                    pass
    # exact-suffix match so e.g. MSSSIM is never mistaken for SSIM
    return {disp: raw[PREFIX + comp] for disp, comp in HEADLINE.items()
            if PREFIX + comp in raw}


per_run = {n: read_metrics(n) for n in RUNS}
COLS = list(HEADLINE)

print("per-run values")
for n, v in per_run.items():
    w = RUNS[n]["weights"]
    tag = f"init={w}" if w else "baseline"
    print(f"  {n:26s} {tag:26s} " +
          ("  ".join(f"{m}={v[m]:.5f}" for m in COLS) if v else "MISSING"))

baseline = per_run["mse"]
comp_names = [n for n in COMPOSITE_WEIGHT_INITS if per_run[n]]
assert baseline, "the mse baseline has no results"
assert comp_names, "no composite runs have results"

# Spread across the composite weight initialisations.
comp_stats = {m: (statistics.mean([per_run[n][m] for n in comp_names]),
                  statistics.stdev([per_run[n][m] for n in comp_names]) if len(comp_names) > 1 else 0.0)
              for m in COLS}

lines = ["| Run | initial weights | " + " | ".join(f"{c} v" for c in COLS) + " | SSIM (plain) ^ |",
         "|" + "---|" * (len(COLS) + 3)]
lines.append("| MSE only (baseline) | - | " +
             " | ".join(f"{baseline[c]:.5f}" for c in COLS) +
             f" | {1.0 - baseline['SSIM']:.4f} |")
for n in comp_names:
    lines.append(f"| {n} | `{RUNS[n]['weights']}` | " +
                 " | ".join(f"{per_run[n][c]:.5f}" for c in COLS) +
                 f" | {1.0 - per_run[n]['SSIM']:.4f} |")
lines.append("| **composite mean +/- std over inits** | - | " +
             " | ".join(f"**{comp_stats[c][0]:.5f} +/- {comp_stats[c][1]:.5f}**" for c in COLS) +
             " | |")
lines.append("| **Delta vs MSE (negative = better)** | - | " +
             " | ".join(f"**{100*(comp_stats[c][0]-baseline[c])/baseline[c]:+.1f}%**" for c in COLS) +
             " | |")

table = "\n".join(lines)
note = (f"\nAll four columns are losses: **lower is better**, including SSIM "
        f"(reported as 1 - SSIM).\n"
        f"Baseline n = 1. Composite n = {len(comp_names)}, varying the INITIAL loss-component "
        f"weights over (MSE, H1SemiNorm, SSIM, InterfaceRMSE);\nall runs share seed "
        f"{SEED}, so the spread is sensitivity to initial weighting, not seed noise.\n"
        f"FFNO {n_params/1e6:.2f} M params, {NUM_TRAIN_EPOCHS} epochs, "
        f"{N_INFER_ROLLOUTS}-step autoregressive rollout on {len(te['groups'])} "
        f"held-out trajectories.\n")
print("\n" + table + "\n" + note)

(RESULTS / "softadapt_minimal_results.md").write_text(
    "# Minimal SoftAdapt reproduction - results\n\n" + table + "\n" + note)
with open(RESULTS / "softadapt_minimal_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run", "initial_weights", "seed"] + COLS)
    for n, v in per_run.items():
        if v:
            w.writerow([n, RUNS[n]["weights"], SEED] + [v[m] for m in COLS])
    w.writerow(["composite", "mean_over_inits", SEED] + [comp_stats[c][0] for c in COLS])
    w.writerow(["composite", "std_over_inits", SEED] + [comp_stats[c][1] for c in COLS])
print(f"written to {RESULTS}/softadapt_minimal_results.{{md,csv}}")

# =============================================================================
# 10. Plot: Comparison between the MSE baseline and the three composite runs, spread across initial weights
# =============================================================================
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(COLS), figsize=(4.4 * len(COLS), 3.8))
order = ["mse"] + comp_names
labels = ["MSE\n(baseline)"] + [f"SA init\n{RUNS[n]['weights']}" for n in comp_names]
colors = ["#8c8c8c"] + ["#1f77b4"] * len(comp_names)

for ax, m in zip(axes, COLS):
    xs = range(len(order))
    ax.bar(xs, [per_run[n][m] for n in order], color=colors, width=0.65)
    # baseline reference line, so "does every init beat it?" is readable off the plot
    ax.axhline(baseline[m], color="#8c8c8c", ls="--", lw=1.2, zorder=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(f"{m} (-inference_metric-lower better)") 
                                        
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("loss value (lower is better)")
fig.suptitle(f"Minimal SoftAdapt reproduction - FFNO {n_params/1e6:.1f}M, "
             f"{NUM_TRAIN_EPOCHS} epochs, seed {SEED}; composite bars vary the INITIAL "
             f"loss weights (MSE, H1, SSIM, IRMSE)", y=1.04, fontsize=10)
fig.tight_layout()
fig.savefig(RESULTS / "softadapt_minimal_comparison.png", dpi=150, bbox_inches="tight")
print(f"saved {RESULTS}/softadapt_minimal_comparison.png")
plt.close(fig)
