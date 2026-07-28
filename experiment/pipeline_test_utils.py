#!/usr/bin/env python3
"""Environment bootstrap and small helpers for pipeline_test.py.
"""
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Environment specification
# ---------------------------------------------------------------------------
PYTHON_VERSION = "3.13"

# The venv is created in the current working directory.
VENV_DIR = pathlib.Path(
    os.environ.get("NEPTUNA_REPRO_VENV") or (pathlib.Path.cwd() / ".venv_neptuna_repro")
).resolve()

TORCH_SPEC = "torch==2.10.0"

REQUIREMENTS = [
    TORCH_SPEC,
    "transformers==4.56.0",        # API-critical: Trainer / TrainingArguments
    "accelerate==1.12.0",
    "hydra-core>=1.3", "omegaconf==2.3.0",
    "numpy>=2.2", "scipy", "h5py>=3.12", "matplotlib",
    "ptwt==1.0.1",                 # pulls PyWavelets; needed for the MLW metric
    "einops",                      # models/DPOT, imported via the model registry
    "optuna", "psutil", "wandb", "pyyaml", "tqdm", "colorlog",
]

# import name -> what needs it
NEEDED = {
    "torch": "models + training", "transformers": "Trainer", "accelerate": "Trainer backend",
    "hydra": "config system", "omegaconf": "config system", "numpy": "everywhere",
    "scipy": "metrics", "h5py": "dataset", "matplotlib": "plots",
    "ptwt": "MLW metric", "pywt": "MLW metric", "einops": "model registry",
    "optuna": "imported by bench.run", "psutil": "imported by bench.run",
    "wandb": "imported by utils.custom_callbacks", "yaml": "configs", "tqdm": "progress bars",
}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def missing_packages():
    """Which requirements the *current* interpreter cannot import."""
    out = []
    for mod in NEEDED:
        try:
            if importlib.util.find_spec(mod) is None:
                out.append(mod)
        except (ImportError, ValueError):
            out.append(mod)
    return out


def venv_python(venv):
    sub, exe = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return venv / sub / exe


def sqlite3_works(python_exe):
    """A venv shares its base interpreter's stdlib extension modules, so a broken
    _sqlite3 in the base is inherited. optuna -> bench.run needs it, so check early."""
    return subprocess.run([str(python_exe), "-c", "import sqlite3"],
                          capture_output=True).returncode == 0


def missing_in(python_exe):
    """Which requirements are not importable by *another* interpreter."""
    code = (
        "import importlib.util as u, sys\n"
        "bad=[]\n"
        f"for m in {list(NEEDED)!r}:\n"
        "    try:\n"
        "        if u.find_spec(m) is None: bad.append(m)\n"
        "    except Exception: bad.append(m)\n"
        "print(','.join(bad))\n"
    )
    p = subprocess.run([str(python_exe), "-c", code], capture_output=True, text=True)
    if p.returncode != 0:
        return list(NEEDED)
    return [m for m in p.stdout.strip().split(",") if m]


def install_into(python_exe):
    uv = shutil.which("uv")
    print("\ninstalling dependencies ...\n", flush=True)
    if uv:
        subprocess.run([uv, "pip", "install", "--python", str(python_exe), *REQUIREMENTS],
                       check=True)
    else:
        subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip", "-q"],
                       check=True)
        subprocess.run([str(python_exe), "-m", "pip", "install", *REQUIREMENTS], check=True)


def create_environment():
    """Create VENV_DIR from scratch and return its interpreter (does not install).

    Prefers `uv` with a *managed* standalone CPython. `python -m venv` does not copy an
    interpreter - it references the one that created it - so a stdlib venv inherits that
    interpreter's stdlib and C extensions, including any breakage. A managed CPython is
    self-contained and independent of whatever conda/system Python happens to be on PATH.
    """
    uv = shutil.which("uv")

    if uv:
        print(f"creating environment at {VENV_DIR}")
        print(f"  using uv with a managed, standalone CPython {PYTHON_VERSION}", flush=True)
        # --clear: never prompt. We only get here when the directory is absent or was
        # judged unusable, so replacing it is always the intent.
        subprocess.run([uv, "venv", "--python", PYTHON_VERSION,
                        "--python-preference", "only-managed", "--clear",
                        str(VENV_DIR)], check=True)
        return venv_python(VENV_DIR)

    if sys.version_info < (3, 11):
        raise SystemExit(
            f"This interpreter is Python {sys.version_info.major}.{sys.version_info.minor}; "
            "ptwt==1.0.1 requires >= 3.11.\n"
            "Either install uv (https://docs.astral.sh/uv/), which provisions its own "
            f"Python and makes this irrelevant, or re-run with:\n"
            f"    python3.11 {' '.join(sys.argv)}")

    if not sqlite3_works(sys.executable):
        raise SystemExit(
            f"{sys.executable} cannot `import sqlite3`, and a venv built from it would\n"
            "inherit that fault (venvs share the base interpreter's stdlib), so optuna and\n"
            "bench.run would fail after a multi-GB install. Refusing to build.\n\n"
            "Fix: install uv (https://docs.astral.sh/uv/) so this script can provision its\n"
            "own standalone Python, or re-run with an interpreter whose sqlite3 works.")

    print(f"creating environment at {VENV_DIR}")
    print(f"  uv not found; using `python -m venv` from {sys.executable}", flush=True)
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    return venv_python(VENV_DIR)


def bootstrap(script_path):
    """Ensure the requirements are importable, re-executing `script_path` if needed.

    `script_path` must be the ENTRY script (pass __file__ from it), not this module -
    the re-exec has to restart the experiment, not the helpers.
    Returns normally when the current interpreter is already usable; otherwise it
    replaces the process and never returns.
    """
    missing = missing_packages()
    if not missing:
        print(f"All dependencies are already available in {sys.prefix}")
        print("Nothing to install - continuing.")
        return

    print(f"Missing here: {', '.join(missing)}\n")

    # Guard against an exec loop: if we already re-executed once and packages are still
    # missing, something is wrong with the install and looping would hide it.
    if os.environ.get("NEPTUNA_REPRO_BOOTSTRAPPED") == "1":
        raise SystemExit(
            "Still missing packages after bootstrapping into "
            f"{sys.prefix}: {missing}\nInstall them manually and re-run.")

    py = venv_python(VENV_DIR)
    reuse = False

    if VENV_DIR.exists():
        if not sqlite3_works(py):
            print(f"existing environment at {VENV_DIR} has a broken sqlite3 "
                  "(inherited from the interpreter that created it) - rebuilding it\n",
                  flush=True)
            shutil.rmtree(VENV_DIR)
        else:
            reuse = True
            print(f"reusing existing environment at {VENV_DIR}", flush=True)

    if not reuse:
        py = create_environment()
        install_into(py)
    else:
        still_missing = missing_in(py)
        if still_missing:
            print(f"  missing there: {', '.join(still_missing)}", flush=True)
            install_into(py)
        else:
            print("  it already has everything", flush=True)

    print("\n" + "=" * 72)
    print(f"Environment ready. Re-executing {os.path.basename(script_path)} inside {py}")
    print("=" * 72 + "\n", flush=True)
    os.environ["NEPTUNA_REPRO_BOOTSTRAPPED"] = "1"
    script = os.path.abspath(script_path)
    os.execv(str(py), [str(py), script, *sys.argv[1:]])   # never returns


def ensure_sqlite3():
    """Import sqlite3 while the loader state is still clean. Call BEFORE torch.
    """
    try:
        import sqlite3  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "sqlite3 could not be imported, so optuna / bench.run will not load.\n"
            "Start from an activated environment, or export "
            f"LD_LIBRARY_PATH=<env prefix>/lib before launching.\nOriginal error: {e}")


# ---------------------------------------------------------------------------
# Repo / device / dataset helpers
# ---------------------------------------------------------------------------
def find_repo_root(start=None):
    """Locate the benchmark checkout by walking up from `start` (or the cwd)."""
    candidates = []
    if start is not None:
        candidates.append(pathlib.Path(start).resolve())
    candidates.append(pathlib.Path(os.getcwd()).resolve())
    for base in candidates:
        base = base if base.is_dir() else base.parent
        for cand in (base, *base.parents):
            if (cand / "main.py").is_file() and (cand / "bench" / "run.py").is_file():
                return cand
    raise SystemExit(
        "Could not find the benchmark repo root (a directory containing main.py and "
        f"bench/run.py) above {candidates}.\nSet REPO_ROOT manually.")


def resolve_device(preference):
    """Resolve "auto" | "cuda" | "cpu" into the device actually used.

    "auto" prefers CUDA -> CPU; naming one explicitly forces it, or fails with a
    clear message rather than silently falling back.
    """
    import torch                                    # lazy: may not exist pre-bootstrap

    available = {
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }
    if preference != "auto":
        if preference not in available:
            raise SystemExit(
                f"DEVICE_PREFERENCE = {preference!r} is not supported; "
                f"choose one of 'auto', {', '.join(repr(k) for k in available)}.")
        if not available[preference]:
            raise SystemExit(
                f"DEVICE_PREFERENCE = {preference!r}, but no usable {preference} device "
                f"was found.\n  availability: {available}\n"
                "Set DEVICE_PREFERENCE = 'auto' to fall back automatically.")
        return preference
    for candidate in ("cuda", "cpu"):                 # preference order
        if available[candidate]:
            return candidate


def report_device(device, preference, num_train_epochs_hint=True):
    """Print what was selected, plus the caveats that actually bite per backend."""
    import torch                                    # lazy

    print(f"\nselected device : {device}"
          f"{'  (auto)' if preference == 'auto' else '  (forced via DEVICE_PREFERENCE)'}")

    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB), "
              f"torch built against CUDA {torch.version.cuda}")
    else:
        print("  CPU requested." if preference == "cpu" else "  No accelerator found.")
        print("  Supported but SLOW: the reference timing is ~10 min/run on an RTX A6000.")
        if num_train_epochs_hint:
            print("  Consider NUM_TRAIN_EPOCHS = 2, and trimming COMPOSITE_WEIGHT_INITS to")
            print("  one entry, for a quick directional check.")

    if device != "cuda" and preference == "auto" and shutil.which("nvidia-smi"):
        try:
            drv = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True).stdout.strip().splitlines()
        except OSError:                               # driver present but unusable
            drv = []
        if drv:
            print(f"\n  NOTE: nvidia-smi reports GPU driver(s) {', '.join(sorted(set(drv)))}, "
                  f"but torch {torch.__version__}\n  (built against CUDA {torch.version.cuda}) "
                  "cannot use them. A CUDA-13 wheel needs driver >= 580;\n  CUDA-12 wheels "
                  "work on any 12.x driver. Install a matching torch from\n"
                  "  https://pytorch.org/get-started/locally/ .")


def describe_h5(path):
    """Summarise one HDF5 trajectory file. Opened read-only; nothing is written."""
    import h5py                                     # lazy

    with h5py.File(path, "r") as h:
        groups = sorted(h.keys())
        chans = sorted(h[groups[0]].keys())
        shape = h[groups[0]][chans[0]].shape           # (T, 1, H, W)
        dtype = h[groups[0]][chans[0]].dtype
        mach = sorted({g.split("_Mas")[1][:4] for g in groups if "_Mas" in g})
        dens = h[groups[0]]["density"][5, 0]
        return dict(groups=groups, chans=chans, shape=shape, dtype=dtype, mach=mach,
                    dens=(float(dens.min()), float(dens.max())),
                    size_gb=os.path.getsize(path) / 1e9)
