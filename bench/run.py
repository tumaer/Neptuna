import time
import os
import atexit
import socket
import platform
from transformers import TrainingArguments
from train.trainer import Trainer, compute_curriculum_start_epochs
from metrics.inference_metrics import compute_metrics_for_n_rollouts, StreamingRolloutMetrics
from metrics.loss_weighting_strategy_registry import get_loss_weighting_strategy_entry
from transformers.trainer import EvalPrediction
from utils.load_model import fetch_model
from utils.dataset_utils import make_datasets
from utils.hp_optimization import (
    compute_objective_function,
    optuna_hp_space_factory,
    get_optuna_sampler,
)
from optuna.pruners import NopPruner
from utils.custom_callbacks import (
    PlotOnEvalAndSaveCallback,
    NaNCallback,
    LossStatisticsCallback,
    AdaptiveWeightCallback,
    TrainingJsonLoggerCallback,
)
import csv
from utils.hp_optimization import trial_name_factory
from utils.loss_utils import fetch_loss_metric, create_loss_weighting_strategy
from utils.plot_progress import preprocess_for_plotting, plot_rollout_metrics
from utils.plot_progress import LayoutConfig, Slice3DConfig, create_plotter
from utils.plot_progress import build_info_strings, strip_validation_loss
from utils.plot_progress import calculate_and_save_results_all_channels
from utils.plot_progress import calculate_and_save_results_all_channels
from utils.seed_utils import set_global_seed
import psutil
from only_inference import save_errors_to_structured_csv, save_overall_errors_to_csv
import numpy as np
import torch
from bench.runner_utils import StreamingMetrics
import torch.distributed as dist
from omegaconf import ListConfig, OmegaConf
from only_inference import build_train_and_infer_loss
import glob
from utils.telemetry_log_utils import (
    RuntimeTelemetryScope,
    get_rank_world,
    aggregate_runtime_report,
    detect_runtime_backend,
    write_runtime_log,
    estimate_local_sample_count,
    now_local_iso,
)

__all__ = ["run"]

_CLEANUP_DONE = False

def _resolve_metric_for_best_model(metric_cfg) -> str | None:
    """Return the metric identifier string, preferring name over type."""

    def _from_entry(entry):
        name = entry.get("name", None)
        mtype = entry.get("type", None)
        #if name is None then returns the type otherwise returns the name
        return name or mtype  

    if isinstance(metric_cfg, (list, tuple, ListConfig)):
        if len(metric_cfg) == 0:
            raise ValueError("metric_cfg is empty")
        return _from_entry(metric_cfg[0])

    return _from_entry(metric_cfg)

def get_device_string() -> str:
    """Return a human-readable device identifier for logging."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            idx = torch.xpu.current_device()
            name = torch.xpu.get_device_name(idx)
            return f"xpu:{idx} ({name})"
        except Exception:
            return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps:0"
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        return f"cuda:{idx} ({name})"
    return "cpu"


def get_metric_device() -> torch.device:
    """Return the best available device for metrics (XPU > CUDA > CPU)."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        idx = torch.xpu.current_device()
        return torch.device(f"xpu:{idx}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")

def cleanup_distributed(rank: int) -> None:
    """Best-effort teardown so distributed/XPU jobs exit cleanly."""
    global _CLEANUP_DONE
    if _CLEANUP_DONE:
        return
    _CLEANUP_DONE = True
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
    except Exception as exc:
        print(f"[rank {rank}] torch.xpu.synchronize() failed: {exc}", flush=True)

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as exc:
        print(f"[rank {rank}] torch.cuda.synchronize() failed: {exc}", flush=True)

    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as exc:
            print(f"[rank {rank}] dist.destroy_process_group() failed: {exc}", flush=True)

    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except Exception as exc:
        print(f"[rank {rank}] torch.xpu.empty_cache() failed: {exc}", flush=True)

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        print(f"[rank {rank}] torch.cuda.empty_cache() failed: {exc}", flush=True)

def run(cfg):
    """Entry-point called by main.py after Hydra config is prepared."""
    RANK = int(os.environ.get("RANK", -1))
    IS_MAIN_PROCESS = RANK in [-1, 0]
    print(f"RANK: {RANK}")
    atexit.register(cleanup_distributed, rank=RANK)
    try:
        affinity = psutil.Process().cpu_affinity()
        CPU_CORES = len(affinity) if affinity else (psutil.cpu_count())
    except Exception:
        CPU_CORES = psutil.cpu_count()
    print(f"Detected {CPU_CORES} CPU cores")
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Global seeding
    # ------------------------------------------------------------------
    seed = int(cfg["data_config"].get("seed", 0))
    set_global_seed(seed, deterministic=True)

    # ------------------------------------------------------------------
    # Setup WANDB Project
    # ------------------------------------------------------------------
    if cfg["output_log_config"]["logging"]["wandb"] and IS_MAIN_PROCESS:
        # Set WANDB_PROJECT environment variable
        wandb_project = cfg["output_log_config"]["logging"].get(
            "wandb_project", "neptuna"
        )
        os.environ["WANDB_PROJECT"] = wandb_project
        print(f"Setting WANDB_PROJECT to: {wandb_project}")
        os.environ['WANDB_API_KEY'] = cfg["output_log_config"]["logging"]["wandb_api_key"]
        if cfg["output_log_config"]["logging"]["wandb_offline"]:
            os.environ["WANDB_MODE"] = "offline"

    use_batch_eval_metrics = cfg["train_strategy_config"].get("batch_eval_metrics", False)

    # ------------------------------------------------------------------
    # Build TrainingArguments
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        # ------------------------------------------------------------------
        # Directory & checkpointing
        # ------------------------------------------------------------------
        output_dir=cfg["output_log_config"]["logging"][
            "output_dir"
        ],  # add model name & timestamp
        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        eval_strategy=cfg["train_config"]["eval_strategy"], 
        eval_steps=cfg["train_config"]["eval_steps"],  # update-steps between evaluations, has no meaning if eval_strategy is "epoch"
        # ------------------------------------------------------------------
        # Testing 
        # ------------------------------------------------------------------
        load_best_model_at_end=True, # Ensure that the save & eval strategy align
        # ------------------------------------------------------------------
        # Batching
        # ------------------------------------------------------------------
        per_device_train_batch_size=cfg["train_config"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["train_config"]["per_device_eval_batch_size"],
        # accumulate predictions on GPU before moving to CPU
        eval_accumulation_steps=cfg["train_config"]["eval_accumulation_steps"],  
        # ------------------------------------------------------------------
        # Optimiser / schedule
        # ------------------------------------------------------------------
        max_grad_norm=1.0,  # gradient clipping (default is 1.0)
        num_train_epochs=cfg["train_strategy_config"]["num_train_epochs"],
        learning_rate=cfg["scheduler_config"]["lr"],
        weight_decay=cfg["scheduler_config"]["weight_decay"],
        optim=cfg["scheduler_config"]["optim"], 
        lr_scheduler_type=cfg["scheduler_config"]["lr_scheduler"],
        warmup_ratio=cfg["scheduler_config"]["warmup_ratio"],  # linear warm-up fraction
        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        # debug / info / warning / ...
        log_level=cfg["train_config"].get("log_level", "info"),  
        logging_strategy=cfg["train_config"]["logging_strategy"],  # switch to "epoch" later if needed
        logging_steps=cfg["train_config"]["logging_steps"],  # only used if logging_strategy is "steps"
        logging_nan_inf_filter=False,  # include NaNs in logs for debugging
        # ------------------------------------------------------------------
        # Saving
        # ------------------------------------------------------------------
        save_strategy=cfg["train_config"]["save_strategy"],  # switch to "epoch" once validation present
        save_steps=cfg["train_config"]["save_steps"],  # only used if save_strategy is "steps"
        save_total_limit=cfg["train_config"]["save_total_limit"],  # keep only last N checkpoints
        push_to_hub=cfg["train_config"]["push_to_hub"],  # push to Hugging Face Hub, requires login before (run `huggingface-cli login` in terminal)
        hub_strategy=cfg["train_config"]["hub_strategy"],  # push last checkpoint to Hub (alternatives: "end", "every_save", "checkpoint", "all_checkpoints")
        # ------------------------------------------------------------------
        # Reproducibility
        # ------------------------------------------------------------------
        seed=seed,  # model-seed
        data_seed=seed,  # sampler-seed for SeedableRandomSampler
        # ------------------------------------------------------------------
        # Mixed precision training
        # ------------------------------------------------------------------
        fp16=cfg["train_config"]["mix_precision_config"]["fp16"], 
        bf16=cfg["train_config"]["mix_precision_config"]["bf16"],  
        tf32=cfg["train_config"]["mix_precision_config"]["tf32"],
        # ------------------------------------------------------------------
        # Misc runtime knobs 
        # ------------------------------------------------------------------    
        dataloader_num_workers=cfg["train_config"]["dataloader_num_workers"],
        metric_for_best_model=_resolve_metric_for_best_model(  # best checkpoint metric string
            cfg["train_strategy_config"]["metric_for_best_model"]
        ),
        include_for_metrics=[
            "inputs",
        ]
        + (
            ["conditioning_inputs"]
            if cfg["data_config"]["conditioning_features"]["conditioning_in_channels"] is not None
            else []
        ),  # keep inputs and optionally conditioning_inputs for plotting
        greater_is_better=False,  # lower loss/error is better
        dataloader_pin_memory=True,
        #dataloader_persistent_workers=True,
        #dataloader_prefetch_factor=2,
        gradient_accumulation_steps=cfg["train_config"]["gradient_accumulation_steps"],
        gradient_checkpointing=False,  # save memory, slower back-prop
        auto_find_batch_size=False,
        full_determinism=False,  # turn on for reproducible distributed training
        torch_compile=False,
        use_cpu=cfg["train_config"]["use_cpu"],  # use_cpu even if other devices are present
        label_names=["label_including_rollouts"],
        disable_tqdm=False,
        # ------------------------------------------------------------------
        # Reporting 
        # ------------------------------------------------------------------
        report_to=(
            "wandb"
            if (
                cfg["output_log_config"]["logging"]["wandb"]
                and cfg["hyperparam_opt_config"]["optimize"] is False
            )
            else "none"
        ),
        run_name=(
            os.path.basename(
                os.path.normpath(cfg["output_log_config"]["logging"]["output_dir"])
            )
            if (cfg["hyperparam_opt_config"]["optimize"] is False)
            else "none"
        ),
        ddp_find_unused_parameters=False,
        batch_eval_metrics=use_batch_eval_metrics,
        dataloader_drop_last=False,
    )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_ds, eval_ds = make_datasets(cfg, mode="train")

    # ------------------------------------------------------------------
    # Model & Trainer 
    # ------------------------------------------------------------------
    if cfg["hyperparam_opt_config"]["optimize"] is False:
        model = fetch_model(cfg["model_config"], cfg["data_config"])
    else:
        model = None

    def model_init():
        return fetch_model(cfg["model_config"], cfg["data_config"])

    def compute_metrics(eval_pred: EvalPrediction):
        preds = eval_pred.predictions
        (
            len_eval_dataloader,
            num_eval_rollouts,
            label_seq_length,
            channel_dim,
            *spatial,
        ) = preds.shape

        # Flatten rollouts into time dimension
        preds = preds.reshape(
            len_eval_dataloader,
            num_eval_rollouts * label_seq_length,
            channel_dim,
            *spatial,
        )
        #print(f"[rank {RANK}] preds shape: {preds.shape}")
        targets = eval_pred.label_ids

        metrics: dict[str, float] = {}

        device = torch.device("cpu") #getattr(trainer, "metric_device", torch.device("cpu"))

        if isinstance(preds, np.ndarray):
            preds_tensor = torch.from_numpy(preds).float()
        else:
            preds_tensor = (
                preds.detach().cpu()
                if torch.is_tensor(preds)
                else torch.tensor(preds, dtype=torch.float32)
            )

        if isinstance(targets, np.ndarray):
            targets_tensor = torch.from_numpy(targets).float()
        else:
            targets_tensor = (
                targets.detach().cpu()
                if torch.is_tensor(targets)
                else torch.tensor(targets, dtype=torch.float32)
            )

        # 1. Create an additional composite_train_loss metric using the metrics from train_loss block.
        # "Can" be used for checkpointing.
        if getattr(trainer, "loss_fn", None) is not None:
            try:
                with torch.no_grad():
                    train_loss_fn = trainer.loss_fn.to(device)
                    composite_train_loss = train_loss_fn(
                        model=None,
                        predictions=preds_tensor.to(device),
                        labels=targets_tensor.to(device),
                        input_frames=None,
                        return_detailed=False,  # scalar only
                    )

                if torch.is_tensor(composite_train_loss):
                    metrics["composite_train_loss"] = composite_train_loss.item()
                else:
                    metrics["composite_train_loss"] = float(composite_train_loss)

            except Exception as e:
                print("[compute_metrics] composite_train_loss failed:", repr(e))

        # 2. Evaluation loss metrics (for logging), cached on trainer
        eval_loss_fn = getattr(trainer, "eval_loss_fn", None)
        if eval_loss_fn is not None:
            try:
                with torch.no_grad():
                    eval_loss_fn = eval_loss_fn.to(device)
                    _, detailed = eval_loss_fn(
                        model=None,
                        predictions=preds_tensor.to(device),
                        labels=targets_tensor.to(device),
                        input_frames=None,
                        return_detailed=True,
                    )

                for component_name, component_detailed in detailed.items():
                    component_total = component_detailed["total"]
                    if torch.is_tensor(component_total):
                        metrics[component_name] = component_total.item()
                    else:
                        metrics[component_name] = float(component_total)

            except Exception as e:
                print("[compute_metrics] eval_loss_fn failed:", repr(e))
        #print(f"[rank {RANK}] metrics: {metrics}")
        return metrics

    # Curriculum start epochs (shared helper from Trainer module)
    curriculum_start_epochs = compute_curriculum_start_epochs(cfg["train_strategy_config"])

    # Build the callbacks list
    callbacks = []
    # Always add PlotOnEvalAndSaveCallback and NaNCallback
    callbacks.append(PlotOnEvalAndSaveCallback)
    callbacks.append(NaNCallback)
    callbacks.append(TrainingJsonLoggerCallback)

    initial_train_strategy_dict = cfg.train_strategy_config.curriculum[0]
    initial_train_loss_dict = initial_train_strategy_dict.train_loss

    #train_loss_dict = fetch_train_loss_dict(cfg)
    initial_loss_weighting_strategy = create_loss_weighting_strategy(initial_train_loss_dict)
    if initial_loss_weighting_strategy is not None:
        
        grad_stats = get_loss_weighting_strategy_entry(initial_train_loss_dict.train_loss_weighting_strategy.type).get("grad_stats", [])
        use_gradients = bool(grad_stats)
        # Create statistics collector
        initial_loss_stats_callback = LossStatisticsCallback(collect_train_losses=True, grad_stats=grad_stats, collect_gradients=use_gradients)
        callbacks.append(initial_loss_stats_callback)
        
        # Create adaptive weight callback
        loss_source = initial_train_loss_dict.train_loss_weighting_strategy.get('loss_source', 'train')
        initial_weight_callback = AdaptiveWeightCallback(
            initial_loss_weighting_strategy, 
            initial_loss_stats_callback,
            loss_source = loss_source,
            use_gradients = use_gradients,
            curriculum_start_epochs=curriculum_start_epochs,
            grad_stats = grad_stats,
        )
        callbacks.append(initial_weight_callback)
    else:
        initial_weight_callback = None

    trainer = Trainer(
        model_config=cfg["model_config"],
        data_config=cfg["data_config"],
        train_config=cfg["train_config"],
        scheduler_config=cfg["scheduler_config"],
        infer_config=cfg["infer_config"],
        output_log_config=cfg["output_log_config"],
        train_strategy_config=cfg["train_strategy_config"],
        # all kwargs below go directly to the base trainer class of HF
        model=model,
        model_init=model_init if cfg["hyperparam_opt_config"]["optimize"] else None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        #compute_metrics=trainer_compute_metrics,
        callbacks=callbacks if callbacks else None,
    )

    streaming_metrics = StreamingMetrics(trainer=trainer, mode="eval")
    trainer.compute_metrics = streaming_metrics if use_batch_eval_metrics else compute_metrics
    
    trainer.set_eval_or_test_rollout_steps(
        rollout_steps=cfg["train_config"]["n_eval_rollouts"], output_all_steps=True
    )

    # Initialize eval_loss_fn
    metric_device = get_metric_device()
    try:
        initial_eval_loss_dict = initial_train_strategy_dict.validation_loss
        #eval_loss_dict = fetch_eval_loss_dict(cfg)
        eval_loss_fn = trainer.get_loss_fn(initial_eval_loss_dict)
        trainer.eval_loss_fn = eval_loss_fn.to(metric_device)
    except Exception as e:
        print("[run] Failed to initialize eval_loss_fn:", repr(e))
        trainer.eval_loss_fn = None

    trainer.metric_device = metric_device

    if initial_weight_callback is not None:
        initial_weight_callback.trainer = trainer
        initial_loss_stats_callback.trainer = trainer
        # Enable detailed loss collection in trainer
        trainer._collect_detailed_losses = initial_train_loss_dict.train_loss_weighting_strategy.get(
            'enabled', False,
        )
        trainer._last_detailed_losses = {}

    # ------------------------------------------------------------------
    # Train vs HP-search
    # ------------------------------------------------------------------
    if cfg["hyperparam_opt_config"]["optimize"] is False:
        start = time.time()
        device_str = get_device_string()
        print(f"Training on device {device_str} \n", flush=True)
        #trainer.train(resume_from_checkpoint=f"/Users/harish/Desktop/cfd_bench/checkpoints/KuramotoSivashinsky_2D_ScOT_02032026_083134/checkpoint-15")
        trainer.train(resume_from_checkpoint=False)
        #print(f"Total train time: {time.time() - start:.2f} s")

        print(f"All done with training, rank: {RANK} \n", flush=True)
        if training_args.push_to_hub and IS_MAIN_PROCESS:
            print("Pushing model to Hugging Face Hub...")
            trainer.push_to_hub()
        
        # ------------------------------------------------------------------
        # Inference (Continued after training)
        # ------------------------------------------------------------------
        
        if cfg["infer_config"]["do_infer"]:
            print("Running inference...")

            def _extract_rollout_metrics(metrics_dict: dict):
                out = {}
                if not isinstance(metrics_dict, dict):
                    return out
                nested = metrics_dict.get("rollout_metrics")
                if isinstance(nested, dict):
                    for k, v in nested.items():
                        if isinstance(v, dict) and "per_rollout_step_mean" in v:
                            out[k] = v
                    if out:
                        return out
                # Backward-compatible fallback
                for k, v in metrics_dict.items():
                    if isinstance(v, dict) and "per_rollout_step_mean" in v:
                        out[k] = v
                return out

            def _find_checkpoint_path(base_dir: str) -> str | None:
                ckpts = glob.glob(os.path.join(base_dir, "checkpoint-*"))
                if not ckpts:
                    return None
                ckpts.sort(key=lambda p: int(os.path.basename(p).split("-")[-1]))
                return ckpts[-1]

            checkpoint_dir = trainer.state.best_model_checkpoint or _find_checkpoint_path(
                cfg["output_log_config"]["logging"]["output_dir"]
            )
            if checkpoint_dir is None:
                raise FileNotFoundError("No checkpoint-* found for inference.")

            checkpoint_parent_dir = os.path.dirname(checkpoint_dir)

            # loss_config from checkpoint
            loss_config_path = os.path.join(checkpoint_dir, "loss_config.json")
            loss_config_ckpt = OmegaConf.load(loss_config_path) if os.path.exists(loss_config_path) else None

            loss_config_for_plotting = strip_validation_loss(loss_config_ckpt)

            for component in loss_config_ckpt.train_loss.components:
                if 'current_weights' in component:
                    # Extract current weights
                    current_weights = component.current_weights
                    # Update the component's weight configuration
                    if 'base_weight' in current_weights:
                        component.weight = current_weights.base_weight
                    if 'timestep_weights' in current_weights:
                        component.timestep_weights = current_weights.timestep_weights
                    if 'channel_weights' in current_weights:
                        component.channel_weights = current_weights.channel_weights
                    if 'component_weights' in current_weights:
                        component.component_weights = current_weights.component_weights
                    
                    print(f" Using checkpoint weights for '{component.get('name', component.type)}'")

            # data_config from checkpoint (fallback to in-memory cfg)
            data_config_path = os.path.join(checkpoint_dir, "data_config.json")
            data_config_ckpt = OmegaConf.load(data_config_path) if os.path.exists(data_config_path) else cfg["data_config"]

            # Determine metric device from infer_config.metrics_device
            _dev_cfg = cfg["infer_config"].get("metrics_device") if cfg["infer_config"] else None
            if _dev_cfg is not None and str(_dev_cfg).strip():
                metric_device = torch.device(str(_dev_cfg).strip())
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                metric_device = torch.device("xpu:0")
            elif torch.cuda.is_available():
                metric_device = torch.device("cuda:0")
            else:
                metric_device = torch.device("cpu")
            
             # Build loss functions
            train_loss_fn_inf, infer_loss_fn, _ = build_train_and_infer_loss(
                loss_config=loss_config_ckpt,
                data_config=data_config_ckpt,
                device=metric_device,
            )

            if infer_loss_fn is not None:
                trainer.infer_loss_fn = infer_loss_fn  #this is taken from train_strategy_config/infer_loss.yaml
            trainer.train_loss_fn = train_loss_fn_inf
            
            # Override compute_metrics for inference to align with only_inference.py
            def compute_metrics_during_inference(eval_pred: EvalPrediction):
                preds = eval_pred.predictions
                (
                    len_eval_dataloader,
                    num_eval_rollouts,
                    label_seq_length,
                    channel_dim,
                    *spatial,
                ) = preds.shape

                preds = preds.reshape(
                    len_eval_dataloader,
                    num_eval_rollouts * label_seq_length,
                    channel_dim,
                    *spatial,
                )
                targets = eval_pred.label_ids

                metrics = {}

                if isinstance(preds, np.ndarray):
                    preds_tensor = torch.from_numpy(preds).float()
                else:
                    preds_tensor = (
                        preds.detach().cpu()
                        if torch.is_tensor(preds)
                        else torch.tensor(preds, dtype=torch.float32)
                    )

                if isinstance(targets, np.ndarray):
                    targets_tensor = torch.from_numpy(targets).float()
                else:
                    targets_tensor = (
                        targets.detach().cpu()
                        if torch.is_tensor(targets)
                        else torch.tensor(targets, dtype=torch.float32)
                    )

                # 1) Training (composite) loss for logging/checkpointing
                if train_loss_fn_inf is not None:
                    try:
                        with torch.no_grad():
                            composite_loss = train_loss_fn_inf(
                                model=None,
                                predictions=preds_tensor.to(metric_device),
                                labels=targets_tensor.to(metric_device),
                                return_detailed=False,
                            )
                        metrics["infer_composite_train_loss"] = float(
                            composite_loss.item()
                            if torch.is_tensor(composite_loss)
                            else composite_loss
                        )
                    except Exception as e:
                        print(f"Failed to compute composite loss metrics: {e}")

                # 2) Evaluation loss components for logging
                if infer_loss_fn is not None:
                    try:
                        with torch.no_grad():
                            _, detailed = infer_loss_fn(
                                model=None, 
                                predictions=preds_tensor.to(metric_device),
                                labels=targets_tensor.to(metric_device),
                                return_detailed=True,
                            )
                        for component_name, component_detailed in detailed.items():
                            component_total = component_detailed["total"]
                            metrics[f"infer_{component_name}"] = (
                                component_total.item()
                                if torch.is_tensor(component_total)
                                else component_total
                            )
                    except Exception as e:
                        print(f"Failed to compute evaluation loss metrics: {e}")

                return metrics

            # StreamingMetrics for inference (rollout + overall metrics accumulated on GPU)
            streaming_metrics = StreamingMetrics(trainer=trainer, mode="infer")
            use_batch_eval_metrics = cfg["infer_config"].get("batch_eval_metrics", True)
            trainer.args.batch_eval_metrics = use_batch_eval_metrics
            trainer.compute_metrics = StreamingRolloutMetrics(
                loss_metric=infer_loss_fn,
                include_per_timestep=True,
                device=cfg["infer_config"].get("metrics_device"),
                base_metrics=streaming_metrics,
            ) if cfg["infer_config"].get("batch_eval_metrics", True) else compute_metrics_during_inference
            
            direct_inference_dir = os.path.join(checkpoint_parent_dir, "direct_inference")
            inference_dir = os.path.join(direct_inference_dir, "inference_plots")

            os.makedirs(inference_dir, exist_ok=True)

            # Initialize runtime telemetry for inference_runtime_log.json
            runtime_sample_interval = float(cfg["infer_config"].get("runtime_log_sample_interval_sec", 1.0))
            runtime_log_sections: dict[str, dict] = {}
            runtime_local_samples_total = 0
            runtime_global_samples_total = 0
            overall_runtime_scope = RuntimeTelemetryScope(
                name="overall_inference",
                sample_interval_sec=runtime_sample_interval,
            )
            overall_runtime_scope.start()
            overall_runtime_start_wall = now_local_iso()

            infer_ds, infer_ds_from_ic = make_datasets(cfg, mode="infer")

            if cfg["infer_config"]["infer_from_random_timestep"] and infer_ds is not None:
                runtime_global_samples_total += len(infer_ds)
            if cfg["infer_config"]["infer_from_ic"] and infer_ds_from_ic is not None:
                runtime_global_samples_total += len(infer_ds_from_ic)
            
            random_start_stats_dict = None
            ic_start_stats_dict = None
            
            if cfg["infer_config"]["infer_from_random_timestep"]:
                print(" \n Running inference rollouts using random windows sliced across the test trajectory...")
                trainer.set_eval_or_test_rollout_steps(
                    rollout_steps=cfg["infer_config"]["n_infer_rollouts"], output_all_steps=True
                )

                with RuntimeTelemetryScope(
                    name="eval_loop_random_start",
                    sample_interval_sec=runtime_sample_interval,
                ) as eval_scope_random:
                    predictions_obj, inputs, conditioning_inputs = trainer.predict(infer_ds, metric_key_prefix="")

                random_local_samples = estimate_local_sample_count(predictions_obj, len(infer_ds))
                runtime_local_samples_total += int(random_local_samples)
                runtime_log_sections["eval_loop_random_start"] = aggregate_runtime_report(
                    eval_scope_random.build_local_report(local_samples=random_local_samples),
                    global_samples=len(infer_ds),
                )
                ############################################################
                # predictions_obj.predictions: the output of the model with shape (accumulated_outputs, num_rollouts, label_seq_length, channel_dim, *spatial) 
                # accumulated_outputs and accumulated_gt have the length of number of windows in the test dataset
                # predictions_obj.label_ids: the ground truth with shape (accumulated_gt, num_rollouts*label_seq_length, channel_dim, *spatial)
                # predictions_obj.metrics: the metrics computed after accumulating the outputs and ground truth
                ############################################################

                # pretty print the keys which have the word error in them
                print('Accumulated error for the whole test set (random start):')
                overall_errors = {} 
                for key, value in predictions_obj.metrics.items():
                    # omit throughput-style metrics
                    if key.endswith(("runtime", "samples_per_second", "steps_per_second")):
                        continue
                    if isinstance(value, dict):
                        continue
                    print(f"{key}: {value}")
                    overall_errors["random_start_"+key] = value
                save_overall_errors_to_csv(overall_errors, direct_inference_dir)
                # ----------------------------------------------------------
                # Prepare prediction, target and input arrays
                # ----------------------------------------------------------
                preds = predictions_obj.predictions  # (N, R, T, C, *spatial)

                # Flatten rollout and label sequence dimensions if necessary
                if preds.ndim >= 5:
                    n, n_rollouts, seq_len, c = preds.shape[:4]
                    outputs_per_rollout = seq_len
                    extra_dims = preds.shape[4:]
                    preds = preds.reshape(n, n_rollouts * seq_len, c, *extra_dims) # (N, R*T, C, *spatial)

                targets = predictions_obj.label_ids  # Expected shape: (N, R*T, C, *spatial)
                
                per_rollout_step_metrics_random = _extract_rollout_metrics(predictions_obj.metrics)
                if not per_rollout_step_metrics_random:
                    raise RuntimeError(
                        "Streaming rollout metrics were not returned from Trainer.inference_loop. "
                        "Please verify batch_eval_metrics=True and StreamingRolloutMetrics wiring."
                    )

                # Inputs already returned by `trainer.predict`
                inp_arr = inputs  # Shape: (N, T_in, C_in, *spatial)

                # Conditioning inputs may be None
                cond_inp_arr = conditioning_inputs if conditioning_inputs is not None else None

                # ----------------------------------------------------------
                # Renormalise data and reconstruct residuals for plotting
                # ----------------------------------------------------------
                (inp_renorm,
                    tgt_renorm,
                    pred_renorm,
                    only_input_channel_names,
                    output_channel_names,
                    cond_inp_renorm,
                    cond_inp_channel_names) = preprocess_for_plotting(
                    inputs=inp_arr,
                    labels=targets,
                    predictions=preds,
                    data_config=cfg["data_config"],
                    dataset=infer_ds,
                    residual_config=cfg["data_config"].get("residual_config", None),
                    conditioning_inputs=cond_inp_arr,
                )

                per_step_errors = {}
                for metric_name, values in per_rollout_step_metrics_random.items():
                    #print(f"{metric_name} per-step (random start): {values}")
                    per_step_errors[metric_name] = values
                save_errors_to_structured_csv(per_step_errors, 
                                              direct_inference_dir, 
                                              channel_names=output_channel_names, 
                                              file_name="results_structured_random_start.csv")

                # Infer spatial dimensionality (1D / 2D / 3D)
                ndim = pred_renorm.ndim - 3  # subtract batch, time, channel dims

                stride_val = cfg["data_config"].get("sequence_info")[2]

                plot_save_dir = os.path.join(inference_dir, "random_start")

                model_info_str, data_info_str, train_info_str, _ = build_info_strings(model_obj=trainer.model, 
                                                                                                    data_config=cfg["data_config"],
                                                                                                    model_config=cfg["model_config"],
                                                                                                    train_config=cfg["train_config"],
                                                                                                    scheduler_config=cfg["scheduler_config"]
                                                                        )
                layout_config = LayoutConfig(
                    base_visual_size=3.5,
                    margin_between_plots_h=0.65,
                    margin_between_plots_v=0.65
                )

                slice_config = Slice3DConfig(
                    slice_axis=0,
                    num_slices=4
                )

                plotter = create_plotter(
                    orientation=cfg["infer_config"].get("plot_orientation", "vertical"),
                    input_array=inp_renorm,
                    prediction_array=pred_renorm,
                    target_array=tgt_renorm,
                    input_channel_names=only_input_channel_names,
                    output_channel_names=output_channel_names,
                    conditioning_input_array=cond_inp_renorm,
                    conditioning_channel_names=cond_inp_channel_names,
                    checkpoint_step=None,
                    epoch=None,
                    extra_info=cfg["data_config"].get("dataset_name")+"_Inference_plot_from_random_timestep",
                    ndim=ndim,
                    slice_config=slice_config,
                    num_examples=cfg["infer_config"]["n_infer_plot_examples"],
                    stride=stride_val,
                    save_dir=plot_save_dir,
                    log_to_wandb=False,
                    best_plot_at_train_end=False,
                    layout_config=layout_config,
                    include_relative_error=True,
                    model_info=model_info_str,
                    data_info=data_info_str,
                    train_info=train_info_str,
                    loss_config=loss_config_for_plotting
                )
                
                plotter.plot()

                # Create only rollout metrics (and not timestep metrics) plots as windows are sliced across time steps
                ex_title = f"Per-rollout metrics ({cfg['data_config'].get('dataset_name', 'dataset')} - random start)"
                plot_rollout_metrics(
                    step_metrics=per_rollout_step_metrics_random,
                    output_channel_names=output_channel_names,
                    save_dir=plot_save_dir,
                    mode="random_start",
                    title=ex_title,
                    filename="Metric_evolution_for_random_start.png",
                    sequence_info=cfg["data_config"].get("sequence_info"),
                    num_examples=pred_renorm.shape[0],
                )
                
                random_start_stats_dict = {
                    "metrics": per_rollout_step_metrics_random,
                    "sequence_info": list(cfg["data_config"].get("sequence_info")),
                    "output_channel_names": output_channel_names,
                }
                runs_step_metrics = {}
                run_label = os.path.basename(checkpoint_parent_dir)
                runs_step_metrics[run_label] = random_start_stats_dict["metrics"]

                calculate_and_save_results_all_channels(
                    runs_step_metrics=runs_step_metrics,
                    save_dir=os.path.dirname(direct_inference_dir),
                    output_channel_names=output_channel_names,
                    filename="rollout_metrics_random_start_tabulated.csv"
                )
                
            if cfg["infer_config"]["infer_from_ic"]:
                print(" \n Running inference rollouts using windows starting from the initial conditions...")
                trainer.set_eval_or_test_rollout_steps(
                    rollout_steps=cfg["infer_config"]["n_infer_rollouts"], output_all_steps=True
                )
                # ----------------------------------------------------------
                # Prepare prediction, target and input arrays
                # ----------------------------------------------------------
                with RuntimeTelemetryScope(
                    name="eval_loop_ic_start",
                    sample_interval_sec=runtime_sample_interval,
                ) as eval_scope_ic:
                    predictions_obj, inputs, conditioning_inputs = trainer.predict(infer_ds_from_ic, metric_key_prefix="")

                ic_local_samples = estimate_local_sample_count(predictions_obj, len(infer_ds_from_ic))
                runtime_local_samples_total += int(ic_local_samples)
                runtime_log_sections["eval_loop_ic_start"] = aggregate_runtime_report(
                    eval_scope_ic.build_local_report(local_samples=ic_local_samples),
                    global_samples=len(infer_ds_from_ic),
                )

                print('Accumulated error for the whole test set (IC start):')
                errors = {}
                for key, value in predictions_obj.metrics.items():
                    if key.endswith(("runtime", "samples_per_second", "steps_per_second")):
                        continue
                    if isinstance(value, dict):
                        continue
                    print(f"{key}: {value}")
                    errors["ic_start_"+key] = value
                save_overall_errors_to_csv(errors, direct_inference_dir)

                preds = predictions_obj.predictions
                targets = predictions_obj.label_ids
                inp_arr = inputs
                cond_inp_arr = conditioning_inputs if conditioning_inputs is not None else None

                # Determine outputs per rollout (T_out) before flattening, default to 1
                if preds.ndim >= 5:
                    n, n_rollouts, seq_len, c = preds.shape[:4]
                    outputs_per_rollout = seq_len
                    extra_dims = preds.shape[4:]
                    preds = preds.reshape(n, n_rollouts * seq_len, c, *extra_dims)

                # Read per-rollout errors from streaming metrics computed inside inference_loop
                per_rollout_step_metrics_ic = _extract_rollout_metrics(predictions_obj.metrics)
                if not per_rollout_step_metrics_ic:
                    raise RuntimeError(
                        "Streaming rollout metrics were not returned from Trainer.inference_loop. "
                        "Please verify batch_eval_metrics=True and StreamingRolloutMetrics wiring."
                    )

                # ----------------------------------------------------------
                # Renormalise data and reconstruct residuals for plotting
                # ----------------------------------------------------------
                (inp_renorm,
                    tgt_renorm,
                    pred_renorm,
                    only_input_channel_names,
                    output_channel_names,
                    cond_inp_renorm,
                    cond_inp_channel_names) = preprocess_for_plotting(
                    inputs=inp_arr,
                    labels=targets,
                    predictions=preds,
                    data_config=cfg["data_config"],
                    dataset=infer_ds,
                    residual_config=cfg["data_config"].get("residual_config", None),
                    conditioning_inputs=cond_inp_arr,
                )

                errors = {}
                for metric_name, values in per_rollout_step_metrics_ic.items():
                    #print(f"{metric_name} per-step (IC start): {values}")
                    errors[metric_name] = values
                save_errors_to_structured_csv(errors, direct_inference_dir, channel_names=output_channel_names, file_name="results_structured_ic_start.csv")
                
                # Infer spatial dimensionality (1D / 2D / 3D)
                ndim = pred_renorm.ndim - 3  # subtract batch, time, channel dims

                # Use stride from the config if available
                stride_val = cfg["data_config"].get("sequence_info")[2]

                # Directory for saving inference plots (IC start)
                plot_save_dir = os.path.join(inference_dir, "ic_start")

                # Plot rollout metrics (per metric subplot)
                plot_rollout_metrics(
                    step_metrics=per_rollout_step_metrics_ic,
                    output_channel_names=output_channel_names,
                    save_dir=plot_save_dir,
                    mode="ic_start",
                    title=f"Per-(rollout and time) step metrics ({cfg['data_config'].get('dataset_name', 'dataset')} - IC start)",
                    filename="Metric_evolution_for_ic_start.png",
                    sequence_info=cfg["data_config"].get("sequence_info"),
                    num_examples=pred_renorm.shape[0],
                )

                model_info_str, data_info_str, train_info_str, _ = build_info_strings(
                                                                                        model_obj=trainer.model,
                                                                                        data_config=cfg["data_config"],
                                                                                        model_config=cfg["model_config"],
                                                                                        train_config=cfg["train_config"],
                                                                                        scheduler_config=cfg["scheduler_config"]
                                                                                    )

                # Create rollout sample plots, these plots start from the initial condition in the test dataset

                layout_config = LayoutConfig(
                    base_visual_size=3.5,
                    margin_between_plots_h=0.65,
                    margin_between_plots_v=0.65
                )

                slice_config = Slice3DConfig(
                    slice_axis=0,
                    num_slices=4
                )

                plotter = create_plotter(
                    orientation=cfg["infer_config"].get("plot_orientation", "vertical"),
                    input_array=inp_renorm,
                    prediction_array=pred_renorm,
                    target_array=tgt_renorm,
                    input_channel_names=only_input_channel_names,
                    output_channel_names=output_channel_names,
                    conditioning_input_array=cond_inp_renorm,
                    conditioning_channel_names=cond_inp_channel_names,
                    checkpoint_step=None,
                    epoch=None,
                    extra_info=cfg["data_config"].get("dataset_name")+"_Inference_plot_from_IC",
                    ndim=ndim,
                    slice_config=slice_config,
                    num_examples=cfg["infer_config"]["n_infer_plot_examples"],
                    stride=stride_val,
                    save_dir=plot_save_dir,
                    log_to_wandb=False,
                    best_plot_at_train_end=False,
                    layout_config=layout_config,
                    include_relative_error=True,
                    model_info=model_info_str,
                    data_info=data_info_str,
                    train_info=train_info_str,
                    loss_config=loss_config_for_plotting
                )
                
                plotter.plot()

                ic_start_stats_dict = { #TODO: Need to check if this is correct
                    "metrics": per_rollout_step_metrics_ic,
                    "sequence_info": list(cfg["data_config"].get("sequence_info")),
                    "output_channel_names": output_channel_names,
                }
                runs_step_metrics = {}
                run_label = os.path.basename(checkpoint_parent_dir)
                runs_step_metrics[run_label] = ic_start_stats_dict["metrics"]

                calculate_and_save_results_all_channels(
                    runs_step_metrics=runs_step_metrics,
                    save_dir=os.path.dirname(direct_inference_dir),
                    output_channel_names=output_channel_names,
                    filename="rollout_metrics_ic_start_tabulated.csv"
                )

                print(f"Inference completed for {checkpoint_parent_dir} ")
                print(f"Results saved to: {direct_inference_dir}")

            overall_runtime_scope.stop()
            runtime_log_sections["overall_inference"] = aggregate_runtime_report(
                overall_runtime_scope.build_local_report(local_samples=runtime_local_samples_total),
                global_samples=runtime_global_samples_total if runtime_global_samples_total > 0 else None,
            )
            _rank_now, _world_now = get_rank_world()

            runtime_log_payload = {
                "generated_local": now_local_iso(),
                "overall_runtime_start_local": overall_runtime_start_wall,
                "checkpoint_dir": checkpoint_dir,
                "direct_inference_dir": direct_inference_dir,
                "host": {
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                },
                "accelerator_backend": detect_runtime_backend(),
                "distributed": {
                    "initialized": bool(dist.is_available() and dist.is_initialized()),
                    "rank": int(_rank_now),
                    "world_size": int(_world_now),
                },
                "sections": runtime_log_sections,
            }

            if IS_MAIN_PROCESS:
                runtime_log_path = os.path.join(direct_inference_dir, "inference_runtime_log.json")
                write_runtime_log(runtime_log_path, runtime_log_payload)
                print(f"Runtime log saved to: {runtime_log_path}")
            
    else:
        # get the sampler from the config, it could be GridSampler, RandomSampler, TPESampler etc.
        sampler = get_optuna_sampler(
            cfg["hyperparam_opt_config"]["optuna_sampler"], config=cfg
        )

        best_trial, study = trainer.hyperparameter_search(
            direction="minimize",
            backend="optuna",
            hp_space=optuna_hp_space_factory(cfg),
            n_trials=cfg["hyperparam_opt_config"]["n_trials"],
            hp_name=trial_name_factory(cfg["data_config"]),
            compute_objective=compute_objective_function(
                selected_metrics=cfg["hyperparam_opt_config"]["metric_for_tuning_hp"]
            ),
            sampler=sampler,
            pruner=NopPruner(),
        )

        # --------------------------------------------------------------
        # Save HPO results to CSV
        # --------------------------------------------------------------
        if IS_MAIN_PROCESS:
            results_dir = cfg["output_log_config"]["logging"]["output_dir"]
            os.makedirs(results_dir, exist_ok=True)
            csv_path = os.path.join(results_dir, "hp_search_results.csv")

            # Collect all parameter names across trials to build consistent header
            param_keys = set()
            for t in study.trials:
                param_keys.update(t.params.keys())
            fieldnames = ["trial_number", "value", "state"] + sorted(param_keys)

            with open(csv_path, mode="w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for t in study.trials:
                    row = {
                        "trial_number": t.number,
                        "value": t.value,
                        "state": t.state.name,
                    }
                    row.update(t.params)
                    writer.writerow(row)
            print(f"Saved hyperparameter optimisation results to {csv_path}")
