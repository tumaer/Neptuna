"""
Custom Training Callbacks for CFD Model Training with Enhanced Logging and Visualization.

This module provides specialized callback implementations for training models using the Hugging Face
Transformers library. It extends the standard callback system with advanced Weights & Biases (W&B)
integration, custom plotting capabilities, and enhanced evaluation workflows.

Key Features:
- Enhanced W&B callback with trial parameter logging and model configuration tracking
- Custom evaluation callback handler supporting additional keyword arguments
- Automated plotting and visualization during training with configurable epoch thresholds
- Renormalization of predictions and labels for accurate visualization
- Support for residual learning with proper value reconstruction
- Memory-efficient plotting with configurable example limits
- Integration with custom trainer control flow for plotting events

Main Components:
    CallbackHandler: Extended callback handler with custom evaluation and plotting events
    WandbCallback: Enhanced W&B integration with trial tracking and model configuration
    PlotOnEvalAndSaveCallback: Automated plotting callback with data renormalization
    TrainerControl: Extended trainer control with plotting state management

Callback Event Flow:
    1. on_evaluate: Triggered after model evaluation, stores predictions and metadata
    2. on_plot: Custom event for generating and logging visualizations
    3. on_log: Enhanced logging with W&B integration and plot artifact support
    4. on_train_end: Cleanup and finalization of logging sessions

Example Usage:
    >>> from utils.custom_callbacks import WandbCallback, PlotOnEvalAndSaveCallback
    >>> from transformers import Trainer
    >>> 
    >>> # Initialize callbacks
    >>> wandb_callback = WandbCallback()
    >>> plot_callback = PlotOnEvalAndSaveCallback()
    >>> 
    >>> # Create trainer with custom callbacks
    >>> trainer = Trainer(
    ...     model=model,
    ...     args=training_args,
    ...     train_dataset=train_dataset,
    ...     eval_dataset=eval_dataset,
    ...     callbacks=[wandb_callback, plot_callback]
    ... )
    >>> 
    >>> # Training will automatically trigger plotting and logging
    >>> trainer.train()

Configuration Options:
    The callbacks support various configuration options through training arguments:
    - plot_after_epoch: Epoch threshold(s) for enabling plotting (int or list)
    - trial_number: Trial identifier for hyperparameter optimization
    - run_name: W&B run name for experiment organization

Notes:
    This module modifies the base Transformers callback system by:
    1. Monkey-patching TrainerCallback to add on_plot event support.
    2. Extending TrainerControl with should_plot flag.  
    3. Patching the trainer module to use the extended control class.
"""
from transformers.integrations.integration_utils import WandbCallback as WandbCallback_
from transformers.integrations.integration_utils import rewrite_logs, is_torch_xla_available
from transformers.trainer_callback import CallbackHandler as CallbackHandler_
from transformers.trainer_callback import DefaultFlowCallback as DefaultFlowCallback_
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback as TrainerCallback_
from transformers.trainer_callback import TrainerControl as TrainerControl_
from dataclasses import dataclass
import wandb
from transformers.integrations.integration_utils import logger
from omegaconf import ListConfig
import tempfile
# Import high-level preprocessing helper
from utils.plot_progress import preprocess_for_plotting, build_info_strings, strip_validation_loss
from utils.plot_progress import LayoutConfig, Slice3DConfig, create_plotter
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_callback import TrainerState
import os
from PIL import Image
from pathlib import Path
from typing import Callable, Dict, List, Optional
import torch
from metrics.loss_weighting_strategies import LossWeightingStrategyBase
import torch.distributed as dist
from itertools import zip_longest
import json
import socket
import platform
from utils.telemetry_log_utils import (
    RuntimeTelemetryScope,
    aggregate_runtime_report,
    detect_runtime_backend,
    get_rank_world,
    now_local_iso,
)
import time

class CallbackHandler(CallbackHandler_):
    """
    Extended callback handler with support for custom evaluation and plotting events.
    
    This class extends the standard Transformers CallbackHandler to support additional
    keyword arguments in the on_evaluate event and introduces a new on_plot event for
    custom visualization workflows.
    
    Methods
    -------
    on_evaluate(args, state, control, metrics, **kwargs)
        Handle evaluation events with extended keyword argument support.
    on_plot(args, state, control, **kwargs)
        Handle custom plotting events for visualization generation.
        
    Notes
    -----
    The on_evaluate method is modified to accept **kwargs, allowing callbacks to
    receive additional data such as predictions, labels, and configuration parameters
    that are not part of the standard Transformers callback interface.
    """
    #NOTE: on_evaluate is modified to accept **kwargs
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        Handle evaluation completion events with extended parameter support.
        
        This method extends the base on_evaluate to accept additional keyword arguments,
        enabling callbacks to receive prediction data, labels, and configuration
        parameters for advanced post-evaluation processing.
        
        Parameters
        ----------
        args : TrainingArguments
            Training configuration and hyperparameters.
        state : TrainerState
            Current training state including epoch, step, and metrics.
        control : TrainerControl
            Training control flags and flow management.
        metrics : Dict[str, float]
            Evaluation metrics computed during the evaluation phase.
        **kwargs : dict
            Additional keyword arguments that may include:
            - predictions: Model predictions on evaluation data
            - labels: Ground truth labels for evaluation data
            - inputs: Input data used for evaluation
            - conditioning_inputs: Conditioning inputs if applicable
            - eval_dataset: Evaluation dataset object
            - data_config: Data configuration parameters
            - train_config: Training configuration parameters
            - output_log_config: Logging configuration
            
        Returns
        -------
        TrainerControl
            Updated control object with evaluation flag reset.
        """
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics, **kwargs)

    # ------------------------------------------------------------------
    # New optional event that propagates custom plotting callbacks.
    # ------------------------------------------------------------------
    def on_plot(self, args, state, control, **kwargs):
        """
        Forward the `on_plot` event to all registered callbacks.
        
        This method introduces a new callback event specifically for handling
        visualization and plotting operations. It allows callbacks to generate
        and log plots, figures, and other visual artifacts during training.
        
        Parameters
        ----------
        args : TrainingArguments
            Training configuration and hyperparameters.
        state : TrainerState
            Current training state including epoch, step, and metrics.
        control : TrainerControl
            Training control flags including should_plot flag.
        **kwargs : dict
            Additional keyword arguments for plotting operations.
            
        Returns
        -------
        TrainerControl
            Updated control object after plot event processing.
        """
        return self.call_event("on_plot", args, state, control, **kwargs)

class NaNCallback(TrainerCallback):
    """
    Callback to stop training if NaN is encountered in the loss. Training will stop at the nearest step where on_log is called.
    """
    # initusuper
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = logs.get("loss") or logs.get("eval_l2_error") or logs.get("eval_l1_error")
        if loss is not None and (loss != loss):  # NaN check
            print("🛑 NaN encountered in loss. Stopping training.")
            control.should_training_stop_due_to_nan = True
            control.should_evaluate = False
            control.should_save = False
            control.should_plot = False


class TrainingJsonLoggerCallback(TrainerCallback):
    """
    Write training/eval metrics to ``training_log.json``.

    Design goals
    ------------
    - Reuse existing trainer logs/metrics (no extra model compute).
    - Rank-0-only file writes.
    - Optional low-frequency background telemetry sampling.

    What gets recorded
    ------------------
    - Static run metadata at train-begin (host/platform/backend/distribution).
    - One record per evaluation event:
        - latest train scalars seen in ``on_log`` (`loss`, `learning_rate`, `grad_norm`, `epoch`),
        - evaluation metrics dictionary from ``on_evaluate``,
        - elapsed wall time since train start,
        - optional telemetry snapshot (`device_telemetry`, `peak_memory`).
    - Final run summary at train-end (end time + total duration + final telemetry).

    Distributed behavior
    --------------------
    ``aggregate_runtime_report()`` internally performs rank synchronization/gathering
    when distributed is active. Therefore all ranks execute the aggregation call,
    while only rank 0 writes the aggregated JSON payload.
    """

    def __init__(self, telemetry_sample_interval_sec: float = 1.0, enable_device_telemetry: bool = True):
        # Keep sampling conservative by default so telemetry overhead remains low.
        self.telemetry_sample_interval_sec = max(0.01, float(telemetry_sample_interval_sec))
        self.enable_device_telemetry = bool(enable_device_telemetry)

        # Internal run state (initialized in `on_train_begin`).
        self._train_start_perf = None
        self._train_start_local = None

        # Latest train-side scalars captured from `on_log`, later attached to
        # each eval record in `on_evaluate`.
        self._latest_train_log = {}
        self._last_eval_end_epoch = None

        # In-memory JSON payload + destination path.
        self._payload = None
        self._log_path = None

        # Optional long-lived telemetry scope covering the full train window.
        self._runtime_scope = None
        self._runtime_scope_full = None
        self._last_telemetry_global_step = 0

    def _start_new_runtime_scope(self):
        """Start a fresh telemetry scope for the next logging interval."""
        self._runtime_scope = RuntimeTelemetryScope(
            name="training_runtime",
            sample_interval_sec=self.telemetry_sample_interval_sec,
        )
        self._runtime_scope.start()

    def _run_dir(self, args, state) -> str:
        """
        Resolve the per-run output directory.
        """
        trial_name = getattr(state, "trial_name", None)
        return os.path.join(args.output_dir, trial_name) if trial_name else args.output_dir

    def _write_payload_atomic(self):
        """Write the in-memory payload to disk.

        Using a temporary file prevents partially-written JSON files if the
        process is interrupted during write.
        """
        if self._log_path is None or self._payload is None:
            return
        tmp_path = f"{self._log_path}.tmp"
        rounded_payload = self._round_payload_for_write(self._payload)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(rounded_payload, f, indent=2, sort_keys=False)
        os.replace(tmp_path, self._log_path)

    def _round_payload_for_write(self, payload):
        """Round numeric values before writing training log JSON.

        Default precision is 2 decimals, except:
        - `eval_metrics`: 6 decimals
        - `training_metrics` / `train_metrics`: 6 decimals
        """
        high_precision_keys = {"eval_metrics", "training_metrics", "train_metrics"}

        def _round(obj, digits: int):
            if isinstance(obj, bool) or obj is None:
                return obj
            if isinstance(obj, float):
                return round(obj, digits)
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    child_digits = 6 if str(k) in high_precision_keys else digits
                    out[k] = _round(v, child_digits)
                return out
            if isinstance(obj, list):
                return [_round(v, digits) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_round(v, digits) for v in obj)
            return obj

        return _round(payload, 2)

    def _safe_float(self, v):
        """Convert value to float, returning ``None`` when conversion fails."""
        try:
            return float(v)
        except Exception:
            return None

    def _estimate_seen_samples(self, args, global_step: int) -> Optional[float]:
        """Estimate cumulative seen training samples from global step."""
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        try:
            total_train_batch_size = int(trainer.get_total_train_batch_size(args))
            if total_train_batch_size <= 0:
                return None
            return float(int(global_step) * total_train_batch_size)
        except Exception:
            return None

    def _infer_interval_throughput(
        self,
        state,
        elapsed_sec: Optional[float],
        *,
        preferred_log_keys,
        total_from_step: Callable[[int], Optional[float]],
    ) -> Optional[float]:
        """Infer interval throughput from logged keys or record deltas.

        Args:
            state: Trainer state (uses `global_step`).
            elapsed_sec: Wall-clock seconds since training start.
            preferred_log_keys: Throughput keys to reuse from `_latest_train_log`.
            total_from_step: Callable mapping `global_step` -> cumulative total
                for the quantity (e.g. samples or steps).
        """
        # 1) Prefer explicitly logged throughput values when available.
        for key in preferred_log_keys:
            v = self._safe_float(self._latest_train_log.get(key))
            if v is not None:
                return v

        # 2) Estimate from current-vs-previous record deltas.
        if elapsed_sec is None or elapsed_sec <= 0:
            return None

        current_step = int(getattr(state, "global_step", 0) or 0)
        current_total = total_from_step(current_step)
        if current_total is None:
            return None

        records = (self._payload or {}).get("records", []) if isinstance(self._payload, dict) else []
        if not records:
            # First record: cumulative throughput since train start.
            return round(float(current_total) / float(elapsed_sec), 4)

        prev = records[-1] if isinstance(records[-1], dict) else {}
        prev_elapsed_min = self._safe_float(prev.get("wallclock_time_elapsed_since_training_start_min"))
        prev_elapsed_sec = (prev_elapsed_min * 60.0) if prev_elapsed_min is not None else None
        prev_step = int(prev.get("global_step", 0) or 0)
        if prev_elapsed_sec is None:
            return None

        delta_t = float(elapsed_sec) - float(prev_elapsed_sec)
        if delta_t <= 0:
            return None

        if current_step < prev_step:
            return None

        prev_total = total_from_step(prev_step)
        if prev_total is None:
            return None

        delta_total = float(current_total) - float(prev_total)
        if delta_total < 0:
            return None

        return round(delta_total / delta_t, 4)

    def _infer_train_samples_per_second(self, args, state, elapsed_sec: Optional[float]) -> Optional[float]:
        """Infer interval train throughput (samples/sec) for current eval window.

        Priority:
        1. Reuse a logged value if already present.
        2. Estimate from deltas between current and previous eval record:
           `delta_samples / delta_time_sec`.
        """
        return self._infer_interval_throughput(
            state,
            elapsed_sec,
            preferred_log_keys=("train_samples_per_second", "samples_per_second"),
            total_from_step=lambda step: self._estimate_seen_samples(args, step),
        )

    def _infer_train_steps_per_second(self, state, elapsed_sec: Optional[float]) -> Optional[float]:
        """Infer interval train throughput (steps/sec) for current eval window."""
        return self._infer_interval_throughput(
            state,
            elapsed_sec,
            preferred_log_keys=("train_steps_per_second", "steps_per_second"),
            total_from_step=lambda step: float(step),
        )

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize runtime state and create the initial JSON payload.

        Notes:
        - Starts optional telemetry sampler for the whole training window.
        """
        # Rank/world are used both for metadata and rank-0-only persistence.
        rank, world = get_rank_world()

        self._train_start_perf = time.perf_counter()
        self._train_start_local = now_local_iso()

        run_dir = self._run_dir(args, state)
        os.makedirs(run_dir, exist_ok=True)
        self._log_path = os.path.join(run_dir, "training_log.json")

        # Start with a minimal, stable schema and append records incrementally.
        self._payload = {
            "generated_local": self._train_start_local,
            "training_start_local": self._train_start_local,
            "host": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            "accelerator_backend": detect_runtime_backend(),
            "distributed": {
                "initialized": bool(dist.is_available() and dist.is_initialized()),
                "rank": int(rank),
                "world_size": int(world),
            },
            "records": [],
        }

        # Telemetry sampling is optional and runs in a lightweight background
        # sampler thread managed by RuntimeTelemetryScope.
        if self.enable_device_telemetry:
            # Full-window scope for end-of-training summary across the entire run.
            self._runtime_scope_full = RuntimeTelemetryScope(
                name="training_runtime_full",
                sample_interval_sec=self.telemetry_sample_interval_sec,
            )
            self._runtime_scope_full.start()

            # Interval scope used for per-evaluation records.
            self._start_new_runtime_scope()

        # Only rank 0 writes files to avoid write races in distributed runs.
        if rank == 0:
            self._write_payload_atomic()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Cache latest train-side logs. We keep only the latest values
        and attach them to the next evaluation record. 
        """
        logs = logs or {}
        tracked = {}
        # Keys emitted by the final train summary log; we don't want these to
        # pollute the next `training_metrics` entry created on end-of-train
        # evaluation/plot passes.
        excluded_train_metric_keys = {
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "total_flos",
            "train_loss",
        }
        # Keep this intentionally small and stable so the JSON remains easy to
        # diff/parse across long runs.
        for key, value in logs.items():
            # Keep numeric train-side metrics only. Eval metrics are handled in
            # `on_evaluate` under `eval_metrics`.
            if key.startswith("eval_"):
                continue
            if key in excluded_train_metric_keys:
                continue
            if isinstance(value, (int, float)):
                tracked[key] = value
        if tracked:
            self._latest_train_log.update(tracked)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Append one evaluation record to ``training_log.json``.

        The record includes:
        - latest cached train scalars,
        - all numeric evaluation metrics,
        - elapsed training time,
        - optional telemetry snapshot.

        In distributed mode, aggregation is executed on all ranks; only rank 0
        performs the actual file write.
        """
        if self._payload is None:
            return

        rank, _ = get_rank_world()
        metrics = metrics or {}

        # Elapsed wall time is measured from training start, independent of
        # trainer's own runtime metrics.
        elapsed_sec = None
        if self._train_start_perf is not None:
            elapsed_sec = max(0.0, time.perf_counter() - self._train_start_perf)

        end_epoch = self._safe_float(self._latest_train_log.get("epoch", getattr(state, "epoch", None)))
        if self._last_eval_end_epoch is None:
            start_epoch = 0.0 if end_epoch is not None else None
        else:
            start_epoch = self._last_eval_end_epoch

        training_metrics = {
            k: self._safe_float(v)
            for k, v in self._latest_train_log.items()
            if isinstance(v, (int, float))
            and k != "epoch"
            and k not in {"train_steps_per_second", "total_flos", "train_loss"}
        }

        # Ensure throughput is present in training metrics.
        if "train_samples_per_second" not in training_metrics:
            inferred_sps = self._infer_train_samples_per_second(args, state, elapsed_sec)
            if inferred_sps is not None:
                training_metrics["train_samples_per_second"] = inferred_sps
        if "train_steps_per_second" not in training_metrics:
            inferred_steps_ps = self._infer_train_steps_per_second(state, elapsed_sec)
            if inferred_steps_ps is not None:
                training_metrics["train_steps_per_second"] = inferred_steps_ps

        record = {
            "timestamp_local": now_local_iso(),
            "global_step": int(getattr(state, "global_step", 0) or 0),
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "wallclock_time_elapsed_since_training_start_min": (
                (float(elapsed_sec) / 60.0) if elapsed_sec is not None else None
            ),
            "training_metrics": training_metrics,
            # Keep numeric metrics only for schema consistency and safe JSON
            # downstream processing.
            "eval_metrics": {
                k: self._safe_float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and k != "epoch"
            },
        }

        self._last_eval_end_epoch = end_epoch

        # Optional lightweight telemetry snapshot, aggregated across devices/ranks.
        if self._runtime_scope is not None:
            # Close the current interval scope and aggregate it.
            self._runtime_scope.stop()
            current_step = int(getattr(state, "global_step", 0) or 0)
            interval_step_count = max(0, current_step - int(self._last_telemetry_global_step or 0))
            telemetry_snapshot = aggregate_runtime_report(
                self._runtime_scope.build_local_report(
                    # Use interval step delta as proxy so telemetry throughput
                    # corresponds to this logging window only.
                    local_samples=interval_step_count
                )
            )
            record["device_telemetry"] = telemetry_snapshot.get("device_telemetry", {})
            record["telemetry_sampling"] = telemetry_snapshot.get("telemetry_sampling", {})
            record["peak_memory"] = telemetry_snapshot.get("peak_memory", {})

            # Prepare next interval scope immediately (on all ranks).
            self._last_telemetry_global_step = current_step
            self._start_new_runtime_scope()

        # All ranks must pass through aggregation above; only rank 0 writes.
        if rank != 0:
            return

        self._payload["records"].append(record)
        self._payload["generated_local"] = now_local_iso()
        self._write_payload_atomic()

    def on_train_end(self, args, state, control, **kwargs):
        """Finalize training JSON log and write end-of-run summary.

        Stops optional telemetry scope, appends final telemetry summaries and
        total wall time, then performs one last atomic write on rank 0.
        """
        rank, _ = get_rank_world()

        if self._runtime_scope is not None:
            # Stop first so no new samples are appended while we aggregate.
            self._runtime_scope.stop()
            current_step = int(getattr(state, "global_step", 0) or 0)
            interval_step_count = max(0, current_step - int(self._last_telemetry_global_step or 0))
            final_telemetry = aggregate_runtime_report(
                self._runtime_scope.build_local_report(
                    local_samples=interval_step_count
                )
            )
            if self._payload is not None:
                self._payload["final_device_telemetry"] = final_telemetry.get("device_telemetry", {})
                self._payload["final_telemetry_sampling"] = final_telemetry.get("telemetry_sampling", {})
                self._payload["final_peak_memory"] = final_telemetry.get("peak_memory", {})

        # Full-window telemetry summary across the entire training run.
        if self._runtime_scope_full is not None:
            self._runtime_scope_full.stop()
            final_full_telemetry = aggregate_runtime_report(
                self._runtime_scope_full.build_local_report(
                    local_samples=int(getattr(state, "global_step", 0) or 0)
                )
            )
            if self._payload is not None:
                self._payload["final_device_telemetry"] = final_full_telemetry.get("device_telemetry", {})
                self._payload["final_telemetry_sampling"] = final_full_telemetry.get("telemetry_sampling", {})
                self._payload["final_peak_memory"] = final_full_telemetry.get("peak_memory", {})

        if self._payload is not None:
            self._payload["training_end_local"] = now_local_iso()
            if self._train_start_perf is not None:
                self._payload["training_elapsed_min"] = max(0.0, time.perf_counter() - self._train_start_perf) / 60.0

        if rank == 0:
            self._write_payload_atomic()

class WandbCallback(WandbCallback_):
    """
    Enhanced Weights & Biases callback with trial tracking and model configuration logging.
    
    This class extends the standard Transformers WandbCallback to provide enhanced
    experiment tracking capabilities specifically designed for CFD model training.
    It adds support for trial parameter logging, model configuration tracking,
    and custom plot artifact logging.
        
    Attributes
    ----------
    _wandb : wandb module or None
        Reference to the wandb module if available.
    _initialized : bool
        Flag indicating whether the callback has been initialized.
    _log_model : LoggingMode
        Configuration for model artifact logging.
        
    Methods
    -------
    setup(args, state, model, **kwargs)
        Initialize W&B logging with enhanced configuration tracking.
    on_log(args, state, control, model, logs, **kwargs)
        Handle logging events with custom plot artifact support.
    on_train_end(args, state, control, **kwargs)
        Clean up W&B session at training completion.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # When True, metric logs (e.g., eval_*) are suppressed from being sent to W&B.
        # Plot artifacts (keys starting with "plot_") are still allowed through.
        self.suppress_metrics_logging = False

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.
        
        This method initializes W&B logging with enhanced configuration tracking,
        including trial parameters, model configuration, and data configuration.
        
        Parameters
        ----------
        args : TrainingArguments
            Training configuration including output directory and run settings.
        state : TrainerState
            Current training state with trial information and parameters.
        model : torch.nn.Module
            The model being trained, used for configuration extraction.
        **kwargs : dict
            Additional keyword arguments for setup customization.
        """
        if self._wandb is None:
            return
        self._initialized = True

        from wandb.sdk.lib.config_util import ConfigError as WandbConfigError

        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_dict()}
            #NOTE:add trial number if available (additional to base class)
            if hasattr(args, "trial_number"):
                combined_dict["trial_number"] = args.trial_number
            model_config = {}
            #NOTE: add model_config and data_config if available (additional to base class)
            if hasattr(model, "config") and model.config is not None:
                model_config = model.config if isinstance(model.config, dict) else model.config.to_dict()
            if hasattr(state, "trial_params") and state.trial_params is not None:
                if isinstance(state.trial_params, dict):
                    trial_params = {
                        k.replace("model_config.", "").replace("data_config.", ""): v
                        for k, v in state.trial_params.items()
                        if k.startswith("model_config.") or k.startswith("data_config.")
                    }
                    model_config = {**model_config, **trial_params}
            combined_dict = {**model_config, **combined_dict}

            if hasattr(model, "peft_config") and model.peft_config is not None:
                peft_config = model.peft_config
                combined_dict = {**{"peft_config": peft_config}, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                init_args["name"] = trial_name
                init_args["group"] = args.run_name
            elif args.run_name is not None:
                init_args["name"] = args.run_name
                if args.run_name == args.output_dir:
                    self._wandb.termwarn(
                        "The `run_name` is currently set to the same value as `TrainingArguments.output_dir`. If this was "
                        "not intended, please specify a different run name by setting the `TrainingArguments.run_name` parameter.",
                        repeat=False,
                    )

            if self._wandb.run is None:
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    **init_args,
                )
            self._wandb.config.update(combined_dict, allow_val_change=True)

            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

            _watch_model = os.getenv("WANDB_WATCH", "false")
            if not is_torch_xla_available() and _watch_model in ("all", "parameters", "gradients"):
                self._wandb.watch(model, log=_watch_model, log_freq=max(100, state.logging_steps))
            self._wandb.run._label(code="transformers_trainer")

            try:
                self._wandb.config["model/num_parameters"] = model.num_parameters()
            except AttributeError:
                logger.info(
                    "Could not log the number of model parameters in Weights & Biases due to an AttributeError."
                )
            except WandbConfigError:
                logger.warning(
                    "A ConfigError was raised whilst setting the number of model parameters in Weights & Biases config."
                )

            if self._log_model.is_enabled:
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_name = (
                        f"model-{self._wandb.run.id}"
                        if (args.run_name is None or args.run_name == args.output_dir)
                        else f"model-{self._wandb.run.name}"
                    )
                    model_artifact = self._wandb.Artifact(
                        name=model_name,
                        type="model",
                        metadata={
                            "model_config": model.config.to_dict() if hasattr(model, "config") else None,
                            "num_parameters": self._wandb.config.get("model/num_parameters"),
                            "initial_model": True,
                        },
                    )
                    save_model_architecture_to_file(model, temp_dir)

                    for f in Path(temp_dir).glob("*"):
                        if f.is_file():
                            with model_artifact.new_file(f.name, mode="wb") as fa:
                                fa.write(f.read_bytes())
                    self._wandb.run.log_artifact(model_artifact, aliases=["base_model"])

                    badge_markdown = (
                        f'[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge'
                        f'-28.svg" alt="Visualize in Weights & Biases" width="20'
                        f'0" height="32"/>]({self._wandb.run.get_url()})'
                    )

                    modelcard.AUTOGENERATED_TRAINER_COMMENT += f"\n{badge_markdown}"
                    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """
        Handle logging events with enhanced plot artifact support.
        
        This method extends the base on_log to provide special handling for
        plot artifacts while maintaining compatibility with standard metric logging.
        It preserves original key names for plot artifacts and applies standard
        key rewriting for regular metrics.
        
        Parameters
        ----------
        args : TrainingArguments
            Training configuration and hyperparameters.
        state : TrainerState
            Current training state including global step and epoch.
        control : TrainerControl
            Training control flags and flow management.
        model : torch.nn.Module, optional
            The model being trained (used for initialization if needed).
        logs : Dict[str, Any], optional
            Dictionary of values to log, may contain metrics and plot artifacts.
        **kwargs : dict
            Additional keyword arguments for logging customization.
        """
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]
        
        if self._wandb is None:
            return
        # Make sure a W&B run exists **and** has the proper name.
        #
        #   • First-time call → _initialized is False → setup() launches a new run.
        #   • Later calls after wandb.finish() → _wandb.run is None → we clear
        #     the flag and call setup() again, which re-creates a run using the
        #     same naming logic (trial_name, run_name, etc.).
        if not self._initialized or self._wandb.run is None:
            self._initialized = False  # force full re-initialisation path
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = logs or {}
            # When suppression is enabled, block metric logs that do not contain plot artifacts.
            has_plot_keys = any(k.startswith("plot_") for k in logs.keys())
            if self.suppress_metrics_logging and not has_plot_keys:
                return

            # If suppression is enabled and there are plot keys mixed with metrics, keep only plot keys.
            if self.suppress_metrics_logging and has_plot_keys:
                logs = {k: v for k, v in logs.items() if k.startswith("plot_")}

            # Update summary only for scalar metrics and only when not suppressing metrics.
            if not self.suppress_metrics_logging:
                for k, v in logs.items():
                    if k in single_value_scalars:
                        self._wandb.run.summary[k] = v

            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}
            # If all non-scalar log keys start with the special "plot_" prefix, we skip the standard
            # key rewrite performed by 🤗 Transformers so that these entries are logged verbatim.
            # This allows logging arbitrary plot artifacts (e.g. PNG images) without their keys being
            # altered to the train/eval namespace.
            if not non_scalar_logs or not any(k.startswith("plot_") for k in non_scalar_logs):
                non_scalar_logs = rewrite_logs(non_scalar_logs)
            # else: leave non_scalar_logs untouched to preserve the original keys.
            self._wandb.log({**non_scalar_logs, "train/global_step": state.global_step}, step=state.global_step)
    
    def on_predict(self, args, state, control, metrics, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, **kwargs)
        if state.is_world_process_zero:
            metrics = rewrite_logs(metrics)
            self._wandb.log(metrics)

    def on_train_end(self, args, state, control, **kwargs):
        """
        Clean up W&B session at training completion.
        
        This method ensures proper cleanup of the W&B session when training
        ends, preventing resource leaks and ensuring all data is properly
        synchronized with the W&B servers.
        
        Parameters
        ----------
        args : TrainingArguments
            Training configuration and hyperparameters.
        state : TrainerState
            Final training state with completion metrics.
        control : TrainerControl
            Training control flags and flow management.
        **kwargs : dict
            Additional keyword arguments for cleanup customization.
            
        Notes
        -----
        This method calls wandb.finish() to properly close the current run
        and upload any remaining data. It includes safety checks to handle
        cases where no active run exists.
        """
        
        # --------------------------------------------------------------
        # At the end of training upload the *best* plot (if any) residing
        # in ``<output_dir>/plots`` to the W&B summary so it is easily
        # accessible from the run overview page.
        # --------------------------------------------------------------

        if wandb.run is not None:
            plots_dir = os.path.join(args.output_dir, "validation_plots")
            if os.path.isdir(plots_dir):
                best_pngs = [f for f in os.listdir(plots_dir) if f.endswith("_best.png")]
                if best_pngs:
                    best_path = os.path.join(plots_dir, best_pngs[0])  # take the first one
                    try:
                        with Image.open(best_path) as img:
                            wandb.run.summary["best_eval_plot"] = wandb.Image(img)
                    except Exception as e:
                        logger.warning(f"Could not upload best plot '{best_path}' to W&B: {e}")

        # Finish the run last so the image gets logged.
        wandb.finish() if wandb.run is not None else None

class PlotOnEvalAndSaveCallback(TrainerCallback):
    """
    Automated plotting callback with data renormalization and visualization generation.
    
    This callback automatically generates and logs visualizations of model predictions
    during training. It handles data renormalization, residual value reconstruction,
    and configurable plotting schedules based on epoch thresholds.
    
    Key features:
    - Automatic data renormalization for accurate visualization
    - Support for residual learning with proper value reconstruction
    - Configurable epoch thresholds for plotting activation
    - Memory-efficient plotting with example limits
    - Integration with W&B for artifact logging
    - Support for both 2D and 3D visualizations
    
    Attributes
    ----------
    _should_plot : bool
        Internal flag indicating whether plotting should occur.
    global_step : int or None
        Current global training step for plot labeling.
    trial : Any or None
        Trial information for hyperparameter optimization.
    _plotted_thresholds : set
        Set of epoch thresholds that have already triggered plotting.
        
    Methods
    -------
    on_evaluate(args, state, control, **kwargs)
        Store evaluation data and mark plotting readiness.
    on_plot(args, state, control, **kwargs)
        Generate and log visualizations with data renormalization.
    wandb_fig_log(log_dict)
        Log figure dictionary through trainer or directly to W&B.
    """
    
    def __init__(self):
        """
        Initialize the plotting callback with default state.
        
        Sets up internal state tracking for plot generation, including
        threshold tracking to plot only once after the threshold is reached.
        """
        self.trainer = None
        self._should_plot = False
        self.global_step = None
        self.trial = None
        # Track which epoch thresholds (if any) have already triggered a plot so that we don't
        # repeatedly plot for the same threshold when `plot_after_epoch` is a list.
        self._plotted_thresholds = set()
        
    def on_evaluate(self, args, state, control, **kwargs):
        """
        Store evaluation data and mark plotting readiness.
        
        This method is called after model evaluation and stores all necessary
        data for subsequent plotting operations. It extracts predictions, labels,
        inputs, and configuration parameters needed for renormalization and
        visualization.
        
        Parameters
        ----------
        args : TrainingArguments
            Training configuration and hyperparameters.
        state : TrainerState
            Current training state including epoch and global step.
        control : TrainerControl
            Training control flags, will be modified to set should_plot.
        **kwargs : dict
            Evaluation data and configuration including:
            - predictions: Model predictions on evaluation data
            - labels: Ground truth labels
            - inputs: Input data used for evaluation
            - conditioning_inputs: Conditioning inputs if applicable
            - eval_dataset: Evaluation dataset object
            - data_config: Data configuration with normalization parameters
            - train_config: Training configuration with plotting settings
            - output_log_config: Logging configuration
            
        Notes
        -----
        The method reshapes predictions from evaluation format to plotting format
        by combining rollout and sequence dimensions. It also extracts residual
        configuration for proper value reconstruction during plotting.
        """
        # Mark that an evaluation just happened
        self.predictions = kwargs['predictions']
        self.labels = kwargs['labels']
        self.inputs = kwargs['inputs']
        self.conditioning_inputs = kwargs['conditioning_inputs']
        self.eval_dataset = kwargs['eval_dataset']
        self.data_config = kwargs['data_config']
        self.train_config = kwargs['train_config']
        self.train_strategy_config = kwargs['train_strategy_config']
        self.scheduler_config = kwargs['scheduler_config']
        self.output_log_config = kwargs['output_log_config']
        self.model_config = kwargs['model_config']
        self.global_step = state.global_step
        self.residual_config = kwargs['data_config']['residual_config']
        control.should_plot = True

    def on_plot(self, args, state, control, **kwargs):
        """
        Generate and log visualizations with comprehensive data renormalization.
        
        This method handles the complete plotting workflow including:
        1. Epoch threshold checking for conditional plotting
        2. Data renormalization using dataset statistics
        3. Residual value reconstruction for residual learning
        4. Visualization generation and saving
        5. W&B artifact logging
        
        Parameters
        ----------
        args : TrainingArguments
            Training configuration including output directory.
        state : TrainerState
            Current training state with epoch and step information.
        control : TrainerControl
            Training control flags, should_plot will be reset after plotting.
        **kwargs : dict
            Additional plotting parameters including:
            - is_new_best_metric: Boolean indicating if current results are best
            
        Notes
        -----
        Plotting Logic:
        - Supports both single epoch threshold and list of thresholds
        - For lists, plots only once per threshold when first crossed
        - For single values, plots continuously after threshold is reached
        
        Data Processing:
        - Renormalizes all data using original dataset statistics
        - Handles input, conditioning, label, and prediction channels separately
        - Reconstructs raw values from residuals when applicable
        - Preserves channel ordering and naming from dataset
        
        Output Management:
        - Saves plots to organized directory structure
        - Logs to W&B if enabled in configuration
        - Includes metadata like epoch, step, and performance indicators
        """

        RANK = int(os.environ.get("RANK", -1))
        IS_MAIN_PROCESS = RANK in [-1, 0]
        
        if not IS_MAIN_PROCESS:
            control.should_plot = False
            return control

        # Only plot if an evaluation just happened before this save
        if control.should_plot:
            ##NOTE: This plotting is not present in the base class
            part_1 = args.output_dir
            part_2 = state.trial_name or ""
            part_3 = f"validation_plots"
            output_dir = os.path.join(part_1, part_2, part_3)
            
            len_eval_dataloader, num_eval_rollouts, label_seq_length, channel_dim, *spatial_dims = self.predictions.shape
            self.predictions=self.predictions.reshape(len_eval_dataloader, num_eval_rollouts*label_seq_length, channel_dim, *spatial_dims)

            # Plot only after the configured epoch threshold (default 0 when null)
            plot_after_epoch = self.train_config.get("plot_after_epoch") or 0

            # --------------------------------------------------------------
            # Decide whether to plot for the current epoch.
            # --------------------------------------------------------------
            should_plot = False
            # Case 1: `plot_after_epoch` is an iterable (list/tuple/set) of
            # epoch numbers. We want to plot *once* for each threshold when
            # it is first crossed.
            if isinstance(plot_after_epoch, (ListConfig, list, tuple, set)):
                for thr in plot_after_epoch:
                    if state.epoch >= thr and thr not in self._plotted_thresholds:
                        should_plot = True
                        self._plotted_thresholds.add(thr)
                        break  # Plot at most once per call
            # Case 2: `plot_after_epoch` is a single number – plot
            # continuously for every evaluation after the threshold is
            # reached.
            else:
                should_plot = state.epoch >= plot_after_epoch

            # Always plot at the end of training regardless of thresholds
            # Also flag this condition so we can mark the saved image as "best".
            best_plot_at_train_end = False
            try:
                final_epoch = self.train_strategy_config.get("num_train_epochs", None)
                if final_epoch is not None and kwargs["best_plot_at_train_end"]:
                    should_plot = True
                    best_plot_at_train_end = True
            except Exception:
                pass
 
            if should_plot:
                # ------------------------------------------------------------------
                # Preprocessing utility to renormalise data.
                # ------------------------------------------------------------------
                (
                    inputs_renormed,
                    labels_renormed,
                    predictions_renormed,
                    only_input_channel_names,
                    output_channel_names,
                    conditioning_inputs_renormed,
                    conditioning_input_channel_names,
                ) = preprocess_for_plotting(
                    self.inputs,
                    self.labels,
                    self.predictions,
                    data_config=self.data_config,
                    dataset=self.eval_dataset,
                    residual_config=self.residual_config,
                    conditioning_inputs=self.conditioning_inputs,
                )
                run_dir = os.path.join(part_1, part_2) if part_2 is not None else part_1
                # ----------------------------------------------------------
                # Get model and configuration information 
                # ----------------------------------------------------------    
                model_info_str, data_info_str, train_info_str, scheduler_info_str = build_info_strings(
                    model_obj = kwargs.get("model", None),
                    model_config=self.model_config,
                    data_config=self.data_config,
                    train_config=self.train_config,
                    scheduler_config=self.scheduler_config
                )

                loss_config_for_plotting = strip_validation_loss(self.train_strategy_config)

                layout_config = LayoutConfig(
                    base_visual_size=3.5,
                    margin_between_plots_h=0.65,
                    margin_between_plots_v=0.65
                )

                slice_config = Slice3DConfig(
                    slice_axis=0,
                    num_slices=4
                )
                
                #plotter used during validation
                plotter = create_plotter(
                    orientation=self.train_config.get("plot_orientation", "vertical"),
                    input_array=inputs_renormed,
                    prediction_array=predictions_renormed,
                    target_array=labels_renormed,
                    input_channel_names=only_input_channel_names,
                    output_channel_names=output_channel_names,
                    conditioning_input_array=conditioning_inputs_renormed,
                    conditioning_channel_names=conditioning_input_channel_names,
                    checkpoint_step=state.global_step,
                    epoch=round(state.epoch, 3),
                    extra_info=run_dir.split('/')[-2],
                    ndim=self.data_config["dimension"],
                    slice_config=slice_config,
                    num_examples=1,
                    stride=self.data_config["sequence_info"][-1],
                    save_dir=output_dir,
                    log_to_wandb=self.output_log_config["logging"]["wandb"],
                    best_plot_at_train_end=best_plot_at_train_end,
                    layout_config=layout_config,
                    include_relative_error=True,
                    model_info=model_info_str,
                    data_info=data_info_str,
                    train_info=train_info_str,
                    loss_config=loss_config_for_plotting
                )
                
                fig_dict = plotter.plot()

                # If W&B logging is enabled, log the figures now.
                if self.output_log_config["logging"].get("wandb", False) and wandb.run is not None:
                    self.wandb_fig_log(fig_dict)
        # Reset the flag
        control.should_plot = False

    # ------------------------------------------------------------------
    # Utility logger so that callbacks can perform Trainer-like logging.
    # The `trainer` attribute will be injected by our custom CallbackHandler.
    # ------------------------------------------------------------------
    def wandb_fig_log(self, log_dict: dict):
        """
        Log figure dictionary through trainer or directly to W&B.

        Parameters
        ----------
        log_dict : dict
            Dictionary containing figure objects and metadata to log.
            Keys should follow W&B conventions for plot artifacts.
            
        Notes
        -----
        Logging Priority:
        1. Trainer logging (if trainer attribute is available)
        2. Direct W&B logging (if wandb.run is active)
        3. Console output (fallback for debugging)
        
        The trainer-based logging is preferred as it maintains consistency
        with the overall training logging workflow and triggers other callbacks
        that may need to process the logged data.
        """
        if hasattr(self, "trainer") and self.trainer is not None:
            # Delegate to the Trainer's own logging facility. This will also trigger other callbacks' on_log.
            self.trainer.log(log_dict)
        elif wandb.run is not None:
            wandb.log(log_dict)
        else:
            print(log_dict)

    def on_train_end(self, args, state, control, **kwargs):
        # --------------------------------------------------------------
        # Perform one last validation on the best checkpoint found inside
        # output_dir and save plots with the `_best` suffix. This runs
        # regardless of W&B being on or off and happens before W&B is
        # finished by its own callback.
        # --------------------------------------------------------------
        logger.info("\n One last validation with the best model to save the plot with _best.png suffix.")
        trainer = getattr(self, "trainer", None)

        # Temporarily suppress metric logging to W&B so we don't overwrite
        # the final-epoch metrics with this last validation pass.
        callbacks = list(getattr(getattr(trainer, "callback_handler", None), "callbacks", []) or [])
        wandb_callbacks = [cb for cb in callbacks if isinstance(cb, WandbCallback)]
        for cb in wandb_callbacks:
            if hasattr(cb, "suppress_metrics_logging"):
                cb.suppress_metrics_logging = True

        # Force logging with the best epoch during this final evaluate
        ################################
        trainer.last_evaluate()
        ################################
        setattr(trainer, "_force_best_epoch_for_logging", False)
        for cb in wandb_callbacks:
            if hasattr(cb, "suppress_metrics_logging"):
                cb.suppress_metrics_logging = False

        # ----------------------------------------------------------
        # After evaluation, reload TrainerState from the best/latest
        # checkpoint-# directory inside the run directory.
        # ----------------------------------------------------------
        run_dir = os.path.join(
            trainer.args.output_dir,
            trainer.state.trial_name,
        ) if getattr(trainer.state, "trial_name", None) else trainer.args.output_dir

        # Prefer the best checkpoint path recorded by TrainerState
        ckpt_path = getattr(trainer.state, "best_model_checkpoint", None)
        if ckpt_path and os.path.isdir(ckpt_path):
            state_path = os.path.join(ckpt_path, "trainer_state.json")
            if os.path.isfile(state_path):
                trainer.state = TrainerState.load_from_json(state_path)

        # Flag a plot pass and enforce best-suffix saving. We suppress metric
        # logging again during the plot logging, since trainer.log will invoke
        # callback on_log; we only want plot_* entries to go through.
        for cb in wandb_callbacks:
            if hasattr(cb, "suppress_metrics_logging"):
                cb.suppress_metrics_logging = True

        # Ensure that the epoch we log corresponds to the best checkpoint's epoch
        setattr(trainer, "_force_best_epoch_for_logging", True)
        trainer.control.should_plot = True
        self.on_plot(
            trainer.args, trainer.state, trainer.control, is_new_best_metric=True, model=trainer.model, best_plot_at_train_end=True
        )
        setattr(trainer, "_force_best_epoch_for_logging", False)
        for cb in wandb_callbacks:
            if hasattr(cb, "suppress_metrics_logging"):
                cb.suppress_metrics_logging = False

        # Reset local state
        self._plotted_thresholds = set()
        self._should_plot = False

class LossStatisticsCallback(TrainerCallback):
    """
    Callback to accumulate loss values during training and evaluation.
    
    Efficiently handles distributed training by accumulating losses on GPU
    during training and only transferring/aggregating at epoch end.
    """
    
    def __init__(self, collect_train_losses: bool = True, grad_stats: List = [], collect_gradients: bool = False, trainer=None):
        self.collect_train_losses = collect_train_losses
        self.grad_stats = grad_stats
        self.train_losses: Dict[str, List[float]] = {}
        self.eval_losses: Dict[str, List[float]] = {}
        self.grad_stats_history: Dict[str, Dict[str, List[float]]] = {}
        self.trainer = trainer
        #self.current_epoch = -1
        #self.trainer = None
        self.collect_gradients = collect_gradients
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Initialize loss accumulator at start of epoch."""
        self.train_losses = {}
        self.grad_stats_history = {}
        #self.current_epoch = state.epoch
        
        # Initialize fresh accumulator for this epoch
        #if self.trainer is not None:
        self.trainer._detailed_loss_accumulator = {} 
        if self.collect_gradients:
            self.trainer._gradient_accumulator = {}
            self.trainer._grad_stat_names = self.grad_stats
            self.trainer._collect_gradients = True
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Transfer and aggregate losses and gradient norms at end of epoch."""
        # if self.trainer is None:
        #     return
        
        # Transfer losses
        if self.collect_train_losses:
            self._transfer_losses()
        
        # Transfer gradient norms
        if self.collect_gradients:
            self._transfer_gradient_stats()
        
        # Clear accumulators to free GPU memory
        if hasattr(self.trainer, '_detailed_loss_accumulator'):
            self.trainer._detailed_loss_accumulator = {}
        if hasattr(self.trainer, '_gradient_accumulator'):
            self.trainer._gradient_accumulator = {}

        # Log loss/grad histories to W&B (rank 0 only)
        self._log_histories_to_wandb(state)
    
    def _transfer_losses(self):
        """Transfer loss values from GPU to CPU."""
        if not hasattr(self.trainer, '_detailed_loss_accumulator'):
            return
        
        accumulator = self.trainer._detailed_loss_accumulator
        
        if not accumulator:
            return
        
        # Transfer from GPU to CPU and convert to Python floats
        for component_name, loss_tensors in accumulator.items():
            if component_name not in self.train_losses:
                self.train_losses[component_name] = []
            
            # Stack tensors and transfer to CPU in one operation
            if loss_tensors:
                stacked = torch.stack(loss_tensors)  # [num_steps]
                cpu_values = stacked.cpu().tolist()  # Single transfer to CPU
                self.train_losses[component_name].extend(cpu_values)
    
    def _transfer_gradient_stats(self):
        """Transfer gradient stats from GPU to CPU (parallel to loss transfer)."""
        if not hasattr(self.trainer, '_gradient_accumulator'):
            return
        
        accumulator = self.trainer._gradient_accumulator
        
        if not accumulator:
            return
        
        # Transfer from GPU to CPU and convert to Python floats
        for component_name, stat_dict in accumulator.items():
            if component_name not in self.grad_stats_history:
                self.grad_stats_history[component_name] = {}
            
            for stat_name, stat_tensors in stat_dict.items():
                if stat_name not in self.grad_stats_history[component_name]:
                    self.grad_stats_history[component_name][stat_name] = []
                
                if stat_tensors:
                    stacked = torch.stack(stat_tensors)
                    cpu_values = stacked.cpu().tolist()
                    self.grad_stats_history[component_name][stat_name].extend(cpu_values)

    def _log_histories_to_wandb(self, state: TrainerState) -> None:
        """Log loss and gradient stat histories to W&B via Trainer.log."""
        trainer = getattr(self, "trainer", None)
        if trainer is None or not hasattr(trainer, "log"):
            return

        is_distributed = dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0

        # IMPORTANT: all ranks must participate in the gather_object calls below.
        # We aggregate first on *every* rank, then only rank 0 logs to W&B.
        log_dict: Dict[str, float] = {}

        loss_history = self.get_loss_history(source='train', aggregate_distributed=True)
        for component_name, losses in loss_history.items():
            if not losses:
                continue
            loss_tensor = torch.tensor(losses)
            log_dict[f"loss_history/{component_name}"] = float(loss_tensor.mean())

        grad_stats = self.get_grad_stats_history(aggregate_distributed=True)
        for component_name, stat_dict in grad_stats.items():
            for stat_name, values in stat_dict.items():
                if not values:
                    continue
                values_tensor = torch.tensor(values)
                prefix = f"grad_stats_history/{component_name}/{stat_name}"
                log_dict[f"{prefix}"] = float(values_tensor.mean())
        
        if is_distributed and rank != 0:
            return

        if log_dict:
            log_dict["histories/epoch"] = float(state.epoch) if state.epoch is not None else -1.0
            trainer.log(log_dict)

    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Accumulate loss values from evaluation metrics."""
        if metrics is None:
            return
        
        for key, value in metrics.items():
            clean_key = key.replace('eval_', '') if key.startswith('eval_') else key
            
            if clean_key in ['loss', 'runtime', 'samples_per_second', 
                            'steps_per_second', 'epoch', 'step', 'loop_time']:
                continue
            
            if not isinstance(value, (int, float)):
                continue
            
            if clean_key not in self.eval_losses:
                self.eval_losses[clean_key] = []
            self.eval_losses[clean_key].append(float(value))
    
    def _aggregate_distributed_losses(self, losses: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Aggregate losses across all distributed processes.
        
        Called at epoch end, after GPU->CPU transfer.
        
        Args:
            losses: Dictionary of component names to loss lists
            
        Returns:
            Aggregated losses from all ranks (broadcast to all ranks)
        """
        if not dist.is_initialized():
            return losses
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        logger.info(
            f"[LossStatisticsCallback][rank {rank}/{world_size}] Aggregating losses across ranks"
        )
        
        # Gather all losses on rank 0
        gathered = [None] * world_size
        dist.gather_object(losses, gathered if rank == 0 else None, dst=0)

        combined_losses = {}
        if rank == 0:
            # Combine losses from all processes
            for process_losses in gathered:
                if process_losses is None:
                    continue
                for component_name, loss_list in process_losses.items():
                    if component_name not in combined_losses:
                        combined_losses[component_name] = []
            
            # Interleave per-rank histories: first elements from each rank,
            # then second elements, etc.
            for component_name in list(combined_losses.keys()):
                per_rank_lists = [
                    pl.get(component_name, []) if pl is not None else []
                    for pl in gathered
                ]
                interleaved = []
                for row in zip_longest(*per_rank_lists, fillvalue=None):
                    for v in row:
                        if v is not None:
                            interleaved.append(v)
                combined_losses[component_name] = interleaved

            summary = {k: len(v) for k, v in combined_losses.items()}
            logger.info(
                f"[LossStatisticsCallback][rank {rank}] Combined loss counts: {summary}"
            )

        # Broadcast combined losses to all ranks so everyone is in sync
        obj_list = [combined_losses]
        dist.broadcast_object_list(obj_list, src=0)
        if rank != 0:
            recv_summary = {k: len(v) for k, v in obj_list[0].items()}
            logger.info(
                f"[LossStatisticsCallback][rank {rank}] Received combined loss counts: {recv_summary}"
            )
        return obj_list[0]

    def _aggregate_distributed_grad_stats(
        self, grad_stats: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Aggregate gradient statistics across all distributed processes.

        Returns:
            Aggregated grad stats from all ranks (broadcast to all ranks)
        """
        if not dist.is_initialized():
            return grad_stats

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        logger.info(
            f"[LossStatisticsCallback][rank {rank}/{world_size}] Aggregating gradient stats across ranks"
        )

        gathered = [None] * world_size
        dist.gather_object(grad_stats, gathered if rank == 0 else None, dst=0)

        combined: Dict[str, Dict[str, List[float]]] = {}
        if rank == 0:
            for process_stats in gathered:
                if process_stats is None:
                    continue
                for component_name, stat_dict in process_stats.items():
                    if component_name not in combined:
                        combined[component_name] = {}
                    for stat_name, values in stat_dict.items():
                        if stat_name not in combined[component_name]:
                            combined[component_name][stat_name] = []
            
            # Interleave per-rank histories for each component/stat.
            for component_name, stat_dict in list(combined.items()):
                for stat_name in list(stat_dict.keys()):
                    per_rank_lists = [
                        ps.get(component_name, {}).get(stat_name, []) if ps is not None else []
                        for ps in gathered
                    ]
                    interleaved = []
                    for row in zip_longest(*per_rank_lists, fillvalue=None):
                        for v in row:
                            if v is not None:
                                interleaved.append(v)
                    combined[component_name][stat_name] = interleaved

            summary = {
                comp: {stat: len(vals) for stat, vals in stats.items()}
                for comp, stats in combined.items()
            }
            logger.info(
                f"[LossStatisticsCallback][rank {rank}] Combined grad stat counts: {summary}"
            )

        obj_list = [combined]
        dist.broadcast_object_list(obj_list, src=0)
        if rank != 0:
            recv_summary = {
                comp: {stat: len(vals) for stat, vals in stats.items()}
                for comp, stats in obj_list[0].items()
            }
            logger.info(
                f"[LossStatisticsCallback][rank {rank}] Received grad stat counts: {recv_summary}"
            )
        return obj_list[0]
    
    def get_grad_stats_history(self, aggregate_distributed: bool = True) -> Dict[str, Dict[str, List[float]]]:
        """
        Get accumulated gradient stats history.

        Returns
        -------
        Dict[str, Dict[str, List[float]]]
            Dictionary mapping component -> stat -> list of values
        """
        if aggregate_distributed and dist.is_initialized():
            return self._aggregate_distributed_grad_stats(self.grad_stats_history)
        
        return {k: {sk: sv.copy() for sk, sv in v.items()} for k, v in self.grad_stats_history.items()}

    # Back-compat helper (norm-only)
    def get_gradient_norm_history(self, aggregate_distributed: bool = True) -> Dict[str, List[float]]:
        grad_stats = self.get_grad_stats_history(aggregate_distributed=aggregate_distributed)
        return {k: v.get("norm", []) for k, v in grad_stats.items()}

    def get_loss_history(self, source: str = 'train', aggregate_distributed: bool = True) -> Dict[str, List[float]]:
        """
        Get raw loss history for the current epoch.
        
        Args:
            source: Either 'train', 'eval', or 'both'
            aggregate_distributed: Whether to aggregate across distributed processes
            
        Returns:
            Dictionary mapping component names to lists of loss values
        """
        if source == 'train':
            losses = self.train_losses.copy()
        elif source == 'eval':
            losses = self.eval_losses.copy()
        # elif source == 'both':
        #     combined = self.train_losses.copy()
        #     for key, values in self.eval_losses.items():
        #         if key in combined:
        #             combined[key].extend(values)
        #         else:
        #             combined[key] = values
        #     losses = combined
        else:
            raise ValueError(f"Invalid source: {source}")
        
        # Aggregate across distributed processes if needed
        if aggregate_distributed and dist.is_initialized():
            losses = self._aggregate_distributed_losses(losses)
        
        return losses

class AdaptiveWeightCallback(TrainerCallback):
    """
    Callback to update loss weights using a weight scheduler.
    
    Can use training losses, evaluation losses, or both for weight updates.
    """
    
    def __init__(
        self,
        loss_weighting_strategy: LossWeightingStrategyBase,
        stats_callback: LossStatisticsCallback,
        trainer=None,
        loss_source: str = 'train',  # 'train', 'eval', or 'both'
        use_gradients: bool = False,
        curriculum_start_epochs: List[int] = [0],
        grad_stats: List = [],
        weight_per_channel: bool = False,
        weight_sub_components: bool = False,
    ):
        """
        Args:
            loss_weighting_strategy: Scheduler to compute new weights
            stats_callback: Callback collecting loss statistics
            trainer: Reference to trainer (set after trainer creation)
            loss_source: Which losses to use for weight updates ('train', 'eval', or 'both')
        """
        self.loss_weighting_strategy = loss_weighting_strategy
        self.stats_callback = stats_callback
        self.trainer = trainer
        #self.last_update_epoch = -1
        self.loss_source = loss_source
        self.use_gradients = use_gradients
        self.curriculum_start_epochs = curriculum_start_epochs
        self.weight_per_channel = weight_per_channel
        self.weight_sub_components = weight_sub_components
        self.grad_stats = list(grad_stats) if grad_stats is not None else []
        
        if loss_source not in ['train', 'eval', 'both']:
            raise ValueError(f"loss_source must be 'train', 'eval', or 'both', got {loss_source}")
    
    def _filter_loss_history(
        self,
        loss_history: Dict[str, List[float]],
        training_components: set
    ) -> Dict[str, List[float]]:
        """
        Filter loss history to include only relevant components based on flags.
        
        Args:
            loss_history: Full history dictionary with all loss components
            training_components: Set of base component names being trained
            
        Returns:
            Filtered history including base components and optionally hierarchical ones
        """
        filtered = {}
        
        for name, losses in loss_history.items():
            # Check if this is a base component
            if name in training_components:
                filtered[name] = losses
                continue
            
            # Check if this is a hierarchical component (contains '/')
            if '/' in name:
                # Extract base component name (before first '/')
                base_name = name.split('/')[0]
                
                # Only include if base component is being trained
                if base_name not in training_components:
                    continue
                
                # Check if it's a per-channel component
                if '/channel_' in name:
                    if self.weight_per_channel:
                        filtered[name] = losses
                # Otherwise it's a per-component
                elif self.weight_sub_components:
                    filtered[name] = losses
        
        return filtered

    def _update_loss_weights(
        self, 
        current_epoch: int, 
        loss_history: Dict[str, List[float]],
        grad_stats_history: Optional[Dict[str, Dict[str, List[float]]]] = None,
        source_label: str = 'train'
    ):
        """Helper method to perform weight update."""
        if not loss_history:
            logger.warning(f"[AdaptiveWeightCallback] No {source_label} loss history for epoch {current_epoch}")
            return False

        is_distributed = dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        world_size = dist.get_world_size() if is_distributed else 1
        
        # Get current loss weights
        # Only the train loss function metrics are used for weight updates, but the loss history source can be train or eval.
        current_weights = self.trainer.loss_fn.get_loss_weight_dict()

        training_component_names = set(current_weights.keys()) #TODO_MAX:
        
        # Only filter if using eval losses (train losses are already filtered during collection)
        if source_label == 'train' or self.loss_source == 'train':
            filtered_loss_history = loss_history
            filtered_grad_stats_history = grad_stats_history
        else:
            filtered_loss_history = self._filter_loss_history(loss_history, training_component_names)
            filtered_grad_stats_history = None
            if grad_stats_history is not None:
                filtered_grad_stats_history = self._filter_grad_stats_history(
                    grad_stats_history, training_component_names
                )

        if not filtered_loss_history:
            logger.warning(
                f"[AdaptiveWeightCallback] No matching loss components in {source_label} loss history\n"
                f"  Loss components: {training_component_names}\n"
                f"  Available in {source_label}: {set(loss_history.keys())}"
            )
            return False
        
        # Compute new loss weights only on rank 0, then broadcast
        new_weights = None
        if (not is_distributed) or rank == 0:
            new_weights = self.loss_weighting_strategy.step(
                epoch=current_epoch,
                loss_history=filtered_loss_history,
                current_weights=current_weights,
                grad_stats_history=filtered_grad_stats_history
            )

        if is_distributed:
            obj_list = [new_weights]
            dist.broadcast_object_list(obj_list, src=0)
            new_weights = obj_list[0]
        
        # Apply new loss weights if scheduler returned them
        if new_weights is None:
            return False
        
        self.trainer.loss_fn.update_loss_weights(new_weights)
        self.last_update_epoch = current_epoch

        # Log weights to W&B via Trainer log (if available)
        self._log_loss_weights(new_weights, current_epoch)
        
        # Collect all weight information for table formatting
        table_rows = []
        
        for component_name, weight_dict in new_weights.items():
            base_weight = weight_dict.get('base_weight', 1.0)
            
            # Only add row if we have statistics for this component
            if component_name in filtered_loss_history:
                losses_tensor = torch.tensor(filtered_loss_history[component_name])
                mean_loss = float(losses_tensor.mean())
                std_loss = float(losses_tensor.std())
                num_samples = len(filtered_loss_history[component_name])
                stats_str = f"mean={mean_loss:.4e}, std={std_loss:.4e}, n={num_samples}"
                
                table_rows.append({
                    'component': component_name,
                    'weight': f"{base_weight:.4f}",
                    'statistics': stats_str
                })
            
            # Per-channel weights if present
            if 'channel_weights' in weight_dict:
                channel_weights = weight_dict['channel_weights']
                for ch_idx in range(len(channel_weights)):
                    ch_key = f"{component_name}/channel_{ch_idx}"
                    
                    # Only add row if we have statistics for this channel
                    if ch_key in filtered_loss_history:
                        ch_weight = float(channel_weights[ch_idx])
                        ch_losses = torch.tensor(filtered_loss_history[ch_key])
                        ch_mean = float(ch_losses.mean())
                        ch_std = float(ch_losses.std())
                        ch_n = len(filtered_loss_history[ch_key])
                        ch_stats_str = f"mean={ch_mean:.4e}, std={ch_std:.4e}, n={ch_n}"
                        
                        table_rows.append({
                            'component': f"  └─ channel_{ch_idx}",
                            'weight': f"{ch_weight:.4f}",
                            'Loss history statistics': ch_stats_str
                        })
            
            # Per-component weights if present
            if 'component_weights' in weight_dict:
                component_weights = weight_dict['component_weights']
                for sub_name, sub_weight in component_weights.items():
                    comp_key = f"{component_name}/{sub_name}"
                    
                    # Only add row if we have statistics for this sub-component
                    if comp_key in filtered_loss_history:
                        comp_losses = torch.tensor(filtered_loss_history[comp_key])
                        comp_mean = float(comp_losses.mean())
                        comp_std = float(comp_losses.std())
                        comp_n = len(filtered_loss_history[comp_key])
                        comp_stats_str = f"mean={comp_mean:.4e}, std={comp_std:.4e}, n={comp_n}"
                        
                        table_rows.append({
                            'component': f"  └─ {sub_name}",
                            'weight': f"{sub_weight:.4f}",
                            'statistics': comp_stats_str
                        })
        
        # Calculate column widths and print table
        if table_rows:
            max_component_len = max(len(row['component']) for row in table_rows)
            max_weight_len = max(len(row['weight']) for row in table_rows)
            max_stats_len = max(len(row['statistics']) for row in table_rows)
            
            # Add some padding
            component_width = max(max_component_len, len("Component")) + 2
            weight_width = max(max_weight_len, len("Weight")) + 2
            stats_width = max(max_stats_len, len("Statistics")) + 2
            
            # Print table header
            header = f"{'Component':<{component_width}} {'Weight':<{weight_width}} {'Statistics':<{stats_width}}"
            separator = "─" * len(header)
            logger.info(separator)
            logger.info(header)
            logger.info(separator)
            
            # Print table rows
            for row in table_rows:
                logger.info(f"{row['component']:<{component_width}} {row['weight']:<{weight_width}} {row['statistics']:<{stats_width}}")
            
            logger.info(separator)
        
        return True

    def _log_loss_weights(self, weight_dict: Dict[str, Dict], epoch: int) -> None:
        """
        Log loss weights to W&B (through Trainer.log) in a flat, readable format.

        Args:
            weight_dict: Nested weight dict from CompositeLoss.get_loss_weight_dict()
            epoch: Current epoch number
        """
        trainer = getattr(self, "trainer", None)
        if trainer is None or not hasattr(trainer, "log"):
            return

        log_dict: Dict[str, float] = {}

        for component_name, cfg in weight_dict.items():
            base_weight = float(cfg.get("base_weight", 1.0))
            log_dict[f"loss_weights/{component_name}"] = base_weight

            if "channel_weights" in cfg:
                channel_weights = cfg["channel_weights"]
                for ch_idx in range(len(channel_weights)):
                    log_dict[
                        f"loss_weights/{component_name}/channel_{ch_idx}"
                    ] = float(channel_weights[ch_idx])

            if "component_weights" in cfg:
                for sub_name, sub_weight in cfg["component_weights"].items():
                    log_dict[
                        f"loss_weights/{component_name}/{sub_name}"
                    ] = float(sub_weight)

        if log_dict:
            log_dict["loss_weights/epoch"] = float(epoch)
            trainer.log(log_dict)

    def on_epoch_end(self, args, state, control, **kwargs): #This happens before on_evaluate
        """Update weights at end of epoch using "training" losses."""
        next_epoch = round(state.epoch)
        #skip update_loss_weights if the curriculum block changes in the next epoch.
        if next_epoch in self.curriculum_start_epochs:
            return
        
        if self.loss_source == 'train':
            next_epoch = int(state.epoch)
            train_losses = self.stats_callback.get_loss_history(source=self.loss_source)
            
            grad_stats_history = None
            if self.use_gradients:
                grad_stats_history = self.stats_callback.get_grad_stats_history()

            self._update_loss_weights(
                next_epoch, 
                train_losses, 
                grad_stats_history=grad_stats_history,
                source_label='train'
            )
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Update weights after evaluation, optionally combining with training losses."""        
        # Get appropriate loss history based on configuration (LossStatisticsCallback collects both train and eval losses)
        next_epoch = round(state.epoch)
        if next_epoch in self.curriculum_start_epochs:
            return
        # Update the train loss weights using the eval loss history for the next epoch.
        else:
            if self.loss_source == 'eval':
                #if state.epoch is not None else -1
                loss_history = self.stats_callback.get_loss_history(source=self.loss_source)
                self._update_loss_weights(next_epoch, loss_history, source_label=self.loss_source)

# ------------------------------------------------------------------
# Extend the 🤗 Transformers callback interface with an optional
# `on_plot` event **without** modifying the original library files.
# We monkey-patch the base `TrainerCallback` to include a no-op
# implementation so that any callback can safely override it.
# ------------------------------------------------------------------

if not hasattr(TrainerCallback_, "on_plot"):
    def _noop_on_plot(self, args, state, control, **kwargs):  # type: ignore[unused-argument]
        """Event called after a plotting operation (user-defined)."""
        return control

    setattr(TrainerCallback_, "on_plot", _noop_on_plot)

# ------------------------------------------------------------------
# Extended TrainerControl that adds a `should_plot` flag.
# ------------------------------------------------------------------

@dataclass
class TrainerControl(TrainerControl_):
    """
    TrainerControl subclass with an additional `should_plot` and `should_training_stop_due_to_nan`
    switch.
    
    This extended control class adds plotting state management to the standard
    Transformers training control flow. It maintains all original functionality
    while adding support for conditional plotting operations during training.
    
    Attributes
    ----------
    should_plot : bool, default=False
        Flag indicating whether plotting operations should be performed.
        Set by evaluation callbacks and consumed by plotting callbacks.
    should_training_stop_due_to_nan : bool, default=False
        Flag indicating if training should stop due to NaN in loss.
    Methods
    -------
    _new_step()
        Reset control flags for a new training step, including should_plot.
    state()
        Serialize control state including the new should_plot flag.
        
    Notes
    -----
    This class extends the base TrainerControl to support the custom plotting
    workflow while maintaining full compatibility with the existing training
    infrastructure. The should_plot flag follows the same pattern as other
    control flags like should_save and should_evaluate.
    """

    should_plot: bool = False
    should_training_stop_due_to_nan: bool = False  # New flag to indicate NaN-induced stop

    def _new_step(self):
        """
        Reset flags for a new step, including the new `should_plot`.
        
        This method is called at the beginning of each training step to reset
        all control flags to their default state. It extends the base implementation
        to include the custom should_plot flag.
        """
        super()._new_step()
        self.should_plot = False

    # Ensure serialization/deserialization captures the new flag
    def state(self) -> dict:
        """
        Serialize the control state including the should_plot flag.
        
        This method extends the base state serialization to include the
        custom should_plot flag, ensuring it is properly preserved during
        checkpoint saving and loading operations.
        
        Returns
        -------
        dict
            Serialized control state with all flags including should_plot.
        """
        base_state = super().state()
        base_state["args"]["should_plot"] = self.should_plot
        return base_state

# Also patch the reference imported in `transformers.trainer` so that
# Trainer.__init__ uses the extended control when it instantiates.
import transformers.trainer as _tr_mod
if getattr(_tr_mod, "TrainerControl", None) is not TrainerControl:
    _tr_mod.TrainerControl = TrainerControl
