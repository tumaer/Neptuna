
from omegaconf import OmegaConf
import os
import glob
from utils.load_data import fetch_dataset
from utils.plot_progress import build_info_strings
from utils.plot_progress import preprocess_for_plotting, plot_rollout_metrics
from utils.plot_progress import LayoutConfig, Slice3DConfig, create_plotter
from utils.plot_progress import plot_rollout_metrics_bar_chart, calculate_and_save_results_all_channels
from utils.plot_progress import plot_multi_run_rollout_metrics
from utils.loss_utils import fetch_loss_metric, fetch_infer_loss_dict, fetch_train_loss_dict
from metrics.inference_metrics import compute_metrics_for_n_rollouts
from transformers.trainer import EvalPrediction
from transformers import TrainingArguments
from train.trainer import Trainer
import hydra
from omegaconf import DictConfig
import json
from utils.seed_utils import set_global_seed
import torch
import numpy as np
import csv
import ast


def load_pretrained_model(model_config):
    """
    Factory function to load any pretrained model based on model_name in config.
    
    Args:
        model_config: Dictionary containing model configuration with 'model_name' and 'model_checkpoint_path'
    
    Returns:
        Loaded pretrained model instance
    """
    model_name = model_config.get("model_name", "").lower()
    checkpoint_path = model_config["model_checkpoint_path"]
    
    # Map model names to their corresponding classes
    model_registry = {
        "unet": ("models.UNet.unet", "UNet"),
        "fno": ("models.FNO.fno", "FNO"),
        "resnet": ("models.ResNet.resnet", "ResNet"),
        "autodeeponet": ("models.DeepONet.deeponet", "AutoDeepONet"),
        "cno": ("models.CNO.cno", "CNO"),
        "scot": ("models.ScOT.scot", "ScOT"),
        "vit": ("models.ViT.vit", "ViT"),
        "kfno": ("models.kFNO.kfno", "kFNO"),
    }
    
    if model_name not in model_registry:
        supported_models = ", ".join(model_registry.keys())
        raise ValueError(f"Model '{model_name}' is not supported for inference loading. Supported models: {supported_models}")
    
    module_path, class_name = model_registry[model_name]
    
    # Dynamic import and model loading
    import importlib
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    model, loading_info = model_class.from_pretrained(
        checkpoint_path,
        output_loading_info=True,
        ignore_mismatched_sizes=False,
        local_files_only=True,
    )

    assert not loading_info["missing_keys"], f"Missing keys: {loading_info['missing_keys']}"
    assert not loading_info["unexpected_keys"], f"Unexpected keys: {loading_info['unexpected_keys']}"

    return model

def inverse_log_transform_channels(data_array, channel_names, log_transform_channels):
    """
    Apply inverse log transform (exp) to specified channels.
    
    Args:
        data_array: numpy array with shape (N, T, C, *spatial)
        channel_names: list of channel names corresponding to C dimension
        log_transform_channels: list of channel names that were log-transformed
    
    Returns:
        Modified array with inverse log transform applied
    """
    if not log_transform_channels:
        return data_array
    
    data_array = data_array.copy()  # Avoid modifying original
    
    for c_idx, ch_name in enumerate(channel_names):
        if ch_name in log_transform_channels:
            # Apply exp to invert log transform
            data_array[:, :, c_idx] = np.exp(data_array[:, :, c_idx])
    
    return data_array

def build_train_and_eval_loss(loss_config, data_config, device: torch.device):
    """
    Construct training and eval CompositeLoss from configs.
    For inference, both are used as metrics -> keep them on CPU.
    """
    if loss_config is None:
        return None, None

    full_train_cfg = OmegaConf.create({
        "loss_config": loss_config,
        "data_config": data_config,
    })

    # Always put metric losses on CPU
    metric_device = torch.device("cpu")

    train_loss_dict = fetch_train_loss_dict(full_train_cfg)
    train_loss_fn = fetch_loss_metric(data_config, train_loss_dict).to(metric_device)

    infer_loss_dict = fetch_infer_loss_dict(full_train_cfg)
    infer_loss_fn = fetch_loss_metric(data_config, infer_loss_dict).to(metric_device)

    return train_loss_fn, infer_loss_fn

def get_trainer(
    model_config,
    data_config,
    output_dir,
    train_config=None,
    scheduler_config=None,
    infer_config=None,
    output_log_config=None,
    loss_config=None,
    compute_metrics=None
):
    # Function to read train_batch_size from trainer_state.json
    def get_train_batch_size(checkpoint_path):
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if not os.path.exists(trainer_state_path):
            raise FileNotFoundError(f"trainer_state.json not found in {checkpoint_path}")
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
        return trainer_state["train_batch_size"]

    # Helper to read mixed precision flags from training_args.bin
    def get_mixed_precision_flags(checkpoint_path):
        training_args_path = os.path.join(checkpoint_path, "training_args.bin")
        default_flags = {"fp16": False, "bf16": False, "tf32": False}
        if not os.path.exists(training_args_path):
            return default_flags
        try:
            args_obj = torch.load(training_args_path, map_location="cpu", weights_only=False)
            if isinstance(args_obj, dict):
                return {
                    "fp16": bool(args_obj.get("fp16", False)),
                    "bf16": bool(args_obj.get("bf16", False)),
                    "tf32": bool(args_obj.get("tf32", False)),
                }
            return {
                "fp16": bool(getattr(args_obj, "fp16", False)),
                "bf16": bool(getattr(args_obj, "bf16", False)),
                "tf32": bool(getattr(args_obj, "tf32", False)),
            }
        except Exception as exc:
            print(f"Warning: could not load mixed precision flags from {training_args_path}: {exc}")
            return default_flags

    # Extract train_batch_size from trainer_state.json
    train_batch_size = get_train_batch_size(model_config["model_checkpoint_path"])

    # Read mixed precision flags from checkpoint
    mp_flags = get_mixed_precision_flags(model_config["model_checkpoint_path"])

    # Seed from data config (default 0)
    seed_value = int(data_config.get("seed", 0))

    # Use train_batch_size for per_device_eval_batch_size
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=train_batch_size,
        seed=seed_value,  # model-seed
        data_seed=seed_value,  # sampler-seed for SeedableRandomSampler
        eval_accumulation_steps=16,
        dataloader_num_workers=0,
        report_to="none",
        use_cpu=False,  # use_cpu even if other devices are present
        label_names=["label_including_rollouts"],
        include_for_metrics=[
            "inputs",
        ]
        + (
            ["conditioning_inputs"]
            if data_config["conditioning_features"]["conditioning_in_channels"] is not None
            else []
        ), 
        fp16=mp_flags["fp16"],
        bf16=mp_flags["bf16"],
        tf32=mp_flags["tf32"],
    )

    # Load pretrained model using the generic factory function
    model = load_pretrained_model(model_config)

    trainer = Trainer(
        model=model,
        args=args,
        compute_metrics=compute_metrics,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        scheduler_config=scheduler_config,
        infer_config=infer_config,
        output_log_config=output_log_config,
        loss_config=loss_config,
    )
    return trainer



def find_checkpoint_path(experiment_dir):
    """
    Find the checkpoint folder within an experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Path to the checkpoint folder, or None if not found
    """
    checkpoint_pattern = os.path.join(experiment_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        return None
    
    # If multiple checkpoints exist, take the one with the highest number
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
    return checkpoint_dirs[-1]

def save_errors_to_csv(errors, output_dir, file_name):
    """
    Save errors to 2 CSV files in the specified output directory.

    Args:
        errors: Dictionary of errors to save.
        output_dir: Directory where the CSV file will be saved.
    """
    csv_file = os.path.join(output_dir, file_name)
    file_is_empty = not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header only if the file is empty
        if file_is_empty:
            writer.writerow(["Metric", "Value"])
        for key, value in errors.items():
            writer.writerow([key, value])

    # Structured CSV file
    # --------------------------------------------------------------
    # Collect data into a dict-of-dicts before writing
    # rows[(index_type, index)][col_name] = value
    rows: Dict[tuple, Dict[str, float]] = {}
    all_col_names: set = set()

    for metric_name, value in errors.items():
        # Value can be:
        #   - dict from compute_metrics_for_n_rollouts
        #   - stringified dict as written earlier
        metrics_dict = None

        if isinstance(value, dict):
            metrics_dict = value
        elif isinstance(value, str):
            # Quick filter to avoid parsing arbitrary strings
            if "per_rollout_step_mean" not in value and "per_timestep_mean" not in value:
                continue
            try:
                metrics_dict = ast.literal_eval(value)
            except Exception:
                continue
        else:
            continue  # scalar metric, skip

        if not isinstance(metrics_dict, dict):
            continue

        for summary_kind, arr in metrics_dict.items():
            try:
                np_arr = np.asarray(arr)
            except Exception:
                continue

            if np_arr.ndim != 2:
                # expect (R, C+1) or (T, C+1)
                continue

            n_idx, n_channels = np_arr.shape  # last column == overall

            # Decide index_type based on summary_kind
            if "rollout_step" in summary_kind:
                index_type = "rollout"
            elif "timestep" in summary_kind:
                index_type = "timestep"
            else:
                index_type = "index"

            # ------------------------------------------------------------------
            # Map summary_kind -> a more generic "stat" so rollout/timestep
            # share the same columns:
            #   per_rollout_step_mean, per_timestep_mean      -> "mean"
            #   per_rollout_step_std,  per_timestep_std       -> "std"
            #   cumulative_*_mean                             -> "cumulative_mean"
            #   cumulative_*_std                              -> "cumulative_std"
            # ------------------------------------------------------------------
            if "mean" in summary_kind and "cumulative" not in summary_kind:
                stat = "mean"
            elif "std" in summary_kind and "cumulative" not in summary_kind:
                stat = "std"
            elif "cumulative" in summary_kind and "mean" in summary_kind:
                stat = "cumulative_mean"
            elif "cumulative" in summary_kind and "std" in summary_kind:
                stat = "cumulative_std"
            else:
                # fallback: keep full summary_kind if it doesn't match patterns
                stat = summary_kind

            # Columns:
            #   <metric>__<stat>__overall
            #   <metric>__<stat>__ch0
            #   <metric>__<stat>__ch1
            base = f"{metric_name}__{stat}"

            overall_col = f"{base}__overall"
            all_col_names.add(overall_col)
            for ch in range(n_channels - 1):
                ch_col = f"{base}__ch{ch}"
                all_col_names.add(ch_col)

            for idx in range(n_idx):
                row_key = (index_type, idx)
                if row_key not in rows:
                    rows[row_key] = {}

                # overall column from last channel
                rows[row_key][overall_col] = float(np_arr[idx, n_channels - 1])

                # per-channel columns
                for ch in range(n_channels - 1):
                    ch_col = f"{base}__ch{ch}"
                    rows[row_key][ch_col] = float(np_arr[idx, ch])

    # If nothing to write, just ensure header exists
    pretty_csv_file = os.path.join(output_dir, "results_structured.csv")

    # Sort columns for deterministic order
    all_col_names_sorted = sorted(all_col_names)

    with open(pretty_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Header: no "channel" column, channels are in the metric column names
        header = ["index_type", "index"] + all_col_names_sorted
        writer.writerow(header)

        # Sort rows: index_type, then index
        for (index_type, idx) in sorted(rows.keys(), key=lambda k: (k[0], k[1])):
            row_data = rows[(index_type, idx)]
            row = [index_type, idx]
            for col_name in all_col_names_sorted:
                row.append(row_data.get(col_name, ""))  # empty if missing
            writer.writerow(row)

def run_inference_for_each_experiment(experiment_dir, infer_config):
    """
    Run inference for a single experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory
        infer_config: Inference configuration
    """
    print(f"\n{'#'*88}")
    BOX_WIDTH = 88
    header_sep = "*" * BOX_WIDTH
    print("\n" + header_sep)
    print(f"Processing experiment: {os.path.basename(experiment_dir)}")
    print(header_sep)
    
    # Check if data_config.json exists
    data_config_path = os.path.join(experiment_dir, "data_config.json")
    if not os.path.exists(data_config_path):
        print(f"Warning: data_config.json not found in {experiment_dir}. Skipping...")
        return
    
    # Find checkpoint directory
    checkpoint_path = find_checkpoint_path(experiment_dir)
    if not checkpoint_path:
        print(f"Warning: No checkpoint folder found in {experiment_dir}. Skipping...")
        return
    
    # Check if config.json exists in checkpoint
    model_config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(model_config_path):
        print(f"Warning: config.json not found in {checkpoint_path}. Skipping...")
        return
    
    print(f"Data config: {data_config_path}")
    print(f"Model checkpoint: {checkpoint_path}")
    
    try:
        # Load configurations
        data_config = OmegaConf.load(data_config_path)
        # Post-load fix: convert parameter_min_max_stats keys to integers
        try:
            cond_cfg = data_config.get("conditioning_features", None)
            if cond_cfg is not None:
                param_stats = cond_cfg.get("parameter_min_max_stats", None)
                if isinstance(param_stats, DictConfig):
                    coerced = {}
                    for k, v in param_stats.items():
                        try:
                            coerced[int(k)] = v
                        except Exception:
                            coerced[k] = v
                    data_config["conditioning_features"]["parameter_min_max_stats"] = coerced
        except Exception:
            print(f" Parameter_min_max_stats not available, skipping...")
        model_config = OmegaConf.load(model_config_path)

        # Load loss config from checkpoint
        loss_config = None
        loss_config_path = os.path.join(checkpoint_path, "loss_config.json")
        
        if os.path.exists(loss_config_path):
            try:
                with open(loss_config_path, 'r') as f:
                    loss_config_data = json.load(f)
                
                # Convert to OmegaConf for compatibility
                loss_config = OmegaConf.create(loss_config_data)
                
                # Replace configured weights with current_weights from checkpoint
                # This uses the weights that were active when the model was saved
                if 'loss' in loss_config and 'components' in loss_config.loss:
                    for component in loss_config.loss.components:
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
                         
            except Exception as e:
                import traceback
                traceback.print_exc()
                loss_config = None
        else:
            print(f"No loss_config.json found in checkpoint: {loss_config_path}")
            print("Inference will only compute legacy L1/L2 errors")

        # Determine device once
        metric_device = torch.device("cpu")

        # Build loss functions
        train_loss_fn, eval_loss_fn = build_train_and_eval_loss(
            loss_config=loss_config,
            data_config=data_config,
            device=metric_device,
        )

        # Define metrics for trainer
        def compute_metrics(eval_pred: EvalPrediction):
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
            if train_loss_fn is not None:
                try:
                    with torch.no_grad():
                        composite_loss = train_loss_fn(
                            model=None,
                            predictions=preds_tensor.to(metric_device),
                            labels=targets_tensor.to(metric_device),
                            return_detailed=False,
                        )
                    metrics["eval_composite_loss"] = float(
                        composite_loss.item()
                        if torch.is_tensor(composite_loss)
                        else composite_loss
                    )
                except Exception as e:
                    print(f"Failed to compute composite loss metrics: {e}")

            # 2) Evaluation loss components for logging
            if eval_loss_fn is not None:
                try:
                    with torch.no_grad():
                        _, detailed = eval_loss_fn(
                            model=None,
                            predictions=preds_tensor.to(metric_device),
                            labels=targets_tensor.to(metric_device),
                            return_detailed=True,
                        )
                    for component_name, component_detailed in detailed.items():
                        component_total = component_detailed["total"]
                        metrics[f"eval_{component_name}"] = (
                            component_total.item()
                            if torch.is_tensor(component_total)
                            else component_total
                        )
                except Exception as e:
                    print(f"Failed to compute evaluation loss metrics: {e}")

            return metrics
        
        # Set global seed from data_config (default 0)
        seed_value = int(data_config.get("seed", 0))
        set_global_seed(seed_value)
        
        # Add the model checkpoint path to the model config
        model_config["model_checkpoint_path"] = checkpoint_path
        model_config["model_name"] = model_config["architectures"][0]
        
        # Attempt to reconstruct train_config and scheduler_config from training_args.bin
        def _safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        def _enum_to_str(value):
            try:
                # HF enums typically expose .value or .name
                if hasattr(value, "value"):
                    return str(value.value)
                if hasattr(value, "name"):
                    return str(value.name)
            except Exception:
                pass
            return str(value)

        def extract_configs_from_training_args(checkpoint_path):
            training_args_path = os.path.join(checkpoint_path, "training_args.bin")
            if not os.path.exists(training_args_path):
                return None, None
            try:
                args_obj = torch.load(training_args_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                print(f"Warning: could not load {training_args_path}: {exc}")
                return None, None

            # Training config (best-effort reconstruction from HF TrainingArguments)
            train_cfg = {
                "use_cpu": bool(_safe_get(args_obj, "use_cpu", False)),
                "per_device_train_batch_size": int(_safe_get(args_obj, "per_device_train_batch_size", 1)),
                "per_device_eval_batch_size": int(_safe_get(args_obj, "per_device_eval_batch_size", 1)),
                "num_train_epochs": float(_safe_get(args_obj, "num_train_epochs", 0)),
                "mix_precision_config": {
                    "fp16": bool(_safe_get(args_obj, "fp16", False)),
                    "bf16": bool(_safe_get(args_obj, "bf16", False)),
                    "tf32": bool(_safe_get(args_obj, "tf32", False)),
                },
                "dataloader_num_workers": int(_safe_get(args_obj, "dataloader_num_workers", 0)),
                "gradient_accumulation_steps": int(_safe_get(args_obj, "gradient_accumulation_steps", 1)),
                # Map HF naming to our config naming
                "eval_strategy": _enum_to_str(_safe_get(args_obj, "evaluation_strategy", "steps")),
                "eval_steps": int(_safe_get(args_obj, "eval_steps", 0) or 0),
                "logging_strategy": _enum_to_str(_safe_get(args_obj, "logging_strategy", "steps")),
                "logging_steps": int(_safe_get(args_obj, "logging_steps", 0) or 0),
                "save_strategy": _enum_to_str(_safe_get(args_obj, "save_strategy", "steps")),
                "save_steps": int(_safe_get(args_obj, "save_steps", 0) or 0),
                "save_total_limit": int(_safe_get(args_obj, "save_total_limit", 0) or 0),
                "eval_accumulation_steps": int(_safe_get(args_obj, "eval_accumulation_steps", 0) or 0),
                "metric_for_best_model": _safe_get(args_obj, "metric_for_best_model", None),
                # Not recoverable from HF TrainingArguments; keep sensible defaults or None
                "pushforward_config": None,
                "n_eval_rollouts": None,
                "eval_split_ratio": None,
            }

            # Scheduler/optimizer config
            scheduler_cfg = {
                "optim": _enum_to_str(_safe_get(args_obj, "optim", "adamw_torch")),
                "lr": float(_safe_get(args_obj, "learning_rate", 5e-5)),
                "weight_decay": float(_safe_get(args_obj, "weight_decay", 0.0)),
                "lr_scheduler": _enum_to_str(_safe_get(args_obj, "lr_scheduler_type", "linear")),
                # Prefer ratio if present, else provide steps as a fallback via a separate key
                "warmup_ratio": float(_safe_get(args_obj, "warmup_ratio", 0.0) or 0.0),
            }
            warmup_steps = int(_safe_get(args_obj, "warmup_steps", 0) or 0)
            if warmup_steps and not scheduler_cfg.get("warmup_ratio"):
                scheduler_cfg["warmup_steps"] = warmup_steps

            return train_cfg, scheduler_cfg

        train_config, scheduler_config = extract_configs_from_training_args(checkpoint_path)
        output_log_config = None
        
        # Log all available configs
        def _print_config_block(name, cfg_obj):
            import textwrap
            BOX_WIDTH = 90
            double_sep = "=" * BOX_WIDTH
            single_sep = "-" * BOX_WIDTH
            title = f"[ {name} ]"
            print("\n")
            print(double_sep)
            print(title.center(BOX_WIDTH))
            print(double_sep)
            try:
                if cfg_obj is None:
                    body = "<None>"
                elif isinstance(cfg_obj, DictConfig) or OmegaConf.is_config(cfg_obj):
                    body = OmegaConf.to_yaml(cfg_obj, resolve=True)
                else:
                    body = json.dumps(cfg_obj, indent=2, default=str)
                print(textwrap.indent(body.rstrip(), "  "))
            except Exception as exc:
                print(f"(Could not render {name}: {exc})")
                print(textwrap.indent(str(cfg_obj), "  "))
            print(single_sep)

        print("\n" + "=" * 90)
        print("CONFIGURATION OVERVIEW (INFERENCE)".center(90))
        print("=" * 90)
        _print_config_block("INFER CONFIG", infer_config)
        _print_config_block("DATA CONFIG", data_config)
        _print_config_block("MODEL CONFIG", model_config)
        _print_config_block("TRAIN CONFIG", train_config)
        _print_config_block("SCHEDULER CONFIG", scheduler_config)
        #_print_config_block("OUTPUT LOG CONFIG", output_log_config)
        
        # Function to create a unique solo_inference directory
        def create_unique_inference_dir(base_dir):
            solo_inference_dir = os.path.join(base_dir, "solo_inference")
            if not os.path.exists(solo_inference_dir):
                os.makedirs(solo_inference_dir)
                return solo_inference_dir

            # If solo_inference directory already exists, append a number to create a unique directory
            counter = 1
            while True:
                new_dir = f"{solo_inference_dir}_{counter}"
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                    return new_dir
                counter += 1

        # Create a unique solo_inference directory within the experiment directory
        solo_inference_dir = create_unique_inference_dir(experiment_dir)
        
        # Override filter parameters from infer_config if they are specified (not None)
        infer_filter_groups = data_config["filter_features"]["infer_filter_groups"]
        infer_filter_frames = data_config["filter_features"]["infer_filter_frames"]
        
        # Check if infer_config has filter_features and override if not None
        if "filter_features" in infer_config:
            if infer_config["filter_features"].get("infer_filter_groups") is not None:
                print(f"Original infer_filter_groups from data_config: {infer_filter_groups}")
                infer_filter_groups = infer_config["filter_features"]["infer_filter_groups"]
                print(f"** Using infer_filter_groups from inference config: {infer_filter_groups} **")
            
            if infer_config["filter_features"].get("infer_filter_frames") is not None:
                print(f"Original infer_filter_frames from data_config: {infer_filter_frames}")
                infer_filter_frames = infer_config["filter_features"]["infer_filter_frames"]
                print(f"** Using infer_filter_frames from inference config: {infer_filter_frames} **")
        
        print("Running solo inference...")

        if infer_config["dataset_directory_path"] is not None:
            dataset_directory_path = infer_config["dataset_directory_path"]
        else:
            dataset_directory_path = data_config["dataset_directory_path"]

        print(f"Dataset directory path: {dataset_directory_path}")

        infer_ds, infer_ds_from_ic = fetch_dataset(
                                                data_config["dataset_name"], 
                                                mode="infer",
                                                dataset_directory_path=dataset_directory_path,
                                                sequence_info=data_config["sequence_info"],
                                                infer_filter_groups=infer_filter_groups,
                                                infer_filter_frames=infer_filter_frames,
                                                filter_in_channels=data_config["filter_features"]["filter_in_channels"],
                                                filter_out_channels=data_config["filter_features"]["filter_out_channels"],
                                                conditioning_in_channels=data_config["conditioning_features"]["conditioning_in_channels"],
                                                include_conditioning_parameters=data_config["conditioning_features"]["include_conditioning_parameters"],
                                                parameter_min_max_stats=data_config["conditioning_features"]["parameter_min_max_stats"],
                                                data_normalization_stats=data_config["data_normalization_stats"],
                                                data_normalization_strategy=data_config["data_normalization_strategy"],
                                                is_steady_state_prediction=data_config["is_steady_state_prediction"],
                                                residual_config=data_config["residual_config"],
                                                n_infer_rollouts=infer_config["n_infer_rollouts"],
                                                infer_from_random_timestep=infer_config["infer_from_random_timestep"],
                                                infer_from_ic=infer_config["infer_from_ic"],
                                                log_transform_channels=data_config["log_transform_channels"],
                                                )
        
        trainer = get_trainer(
            model_config=model_config, 
            data_config=data_config,
            output_dir=solo_inference_dir,
            train_config=train_config,
            scheduler_config=scheduler_config,
            infer_config=infer_config,
            output_log_config=output_log_config,
            loss_config=loss_config,
            compute_metrics=compute_metrics,
        )

        if infer_config["infer_from_random_timestep"]:
            trainer.set_eval_or_test_rollout_steps(
                rollout_steps=infer_config["n_infer_rollouts"], output_all_steps=True
            )

            predictions_obj, inputs, conditioning_inputs = trainer.predict(infer_ds, metric_key_prefix="")
            ############################################################
            # predictions_obj.predictions: the output of the model with shape (accumulated_outputs, num_rollouts, label_seq_length, channel_dim, *spatial) 
            # accumulated_outputs and accumulated_gt have the length of number of windows in the test dataset
            # predictions_obj.label_ids: the ground truth with shape (accumulated_gt, num_rollouts*label_seq_length, channel_dim, *spatial)
            # predictions_obj.metrics: the metrics computed after accumulating the outputs and ground truth
            ############################################################

            # pretty print the keys which have the word error in them
            print('Accumulated error for the whole test set (random start):')
            errors = {} 
            for key, value in predictions_obj.metrics.items():
                if ("error" in key) or ("eval" in key):
                    print(f"{key}: {value}")
                    errors["random_start"+key] = value
            save_errors_to_csv(errors, solo_inference_dir, "results.csv")
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
            # else:
            #     outputs_per_rollout = 1

            targets = predictions_obj.label_ids  # Expected shape: (N, R*T, C, *spatial)

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
                data_config=data_config,
                dataset=infer_ds,
                residual_config=data_config.get("residual_config", None),
                conditioning_inputs=cond_inp_arr,
            )

            log_transform_channels = data_config["log_transform_channels"]

            inp_renorm = inverse_log_transform_channels(
                inp_renorm, only_input_channel_names, log_transform_channels
            )

            tgt_renorm = inverse_log_transform_channels(
                tgt_renorm, output_channel_names, log_transform_channels
            )

            pred_renorm = inverse_log_transform_channels(
                pred_renorm, output_channel_names, log_transform_channels
            )

            # Infer spatial dimensionality (1D / 2D / 3D)
            ndim = pred_renorm.ndim - 3  # subtract batch, time, channel dims

            # Use stride from the config if available
            stride_val = data_config.get("sequence_info", [1, 1, 1])[2]

            # Directory for saving inference plots
            plot_save_dir = os.path.join(solo_inference_dir, "inference_plots/random_start")

            # Build formatted info strings  
            model_info_str, data_info_str, train_info_str, sched_info_str = build_info_strings(
                                                                                            model_obj=trainer.model,
                                                                                            data_config=data_config,
                                                                                            model_config=model_config,
                                                                                            train_config=train_config,
                                                                                            scheduler_config=scheduler_config
                                                                                        )

            # Create rollout sample plots and per-example rollout metrics, saved per example folder
            # Compute the exact example indices used for plotting to reuse for per-example rollout metrics
            N_examples = preds.shape[0]
            num_plot = min(infer_config["n_infer_plot_examples"], N_examples)
            np.random.seed(42)
            chosen_example_indices = np.random.choice(N_examples, size=num_plot, replace=False)
            # For each selected example, save the example plot and the rollout metrics into the same folder
            for example_idx in chosen_example_indices:
                ex_save_dir = os.path.join(plot_save_dir, f"example_{int(example_idx)}")
                # Save the visual comparison plot for this example

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
                    orientation='vertical',
                    input_array=inp_renorm,
                    prediction_array=pred_renorm,
                    target_array=tgt_renorm,
                    input_channel_names=only_input_channel_names,
                    output_channel_names=output_channel_names,
                    conditioning_input_array=cond_inp_renorm,
                    conditioning_channel_names=cond_inp_channel_names,
                    checkpoint_step=None,
                    epoch=None,
                    extra_info=data_config.get("dataset_name")+"_Inference_plot_from_random_timestep",
                    ndim=ndim,
                    slice_config=slice_config,
                    num_examples=1,
                    stride=stride_val,
                    save_dir=ex_save_dir,
                    log_to_wandb=False,
                    best_plot_at_train_end=False,
                    layout_config=layout_config,
                    include_relative_error=True,
                    model_info=model_info_str,
                    data_info=data_info_str,
                    train_info=train_info_str
                )
                
                plotter.plot()

                # Create a rollout metrics plot for this example (no batch aggregation)
                ex_preds = preds[example_idx:example_idx+1]      # shape (1, R*T, C, *spatial)
                ex_targets = targets[example_idx:example_idx+1]
                
                per_rollout_metrics_ex = compute_metrics_for_n_rollouts(
                    ex_preds, ex_targets, outputs_per_rollout=outputs_per_rollout, loss_metric=eval_loss_fn
                )
                ex_title = f"Per-rollout metrics ({data_config.get('dataset_name', 'dataset')} - random start, example {int(example_idx)})"
                plot_rollout_metrics(
                    step_metrics=per_rollout_metrics_ex,
                    output_channel_names=output_channel_names,
                    save_dir=ex_save_dir,
                    title=ex_title,
                    filename="rollout_metrics.png",
                    sequence_info=data_config.get("sequence_info", [1, 1, 1]),
                )

        ic_return = None
        if infer_config["infer_from_ic"]:
            trainer.set_eval_or_test_rollout_steps(
                rollout_steps=infer_config["n_infer_rollouts"], output_all_steps=True
            )
            # ----------------------------------------------------------
            # Prepare prediction, target and input arrays
            # ----------------------------------------------------------
            predictions_obj, inputs, conditioning_inputs = trainer.predict(infer_ds_from_ic, metric_key_prefix="")

            print('Accumulated error for the whole test set (IC start):')
            errors = {}
            for key, value in predictions_obj.metrics.items():
                if ("error" in key) or ("eval" in key):
                    print(f"{key}: {value}")
                    errors["ic_start"+key] = value
            save_errors_to_csv(errors, solo_inference_dir, "results.csv")

            preds = predictions_obj.predictions
            targets = predictions_obj.label_ids
            inp_arr = inputs
            cond_inp_arr = conditioning_inputs if conditioning_inputs is not None else None

            # Determine outputs per rollout (T_out) before flattening, default to 1
            if preds.ndim >= 5:
                n, n_rollouts, seq_len, c = preds.shape[:4]
                outputs_per_rollout = seq_len
                extra_dims = preds.shape[4:]
                preds = preds.reshape(n, n_rollouts * seq_len, c, *extra_dims) # (N, R*T, C, *spatial)

            per_rollout_step_metrics_ic = compute_metrics_for_n_rollouts(
                preds, targets, outputs_per_rollout=outputs_per_rollout, include_per_timestep=True, loss_metric=eval_loss_fn
            )
            
            errors = {}
            for metric_name, values in per_rollout_step_metrics_ic.items():
                print(f"{metric_name} per-step (IC start): {values}")
                errors[metric_name] = values
            save_errors_to_csv(errors, solo_inference_dir, "results.csv")

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
                data_config=data_config,
                dataset=infer_ds,
                residual_config=data_config.get("residual_config", None),
                conditioning_inputs=cond_inp_arr,
            )

            log_transform_channels = data_config["log_transform_channels"]

            inp_renorm = inverse_log_transform_channels(
                inp_renorm, only_input_channel_names, log_transform_channels
            )

            tgt_renorm = inverse_log_transform_channels(
                tgt_renorm, output_channel_names, log_transform_channels
            )

            pred_renorm = inverse_log_transform_channels(
                pred_renorm, output_channel_names, log_transform_channels
            )
            

            # Compute renormalized metrics: necessary for comparing performance between runs with different data preprocessing
            per_rollout_step_metrics_ic = compute_metrics_for_n_rollouts(
                pred_renorm, tgt_renorm, outputs_per_rollout=outputs_per_rollout, include_per_timestep=True, loss_metric=eval_loss_fn
            )
            
            errors = {}
            for metric_name, values in per_rollout_step_metrics_ic.items():
                print(f"{metric_name} per-step (IC start, renorm): {values}")
                errors[metric_name] = values
            save_errors_to_csv(errors, solo_inference_dir, "results_renorm.csv")

            # Infer spatial dimensionality (1D / 2D / 3D)
            ndim = pred_renorm.ndim - 3  # subtract batch, time, channel dims

            stride_val = data_config.get("sequence_info", [1, 1, 1])[2]

            plot_save_dir = os.path.join(solo_inference_dir, "inference_plots/ic_start")

            plot_rollout_metrics(
                step_metrics=per_rollout_step_metrics_ic,
                output_channel_names=output_channel_names,
                save_dir=plot_save_dir,
                title=f"Per-(rollout and time) step metrics ({data_config.get('dataset_name', 'dataset')} - IC start)",
                filename="per_step_metrics.png",
                sequence_info=data_config.get("sequence_info", [1, 1, 1]),
            )

            model_info_str, data_info_str, train_info_str, sched_info_str = build_info_strings(
                                                                                            model_obj=trainer.model,
                                                                                            data_config=data_config,
                                                                                            model_config=model_config,
                                                                                            train_config=train_config,
                                                                                            scheduler_config=scheduler_config
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
                orientation='vertical',
                input_array=inp_renorm,
                prediction_array=pred_renorm,
                target_array=tgt_renorm,
                input_channel_names=only_input_channel_names,
                output_channel_names=output_channel_names,
                conditioning_input_array=cond_inp_renorm,
                conditioning_channel_names=cond_inp_channel_names,
                checkpoint_step=None,
                epoch=None,
                extra_info=data_config.get("dataset_name")+"_Inference_plot_from_IC",
                ndim=ndim,
                slice_config=slice_config,
                num_examples=infer_config["n_infer_plot_examples"],
                stride=stride_val,
                save_dir=plot_save_dir,
                log_to_wandb=False,
                best_plot_at_train_end=False,
                layout_config=layout_config,
                include_relative_error=True,
                model_info=model_info_str,
                data_info=data_info_str,
                train_info=train_info_str
            )
            
            plotter.plot()

            # Prepare return payload for top-level multi-run plotting
            ic_return = {
                "metrics": per_rollout_step_metrics_ic,
                "sequence_info": list(data_config.get("sequence_info", [1, 1, 1])),
                "output_channel_names": output_channel_names,
            }

        print(f"Inference completed for {os.path.basename(experiment_dir)}")
        print(f"Results saved to: {solo_inference_dir}")

        return ic_return
        
    except Exception as e:
        print(f"Error processing {experiment_dir}: {str(e)}")
        import traceback
        traceback.print_exc()


@hydra.main(version_base="1.3", config_path="config/infer_config", config_name="only_inference")
def main(cfg: DictConfig):
    infer_config = cfg
    
    # Get all subdirectories in the inference directory
    inference_dir = infer_config.inference_directory
    if not os.path.exists(inference_dir):
        print(f"Error: Inference directory {inference_dir} does not exist!")
        return
    
    # Discover experiment directories supporting both flat and checkpoint_prefix layouts
    def discover_experiment_dirs(root_dir):
        discovered = []
        seen = set()

        def _add_if_run_dir(path_dir):
            if not os.path.isdir(path_dir):
                return
            cfg_path = os.path.join(path_dir, "data_config.json")
            if os.path.exists(cfg_path):
                key = os.path.realpath(path_dir)
                if key not in seen:
                    seen.add(key)
                    discovered.append(path_dir)

        # Case A: flat layout -> runs are direct children of root_dir
        for name in os.listdir(root_dir):
            child = os.path.join(root_dir, name)
            _add_if_run_dir(child)

        # Case B: root_dir/checkpoints/*
        checkpoints_dir = os.path.join(root_dir, "checkpoints")
        if os.path.isdir(checkpoints_dir):
            for name in os.listdir(checkpoints_dir):
                _add_if_run_dir(os.path.join(checkpoints_dir, name))

        # Case C: checkpoint_prefix layout -> root_dir/*/checkpoints/*
        for name in os.listdir(root_dir):
            prefix_dir = os.path.join(root_dir, name)
            if not os.path.isdir(prefix_dir):
                continue
            pref_ckpt = os.path.join(prefix_dir, "checkpoints")
            if os.path.isdir(pref_ckpt):
                for run_name in os.listdir(pref_ckpt):
                    _add_if_run_dir(os.path.join(pref_ckpt, run_name))

        # Fallback: recursive search for any data_config.json under root_dir
        if not discovered:
            for cfg_path in glob.glob(os.path.join(root_dir, "**", "data_config.json"), recursive=True):
                _add_if_run_dir(os.path.dirname(cfg_path))

        return discovered

    experiment_dirs = discover_experiment_dirs(inference_dir)
    
    if not experiment_dirs:
        print(f"No experiment directories found in {inference_dir}")
        return
    
    print(f"Found {len(experiment_dirs)} experiment directories:")
    for exp_dir in experiment_dirs:
        print(f"  - {os.path.basename(exp_dir)}")
    
    # Process each experiment directory and collect IC-start metrics for multi-run overlay (no aggregation)
    runs_step_metrics = {}
    runs_sequence_info = {}
    run_configs = {}
    sequence_info_ref = None
    output_channel_names_ref = None
    
    for experiment_dir in experiment_dirs:
        res = run_inference_for_each_experiment(experiment_dir, infer_config)
        if isinstance(res, dict) and res.get("metrics") is not None:
            run_label = os.path.basename(experiment_dir)
            runs_step_metrics[run_label] = res["metrics"]
            if sequence_info_ref is None and res.get("sequence_info") is not None:
                sequence_info_ref = res.get("sequence_info")
            # Keep per-run sequence info for correct timestep x-axis
            if res.get("sequence_info") is not None:
                runs_sequence_info[run_label] = res.get("sequence_info")
            if output_channel_names_ref is None and res.get("output_channel_names") is not None:
                output_channel_names_ref = res.get("output_channel_names")
        try:
            checkpoint_path = find_checkpoint_path(experiment_dir)
            if checkpoint_path:
                model_config_path = os.path.join(checkpoint_path, "config.json")
                if os.path.exists(model_config_path):
                    # Load the config as a dictionary
                    with open(model_config_path, 'r') as f:
                        run_config = json.load(f)
                    run_configs[os.path.basename(experiment_dir)] = run_config
                    print(f"Loaded config for {os.path.basename(experiment_dir)}")
        except Exception as e:
            print(f"Error loading config for {experiment_dir}: {str(e)}")

    # Create a single overlay plot of all runs in inference_dir if IC metrics are available
    # Here the overall all-channel combinedmetrics of each run are plotted and NOT the channel-wise metrics
    if bool(infer_config.get("infer_from_ic", False)) and len(runs_step_metrics) > 0:
        plot_multi_run_rollout_metrics(
            runs_step_metrics=runs_step_metrics,
            save_dir=inference_dir,
            title="Summary Plot: per-(rollout and time) step all channels combined metrics (IC start)",
            filename="all_runs_ic_rollout_timestep_metrics.png",
            sequence_info=sequence_info_ref if sequence_info_ref is not None else [1, 1, 1],
            runs_sequence_info=runs_sequence_info if len(runs_sequence_info) > 0 else None,
        )


    plot_rollout_metrics_bar_chart(
        runs_step_metrics=runs_step_metrics,
        save_dir=inference_dir,
        run_configs=run_configs,
        filename="rollout_metrics_bar_chart.png"
    )

    calculate_and_save_results_all_channels(
        runs_step_metrics=runs_step_metrics,
        save_dir=inference_dir,
        output_channel_names=output_channel_names_ref,
        filename="rollout_metrics_tabulated.csv"
    )


    print(f"\n{'='*60}")
    print("All inference runs completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
