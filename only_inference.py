
from omegaconf import OmegaConf
import os
import glob
import atexit
from utils.load_data import fetch_dataset
from utils.plot_progress import build_info_strings
from utils.plot_progress import preprocess_for_plotting, plot_rollout_metrics
from utils.plot_progress import LayoutConfig, Slice3DConfig, create_plotter
from utils.plot_progress import plot_rollout_metrics_bar_chart, calculate_and_save_results_all_channels, strip_validation_loss
from utils.plot_progress import plot_multi_run_rollout_metrics
from utils.loss_utils import fetch_loss_metric, fetch_infer_loss_dict
from metrics.inference_metrics import compute_metrics_for_n_rollouts
from transformers.trainer import EvalPrediction
from transformers import TrainingArguments
from train.trainer import Trainer
import hydra
from omegaconf import DictConfig
from typing import Dict, List, Optional
import json
from utils.seed_utils import set_global_seed
import torch
import torch.distributed as dist
import time
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
        "unettransformer": ("models.UNetTransformer.unettransformer", "UNetTransformer"),
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

def build_train_and_infer_loss(loss_config, data_config, device: torch.device):
    """
    Construct training and eval CompositeLoss from configs.
    Device for metric computation is passed in (from infer_config.metrics_device).
    """
    if loss_config is not None:
        train_loss_dict = loss_config.train_loss
        train_loss_fn = fetch_loss_metric(data_config, train_loss_dict).to(device)
    else:
        train_loss_fn = None

    infer_loss_dict = fetch_infer_loss_dict(data_config)
    infer_loss_fn = fetch_loss_metric(data_config, infer_loss_dict).to(device)

    return train_loss_fn, infer_loss_fn, infer_loss_dict

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

    curriculum_block = {
        "name": "block_1",
        "start_epoch": 0,
    }
    if loss_config is not None:
        curriculum_block.update(OmegaConf.to_container(loss_config, resolve=True))
    else:
        curriculum_block["train_loss"] = None
    train_strategy_config = OmegaConf.create(
        {
            "curriculum": [curriculum_block],
            "num_train_epochs": 5,
        }
    )

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
        train_strategy_config=train_strategy_config,
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

def save_overall_errors_to_csv(errors, output_dir):
    """
    Save overall errors to a CSV file named overall_results.csv in the specified output directory.
    Args:
        errors: Dictionary of errors to save.
        output_dir: Directory where the CSV file will be saved.
    """
    file_name = "overall_results.csv"
    csv_file = os.path.join(output_dir, file_name)
    file_is_empty = not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header only if the file is empty
        if file_is_empty:
            writer.writerow(["Metric", "Value"])
        for key, value in errors.items():
            writer.writerow([key, value])

def save_errors_to_structured_csv(errors, output_dir, channel_names: Optional[List[str]] = None, file_name: str = "results_structured.csv"):
    """
    Save 'normalized' errors to a structured CSV named results_structured.csv in the specified output directory.
    Args:
        errors: Dictionary of errors to save.
        output_dir: Directory where the CSV file will be saved.
    """
    # Structured CSV file
    # --------------------------------------------------------------
    # Collect data into a dict-of-dicts before writing
    # rows[(index_type, index)][col_name] = value
    rows: Dict[tuple, Dict[str, float]] = {}
    all_col_names: set = set()

    metric_component_order: Dict[str, List[str]] = {}

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

        metric_names = None
        if isinstance(metrics_dict, dict):
            metric_names = metrics_dict.get("names")
            if isinstance(metric_names, list) and metric_names:
                metric_component_order[metric_name] = metric_names

        for summary_kind, arr in metrics_dict.items():
            if summary_kind == "names":
                continue
            if summary_kind == "component_names":
                continue
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
            effective_names = metric_names if metric_names else channel_names
            for ch in range(n_channels - 1):
                ch_label = (
                    effective_names[ch] if effective_names is not None and ch < len(effective_names) else f"ch{ch}"
                )
                ch_col = f"{base}__{ch_label}"
                all_col_names.add(ch_col)

            for idx in range(n_idx):
                row_key = (index_type, idx)
                if row_key not in rows:
                    rows[row_key] = {}

                # overall column from last channel
                rows[row_key][overall_col] = float(np_arr[idx, n_channels - 1])

                # per-channel columns
                effective_names = metric_names if metric_names else channel_names
                for ch in range(n_channels - 1):
                    ch_label = (
                        effective_names[ch] if effective_names is not None and ch < len(effective_names) else f"ch{ch}"
                    )
                    ch_col = f"{base}__{ch_label}"
                    rows[row_key][ch_col] = float(np_arr[idx, ch])

    # If nothing to write, just ensure header exists
    pretty_csv_file = os.path.join(output_dir, file_name)

    # Sort columns so mean/std for each channel sit next to each other
    def _parse_col(name: str):
        parts = name.split("__")
        if len(parts) < 3:
            return name, "", ""
        metric, stat, *channel_parts = parts
        channel = "__".join(channel_parts) if channel_parts else ""
        return metric, stat, channel

    # Keep channels together: overall first, then provided channel_names order,
    # then any leftover channels in lexicographic order.
    channel_order_map: Dict[str, int] = {}
    if channel_names:
        channel_order_map = {ch: i + 1 for i, ch in enumerate(channel_names)}

    def _channel_rank(channel: str, channel_order_map: Dict[str, int]):
        if channel == "overall":
            return (0, "overall")
        if channel in channel_order_map:
            return (channel_order_map[channel], channel)
        if channel.startswith("ch"):
            try:
                num = int(channel[2:])
                return (len(channel_order_map) + 1 + num, channel)
            except ValueError:
                pass
        # unknown channel, push to the end but keep deterministic
        return (len(channel_order_map) + 100, channel)

    def _col_key(name: str):
        metric, stat, channel = _parse_col(name)
        is_cumulative = 1 if stat.startswith("cumulative_") else 0
        stat_base = stat.replace("cumulative_", "")
        stat_rank = {"mean": 0, "std": 1}.get(stat_base, 99)
        component_names = metric_component_order.get(metric)
        if component_names:
            component_order_map = {comp: i + 1 for i, comp in enumerate(component_names)}
            comp_rank_idx, comp_rank_name = _channel_rank(channel, component_order_map)
            # Component-wise order: group by metric, then component, then stat
            return (is_cumulative, 0, metric, comp_rank_idx, comp_rank_name, stat_rank, stat, name)

        ch_rank_idx, ch_rank_name = _channel_rank(channel, channel_order_map)
        # Channel-wise order (interleaved by channel across metrics)
        return (is_cumulative, 1, ch_rank_idx, ch_rank_name, metric, stat_rank, stat, name)

    all_col_names_sorted = sorted(all_col_names, key=_col_key)

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
    if dist.is_available() and dist.is_initialized():
        RANK = dist.get_rank()
        IS_MAIN_PROCESS = RANK == 0
    else:
        RANK = int(os.environ.get("RANK", -1))
        IS_MAIN_PROCESS = RANK in [-1, 0]

    def _debug_enabled() -> bool:
        if os.environ.get("INFERENCE_DEBUG", "").lower() in ("1", "True", "true"):
            return True
        val = infer_config.get("debug", False)
        if isinstance(val, str):
            return val.lower() in ("1", "True", "true")
        return bool(val)

    _DEBUG = _debug_enabled()

    def _dbg(message: str, main_only: bool = False):
        if not _DEBUG:
            return
        if main_only and not IS_MAIN_PROCESS:
            return
        print(f"[inference][rank={RANK}] {message}", flush=True)

    _dbg(
        "enter run_inference_for_each_experiment | "
        f"dist_initialized={dist.is_available() and dist.is_initialized()} | "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE', '1')} | experiment_dir={experiment_dir}"
    )
    if IS_MAIN_PROCESS:
        print(f"\n{'#'*88}")
        BOX_WIDTH = 88
        header_sep = "*" * BOX_WIDTH
        print("\n" + header_sep)
        print(f"Processing experiment: {os.path.basename(experiment_dir)}")
        print(header_sep)
    
    # Check if data_config.json exists
    data_config_path = os.path.join(experiment_dir, "data_config.json")
    _dbg(f"checking data config at: {data_config_path}")
    if not os.path.exists(data_config_path):
        if IS_MAIN_PROCESS:
            print(f"Warning: data_config.json not found in {experiment_dir}. Skipping...")
        return
    
    # Find checkpoint directory
    checkpoint_path = find_checkpoint_path(experiment_dir)
    _dbg(f"resolved checkpoint_path={checkpoint_path}")
    if not checkpoint_path:
        if IS_MAIN_PROCESS:
            print(f"Warning: No checkpoint folder found in {experiment_dir}. Skipping...")
        return
    
    # Check if config.json exists in checkpoint
    model_config_path = os.path.join(checkpoint_path, "config.json")
    _dbg(f"checking model config at: {model_config_path}")
    if not os.path.exists(model_config_path):
        if IS_MAIN_PROCESS:
            print(f"Warning: config.json not found in {checkpoint_path}. Skipping...")
        return
    
    if IS_MAIN_PROCESS:
        print(f"Data config: {data_config_path}")
        print(f"Model checkpoint: {checkpoint_path}")
    
    #try:
    # Load configurations
    data_config = OmegaConf.load(data_config_path)
    # Post-load fix: convert parameter_min_max_stats keys to integers
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
            if 'train_loss' in loss_config and 'components' in loss_config.train_loss:
                for component in loss_config.train_loss.components:
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
                        
                        if IS_MAIN_PROCESS:
                            print(f" Using checkpoint weights for '{component.get('name', component.type)}'")
                        
        except Exception as e:
            import traceback
            traceback.print_exc()
            loss_config = None
    else:
        if IS_MAIN_PROCESS:
            print(f"No loss_config.json found in checkpoint: {loss_config_path}")
            print("Inference will only compute legacy L1/L2 errors")  # ? Check this

    loss_config_for_plotting = strip_validation_loss(loss_config)

    # Determine metric device from infer_config.metrics_device
    _dev_cfg = infer_config.get("metrics_device") if infer_config else None
    if _dev_cfg is not None and str(_dev_cfg).strip():
        metric_device = torch.device(str(_dev_cfg).strip())
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        metric_device = torch.device("xpu:0")
    elif torch.cuda.is_available():
        metric_device = torch.device("cuda:0")
    else:
        metric_device = torch.device("cpu")

    # Build loss functions
    train_loss_fn, infer_loss_fn, _ = build_train_and_infer_loss(
        loss_config=loss_config,
        data_config=data_config,
        device=metric_device,
    )

    # Define metrics for trainer
    def compute_metrics_during_inference(eval_pred: EvalPrediction):
        print("compute_metrics_during_inference: eval_pred is not None")
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
            print("compute_metrics_during_inference: train_loss_fn is not None")
            try:
                with torch.no_grad():
                    composite_loss = train_loss_fn(
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
                print("finished computing composite train loss")
            except Exception as e:
                if IS_MAIN_PROCESS:
                    print(f"Failed to compute composite loss metrics: {e}")

        # 2) Evaluation loss components for logging
        if infer_loss_fn is not None:
            print("compute_metrics_during_inference: infer_loss_fn is not None")
            print("device: ", metric_device)
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
                print("finished computing infer loss metrics")
            except Exception as e:
                if IS_MAIN_PROCESS:
                    print(f"Failed to compute evaluation loss metrics: {e}")

        return metrics
    
    # Set global seed from data_config (default 0)
    seed_value = int(data_config.get("seed", 0))
    set_global_seed(seed_value, deterministic=True)
    
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

    def extract_configs_from_training_args(checkpoint_path, *, _is_main: bool = True):
        training_args_path = os.path.join(checkpoint_path, "training_args.bin")
        if not os.path.exists(training_args_path):
            return None, None
        try:
            args_obj = torch.load(training_args_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            if _is_main:
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
            "dataloader_num_workers": int(_safe_get(args_obj, "dataloader_num_workers", 2)),
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

    if IS_MAIN_PROCESS:
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
    
    def _write_marker_file(base_dir: str, selected_dir: str) -> None:
        """Write marker file with solo_inference directory path for this run."""
        marker_path = os.path.join(base_dir, ".solo_inference_dir.path")
        temp_marker_path = f"{marker_path}.tmp"
        with open(temp_marker_path, "w", encoding="utf-8") as marker_file:
            marker_file.write(selected_dir)
        os.replace(temp_marker_path, marker_path)

    def get_inference_dir_main_process_only(base_dir):
        # In distributed mode, create directory on rank 0 and share path with all ranks.
        if dist.is_available() and dist.is_initialized():
            _dbg("selecting solo_inference dir via dist.broadcast_object_list")
            dir_holder = [None]
            if dist.get_rank() == 0:
                dir_holder[0] = create_unique_inference_dir(base_dir)
                _write_marker_file(base_dir, dir_holder[0])
                _dbg(f"rank0 created solo_inference directory: {dir_holder[0]}")
            dist.broadcast_object_list(dir_holder, src=0)
            _dbg(f"received distributed solo_inference directory: {dir_holder[0]}")
            return dir_holder[0]
        
        # Before torch.distributed is initialized (e.g., early in torchrun startup),
        # coordinate with env ranks via a marker file written by rank 0.
        marker_path = os.path.join(base_dir, ".solo_inference_dir.path")
        env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        env_rank = int(os.environ.get("RANK", "-1"))
        _dbg(
            "selecting solo_inference dir via pre-init env coordination | "
            f"env_world_size={env_world_size} env_rank={env_rank}"
        )
        if env_world_size > 1:
            _dbg(f"using marker path: {marker_path}")
            if env_rank == 0:
                selected_dir = create_unique_inference_dir(base_dir)
                _write_marker_file(base_dir, selected_dir)
                _dbg(f"rank0 published solo_inference directory: {selected_dir}")
                return selected_dir

            timeout_seconds = 120
            start_time = time.time()
            _dbg("non-main rank waiting for marker file from rank0")
            while True:
                if os.path.exists(marker_path):
                    with open(marker_path, "r", encoding="utf-8") as marker_file:
                        selected_dir = marker_file.read().strip()
                    if selected_dir and os.path.exists(selected_dir):
                        _dbg(f"non-main rank received solo_inference directory: {selected_dir}")
                        return selected_dir
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError(
                        f"Timed out waiting for rank 0 to publish solo inference directory at {marker_path}"
                    )
                time.sleep(0.1)

        # Single-process mode: create dir and marker for each run
        selected_dir = create_unique_inference_dir(base_dir)
        _write_marker_file(base_dir, selected_dir)
        _dbg(f"single-process mode selected solo_inference directory: {selected_dir}")
        return selected_dir

    # Create a unique solo_inference directory within the experiment directory
    solo_inference_dir = get_inference_dir_main_process_only(experiment_dir)
    _dbg(f"using solo_inference_dir={solo_inference_dir}")
    
    # Override filter parameters from infer_config if they are specified (not None)
    infer_filter_groups = data_config["filter_features"]["infer_filter_groups"]
    infer_filter_frames = data_config["filter_features"]["infer_filter_frames"]
    
    # Check if infer_config has filter_features and override if not None
    if "filter_features" in infer_config:
        if infer_config["filter_features"].get("infer_filter_groups") is not None:
            if IS_MAIN_PROCESS:
                print(f"Original infer_filter_groups from data_config: {infer_filter_groups}")
            infer_filter_groups = infer_config["filter_features"]["infer_filter_groups"]
            if IS_MAIN_PROCESS:
                print(f"** Using infer_filter_groups from inference config: {infer_filter_groups} **")
        
        if infer_config["filter_features"].get("infer_filter_frames") is not None:
            if IS_MAIN_PROCESS:
                print(f"Original infer_filter_frames from data_config: {infer_filter_frames}")
            infer_filter_frames = infer_config["filter_features"]["infer_filter_frames"]
            if IS_MAIN_PROCESS:
                print(f"** Using infer_filter_frames from inference config: {infer_filter_frames} **")
    

    if infer_config["dataset_directory_path"] is not None:
        dataset_directory_path = infer_config["dataset_directory_path"]
    else:
        dataset_directory_path = data_config["dataset_directory_path"]

    if IS_MAIN_PROCESS:
        print("Running solo inference...")
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
                                            include_conditioning_parameters=False if data_config["conditioning_features"]["conditioning_method"] is None else True,
                                            parameter_min_max_stats=data_config["conditioning_features"]["parameter_min_max_stats"],
                                            data_normalization_stats=data_config["data_normalization_stats"],
                                            data_normalization_strategy=data_config["data_normalization_strategy"],
                                            is_steady_state_prediction=data_config["is_steady_state_prediction"],
                                            residual_config=data_config["residual_config"],
                                            n_infer_rollouts=infer_config["n_infer_rollouts"],
                                            infer_from_random_timestep=infer_config["infer_from_random_timestep"],
                                            infer_from_ic=infer_config["infer_from_ic"],
                                            log_transform_channels=data_config.get("log_transform_channels", []),
                                            )
    _dbg(
        "dataset loaded | "
        f"infer_ds_len={len(infer_ds) if infer_ds is not None else 'None'} | "
        f"infer_ds_from_ic_len={len(infer_ds_from_ic) if infer_ds_from_ic is not None else 'None'}",
        main_only=True,
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
        compute_metrics=compute_metrics_during_inference,
    )
    _dbg("trainer initialized for inference")

    random_start_stats_dict = None
    ic_start_stats_dict = None
    
    if infer_config["infer_from_random_timestep"]:
        if IS_MAIN_PROCESS:
            print(" \n Running inference rollouts using random windows sliced across the test trajectory...")
        _dbg("starting random-start inference branch")
        trainer.set_eval_or_test_rollout_steps(
            rollout_steps=infer_config["n_infer_rollouts"], output_all_steps=True
        )
        _dbg(
            f"configured rollout steps for random-start: {infer_config['n_infer_rollouts']}",
            main_only=True,
        )

        predictions_obj, inputs, conditioning_inputs = trainer.predict(infer_ds, metric_key_prefix="")
        _dbg(
            "trainer.predict(random-start) completed | "
            f"pred_shape={getattr(predictions_obj.predictions, 'shape', None)} | "
            f"label_shape={getattr(predictions_obj.label_ids, 'shape', None)} | "
            f"metrics_count={len(predictions_obj.metrics) if predictions_obj.metrics is not None else 0}",
            main_only=True,
        )
        ############################################################
        # predictions_obj.predictions: the output of the model with shape (accumulated_outputs, num_rollouts, label_seq_length, channel_dim, *spatial) 
        # accumulated_outputs and accumulated_gt have the length of number of windows in the test dataset
        # predictions_obj.label_ids: the ground truth with shape (accumulated_gt, num_rollouts*label_seq_length, channel_dim, *spatial)
        # predictions_obj.metrics: the metrics computed after accumulating the outputs and ground truth
        ############################################################

        if IS_MAIN_PROCESS:
            # pretty print the keys which have the word error in them
            print('Accumulated error for the whole test set (random start):')
            overall_errors = {} 
            for key, value in predictions_obj.metrics.items():
                if key.endswith(("runtime", "samples_per_second", "steps_per_second", "model_preparation_time", "jit_compilation_time")):
                    continue
                print(f"{key}: {value}")
                overall_errors["random_start_"+key] = value
            save_overall_errors_to_csv(overall_errors, solo_inference_dir)
        # ----------------------------------------------------------
        # Prepare prediction, target and input arrays
        # ----------------------------------------------------------
        #if IS_MAIN_PROCESS:
            preds = predictions_obj.predictions  # (N, R, T, C, *spatial)

            # Flatten rollout and label sequence dimensions if necessary
            if preds.ndim >= 5:
                n, n_rollouts, seq_len, c = preds.shape[:4]
                outputs_per_rollout = seq_len
                extra_dims = preds.shape[4:]
                preds = preds.reshape(n, n_rollouts * seq_len, c, *extra_dims) # (N, R*T, C, *spatial)

            targets = predictions_obj.label_ids  # Expected shape: (N, R*T, C, *spatial)

            per_rollout_step_metrics_random = compute_metrics_for_n_rollouts(
                preds,
                targets,
                outputs_per_rollout=outputs_per_rollout,
                include_per_timestep=True,
                loss_metric=infer_loss_fn,
                device=infer_config.get("metrics_device"),
                input_frames=inputs,
                metric_batch_size=infer_config.get("metrics_batch_size"),
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
                data_config=data_config,
                dataset=infer_ds,
                residual_config=data_config.get("residual_config", None),
                conditioning_inputs=cond_inp_arr,
            )
            per_step_errors = {}
            for metric_name, values in per_rollout_step_metrics_random.items():
                if IS_MAIN_PROCESS:
                    print(f"{metric_name} per-step (random start): {values}")
                per_step_errors[metric_name] = values
            save_errors_to_structured_csv(per_step_errors, 
                                            solo_inference_dir, 
                                            channel_names=output_channel_names, 
                                            file_name="results_structured_random_start.csv")

        # Infer spatial dimensionality (1D / 2D / 3D)
            ndim = pred_renorm.ndim - 3  # subtract l, time, channel dims

        # Use stride from the config if available
            stride_val = data_config.get("sequence_info")[2]

        # Directory for saving inference plots
            plot_save_dir = os.path.join(solo_inference_dir, "inference_plots/random_start")

        # Build formatted info strings  
            model_info_str, data_info_str, train_info_str, _ = build_info_strings(
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
                    train_info=train_info_str,
                    loss_config=loss_config_for_plotting
                )
                
                plotter.plot()

            # Create only rollout metrics (and not timestep metrics) plot for this example (no batch aggregation)
            # Reason: timestep metrics are useful in case 
                ex_preds = preds[example_idx:example_idx+1]      # shape (1, R*T, C, *spatial)
                ex_targets = targets[example_idx:example_idx+1]
                
                per_rollout_metrics_ex = compute_metrics_for_n_rollouts(
                    ex_preds,
                    ex_targets,
                    outputs_per_rollout=outputs_per_rollout,
                    loss_metric=infer_loss_fn,
                    include_per_timestep=False,
                    input_frames=inp_arr[example_idx:example_idx+1],
                    device=infer_config.get("metrics_device"),
                    metric_batch_size=infer_config.get("metrics_batch_size"),
                )
                ex_title = f"Per-rollout metrics ({data_config.get('dataset_name', 'dataset')} - random start, example {int(example_idx)})"
                plot_rollout_metrics(
                    step_metrics=per_rollout_metrics_ex,
                    output_channel_names=output_channel_names,
                    save_dir=ex_save_dir,
                    title=ex_title,
                    filename=f"Metric_evolution_for_random_Ex_{int(example_idx)}.png", 
                    sequence_info=data_config.get("sequence_info"),
                )

            random_start_stats_dict = {
                "metrics": per_rollout_step_metrics_random,
                "sequence_info": list(data_config.get("sequence_info")),
                "output_channel_names": output_channel_names,
            }
            runs_step_metrics = {}
            run_label = os.path.basename(experiment_dir)
            runs_step_metrics[run_label] = random_start_stats_dict["metrics"]

            calculate_and_save_results_all_channels(
                runs_step_metrics=runs_step_metrics,
                save_dir=os.path.dirname(experiment_dir),
                output_channel_names=output_channel_names,
                filename="rollout_metrics_random_start_tabulated.csv"
            )

        if dist.is_available() and dist.is_initialized():
            _dbg("entering dist.barrier() after random-start branch")
            dist.barrier()
            _dbg("finished dist.barrier() after random-start branch")

    if infer_config["infer_from_ic"]:
        if IS_MAIN_PROCESS:
            print(" \n Running inference rollouts using windows starting from the initial conditions...")
        _dbg("starting ic-start inference branch")
        trainer.set_eval_or_test_rollout_steps(
            rollout_steps=infer_config["n_infer_rollouts"], output_all_steps=True
        )
        # ----------------------------------------------------------
        # Prepare prediction, target and input arrays
        # ----------------------------------------------------------
        predictions_obj, inputs, conditioning_inputs = trainer.predict(infer_ds_from_ic, metric_key_prefix="")
        _dbg(
            "trainer.predict(ic-start) completed | "
            f"pred_shape={getattr(predictions_obj.predictions, 'shape', None)} | "
            f"label_shape={getattr(predictions_obj.label_ids, 'shape', None)} | "
            f"metrics_count={len(predictions_obj.metrics) if predictions_obj.metrics is not None else 0}",
            main_only=True,
        )
        #predictions and labels have NO log-transformed channels

        if IS_MAIN_PROCESS:
            print('Accumulated error for the whole test set (IC start):')
            errors = {}
            for key, value in predictions_obj.metrics.items():
                if key.endswith(("runtime", "samples_per_second", "steps_per_second", "model_preparation_time", "jit_compilation_time")):
                    continue
                print(f"{key}: {value}")
                errors["ic_start_"+key] = value
            save_overall_errors_to_csv(errors, solo_inference_dir)
            
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
                preds,
                targets,
                outputs_per_rollout=outputs_per_rollout,
                include_per_timestep=True,
                loss_metric=infer_loss_fn,
                input_frames=inp_arr,
                device=infer_config.get("metrics_device"),
                metric_batch_size=infer_config.get("metrics_batch_size"),
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
                data_config=data_config,
                dataset=infer_ds,
                residual_config=data_config.get("residual_config", None),
                conditioning_inputs=cond_inp_arr,
            )

            errors = {}
            for metric_name, values in per_rollout_step_metrics_ic.items():
                if IS_MAIN_PROCESS:
                    print(f"{metric_name} per-step (IC start): {values}")
                errors[metric_name] = values
            save_errors_to_structured_csv(errors, solo_inference_dir, channel_names=output_channel_names, file_name="results_structured_ic_start.csv")

        # Infer spatial dimensionality (1D / 2D / 3D)
            ndim = pred_renorm.ndim - 3  # subtract batch, time, channel dims

            stride_val = data_config.get("sequence_info")[2]

            plot_save_dir = os.path.join(solo_inference_dir, "inference_plots/ic_start")

            plot_rollout_metrics(
                step_metrics=per_rollout_step_metrics_ic,
                output_channel_names=output_channel_names,
                save_dir=plot_save_dir,
                title=f"Per-(rollout and time) step metrics ({data_config.get('dataset_name', 'dataset')} - IC start)",
                filename="per_step_metrics.png",
                sequence_info=data_config.get("sequence_info"),
            )

            model_info_str, data_info_str, train_info_str, _ = build_info_strings(
                                                                                    model_obj=trainer.model,
                                                                                    data_config=data_config,
                                                                                    model_config=model_config,
                                                                                    train_config=train_config,
                                                                                    scheduler_config=scheduler_config
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
                train_info=train_info_str,
                loss_config=loss_config_for_plotting
            )
            
            plotter.plot()

            ic_start_stats_dict = {
                "metrics": per_rollout_step_metrics_ic,
                "sequence_info": list(data_config.get("sequence_info")),
                "output_channel_names": output_channel_names,
            }
            runs_step_metrics = {}
            run_label = os.path.basename(experiment_dir)
            runs_step_metrics[run_label] = ic_start_stats_dict["metrics"]

            calculate_and_save_results_all_channels(
                runs_step_metrics=runs_step_metrics,
                save_dir=os.path.dirname(experiment_dir),
                output_channel_names=output_channel_names,
                filename="rollout_metrics_ic_start_tabulated.csv"
            )

        if dist.is_available() and dist.is_initialized():
            _dbg("entering dist.barrier() after ic-start branch")
            dist.barrier()
            _dbg("finished dist.barrier() after ic-start branch")

    if IS_MAIN_PROCESS:
        print(f"Inference completed for {os.path.basename(experiment_dir)}")
        print(f"Results saved to: {solo_inference_dir}")

        # Clean up marker file used for pre-init distributed coordination
        marker_path = os.path.join(experiment_dir, ".solo_inference_dir.path")
        try:
            if os.path.exists(marker_path):
                os.remove(marker_path)
        except OSError:
            pass

    _dbg("leaving run_inference_for_each_experiment")

    # Return both if available
    result_dict = {}
    if random_start_stats_dict is not None:
        result_dict["random_start"] = random_start_stats_dict
    if ic_start_stats_dict is not None:
        result_dict["ic_start"] = ic_start_stats_dict

    return result_dict

def _cleanup_distributed() -> None:
    """Best-effort teardown so distributed jobs exit without leaking resources."""
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


@hydra.main(version_base="1.3", config_path="config/infer_config", config_name="only_inference")
def main(cfg: DictConfig):
    infer_config = cfg

    # Register distributed cleanup on exit to avoid NCCL warning
    atexit.register(_cleanup_distributed)

    if dist.is_available() and dist.is_initialized():
        IS_MAIN_PROCESS = dist.get_rank() == 0
    else:
        IS_MAIN_PROCESS = int(os.environ.get("RANK", -1)) in [-1, 0]

    # Get all subdirectories in the inference directory
    inference_dir = infer_config.inference_directory
    if not os.path.exists(inference_dir):
        if IS_MAIN_PROCESS:
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
        if IS_MAIN_PROCESS:
            print(f"No experiment directories found in {inference_dir}")
        return

    if IS_MAIN_PROCESS:
        print(f"Found {len(experiment_dirs)} experiment directories:")
        for exp_dir in experiment_dirs:
            print(f"  - {os.path.basename(exp_dir)}")
    
    # Process each experiment directory and collect IC-start metrics for multi-run overlay (no aggregation)
    runs_step_metrics_random_start = {}
    runs_step_metrics_ic_start = {}
    runs_sequence_info = {}
    run_configs = {}
    
    for experiment_dir in experiment_dirs:
        res = run_inference_for_each_experiment(experiment_dir, infer_config)
        run_label = os.path.basename(experiment_dir)
        if res.get("random_start"):
            runs_step_metrics_random_start[run_label] = res["random_start"]["metrics"]
            runs_sequence_info[run_label] = res["random_start"]["sequence_info"]
        if res.get("ic_start"):
            runs_step_metrics_ic_start[run_label] = res["ic_start"]["metrics"]
            runs_sequence_info[run_label] = res["ic_start"]["sequence_info"]
        try:
            checkpoint_path = find_checkpoint_path(experiment_dir)
            if checkpoint_path:
                model_config_path = os.path.join(checkpoint_path, "config.json")
                if os.path.exists(model_config_path):
                    # Load the config as a dictionary
                    with open(model_config_path, 'r') as f:
                        run_config = json.load(f)
                    run_configs[os.path.basename(experiment_dir)] = run_config
                    if IS_MAIN_PROCESS:
                        print(f"Loaded config for {os.path.basename(experiment_dir)}")
        except Exception as e:
            if IS_MAIN_PROCESS:
                print(f"Error loading config for {experiment_dir}: {str(e)}")

    # Create a single overlay plot of all runs in inference_dir if IC metrics are available
    # Here the overall all-channel combinedmetrics of each run are plotted and NOT the channel-wise metrics
    if bool(infer_config.get("infer_from_ic")) and len(runs_step_metrics_ic_start) > 0:
        plot_multi_run_rollout_metrics(
            runs_step_metrics=runs_step_metrics_ic_start,
            save_dir=inference_dir,
            title="Summary Plot: per-(rollout and time) step all channels combined metrics (IC start)",
            filename="all_runs_ic_rollout_timestep_metrics.png",
            runs_sequence_info=runs_sequence_info if len(runs_sequence_info) > 0 else None,
        )

    # Bar charts for both IC-start and random-start runs
    if len(runs_step_metrics_ic_start) > 0:
        plot_rollout_metrics_bar_chart(
            runs_step_metrics=runs_step_metrics_ic_start,
            save_dir=inference_dir,
            run_configs=run_configs,
            filename="rollout_metrics_ic_start_bar_chart.png"
        )
    if len(runs_step_metrics_random_start) > 0:
        plot_rollout_metrics_bar_chart(
            runs_step_metrics=runs_step_metrics_random_start,
            save_dir=inference_dir,
            run_configs=run_configs,
            filename="rollout_metrics_random_start_bar_chart.png"
        )

    if IS_MAIN_PROCESS:
        print(f"\n{'='*60}")
        print("All inference runs completed!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
