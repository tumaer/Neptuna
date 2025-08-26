import os
import sys
import glob
import argparse
from typing import Dict, Any, List, Tuple

import torch
from omegaconf import OmegaConf

# Prefer absolute imports relative to project root
# Ensure project root is on sys.path for absolute package imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.load_model import fetch_model


def build_synthetic_data_config(dimension: int = 2) -> Dict[str, Any]:
    """
    Build a minimal data_config dictionary sufficient for model instantiation.

    The goal is not to reflect a real dataset but to provide required fields
    with sensible placeholder values so models can be constructed to compute
    parameter counts.
    """
    if dimension == 1:
        grid_resolution = [128]
    elif dimension == 2:
        grid_resolution = [64, 64]
    elif dimension == 3:
        grid_resolution = [32, 32, 32]
    else:
        raise ValueError("dimension must be 1, 2, or 3")

    data_config: Dict[str, Any] = {
        "dataset_name": "synthetic",
        "dimension": dimension,
        "grid_resolution": grid_resolution,
        # [input_len, label_len, stride]
        "sequence_info": [3, 1, 1],
        # In/out channel names are used only for their lengths
        "filter_features": {
            # For transient simulations, input = output + conditioning_in_channels
            # Here we provide 2 outputs and 1 additional input channel
            "filter_in_channels": ["density", "velocity_0", "velocity_1"],
            "filter_out_channels": ["velocity_0", "velocity_1"],
            "train_filter_frames": None,
            "train_filter_groups": None,
            "infer_filter_frames": None,
            "infer_filter_groups": None,
        },
        "conditioning_features": {
            # Default to no conditioning parameters
            "conditioning_in_channels": [],
            "include_conditioning_parameters": False,
            "num_cond_params": 0,
            "parameter_min_max_stats": None,
        },
        "residual_config": None,
        "data_normalization_stats": None,
        "data_normalization_strategy": "z_normalization",
        "coord_features": True,
    }

    return data_config


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def discover_model_config_files(root: str) -> List[str]:
    pattern = os.path.join(root, "**", "*.yaml")
    files = sorted(glob.glob(pattern, recursive=True))
    return files


def normalize_model_name(name: str) -> str:
    # fetch_model expects specific lowercase names
    return name.lower()


def to_millions_str(value: int) -> str:
    """Format a parameter count as millions with two decimals."""
    return f"{value / 1_000_000:.2f}"


def split_long_layer_name(layer_name: str, max_width: int = 60) -> List[str]:
    """
    Split a long layer name into multiple lines if it exceeds max_width.
    
    Args:
        layer_name: The layer name to split
        max_width: Maximum width per line
        
    Returns:
        List of strings, each representing a line
    """
    if len(layer_name) <= max_width:
        return [layer_name]
    
    # Try to split at dots for better readability
    parts = layer_name.split('.')
    lines = []
    current_line = ""
    
    for part in parts:
        # If adding this part would exceed max_width, start a new line
        if current_line and len(current_line + "." + part) > max_width:
            lines.append(current_line)
            current_line = part
        else:
            if current_line:
                current_line += "." + part
            else:
                current_line = part
    
    if current_line:
        lines.append(current_line)
    
    return lines


def get_layerwise_parameter_summary(model: torch.nn.Module) -> List[Tuple[str, int, int, int]]:
    """
    Get layerwise parameter summary similar to PyTorch Lightning's summary callback.
    
    Returns:
        List of (layer_name, total_params, trainable_params, non_trainable_params) tuples
        in the order they appear in the model traversal
    """
    summary = []
    
    # Group parameters by their module names, maintaining order
    layer_params = {}
    layer_order = []  # To maintain the order of appearance
    
    for name, param in model.named_parameters():
        # Extract the layer name (e.g., 'resnet.layer1.0.conv1.weight' -> 'resnet.layer1.0.conv1')
        if '.' in name:
            layer_name = '.'.join(name.split('.')[:-1])
        else:
            layer_name = name
            
        if layer_name not in layer_params:
            layer_params[layer_name] = {'total': 0, 'trainable': 0, 'non_trainable': 0}
            layer_order.append(layer_name)  # Record the order of appearance
        
        layer_params[layer_name]['total'] += param.numel()
        if param.requires_grad:
            layer_params[layer_name]['trainable'] += param.numel()
        else:
            layer_params[layer_name]['non_trainable'] += param.numel()
    
    # Convert to list maintaining the original order
    for layer_name in layer_order:
        counts = layer_params[layer_name]
        summary.append((
            layer_name,
            counts['total'],
            counts['trainable'],
            counts['non_trainable']
        ))
    
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Count parameters for CFD Bench models")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help=(
            "Filter by model name (e.g., unet, fno, resnet, dilresnet, "
            "deeponet_cnn, deeponet_ffn, deeponet_resnet, cno, scot, vit)"
        ),
    )
    args = parser.parse_args()
    model_filter = normalize_model_name(args.model) if args.model else None

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_cfg_root = os.path.join(project_root, "config", "model_config")

    cfg_files = discover_model_config_files(model_cfg_root)
    if len(cfg_files) == 0:
        print(f"No model configs found under: {model_cfg_root}")
        return

    results: List[Tuple[str, str, int, int]] = []  # (cfg_path, model_name, total, trainable)

    for cfg_path in cfg_files:
        try:
            cfg = OmegaConf.load(cfg_path)
            # Work with a plain dict for fetch_model
            model_config: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore

            # Ensure model_name normalized for the factory
            if "model_name" in model_config and isinstance(model_config["model_name"], str):
                model_config["model_name"] = normalize_model_name(model_config["model_name"])  # type: ignore
            else:
                raise KeyError("'model_name' missing in model config")

            # Apply optional filter
            if model_filter is not None and model_config["model_name"] != model_filter:
                continue

            # Choose dimension based on model constraints
            # ScOT supports only 2D per factory
            if model_config["model_name"] == "scot":
                data_config = build_synthetic_data_config(dimension=2)
            else:
                # Default to 2D which works for all implemented models
                data_config = build_synthetic_data_config(dimension=2)

            model = fetch_model(model_config=model_config, data_config=data_config)
            model.eval()
            total, trainable = count_parameters(model)
            results.append((cfg_path, model_config["model_name"], total, trainable))
        except Exception as exc:
            # Print errors only when a specific model filter is requested
            if model_filter is not None:
                cfg_rel = os.path.relpath(cfg_path, project_root)
                print(f"[ERROR] {cfg_rel}: {type(exc).__name__}: {exc}")
            continue

    # Pretty print results
    header = f"{'Config File':<60}  {'Model':<16}  {'Total (M)':>12}  {'Trainable (M)':>15}"
    print(header)
    print("-" * len(header))
    for cfg_path, model_name, total, trainable in results:
        cfg_rel = os.path.relpath(cfg_path, project_root)
        print(f"{cfg_rel:<60}  {model_name:<16}  {to_millions_str(total):>12}  {to_millions_str(trainable):>15}")

    if not results:
        if model_filter is not None:
            print(f"No configs matched model '{model_filter}'.")
        return

    # If filtering by specific model, show layerwise breakdown
    if model_filter is not None and len(results) > 0:
        print(f"\nLayerwise parameter breakdown for model '{model_filter}':")
        print("=" * 85)
        
        # Show layerwise breakdown for each matching config file
        for i, (cfg_path, model_name, total, trainable) in enumerate(results):
            print(f"\nConfig {i+1}: {os.path.basename(cfg_path)}")
            print("-" * 95)
            
            try:
                # Reload the model for layerwise analysis
                cfg = OmegaConf.load(cfg_path)
                model_config = OmegaConf.to_container(cfg, resolve=True)
                model_config["model_name"] = normalize_model_name(model_config["model_name"])
                
                if model_config["model_name"] == "scot":
                    data_config = build_synthetic_data_config(dimension=2)
                else:
                    data_config = build_synthetic_data_config(dimension=2)
                
                model = fetch_model(model_config=model_config, data_config=data_config)
                model.eval()
                
                layer_summary = get_layerwise_parameter_summary(model)
                
                # Print layerwise summary with proper alignment
                print(f"{'Layer Type':<60} {'Total Params':>15} {'Trainable':>15} {'Non-Trainable':>15}")
                print("-" * 105)
                
                # Add summary row showing total parameters for the whole model
                print(f"{'TOTAL':<60} {to_millions_str(total):>15}M {to_millions_str(trainable):>15}M {to_millions_str(total - trainable):>15}M")
                print("-" * 105)
                
                for layer_name, total_params, trainable_params, non_trainable_params in layer_summary:
                    # Split layer name into lines if it's too long
                    split_layer_name = split_long_layer_name(layer_name)
                    
                    # Print first line with parameter counts
                    print(f"{split_layer_name[0]:<60} {total_params:>15} {trainable_params:>15} {non_trainable_params:>15}")
                    
                    # Print continuation lines with proper indentation
                    for line in split_layer_name[1:]:
                        print(f"  {line:<58}")
                
            except Exception as exc:
                print(f"[ERROR] Could not generate layerwise summary for {os.path.basename(cfg_path)}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()


