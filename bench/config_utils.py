from __future__ import annotations

"""Utility functions to enrich and finalise the Hydra DictConfig before training."""

from datetime import datetime
from omegaconf import DictConfig
import os

from utils.feature_utils import get_grid_resolution
from utils.compute_stats import compute_statistics

__all__ = ["prepare_config"]


def prepare_config(cfg: DictConfig) -> DictConfig:  
    """Mutate *cfg* in-place, filling in any derived fields.

    1. Derive *output_dir* when not specified.
    2. Ensure grid resolution is populated.
    3. Compute normalisation statistics (if not provided).
    4. Derive the final channel list and in/out channel counts.
    """

    # ------------------------------------------------------------------
    # 1) Output directory ------------------------------------------------
    # ------------------------------------------------------------------
    out_dir = cfg["output_log_config"]["logging"]["output_dir"]

    if not out_dir:
        ts = datetime.now().strftime("%d%m%Y_%H%M%S")
        out_dir = (
            f"./checkpoints/{cfg['data_config']['dataset_name']}_"
            f"{cfg['data_config']['dimension']}D_{cfg['model_config']['model_name']}_{ts}"
        )

    # Prepend "HP_" if we are running hyper-parameter optimisation and the
    # directory name is not already prefixed.
    if cfg["hyperparam_opt_config"]["optimize"]:
        dir_path, base = os.path.split(out_dir)
        if not base.startswith("HP_"):
            out_dir = os.path.join(dir_path, f"HP_{base}") if dir_path else f"HP_{base}"

    cfg["output_log_config"]["logging"]["output_dir"] = out_dir

    # ------------------------------------------------------------------
    # 2) Grid resolution -------------------------------------------------
    # ------------------------------------------------------------------
    if cfg["data_config"]["grid_resolution"] is None:
        cfg["data_config"]["grid_resolution"] = get_grid_resolution(
            cfg["data_config"]["dataset_directory_path"]
        )

    # ------------------------------------------------------------------
    # 3) Normalisation stats --------------------------------------------
    # ------------------------------------------------------------------
    if cfg["data_config"]["data_normalization_stats"] is None:
        stats, channel_names, _ = compute_statistics(
            h5_paths=[f"{cfg['data_config']['dataset_directory_path']}/train.h5"], 
            residual_config=cfg["data_config"]["residual_config"],
            # the following arguments should be adjusted depending on the h5_paths 
            # (if multiple h5-files are provided)
            filter_groups=cfg["data_config"]["filter_groups"],
            filter_frames=cfg["data_config"]["filter_frames"],
            frame_stride=cfg["data_config"]["sequence_info"][2],
            on_fly_stats=False
        )
        cfg["data_config"]["data_normalization_stats"] = stats
    else:
        channel_names = list(cfg["data_config"]["data_normalization_stats"].keys())
        if (cfg["data_config"]["residual_config"] is not None) and (len(cfg["data_config"]["data_normalization_stats"]) != 2*len(channel_names)):
            raise ValueError(f"Insufficient statistics provided in the data_normalization_stats dictionary. Please provide statistics also for the residual channels.")
        # check if the desired quantities are present for the given normalization strategy
        if cfg["data_config"]["data_normalization_strategy"] == "z_normalization":
            missing_stats = {}
            for ch_name, stat_dict in cfg["data_config"]["data_normalization_stats"].items():
                missing = [k for k in ("mean", "std") if k not in stat_dict]
                if missing:
                    missing_stats[ch_name] = missing
                    
            if missing_stats:
                details = "; ".join(f"{ch}: {', '.join(miss)}" for ch, miss in missing_stats.items())
                raise ValueError(
                    "Insufficient statistics. The following keys are missing -> " + details
                )
        elif cfg["data_config"]["data_normalization_strategy"] == "min_max_normalization":
            missing_stats = {}
            for ch_name, stat_dict in cfg["data_config"]["data_normalization_stats"].items():
                missing = [k for k in ("min", "max") if k not in stat_dict]
                if missing:
                    missing_stats[ch_name] = missing

            if missing_stats:
                details = "; ".join(f"{ch}: {', '.join(miss)}" for ch, miss in missing_stats.items())
                raise ValueError(
                    "Insufficient statistics for min-max normalization. The following keys are missing -> "
                    + details
                )
        elif cfg["data_config"]["data_normalization_strategy"] == "robust_normalization":
            missing_stats = {}
            for ch_name, stat_dict in cfg["data_config"]["data_normalization_stats"].items():
                missing = [k for k in ("median", "iqr") if k not in stat_dict]
                if missing:
                    missing_stats[ch_name] = missing

            if missing_stats:
                details = "; ".join(f"{ch}: {', '.join(miss)}" for ch, miss in missing_stats.items())
                raise ValueError(
                    "Insufficient statistics for robust normalization. The following keys are missing -> "
                    + details
                )

    # ------------------------------------------------------------------
    # 4) Channel selection ----------------------------------------------
    # ------------------------------------------------------------------
    #NOTE: filter_in_channels has also the conditioning_in_channels (if any)
    filter_in_keywords = cfg["data_config"]["filter_in_channels"]
    filtered_in_channels = (
        [n for n in channel_names if any(n.startswith(k) for k in filter_in_keywords)]
        if filter_in_keywords
        else channel_names
    )

    filter_cond_in_keywords = cfg["data_config"]["conditioning_in_channels"]
    filtered_cond_in_channels = (
        [n for n in channel_names if any(n.startswith(k) for k in filter_cond_in_keywords)]
        if filter_cond_in_keywords
        else None
    )

    filter_out_keywords = cfg["data_config"]["filter_out_channels"]
    filtered_out_channels = (
        [n for n in channel_names if any(n.startswith(k) for k in filter_out_keywords)]
        if filter_out_keywords
        else channel_names
    )

    cfg["data_config"]["filter_in_channels"] = filtered_in_channels
    cfg["data_config"]["filter_out_channels"] = filtered_out_channels
    cfg["data_config"]["conditioning_in_channels"] = filtered_cond_in_channels

    return cfg 