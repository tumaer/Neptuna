from __future__ import annotations

"""Utility functions to enrich and finalise the Hydra DictConfig before training."""

from datetime import datetime
from omegaconf import DictConfig
import os
import json
import time

from omegaconf import OmegaConf
from utils.grid_utils import get_grid_resolution
from utils.compute_stats import compute_statistics_parallel, compute_parameter_statistics

__all__ = ["prepare_config"]


def prepare_config(cfg: DictConfig) -> DictConfig:
    """
    Prepare and validate configuration by enriching it with derived fields.

    This function mutates the input configuration in-place, performing several
    essential preprocessing steps before training can begin:

    1. Derive output directory path when not specified
    2. Populate grid resolution from dataset if missing
    3. Compute normalization statistics if not provided
    3b. Compute parameter ranges for min-max normalization if not provided
    4. Validate normalization statistics against chosen strategy
    5. Filter and finalize channel lists based on configuration
    6. Load loss component configurations and merge them

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra configuration object containing nested dictionaries for
        `data_config`, `model_config`, `train_config`, `output_log_config`,
        and `hyperparam_opt_config`. The configuration is modified in-place.

    Returns
    -------
    omegaconf.DictConfig
        The same configuration object (modified in-place) with all derived
        fields populated and validated.

    Raises
    ------
    ValueError
        If normalization statistics are incomplete for the chosen strategy,
        or if residual configuration requires additional statistics that
        are not provided.

    Notes
    -----
    The function performs extensive validation of normalization statistics
    based on the chosen strategy:
    - 'z_normalization': requires 'mean' and 'std' for each channel
    - 'min_max_normalization': requires 'min' and 'max' for each channel  
    - 'robust_normalization': requires 'median' and 'iqr' for each channel

    Channel filtering is performed using keyword matching where channel names
    are included if they start with any of the specified filter keywords.
    """

    # ------------------------------------------------------------------
    # 1) Output directory
    # ------------------------------------------------------------------
    out_dir = cfg["output_log_config"]["logging"]["output_dir"]

    if not out_dir:
        ts = datetime.now().strftime("%d%m%Y_%H%M%S")
        # Get checkpoint prefix if specified in config
        checkpoint_prefix = cfg["output_log_config"]["logging"].get("checkpoint_prefix", "")
        
        run_name = (
            f"{cfg['data_config']['dataset_name']}_"
            f"{cfg['data_config']['dimension']}D_{cfg['model_config']['model_name']}_{ts}"
        )
        if checkpoint_prefix:
            run_name = f"{checkpoint_prefix}_{run_name}"
        
        # Build the output directory path, optionally nesting under a top-level checkpoint_prefix directory
        if checkpoint_prefix:
            out_dir = os.path.join(checkpoint_prefix, "checkpoints", run_name)
        else:
            out_dir = os.path.join("checkpoints", run_name)
        
        print(f"Location of the checkpoints directory: {out_dir}")

    # Prepend "HP_" if we are running hyper-parameter optimisation and the
    # directory name is not already prefixed.
    if cfg["hyperparam_opt_config"]["optimize"]:
        dir_path, base = os.path.split(out_dir)
        if not base.startswith("HP_"):
            out_dir = os.path.join(dir_path, f"HP_{base}") if dir_path else f"HP_{base}"

    cfg["output_log_config"]["logging"]["output_dir"] = out_dir

    # ------------------------------------------------------------------
    # 2) Grid resolution
    # ------------------------------------------------------------------
    if cfg["data_config"]["grid_resolution"] is None:
        cfg["data_config"]["grid_resolution"] = get_grid_resolution(
            cfg["data_config"]["dataset_directory_path"]
        )

    # ------------------------------------------------------------------
    # 3) Normalisation stats
    # ------------------------------------------------------------------
    if cfg["data_config"]["data_normalization_stats"] is None:
        _t_start = time.perf_counter()
        #TODO: compute_statistics_parallel doesnt provide median and iqr (on_fly_stats=True by default in parallel mode)
        #NOTE: compute only the stats for the train dataset(test data is assumed to be inside/close to the train distribution)
        stats, channel_names, _ = compute_statistics_parallel(
            h5_paths=[os.path.join(cfg["data_config"]["dataset_directory_path"], "train.h5")],
            residual_config=cfg["data_config"]["residual_config"],
            filter_groups=cfg["data_config"]["filter_features"]["train_filter_groups"] ,
            filter_frames=cfg["data_config"]["filter_features"]["train_filter_frames"],
            frame_stride=cfg["data_config"]["sequence_info"][2],
            on_fly_stats=True,
            num_workers=4
        )
        print(f"compute_statistics took {time.perf_counter() - _t_start:.2f} seconds")
        cfg["data_config"]["data_normalization_stats"] = stats
    else:
        raw_keys = list(cfg["data_config"]["data_normalization_stats"].keys())
        # Collapse residual keys: if both "foo" and "foo_residual" exist, keep only "foo"
        channel_names = []
        for k in raw_keys:
            base = k[:-9] if k.endswith("_residual") else k
            if base not in channel_names:
                channel_names.append(base)
        if (cfg["data_config"]["residual_config"] is not None) and (
            len(cfg["data_config"]["data_normalization_stats"])
            != 2* len(channel_names)
        ):
            raise ValueError(
                f"Insufficient statistics provided in the data_normalization_stats dictionary. Please provide statistics also for the residual channels."
            )
        # check if the desired quantities are present for the given normalization strategy
        if cfg["data_config"]["data_normalization_strategy"] == "z_normalization":
            missing_stats = {}
            for ch_name, stat_dict in cfg["data_config"][
                "data_normalization_stats"
            ].items():
                missing = [k for k in ("mean", "std") if k not in stat_dict]
                if missing:
                    missing_stats[ch_name] = missing

            if missing_stats:
                details = "; ".join(
                    f"{ch}: {', '.join(miss)}" for ch, miss in missing_stats.items()
                )
                raise ValueError(
                    "Insufficient statistics. The following keys are missing -> "
                    + details
                )
        elif (
            cfg["data_config"]["data_normalization_strategy"] == "min_max_normalization"
        ):
            missing_stats = {}
            for ch_name, stat_dict in cfg["data_config"][
                "data_normalization_stats"
            ].items():
                missing = [k for k in ("min", "max") if k not in stat_dict]
                if missing:
                    missing_stats[ch_name] = missing

            if missing_stats:
                details = "; ".join(
                    f"{ch}: {', '.join(miss)}" for ch, miss in missing_stats.items()
                )
                raise ValueError(
                    "Insufficient statistics for min-max normalization. The following keys are missing -> "
                    + details
                )
        elif (
            cfg["data_config"]["data_normalization_strategy"] == "robust_normalization"
        ):
            missing_stats = {}
            for ch_name, stat_dict in cfg["data_config"][
                "data_normalization_stats"
            ].items():
                missing = [k for k in ("median", "iqr") if k not in stat_dict]
                if missing:
                    missing_stats[ch_name] = missing

            if missing_stats:
                details = "; ".join(
                    f"{ch}: {', '.join(miss)}" for ch, miss in missing_stats.items()
                )
                raise ValueError(
                    "Insufficient statistics for robust normalization. The following keys are missing -> "
                    + details
                )

    # ------------------------------------------------------------------
    # Conditioning-parameter normalisation (min / max per dimension)
    # ------------------------------------------------------------------
    if cfg["data_config"]["conditioning_features"].get("include_conditioning_parameters", False):
        # Users can optionally provide ``parameter_min_max_stats`` directly in
        # the config.  When present we *respect* those values.  Otherwise we
        # compute the ranges from the *training* data file for consistency
        # with the data split.

        if cfg["data_config"]["conditioning_features"].get("parameter_min_max_stats") is None:
            h5_dir = cfg["data_config"]["dataset_directory_path"]
            h5_paths_params = [
                os.path.join(h5_dir, fname) for fname in os.listdir(h5_dir) if fname.endswith(".h5")
            ]

            if len(h5_paths_params) == 0:
                raise FileNotFoundError(
                    f"No .h5 files found in directory '{h5_dir}' when computing parameter ranges."
                )

            param_ranges = compute_parameter_statistics(h5_paths_params)
            cfg['data_config']['conditioning_features']['num_cond_params'] = len(param_ranges)
            cfg["data_config"]["conditioning_features"]["parameter_min_max_stats"] = param_ranges
        else:
            # --- Basic schema validation of user-supplied stats ---------
            user_stats = cfg["data_config"]["conditioning_features"]["parameter_min_max_stats"]
            for idx, stat_dict in user_stats.items():
                if not {
                    "min",
                    "max",
                }.issubset(stat_dict):
                    raise ValueError(
                        "parameter_min_max_stats must supply 'min' and 'max' for every parameter dimension "
                        f"(problematic index: {idx})."
                    )

    # ------------------------------------------------------------------
    # 4) Channel selection
    # ------------------------------------------------------------------
    # NOTE: filter_in_channels has also the conditioning_in_channels (if any)
    filter_in_keywords = cfg["data_config"]["filter_features"]["filter_in_channels"]
    filtered_in_channels = (
        [n for n in channel_names if any(n.startswith(k) for k in filter_in_keywords)]
        if filter_in_keywords
        else channel_names
    )

    filter_cond_in_keywords = cfg["data_config"]["conditioning_features"]["conditioning_in_channels"]
    filtered_cond_in_channels = (
        [
            n
            for n in channel_names
            if any(n.startswith(k) for k in filter_cond_in_keywords)
        ]
        if filter_cond_in_keywords
        else None
    )

    filter_out_keywords = cfg["data_config"]["filter_features"]["filter_out_channels"]
    filtered_out_channels = (
        [n for n in channel_names if any(n.startswith(k) for k in filter_out_keywords)]
        if filter_out_keywords
        else channel_names
    )

    cfg["data_config"]["filter_features"]["filter_in_channels"] = filtered_in_channels
    cfg["data_config"]["filter_features"]["filter_out_channels"] = filtered_out_channels
    cfg["data_config"]["conditioning_features"]["conditioning_in_channels"] = filtered_cond_in_channels

    # ------------------------------------------------------------------
    # 5) Persist data_config to JSON in the designated output directory,
    # In case of hyper-parameter optimisation, the data_config is written 
    # inside each trial directory. (Refer trainer.py > _hp_search_setup method)
    # ------------------------------------------------------------------
    if cfg["hyperparam_opt_config"]["optimize"] is False:
        out_dir_path = cfg["output_log_config"]["logging"]["output_dir"]
        # Ensure directory exists before writing
        os.makedirs(out_dir_path, exist_ok=True)

        json_path = os.path.join(out_dir_path, "data_config.json")

        try:
            # Convert OmegaConf section to a regular dict for JSON serialization
            data_config_dict = OmegaConf.to_container(cfg["data_config"], resolve=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data_config_dict, f, indent=4)
        except Exception as exc:
            # We do not want to fail the entire run due to logging issues; print a warning instead.
            print(f"[WARNING] Failed to write data_config to {json_path}: {exc}")

    # ------------------------------------------------------------------
    # 6) Load and merge the relevant loss component configs
    # ------------------------------------------------------------------
    if hasattr(cfg, 'loss_config') and cfg.loss_config is not None:
        if hasattr(cfg.loss_config, 'loss') and hasattr(cfg.loss_config.loss, 'components'):
            # Temporarily disable struct mode to allow adding new fields
            OmegaConf.set_struct(cfg.loss_config, False)
            
            for component_cfg in cfg.loss_config.loss.components:
                # Check if this component references an external config file
                if 'config_file' in component_cfg and not hasattr(component_cfg, 'metric_params'):
                    config_path = f"config/loss_config/{component_cfg.config_file}.yaml"
                    
                    if os.path.exists(config_path):
                        try:
                            # Load metric config and store it separately
                            metric_config = OmegaConf.load(config_path)
                            component_cfg.metric_params = metric_config
                            
                        except Exception as e:
                            component_cfg.metric_params = {}
                    else:
                        component_cfg.metric_params = {}
            
            # Re-enable struct mode
            OmegaConf.set_struct(cfg.loss_config, True)

    return cfg