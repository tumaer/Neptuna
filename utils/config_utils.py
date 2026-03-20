from __future__ import annotations

"""Utility functions to enrich and finalise the Hydra DictConfig before training."""

from datetime import datetime
from omegaconf import DictConfig, ListConfig
import os
import json
import time

from omegaconf import OmegaConf
from utils.grid_utils import get_grid_resolution
from utils.compute_stats import compute_statistics_parallel, compute_parameter_statistics
from metrics.loss_registry import get_loss_entry

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
            frame_stride=1,
            on_fly_stats=True,
            num_workers=4,
            log_transform_channels=cfg["data_config"]["log_transform_channels"],
        )
        print(f"compute_statistics took {time.perf_counter() - _t_start:.2f} seconds")
        cfg["data_config"]["data_normalization_stats"] = stats
    else:
        #if cfg["data_config"]['log_transform_channels'] is not None, then the data_normalization_stats should contain the statistics for the log-transformed channels
        if cfg["data_config"]["log_transform_channels"] is not None:
            for ch_name in cfg["data_config"]["log_transform_channels"]:
                ch_name = f"log_{ch_name}"
                if ch_name not in cfg["data_config"]["data_normalization_stats"]:
                    raise ValueError(f"Statistics for the log-transformed channel {ch_name} are not provided in the data_normalization_stats dictionary.")
        # Exclude log-transformed channels; downstream logic expects raw names
        raw_keys = [
            k
            for k in cfg["data_config"]["data_normalization_stats"].keys()
            if not k.startswith("log_")
        ]
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
    cond_method = cfg["data_config"]["conditioning_features"].get("conditioning_method", 'None')
    if cond_method is not None:
        # Users can optionally provide ``parameter_min_max_stats`` directly in
        # the config.  When present we *respect* those values.  Otherwise we
        # compute the ranges from the *training* data file for consistency
        # with the data split.

        if cfg["data_config"]["conditioning_features"].get("parameter_min_max_stats") is None:
            h5_dir = cfg["data_config"]["dataset_directory_path"]
            h5_paths_params = [
                os.path.join(h5_dir, fname) for fname in os.listdir(h5_dir) if fname.endswith("train.h5")
            ]

            if len(h5_paths_params) == 0:
                raise FileNotFoundError(
                    f"No .h5 files found in directory '{h5_dir}' when computing parameter ranges."
                )

            param_ranges = compute_parameter_statistics(h5_paths_params)
            cfg["data_config"]["conditioning_features"]["parameter_min_max_stats"] = param_ranges
        else:
            # --- Basic schema validation of user-supplied stats ---------
            user_stats = cfg["data_config"]["conditioning_features"]["parameter_min_max_stats"]
            for idx, stat_dict in user_stats.items():
                if idx == "max_number_of_parameters":
                    continue
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
        [k for k in filter_in_keywords if k in channel_names]
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
        [k for k in filter_out_keywords if k in channel_names]
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
    # 6) Update the validation_loss components in the train_strategy_config block
    # by appending the metric_for_best_model to the validation_loss components of each curriculum block.
    # ------------------------------------------------------------------
    train_strategy_cfg = cfg.get("train_strategy_config", None)
    if train_strategy_cfg is not None:
        # Allow inserting fields into the strategy config when struct is enabled
        try:
            OmegaConf.set_struct(train_strategy_cfg, False)
        except Exception:
            pass
        metric_for_best_model_cfg = train_strategy_cfg.get("metric_for_best_model", None)

        def _normalize_metric_entry(entry):
            """Return a dict with keys: type, name, weight, sub_components (optional)."""
            metric_type = None
            metric_name = None
            weight = 1.0
            sub_components = None

            if hasattr(entry, "get"):
                metric_type = entry.get("type", None) or entry.get("name", None)
                metric_name = entry.get("name", None)
                weight = entry.get("weight", 1.0)
                sub_components = entry.get("sub_components", None) or entry.get("components", None)
            elif isinstance(entry, dict):
                metric_type = entry.get("type", None) or entry.get("name", None)
                metric_name = entry.get("name", None)
                weight = entry.get("weight", 1.0)
                sub_components = entry.get("sub_components", None) or entry.get("components", None)
            else:
                metric_type = entry

            return {
                "type": metric_type,
                "name": metric_name,
                "weight": weight,
                "sub_components": sub_components,
            }

        # Support list / single entries for metric_for_best_model
        metric_entries = []
        if metric_for_best_model_cfg is not None:
            if isinstance(metric_for_best_model_cfg, (list, ListConfig, tuple)):
                metric_entries = [_normalize_metric_entry(e) for e in metric_for_best_model_cfg]
            else:
                metric_entries = [_normalize_metric_entry(metric_for_best_model_cfg)]

        curriculum_blocks = train_strategy_cfg.get("curriculum", [])

        if metric_entries and curriculum_blocks:
            for block in curriculum_blocks:
                if block is None or "validation_loss" not in block:
                    continue

                validation_loss_cfg = block["validation_loss"]
                if validation_loss_cfg is None:
                    continue

                # Temporarily disable struct to allow insertions, then restore
                #prev_struct = OmegaConf.get_struct(validation_loss_cfg)
                OmegaConf.set_struct(validation_loss_cfg, False)


                for metric_entry in metric_entries:
                    metric_type = metric_entry.get("type", None)
                    metric_name = metric_entry.get("name", None)
                    metric_weight = metric_entry.get("weight", 1.0)
                    metric_sub_components = metric_entry.get("sub_components", None)

                    if metric_type is None:
                        continue

                    # Ensure the best-model metric is present in validation_loss components
                    components = validation_loss_cfg.get("components", [])
                    if components is None:
                        components = []

                    def _extract_type(obj):
                        if hasattr(obj, "get"):
                            return obj.get("type", None)
                        if isinstance(obj, dict):
                            return obj.get("type", None)
                        return None

                    def _extract_sub_components(obj):
                        if hasattr(obj, "get"):
                            return obj.get("sub_components", None) or obj.get("components", None)
                        if isinstance(obj, dict):
                            return obj.get("sub_components", None) or obj.get("components", None)
                        return None

                    def _extract_name(obj):
                        if hasattr(obj, "get"):
                            return obj.get("name", None)
                        if isinstance(obj, dict):
                            return obj.get("name", None)
                        return None

                    def _extract_weight(obj):
                        if hasattr(obj, "get"):
                            return obj.get("weight", 1.0)
                        if isinstance(obj, dict):
                            return obj.get("weight", 1.0)
                        return 1.0

                    def _sub_components_match(target_sub, existing_sub):
                        target_list = target_sub or []
                        existing_list = existing_sub or []
                        if len(target_list) != len(existing_list):
                            return False
                        for t_item, e_item in zip(target_list, existing_list):
                            t_type = _extract_type(t_item)
                            e_type = _extract_type(e_item)
                            if t_type != e_type:
                                return False
                            t_w = _extract_weight(t_item)
                            e_w = _extract_weight(e_item)
                            if t_w != e_w:
                                return False
                        return True

                    # Skip append if a component with the same type (and for composite, matching sub-components) exists
                    metric_exists = False
                    for comp in components:
                        comp_type = _extract_type(comp)
                        if comp_type != metric_type:
                            continue

                        if metric_type == "CompositeLoss":
                            comp_sub = _extract_sub_components(comp)
                            if _sub_components_match(metric_sub_components, comp_sub):
                                comp_name = _extract_name(comp)
                                # If sub-components match but names differ, raise to avoid ambiguous configs
                                if (metric_name or comp_name) and (metric_name != comp_name):
                                    block_name = block.get("name", "<unknown>")
                                    raise ValueError(
                                        f"CompositeLoss name mismatch in validation_loss of block '{block_name}': "
                                        f"metric_for_best_model name='{metric_name}' vs existing name='{comp_name}' "
                                        "with identical sub_components."
                                    )
                                metric_exists = True
                                break
                        else:
                            metric_exists = True
                            break

                    if not metric_exists:
                        new_comp = {"type": metric_type, "weight": metric_weight}
                        if metric_name is not None:
                            new_comp["name"] = metric_name
                        if metric_type == "CompositeLoss" and metric_sub_components is not None:
                            new_comp["sub_components"] = metric_sub_components
                        components.append(new_comp)
                        validation_loss_cfg["components"] = components
                # Restore original struct setting
                OmegaConf.set_struct(validation_loss_cfg, True)

    # ------------------------------------------------------------------
    # 7) Load and merge the relevant loss component configs. This block updates the 
    # default metric_params for the loss component. For knowing the options for each loss component,
    # refer to the config/train_strategy_config/loss_metrics folder/<loss_component_foler>/*_default.yaml.
    # ------------------------------------------------------------------

    def _load_and_merge_loss_config(component_cfg):
        """Helper to load and merge loss component configuration."""
        loss_type = component_cfg.type
        
        # Handle CompositeLoss with sub_components
        if loss_type == "CompositeLoss" and hasattr(component_cfg, 'sub_components'):
            for sub_comp in component_cfg.sub_components:
                _load_and_merge_loss_config(sub_comp)
        
        # Load config for this component
        try:
            registry_entry = get_loss_entry(loss_type)
            default_config = registry_entry["default_config"]
            config_base_path = registry_entry["config_path"]
        except (KeyError, ValueError):
            # Loss type not in registry or no default config
            return
        
        config_file = component_cfg.get("config_file", default_config)
        
        # Ensure metric_params exists
        existing_metric = component_cfg.get("metric_params", None)
        if existing_metric is None:
            existing_metric = OmegaConf.create({})
        
        # Load defaults (if available)
        defaults_metric = OmegaConf.create({})
        if config_file is not None:
            config_path = f"config/train_strategy_config/{config_base_path}{config_file}.yaml"
            if os.path.exists(config_path):
                try:
                    defaults_metric = OmegaConf.load(config_path)
                except Exception:
                    defaults_metric = OmegaConf.create({})
        
        # Merge: user-provided keys win over defaults
        merged_metric = OmegaConf.merge(defaults_metric, existing_metric)
        was_struct = OmegaConf.is_struct(component_cfg)
        OmegaConf.set_struct(component_cfg, False)
        component_cfg.metric_params = merged_metric
        OmegaConf.set_struct(component_cfg, was_struct)


    if hasattr(cfg, 'train_strategy_config') and cfg.train_strategy_config is not None:
        train_strategy = cfg.train_strategy_config
        
        if hasattr(train_strategy, 'curriculum') and train_strategy.curriculum is not None:
            OmegaConf.set_struct(train_strategy, False)
            
            for block in train_strategy.curriculum:
                # Process train_loss components
                if hasattr(block, 'train_loss') and hasattr(block.train_loss, 'components'):
                    for component_cfg in block.train_loss.components:
                        _load_and_merge_loss_config(component_cfg)
                
                # Process validation_loss components
                if hasattr(block, 'validation_loss') and hasattr(block.validation_loss, 'components'):
                    for component_cfg in block.validation_loss.components:
                        _load_and_merge_loss_config(component_cfg)
            
            OmegaConf.set_struct(train_strategy, True)

    return cfg