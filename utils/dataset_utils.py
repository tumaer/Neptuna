from __future__ import annotations

"""Dataset convenience wrappers."""

from omegaconf import DictConfig
from utils.load_data import fetch_dataset

__all__ = ["make_datasets"]


def make_datasets(cfg: DictConfig):
    """
    Create and return training and evaluation datasets as configured.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra/OmegaConf configuration object containing the keys
        `data_config` and `train_config`. The nested values are forwarded
        unchanged to `utils.load_data.fetch_dataset`; see that function for
        the full list of expected fields.

    Returns
    -------
    train_ds
        Dataset containing samples to be used for training.
    eval_ds
        Dataset containing samples reserved for validation during
        training.

    """

    data_cfg = cfg["data_config"]
    train_cfg = cfg["train_config"]

    return fetch_dataset(
        dataset_name=data_cfg["dataset_name"],
        dataset_directory_path=data_cfg["dataset_directory_path"],
        sequence_info=data_cfg["sequence_info"],
        pushforward_config=train_cfg["pushforward_config"],
        n_eval_rollouts=train_cfg["n_eval_rollouts"],
        filter_frames=data_cfg["filter_features"]["filter_frames"],
        filter_groups=data_cfg["filter_features"]["filter_groups"],
        filter_in_channels=data_cfg["filter_features"]["filter_in_channels"],
        conditioning_in_channels=data_cfg["conditioning_features"]["conditioning_in_channels"],
        include_conditioning_parameters=data_cfg["conditioning_features"]["include_conditioning_parameters"],
        parameter_min_max_stats=data_cfg["conditioning_features"]["parameter_min_max_stats"],
        filter_out_channels=data_cfg["filter_features"]["filter_out_channels"],
        data_normalization_stats=data_cfg["data_normalization_stats"],
        data_normalization_strategy=data_cfg["data_normalization_strategy"],
        eval_split_ratio=train_cfg["eval_split_ratio"],
        eval_groups=data_cfg["eval_groups"],
        is_steady_state_prediction=data_cfg["is_steady_state_prediction"],
        residual_config=data_cfg["residual_config"]
    )
