from __future__ import annotations

"""Dataset convenience wrappers."""

from omegaconf import DictConfig
from utils.load_data import fetch_dataset

__all__ = ["make_datasets"]


def make_datasets(cfg: DictConfig): 
    """Return (train_ds, eval_ds) according to *cfg*."""

    return fetch_dataset(
        dataset_name=cfg["data_config"]["dataset_name"],
        dataset_directory_path=cfg["data_config"]["dataset_directory_path"],
        sequence_info=cfg["data_config"]["sequence_info"],
        max_pf_train_rollouts=cfg["train_config"]["pushforward"]["max_allowed_unroll_steps"][-1],
        n_eval_rollouts=cfg["train_config"]["n_eval_rollouts"],
        filter_frames=cfg["data_config"]["filter_frames"],
        filter_groups=cfg["data_config"]["filter_groups"],
        filter_in_channels=cfg["data_config"]["filter_in_channels"],
        conditioning_in_channels=cfg["data_config"]["conditioning_in_channels"],
        filter_out_channels=cfg["data_config"]["filter_out_channels"],
        data_normalization_stats=cfg["data_config"]["data_normalization_stats"],
        data_normalization_strategy=cfg["data_config"]["data_normalization_strategy"],
        eval_split_ratio=cfg["train_config"]["eval_split_ratio"],
        eval_groups=cfg["data_config"]["eval_groups"],
        is_steady_state_prediction=cfg["data_config"]["is_steady_state_prediction"],
    ) 