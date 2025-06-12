from typing import List, Dict
import torch
from torch import Tensor
import h5py
import os
import copy
import numpy as np

def oned_meshgrid(shape: List[int], device: torch.device) -> Tensor:
    """Creates 1D meshgrid feature

    Parameters
    ----------
    shape : List[int]
        Tensor shape
    device : torch.device
        Device model is on

    Returns
    -------
    Tensor
        Meshgrid tensor
    """
    bsize, size_x = shape[0], shape[2]
    grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1)
    return grid_x

def twod_meshgrid(shape: List[int], device: torch.device) -> Tensor:
    """Creates 2D meshgrid feature

    Parameters
    ----------
    shape : List[int]
        Tensor shape
    device : torch.device
        Device model is on

    Returns
    -------
    Tensor
        Meshgrid tensor
    """
    bsize, size_x, size_y = shape[0], shape[2], shape[3]
    grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
    grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
    return torch.cat((grid_x, grid_y), dim=1)

def threed_meshgrid(shape: List[int], device: torch.device) -> Tensor:
    """Creates 3D meshgrid feature

    Parameters
    ----------
    shape : List[int]
        Tensor shape
    device : torch.device
        Device model is on

    Returns
    -------
    Tensor
        Meshgrid tensor
    """
    bsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]
    grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
    grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
    grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
    grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
    return torch.cat((grid_x, grid_y, grid_z), dim=1)

def get_grid_resolution(dataset_directory_path: str) -> List[int]:
    """Get the grid resolution from the dataset directory path
    """
    train_eval_h5file_path = os.path.abspath(dataset_directory_path + "/train.h5")
    with h5py.File(train_eval_h5file_path, 'r') as f:
        first_group = list(f.keys())[0]
        first_field = list(f[first_group].keys())[0]
        grid_resolution = list(f[first_group][first_field].shape[2:])
    return grid_resolution

def _renormalize(arr: np.ndarray, channel_names: List[str], norm_stats: Dict[str, Dict[str, float]], norm_strategy: str) -> np.ndarray:
    """Reverts normalization applied during loading.
    Expects shape [N, T, C, *spatial_dims]. Returns a copy.
    """
    arr_renormed = copy.deepcopy(arr)

    for c_idx, ch_name in enumerate(channel_names):
        if ch_name not in norm_stats:
            # Raise an error if stats for this channel are unavailable
            raise ValueError(f"Stats for channel {ch_name} are unavailable.")

        stats = norm_stats[ch_name]
        if norm_strategy == "z_normalization":
            arr_renormed[:, :, c_idx] = arr_renormed[:, :, c_idx] * stats["std"] + stats["mean"]
        elif norm_strategy == "min_max_normalization":
            arr_renormed[:, :, c_idx] = arr_renormed[:, :, c_idx] * (stats["max"] - stats["min"]) + stats["min"]

    return arr_renormed