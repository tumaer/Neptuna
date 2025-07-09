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

def re_normalize_data(arr: np.ndarray, stats: Dict[str, float], norm_strategy: str) -> np.ndarray:
    """Reverts normalization applied during loading for a single channel.
    Mainly used for plotting the normalized data.
    
    Parameters
    ----------
    arr : np.ndarray
        The normalized data to revert (any shape).
    stats : Dict[str, float]
        Dictionary containing either {"mean", "std"} or {"min", "max"} or {"median", "iqr"}. 
    norm_strategy : str
        Either "z_normalization" or "min_max_normalization" or "robust_normalization" or "no_normalization".
    
    Returns
    -------
    np.ndarray
        Renormalized array (new copy).
    """
    if norm_strategy == "z_normalization":
        return arr * stats["std"] + stats["mean"]
    elif norm_strategy == "min_max_normalization":
        return arr * (stats["max"] - stats["min"]) + stats["min"]
    elif norm_strategy == "robust_normalization":
        return arr * stats["iqr"] + stats["median"]
    elif norm_strategy == "no_normalization":
        return arr
    else:
        raise ValueError(f"Unknown normalization strategy: {norm_strategy}")

# -----------------------------------------------------------------------------
# Normalization helper used during data loading
# -----------------------------------------------------------------------------

def normalize_data(arr: np.ndarray, stats: Dict[str, float], strategy: str) -> np.ndarray:
    """Apply per-channel normalization to a NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        The data to normalize (any shape).
    stats : Dict[str, float]
        Dictionary containing either {"mean", "std"} or {"min", "max"} or {"median", "iqr"}.
    strategy : str
        Either "z_normalization" or "min_max_normalization" or "robust_normalization" or "no_normalization".

    Returns
    -------
    np.ndarray
        Normalized array (new copy).
    """
    eps = 1e-12  # small constant to prevent divide-by-zero

    if strategy == "z_normalization":
        return (arr - stats["mean"]) / (stats["std"]  + eps)
    elif strategy == "min_max_normalization":
        return (arr - stats["min"]) / ((stats["max"] - stats["min"]) + eps)
    elif strategy == "robust_normalization":
        return (arr - stats["median"]) / (stats["iqr"] + eps)
    elif strategy == "no_normalization":
        return arr
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")