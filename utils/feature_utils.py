from typing import List, Tuple, Union
import torch
from torch import Tensor

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