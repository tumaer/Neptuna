from typing import List, Tuple
import torch.nn as nn
from torch import Tensor
from models.ResNet.resnet_utils import BasicBlockND


class Ffn(nn.Module):
    """
    A general fully connected multi-layer neural network.
    """

    def __init__(
        self, 
        dims: List, 
        activation_fn: nn.Module = nn.GELU(),
        act_on_output: bool = False
    ):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if act_on_output:
            layers.append(activation_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x
    
class CnnBranch(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        kernel_size: int, 
        padding: int, 
        dimension: int,
        hidden_channels: int = 32,
        depth: int = 4,
        activation_fn: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        
        if dimension == 1:
            Conv = nn.Conv1d
            Pool = nn.MaxPool1d
        elif dimension == 2:
            Conv = nn.Conv2d
            Pool = nn.MaxPool2d
        elif dimension == 3:
            Conv = nn.Conv3d
            Pool = nn.MaxPool3d
        else:
            raise ValueError(f"Unsupported dimension: {dimension}. Must be 1, 2, or 3.")
        self.in_conv = Conv(
            in_channels, hidden_channels , kernel_size=kernel_size, padding=padding
        )
        self.out_conv = Conv(
            hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding
        )

        blocks = []
        for i in range(depth):
            blocks += [
                Conv(hidden_channels, hidden_channels, kernel_size, padding=padding),
                Pool(2),
                activation_fn,
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)  # (b, 16, h, w)
        x = self.blocks(x)  # (b, 32, h/16=4, w/16=4)
        x = self.out_conv(x)  # (b, 32, 4, 4)
        return x
    

def grid_to_points(value: Tensor) -> Tuple[Tensor, List[int]]:
    """
    Convert from grid-based (1D, 2D, 3D) representation to point-based representation.

    Parameters
    ----------
    value : Tensor
        Input tensor of shape (B, C, X, Y, Z).

    Returns
    -------
    Tuple
        - Tensor of shape (B, C*X*Y*Z).
    """
    output = value.reshape(value.size(0), -1)  # Reshape to (B, C*X*Y*Z)
    return output

def points_to_grid(value: Tensor, shape: List[int]) -> Tensor:
    """
    Convert from point-based representation back to grid-based (1D, 2D, 3D) representation.

    Parameters
    ----------
    value : Tensor
        Input tensor of shape (B, C*X*Y*Z).
    shape : List[int]
        Original shape as [B, C, X, Y, Z].

    Returns
    -------
    Tensor
        Restored tensor of shape (B, C, X, Y, Z).
    """
    output = value.reshape(shape)  # Reshape back to (B, C, X, Y, Z)
    return output