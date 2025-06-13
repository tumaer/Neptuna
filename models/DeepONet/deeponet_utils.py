from typing import List, Tuple
import torch.nn as nn
from torch import Tensor
import numpy as np


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
        grid_resolution:Tuple[int],
        latent_channels: int = 32,
        depth: int = 4,
        activation_fn: nn.Module = nn.ReLU(),
        stride : int = 1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.depth = depth
        self.stride = stride
        self.grid_resolution = grid_resolution
        
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
            in_channels, latent_channels , kernel_size=3, padding=1)
        self.out_conv = Conv(
            latent_channels, latent_channels, kernel_size=3, padding=1)

        blocks = []
        for i in range(depth):
            blocks += [
                Conv(latent_channels, latent_channels, kernel_size, padding=padding, stride=stride),
                Pool(2),
                activation_fn,
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)  # (b, 16, h, w)
        x = self.blocks(x)  # (b, 32, h/16=4, w/16=4)
        x = self.out_conv(x)  # (b, 32, 4, 4)
        return x
    
    def calc_out_shape(self) -> Tuple[int, ...]:
        out = list(self.grid_resolution)
        for _ in range(self.depth):
            for i in range(len(out)):
                out[i] = (out[i] + 2 * self.padding - self.kernel_size) // self.stride + 1  # Conv
                out[i] = (out[i]  - 2) // 2 + 1  # Pool
                assert out[i] > 0, f"Output shape is non-positive: {out[i]}"
        return tuple(out)

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

def calc_resnet_out_shape(
    in_shape: tuple,
    num_blocks: List[int],
    stride: int=1,
    kernel_size: int=3,
    padding: int=1,
):
    out = list(in_shape)
    count = 0
    for j in range(len(num_blocks)):
        count += num_blocks[j]
    for _ in range(count):
        for i in range(len(out)):
            # 第一个conv
            out[i] = (out[i] + 2 * padding - kernel_size) // stride + 1
            # 第二个conv（stride=1, padding同上）
            out[i] = (out[i] + 2 * padding - kernel_size) // 1 + 1
            # shortcut不改变空间尺寸
            assert out[i] > 0, f"Output shape is non-positive: {out[i]}"
    return tuple(out)
def linspace_int_list(int1: int, int2: int, int3: int, reverse: bool) -> list:
    arr = [int(round(x)) for x in np.linspace(int3, int1, int2)]
    arr[0] = int3
    arr[-1] = int1
    if reverse:
        return arr[::-1]
    else:
        return arr