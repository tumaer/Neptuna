from typing import List, Optional, Union, Callable
import torch
import torch.nn as nn
from torch import Tensor
from utils.activation_func import get_activation

#######################################################################
#######################################################################
class BasicBlock(nn.Module):
    """including two 3x3 convolutions layers with BatchNorm and activation function.

    Parameters
    ----------
    in_planes : int
        Size of hidden channels 
    planes : int
        Size of output channels, normally equal to in_planes
    stride : int
        stride for 2dCNN
    activation_fn : nn.Module
        Activation function, by default nn.GELU
    norm : bool
        Whether to use normalization, by default True
    num_groups : int
        Number of groups for GroupNorm, by default 1 (equivalent with LayerNorm)
    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        activation_fn: nn.Module = nn.GELU(),
        norm: bool = True,
        num_groups: int = 1,
    ) -> None:
        super().__init__()
        
        #2X 3*3 convolutions Layers and corresponding GroupNorm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(num_groups, num_channels=planes)
        self.activation = activation_fn

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, self.expansion * planes) if norm else nn.Identity(),
            )

    def forward(self, x: Tensor) -> Tensor:
        #out = self.conv1(self.activation(self.bn1(x)))
        out = self.activation(self.bn1(self.conv1(x)))
        #out = self.conv2(self.activation(self.bn2(out)))
        out = self.activation(self.bn2(self.conv2(x)))
        out = out + self.shortcut(x)
        return out