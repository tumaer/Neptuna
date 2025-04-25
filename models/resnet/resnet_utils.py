from typing import List, Optional, Union, Callable
import torch
import torch.nn as nn
from torch import Tensor

#######################################################################
#######################################################################
class BasicBlock2D(nn.Module):
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
    
    

class DilatedBasicBlock2D(nn.Module):
    """Basic block for Dilated ResNet

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

    expansion = 1

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

        self.dilation = [1, 2, 4, 8, 4, 2, 1]
        dilation_layers = []
        for dil in self.dilation:
            dilation_layers.append(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=3,
                    stride=stride,
                    dilation=dil,
                    padding=dil,
                    bias=True,
                )
            )
        self.dilation_layers = nn.ModuleList(dilation_layers)
        self.norm_layers = nn.ModuleList(
            nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity() for dil in self.dilation
        )
        self.activation = activation_fn


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer, norm in zip(self.dilation_layers, self.norm_layers): 
            out = self.activation(layer(norm(out)))
        return out + x