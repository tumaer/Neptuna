from typing import List, Optional, Union, Callable
import torch
import torch.nn as nn
from torch import Tensor
from models.DeepONet.deeponet_utils import BasicBlockND4DeepONet
#######################################################################
#######################################################################
class BasicBlockND(nn.Module):
    """including two 3x3 convolutions layers with BatchNorm and activation for 1,2,3D input.

    Parameters
    ----------
    in_planes : int
        Size of hidden channels 
    planes : int
        Size of output channels, normally equal to in_planes
    dimension : int
        Model dimensionality (supports 1,2,3)
    stride : int
        stride for 2dCNN
    activation_fn : nn.Module
        Activation function, by default nn.GELU
    norm : bool
        Whether to use normalization, by default True
    n_groups : int
        Number of groups for GroupNorm, by default 1 (equivalent with LayerNorm)
    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        dimension: int, 
        stride: int = 1,
        activation_fn: nn.Module = nn.GELU(),
        norm: bool = True,
        n_groups: int = 1,
    ) -> None:
        super().__init__()
        
        if dimension == 1:
            Conv = nn.Conv1d
            kernel_size = (3,)
            padding = (1,)
        elif dimension == 2:
            Conv = nn.Conv2d
            kernel_size = (3, 3)
            padding = (1, 1)
        elif dimension == 3:
            Conv = nn.Conv3d
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)
        else:
            raise ValueError(f"Unsupported dimension: {dimension}. Must be 1, 2, or 3.")
        
        #2X 3*3 convolutions Layers and corresponding GroupNorm
        self.conv1 = Conv(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn1 = nn.GroupNorm(n_groups, num_channels=planes) if norm else nn.Identity()
        self.conv2 = Conv(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=True)
        self.bn2 = nn.GroupNorm(n_groups, num_channels=planes)
        self.activation = activation_fn

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(n_groups, self.expansion * planes) if norm else nn.Identity(),
            )

    def forward(self, x: Tensor) -> Tensor:
        #out = self.conv1(self.activation(self.bn1(x)))
        out = self.activation(self.bn1(self.conv1(x)))
        #out = self.conv2(self.activation(self.bn2(out)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = out + self.shortcut(x)
        return out
    
#######################################################################
#######################################################################

class DilatedBasicBlockND(nn.Module):
    """Basic block for Dilated ResNet (1,2,3D)

    Parameters
    ----------
    in_planes : int
        Size of hidden channels 
    planes : int
        Size of output channels, normally equal to in_planes
    dimension : int
        Model dimensionality (supports 1,2,3)
    stride : int
        stride for 2dCNN
    activation_fn : nn.Module
        Activation function, by default nn.GELU
    norm : bool
        Whether to use normalization, by default True
    n_groups : int
        Number of groups for GroupNorm, by default 1 (equivalent with LayerNorm)
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        dimension: int, 
        stride: int = 1,
        activation_fn: nn.Module = nn.GELU(),
        norm: bool = True,
        n_groups: int = 1,
    ) -> None:
        super().__init__()

        if dimension == 1:
            Conv = nn.Conv1d
            kernel_size = (3,)
        elif dimension == 2:
            Conv = nn.Conv2d
            kernel_size = (3, 3)
        elif dimension == 3:
            Conv = nn.Conv3d
            kernel_size = (3, 3, 3)
        else:
            raise ValueError(f"Unsupported dimension: {dimension}. Must be 1, 2, or 3.")
        self.dilation = [1, 2, 4, 8, 4, 2, 1]
        dilation_layers = []
        for dil in self.dilation:
            dilation_layers.append(
                Conv(
                    in_planes,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dil,
                    padding=dil,
                    bias=True,
                )
            )
        self.dilation_layers = nn.ModuleList(dilation_layers)
        self.norm_layers = nn.ModuleList(
            nn.GroupNorm(n_groups, num_channels=planes) if norm else nn.Identity() for dil in self.dilation
        )
        self.activation = activation_fn


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer, norm in zip(self.dilation_layers, self.norm_layers): 
            out = self.activation(layer(norm(out))) 
            #out = self.activation(norm(layer(out)))             
        return out + x
    

def getblock(block):
    """Get the ResNet block"""
    if isinstance(block, str):
        if block == "BasicBlock":
                return BasicBlockND
        elif block == "DilatedBasicBlock":
                return DilatedBasicBlockND
        elif block == "BasicBlockND4DeepONet":
                return BasicBlockND4DeepONet
        else:
            raise NotImplementedError(f"Unknown block: {block}")
    else:
        raise ValueError(f"Unknown block type: {block}")           

def make_layer(
    block: Callable,
    in_planes: int,
    out_planes: int,
    num_blocks: int,
    stride: int,
    dimension: int,
    activation_fn: nn.Module = nn.GELU(),
    norm: bool = True,
    n_groups: int = 1,
) -> nn.Sequential:
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(
            block(
                in_planes,
                out_planes,
                dimension,
                stride,
                activation_fn,
                norm,
                n_groups,
            )
        )
        in_planes = out_planes * block.expansion
    return nn.Sequential(*layers)