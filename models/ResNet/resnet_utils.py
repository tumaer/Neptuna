from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
from models.DeepONet.deeponet_utils import BasicBlockND4DeepONet
from utils.model_utils import PretrainedConfig
from utils.model_utils import CustomNorm

class ResNetConfig(PretrainedConfig):
    """
    Args:
        block (str): 
            BasicBlock, Dilblock only for now
        num_blocks (List[int]): 
            Number of blocks in each stage
        latent_channels (int): 
            Number of channels in the latent space
        activation_fn : str
            Activation function, by default "gelu"
        coord_features : bool, optional
            Use coordinate grid as additional feature map, by default True
        norm : str
            Normalization type, by default "layer"
        norm_layer_eps : float
            Epsilon for the normalization layer, by default 1e-5
    """

    model_type = "ResNet"
    
    def __init__(
        self,
        num_blocks: list = [1, 1, 1, 1],
        block: str = "BasicBlock",
        norm: str = "layer",
        norm_layer_eps: float = 1e-5,
        activation_fn_name : str = "gelu",
        padding: int = 1,
        stride: int = 1, #TODO: get it fixed when stride!=1
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        self.block = block
        self.norm = norm
        self.norm_layer_eps = norm_layer_eps
        self.activation_fn_name = activation_fn_name
        self.padding = padding
        self.stride = stride
        
# -----------------------------------------------------------------------------
# Helper to propagate **kwargs through nn.Sequential
# -----------------------------------------------------------------------------
class SequentialWithKwargs(nn.Sequential):
    """nn.Sequential variant that forwards any additional keyword arguments
    to every sub-module. This makes it compatible with blocks whose forward
    signature is ``forward(x, **kwargs)`` (e.g. blocks containing
    `CustomNorm` layers that need conditioning parameters).
    """

    def forward(self, x, **kwargs): 
        """Forward that is tolerant of modules which do **not** accept the
        extra keyword arguments.  For each sub-module we first attempt to call
        it with ``**kwargs``; if this results in a *TypeError* complaining
        about unexpected keyword arguments we retry without them.  This allows
        mixing plain layers (e.g. ``nn.Conv``) with custom layers (e.g.
        ``CustomNorm``) that require the extra data.
        """

        for module in self:
            if kwargs:
                try:
                    x = module(x, **kwargs)
                    continue  # success
                except TypeError as e:
                    # Only swallow the error if it is about unexpected kwarg
                    # to keep other bugs visible.
                    if "unexpected keyword argument" not in str(e):
                        raise
            # Fallback: call without kwargs
            x = module(x)
        return x

# -----------------------------------------------------------------------------
# Main ResNet building blocks
# -----------------------------------------------------------------------------
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
    """

    expansion: int = 1

    def __init__(
        self,
        config,
        in_planes: int,
        planes: int,
        dimension: int, 
        stride: int = 1,
        activation_fn: nn.Module = nn.GELU(),
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
        
        #2X 3*3 convolutions Layers and corresponding CustomNorm
        self.conv1 = Conv(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)        
        self.norm1 = CustomNorm(config=config, num_channels=planes, array_length=dimension+2, channel_at_last_position=False)
        self.conv2 = Conv(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=True)
        self.norm2 = CustomNorm(config=config, num_channels=planes, array_length=dimension+2, channel_at_last_position=False)
        self.activation = activation_fn

        # Shortcut connection – use SequentialWithKwargs so we can forward **kwargs
        self.shortcut = SequentialWithKwargs() #no modules inside SequentialWithKwargs means it returns the input  (identity-fn)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = SequentialWithKwargs(
                Conv(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                CustomNorm(config=config, num_channels=self.expansion * planes, array_length=dimension+2, channel_at_last_position=False)
            )
    #Forward function of one residual block
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        out=x
        out = self.conv1(out)
        out = self.norm1(out, **kwargs)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out, **kwargs)
        out = self.activation(out)
        out = out+self.shortcut(x, **kwargs) #residual connection
        return out 
    
class DilatedBasicBlockND(nn.Module):
    """Basic block for Dilated ResNet (1,2,3D)

    Parameters
    ----------
    config: ResNetConfig
        Configuration for the ResNet
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
    """

    expansion = 1

    def __init__(
        self,
        config,
        in_planes: int,
        planes: int,
        dimension: int, 
        stride: int = 1,
        activation_fn: nn.Module = nn.GELU(),
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
        # Normalization layers (CustomNorm supports various norms + conditioning)
        self.norm_layers = nn.ModuleList(
            CustomNorm(
                config=config,
                num_channels=planes,
                array_length=dimension + 2,
                channel_at_last_position=False,
            )
            for _ in self.dilation
        )
        self.activation = activation_fn


    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = x
        for layer, norm in zip(self.dilation_layers, self.norm_layers): 
            # Pass **kwargs through the norm layer so conditioning parameters propagate
            out = self.activation(layer(norm(out, **kwargs)))          
        return out + x #residual connection
    

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
    config: ResNetConfig,
    in_planes: int,
    out_planes: int,
    num_blocks: int,
    stride: int,
    dimension: int,
    activation_fn: nn.Module = nn.GELU(),

) -> nn.Sequential:
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(
            block(
                config,
                in_planes,
                out_planes,
                dimension,
                stride,
                activation_fn,
            )
        )
        in_planes = out_planes * block.expansion
    return SequentialWithKwargs(*layers)