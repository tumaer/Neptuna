import torch
from omegaconf import ListConfig
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Callable, Tuple
from models.model_utils import cfd_PretrainedConfig

def _to_tuple(x):
    if isinstance(x, ListConfig):
        return tuple(int(i) for i in x)
    elif isinstance(x, (list, tuple)):
        return tuple(int(i) for i in x)
    else:
        return (int(x),)

class CNOConfig(cfd_PretrainedConfig):

    model_type = "CNO"

    def __init__(
        self,
        cno_depth: int = 4,                                   # Number of (D) or (U) blocks in the network
        n_blocks: int = 4,                                  # Number of (R) blocks per level (except the neck)
        n_blocks_bottleneck: int = 4,                       # Number of (R) blocks in the neck
        channel_multiplier: int = 16, 
        norm: bool = True,             
        latent_channels: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.cno_depth = int(cno_depth)         # Number of (D) & (U) Blocks
        self.n_blocks = n_blocks
        self.n_blocks_bottleneck = n_blocks_bottleneck
        self.norm = norm
        self.latent_channels = latent_channels
        self.lift_dim = channel_multiplier//2 # Input is lifted to the half of channel_multiplier dimension
        self.channel_multiplier = channel_multiplier  # The growth of the channels


class CNO_LReLu(nn.Module):
    def __init__(self,
                in_grid_resolution: Union[int, List[int], Tuple[int]],
                out_grid_resolution: Union[int, list[int], Tuple[int]],
                ):
        super().__init__()

        self.in_grid_resolution = _to_tuple(in_grid_resolution)
        self.out_grid_resolution = _to_tuple(out_grid_resolution)
        self.act = nn.LeakyReLU()
        if len(in_grid_resolution) == 1:
            self.mode = "linear"
        elif len(in_grid_resolution) == 2:
            self.mode = "bicubic"
        elif len(in_grid_resolution) == 3:
            self.mode = "trilinear"

    def forward(self, x):
        up_grid_resolution = tuple(2 * s for s in self.in_grid_resolution)
        x = F.interpolate(x, size=up_grid_resolution, mode=self.mode,antialias = True if self.mode == "bicubi" else False)
        x = self.act(x)
        down_grid_resolution = tuple(self.out_grid_resolution)
        x = F.interpolate(x, size=down_grid_resolution, mode=self.mode ,antialias = True if self.mode == "bicubi" else False)
        return x    

    
class CNOBlock(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                in_grid_resolution: Union[int, List[int], Tuple[int]],
                out_grid_resolution: Union[int, List[int], Tuple[int]],
                dimension: int,
                norm: bool = True
                ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_grid_resolution  = in_grid_resolution
        self.out_grid_resolution = out_grid_resolution
        if dimension == 1:
            Conv = nn.Conv1d
            BN = nn.BatchNorm1d
        elif dimension == 2:
            Conv = nn.Conv2d
            BN = nn.BatchNorm2d
        elif dimension == 3:
            Conv = nn.Conv3d
            BN = nn.BatchNorm3d
        else:
            raise ValueError(f"Unsupported dimension: {dimension}. Must be 1, 2, or 3.")
        #-----------------------------------------

        # We apply Conv -> BN (optional) -> Activation
        # Up/Downsampling happens inside Activation

        self.convolution = Conv(in_channels = self.in_channels,
                                out_channels= self.out_channels,
                                kernel_size = 3,
                                padding     = 1)

        if norm:
            self.batch_norm  = BN(self.out_channels)
        else:
            self.batch_norm  = nn.Identity()
        self.act           = CNO_LReLu(in_grid_resolution  = self.in_grid_resolution,
                                        out_grid_resolution = self.out_grid_resolution)
    def forward(self, x):
        x = self.convolution(x)
        x = self.batch_norm(x)
        return self.act(x)
    
#--------------------
# Lift/Project Block:
#--------------------

class LiftProjectBlock(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                grid_resolution: Union[int, List[int], Tuple[int]],
                dimension: int,
                latent_channels: int = 64
                ):
        super().__init__()

        self.inter_CNOBlock = CNOBlock(in_channels       = in_channels,
                                        out_channels     = latent_channels,
                                        in_grid_resolution          = grid_resolution,
                                        out_grid_resolution         = grid_resolution,
                                        dimension        = dimension,
                                        norm           = False)
        if dimension == 1:
            Conv = nn.Conv1d
        elif dimension == 2:
            Conv = nn.Conv2d
        elif dimension == 3:
            Conv = nn.Conv3d
        else:
            raise ValueError(f"Unsupported dimension: {dimension}. Must be 1, 2, or 3.")
        self.convolution = Conv(in_channels  = latent_channels,
                                out_channels = out_channels,
                                kernel_size  = 3,
                                padding      = 1)


    def forward(self, x):
        x = self.inter_CNOBlock(x)
        x = self.convolution(x)
        return x

#--------------------
# Residual Block:
#--------------------

class ResidualBlock(nn.Module):
    def __init__(self,
                channels: int,
                grid_resolution: Union[int, List[int], Tuple[int]],
                dimension: int, 
                norm: bool = True
                ):
        super().__init__()

        self.channels = channels
        self.grid_resolution = grid_resolution

        #-----------------------------------------

        # We apply Conv -> BN (optional) -> Activation -> Conv -> BN (optional) -> Skip Connection
        # Up/Downsampling happens inside Activation
        if dimension == 1:
            Conv = nn.Conv1d
            BN = nn.BatchNorm1d
        elif dimension == 2:
            Conv = nn.Conv2d
            BN = nn.BatchNorm2d
        elif dimension == 3:
            Conv = nn.Conv3d
            BN = nn.BatchNorm3d
        else:
            raise ValueError(f"Unsupported dimension: {dimension}. Must be 1, 2, or 3.")
        self.convolution1 = Conv(in_channels = self.channels,
                                 out_channels= self.channels,
                                 kernel_size = 3,
                                 padding     = 1)
        self.convolution2 = Conv(in_channels = self.channels,
                                 out_channels= self.channels,
                                 kernel_size = 3,
                                 padding     = 1)

        if norm:
            self.batch_norm1  = BN(self.channels)
            self.batch_norm2  = BN(self.channels)

        else:
            self.batch_norm1  = nn.Identity()
            self.batch_norm2  = nn.Identity()

        self.act           = CNO_LReLu(in_grid_resolution  = self.grid_resolution,
                                       out_grid_resolution = self.grid_resolution)


    def forward(self, x):
        out = self.convolution1(x)
        out = self.batch_norm1(out)
        out = self.act(out)
        out = self.convolution2(out)
        out = self.batch_norm2(out)
        return x + out

#--------------------
# ResNet:
#--------------------

class ResNet(nn.Module):
    def __init__(self,
                channels: int,
                grid_resolution: Union[int, List[int], Tuple[int]],
                num_blocks: int,
                dimension: int, 
                norm: bool = True
                ):
        super(ResNet, self).__init__()

        self.channels = channels
        self.grid_resolution = grid_resolution
        self.num_blocks = num_blocks

        self.res_nets = []
        for _ in range(self.num_blocks):
            self.res_nets.append(ResidualBlock(channels = channels,
                                                grid_resolution = grid_resolution,
                                                dimension = dimension,
                                                norm = norm))

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.res_nets[i](x)
        return x