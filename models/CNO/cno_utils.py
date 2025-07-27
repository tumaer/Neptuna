import torch
from omegaconf import ListConfig
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Tuple
from utils.model_utils import PretrainedConfig
from utils.model_utils import CustomNorm
def _to_tuple(x):
    if isinstance(x, ListConfig):
        return tuple(int(i) for i in x)
    elif isinstance(x, (list, tuple)):
        return tuple(int(i) for i in x)
    else:
        return (int(x),)

class CNOConfig(PretrainedConfig):
    """
    Configuration class for the CNO model.

    Args:
        cno_depth (int): Number of (D) or (U) blocks in the network. Default is 4.
        n_blocks (int): Number of (R) blocks per level (except the neck). Default is 4.
        n_blocks_bottleneck (int): Number of (R) blocks in the neck. Default is 4.
        channel_multiplier (int): Multiplier for the number of channels at each level. Default is 16.
        norm (bool): Whether to apply normalization. Default is True.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        cno_depth: int = 4,                                   # Number of (D) or (U) blocks in the network
        n_blocks: int = 4,                                  # Number of (R) blocks per level (except the neck)
        n_blocks_bottleneck: int = 4,                       # Number of (R) blocks in the neck
        channel_multiplier: int = 16, 
        norm: str = "layer",
        norm_layer_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.cno_depth = int(cno_depth)         # Number of (D) & (U) Blocks
        self.n_blocks = n_blocks
        self.n_blocks_bottleneck = n_blocks_bottleneck
        self.norm = norm
        self.norm_layer_eps = norm_layer_eps
        self.lift_dim = channel_multiplier//2 # Input is lifted to the half of channel_multiplier dimension
        self.channel_multiplier = channel_multiplier  # The growth of the channels

#custom activation function for CNO
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
                config: CNOConfig,
                in_channels: int,
                out_channels: int,
                in_grid_resolution: Union[int, List[int], Tuple[int]],
                out_grid_resolution: Union[int, List[int], Tuple[int]],
                ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_grid_resolution  = in_grid_resolution
        self.out_grid_resolution = out_grid_resolution
        dimension = config.dimension

        if dimension == 1:
            Conv = nn.Conv1d
            #BN = nn.BatchNorm1d
        elif dimension == 2:
            Conv = nn.Conv2d
            #BN = nn.BatchNorm2d
        elif dimension == 3:
            Conv = nn.Conv3d
            #BN = nn.BatchNorm3d
        else:
            raise ValueError(f"Unsupported dimension: {dimension}. Must be 1, 2, or 3.")
        #-----------------------------------------

        # We apply Conv -> BN (optional) -> Activation
        # Up/Downsampling happens inside Activation

        self.convolution = Conv(in_channels = self.in_channels,
                                out_channels= self.out_channels,
                                kernel_size = 3,
                                padding     = 1)

        # if norm:
        #     self.batch_norm  = BN(self.out_channels)
        # else:
        #     self.batch_norm  = nn.Identity()
        self.norm  = CustomNorm(config=config, num_channels=self.out_channels, array_length=dimension+2, channel_at_last_position=False)
        self.act           = CNO_LReLu(in_grid_resolution  = self.in_grid_resolution,
                                        out_grid_resolution = self.out_grid_resolution)
    def forward(self, x, **kwargs):
        x = self.convolution(x)
        x = self.norm(x, **kwargs)
        x = self.act(x)
        return x
    
#--------------------
# Lift/Project Block:
#--------------------

class LiftProjectBlock(nn.Module):
    def __init__(self,
                config: CNOConfig,
                in_channels: int,
                out_channels: int,
                grid_resolution: Union[int, List[int], Tuple[int]],
                dimension: int,
                latent_channels: int = 64
                ):
        super().__init__()

        self.inter_CNOBlock = CNOBlock(
                                        config = config,
                                        in_channels = in_channels,
                                        out_channels = latent_channels,
                                        in_grid_resolution = grid_resolution,
                                        out_grid_resolution = grid_resolution
                                        )
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


    def forward(self, x, **kwargs):
        x = self.inter_CNOBlock(x, **kwargs)
        x = self.convolution(x)
        return x

#--------------------
# Residual Block:
#--------------------

class ResidualBlock(nn.Module):
    def __init__(self,
                config: CNOConfig,
                channels: int,
                grid_resolution: Union[int, List[int], Tuple[int]],
                ):
        super().__init__()

        self.channels = channels
        self.grid_resolution = grid_resolution
        dimension = config.dimension
        #-----------------------------------------

        # We apply Conv -> BN (optional) -> Activation -> Conv -> BN (optional) -> Skip Connection
        # Up/Downsampling happens inside Activation
        if dimension == 1:
            Conv = nn.Conv1d
            #BN = nn.BatchNorm1d
        elif dimension == 2:
            Conv = nn.Conv2d
            #BN = nn.BatchNorm2d
        elif dimension == 3:
            Conv = nn.Conv3d
            #BN = nn.BatchNorm3d
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

        # if norm:
        #     self.batch_norm1  = BN(self.channels)
        #     self.batch_norm2  = BN(self.channels)

        # else:
        #     self.batch_norm1  = nn.Identity()
        #     self.batch_norm2  = nn.Identity()
        self.norm1 = CustomNorm(config=config, num_channels=self.channels, array_length=dimension+2, channel_at_last_position=False)
        self.norm2 = CustomNorm(config=config, num_channels=self.channels, array_length=dimension+2, channel_at_last_position=False)

        self.act           = CNO_LReLu(in_grid_resolution  = self.grid_resolution,
                                       out_grid_resolution = self.grid_resolution)


    def forward(self, x, **kwargs):
        out = self.convolution1(x)
        out = self.norm1(out, **kwargs)
        out = self.act(out)
        out = self.convolution2(out)
        out = self.norm2(out, **kwargs)
        return x + out

#--------------------
# ResNet:
#--------------------
class ResNet(nn.Module):
    def __init__(self,
                config: CNOConfig,
                channels: int,
                grid_resolution: Union[int, List[int], Tuple[int]],
                num_blocks: int,
                ):
        super(ResNet, self).__init__()

        self.channels = channels
        self.grid_resolution = grid_resolution
        self.num_blocks = num_blocks
        dimension = config.dimension

        self.res_nets = []
        for _ in range(self.num_blocks):
            self.res_nets.append(ResidualBlock( 
                config = config,
                channels = channels,
                grid_resolution = grid_resolution,
                ))

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x, **kwargs):
        for i in range(self.num_blocks):
            x = self.res_nets[i](x, **kwargs)
        return x