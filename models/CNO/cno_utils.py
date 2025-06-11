import torch
from omegaconf import ListConfig
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Callable, Tuple

def _to_tuple(x):
    if isinstance(x, ListConfig):
        return tuple(int(i) for i in x)
    elif isinstance(x, (list, tuple)):
        return tuple(int(i) for i in x)
    else:
        return (int(x),)
    
class CNO_LReLu(nn.Module):
    def __init__(self,
                in_size: Union[int, List[int], Tuple[int]],
                out_size: Union[int, list[int], Tuple[int]],
                ):
        super().__init__()

        self.in_size = _to_tuple(in_size)
        self.out_size = _to_tuple(out_size)
        self.act = nn.LeakyReLU()
        if len(in_size) == 1:
            self.mode = "linear"
        elif len(in_size) == 2:
            self.mode = "bicubic"
        elif len(in_size) == 3:
            self.mode = "trilinear"

    def forward(self, x):
        up_size = tuple(2 * s for s in self.in_size)
        x = F.interpolate(x, size=up_size, mode=self.mode,antialias = True if self.mode == "bicubi" else False)
        x = self.act(x)
        out_size = tuple(self.out_size)
        x = F.interpolate(x, size=out_size, mode=self.mode ,antialias = True if self.mode == "bicubi" else False)
        return x    

    
class CNOBlock(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                in_size: Union[int, List[int], Tuple[int]],
                out_size: Union[int, List[int], Tuple[int]],
                dimension: int,
                use_bn: bool = True
                ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size  = in_size
        self.out_size = out_size
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

        if use_bn:
            self.batch_norm  = BN(self.out_channels)
        else:
            self.batch_norm  = nn.Identity()
        self.act           = CNO_LReLu(in_size  = self.in_size,
                                        out_size = self.out_size)
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
                size: Union[int, List[int], Tuple[int]],
                dimension: int,
                latent_dim = 64
                ):
        super().__init__()

        self.inter_CNOBlock = CNOBlock(in_channels       = in_channels,
                                        out_channels     = latent_dim,
                                        in_size          = size,
                                        out_size         = size,
                                        dimension        = dimension,
                                        use_bn           = False)
        if dimension == 1:
            Conv = nn.Conv1d
        elif dimension == 2:
            Conv = nn.Conv2d
        elif dimension == 3:
            Conv = nn.Conv3d
        else:
            raise ValueError(f"Unsupported dimension: {dimension}. Must be 1, 2, or 3.")
        self.convolution = Conv(in_channels  = latent_dim,
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
                size: Union[int, List[int], Tuple[int]],
                dimension: int, 
                use_bn: bool = True
                ):
        super().__init__()

        self.channels = channels
        self.size     = size

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

        if use_bn:
            self.batch_norm1  = BN(self.channels)
            self.batch_norm2  = BN(self.channels)

        else:
            self.batch_norm1  = nn.Identity()
            self.batch_norm2  = nn.Identity()

        self.act           = CNO_LReLu(in_size  = self.size,
                                       out_size = self.size)


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
                size: Union[int, List[int], Tuple[int]],
                num_blocks: int,
                dimension: int, 
                use_bn:bool = True
                ):
        super(ResNet, self).__init__()

        self.channels = channels
        self.size = size
        self.num_blocks = num_blocks

        self.res_nets = []
        for _ in range(self.num_blocks):
            self.res_nets.append(ResidualBlock(channels = channels,
                                                size = size,
                                                dimension = dimension,
                                                use_bn = use_bn))

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.res_nets[i](x)
        return x