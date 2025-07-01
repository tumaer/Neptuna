from typing import List, Tuple, Union, Optional
import torch
from torch import Tensor
import torch.nn as nn
from utils import activation_func

from .unet_utils import DownBlockND, UpBlockND, MiddleBlockND, DownsampleND, UpsampleND
from utils.feature_utils import oned_meshgrid, twod_meshgrid, threed_meshgrid

class UNet(nn.Module): 
    """Modern U-Net architecture for fluid dynamics simulation

    A flexible U-Net implementation that supports 1D, 2D, and 3D data processing for fluid dynamics
    simulation. The architecture includes wide-residual blocks, spatial attention blocks, and optional
    coordinate features for spatial awareness. It processes both scalar and vector channels with
    multi-resolution feature extraction and reconstruction.

    Args:
        in_channels (int): Number of input channels/fields
        out_channels (int): Number of output channels/fields
        latent_channels (int): Number of channels in the hidden layers
        activation_fn_name (str): Name of the activation function (default: "gelu")
        sequence_info (List[int]): Configuration for input/output sequences [input_seq_len, output_seq_len, stride]
        dimension (int): Spatial dimension of the data (1, 2, or 3)
        norm (bool): Whether to use normalization layers (default: False)
        channel_multiplier (Union[Tuple[int, ...], List[int]]): Channel multipliers for each resolution level (default: (1, 2, 2, 4))
        is_attn (Union[Tuple[bool, ...], List[bool]]): Whether to use attention at each resolution (default: (False, False, False, False))
        mid_attn (bool): Whether to use attention in the middle block (default: False)
        n_blocks (int): Number of residual blocks per resolution (default: 2)
        use1x1 (bool): Whether to use 1x1 convolutions in initial/final layers (default: False)

    The architecture includes:
    - Multi-resolution processing with skip connections
    - Optional coordinate features for spatial awareness
    - Wide residual blocks with optional attention
    - Configurable normalization and activation functions
    - Flexible input/output sequence handling
    """
    #TODO: Do this for all models
    main_input_name = "input_data"
    conditioning_input_name = "conditioning_input_data"
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        activation_fn_name: str="gelu",
        sequence_info: Optional[List[int]] = [1,1,1],
        dimension: int = 2,
        norm: bool = False,
        n_groups: int = 1,
        channel_multiplier: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        use1x1: bool = False,
    ) -> None:
        
        super().__init__()
        self.dimension = dimension

        self.activation = activation_func.get_activation(activation_fn_name)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation_fn_name} not implemented")
        # Number of resolutions (depth of unet)
        unet_depth = len(channel_multiplier)

        in_size = in_channels*sequence_info[0]
        out_size = out_channels*sequence_info[1]
        
        self.unet = self.build_UNet()(in_size=in_size, 
                                   out_size=out_size, 
                                   latent_channels=latent_channels, 
                                   unet_depth=unet_depth, 
                                   channel_multiplier=channel_multiplier, 
                                   is_attn=is_attn, 
                                   mid_attn=mid_attn, 
                                   n_blocks=n_blocks, 
                                   activation_fn_name=activation_fn_name, 
                                   norm=norm, 
                                   n_groups=n_groups,
                                   use1x1=use1x1)
       
    def build_UNet(self):
        """Get the appropriate upsampler based on the dimension."""
        if self.dimension == 1:
            return UNet1D
        elif self.dimension == 2:
            return UNet2D
        elif self.dimension == 3:
            return UNet3D
        else:
            raise NotImplementedError(f"UNet not implemented for dimension {self.dimension}")

    
    ### Main Forward function ###
    def forward(self, input_data: Tensor, **kwargs) -> Tensor:
        
        if "conditioning_input_data" in kwargs:
            #NOTE: Conditioning data can be passed into a conv network before concatination with input_data.
            conditioning_input_data = kwargs["conditioning_input_data"]
            input_data = torch.cat([input_data, conditioning_input_data], dim=2)
        else:
            conditioning_input_data = None

        batch, input_seq, input_channels, *spatial = input_data.shape
        x=input_data.reshape(batch, input_seq * input_channels, *spatial)

        x = self.unet(x)

        return x

#Unet based on the dimension
class UNet1D(nn.Module):
    """1D U-Net"""
    def __init__(self, 
                 in_size: int, 
                 out_size: int,
                 latent_channels: int, 
                 unet_depth: int, 
                 channel_multiplier: Union[Tuple[int, ...], List[int]], 
                 is_attn: Union[Tuple[bool, ...], List[bool]], 
                 mid_attn: bool, 
                 n_blocks: int, 
                 activation_fn_name: str,
                 norm: bool,
                 n_groups: int,
                 use1x1: bool,
                 coord_features: bool = True
                 ):
        super().__init__()

        self.activation = activation_func.get_activation(activation_fn_name)
        self.coord_features = coord_features
        if self.coord_features:
            in_size = in_size + 1
        
        # Project image into feature map
        if use1x1: #false by default
            self.image_proj = nn.Conv1d(in_size, latent_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv1d(in_size, latent_channels, kernel_size=(3, ), padding=(1, ))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels_down = in_channels_down = latent_channels
        # For each resolution
        for i in range(unet_depth):
            # Number of output channels at this resolution
            out_channels_down = in_channels_down * channel_multiplier[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlockND(
                        in_channels=in_channels_down,
                        out_channels=out_channels_down,
                        dim=1,
                        has_attn=is_attn[i],
                        activation=activation_fn_name,
                        norm=norm,
                        n_groups=n_groups,
                    )
                )
                in_channels_down = out_channels_down
            # Down sample at all resolutions except the last
            if i < unet_depth - 1:
                down.append(DownsampleND(n_channels=in_channels_down, dim=1))

        # Combine the set of modules
        self.down = nn.ModuleList(down)
        out_channels_mid=out_channels_down
        self.middle = MiddleBlockND(n_channels=out_channels_mid, 
                                    dim=1, 
                                    has_attn=mid_attn, 
                                    activation=activation_fn_name, 
                                    norm=norm,
                                    n_groups=n_groups)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels_up = out_channels_mid
        # For each resolution
        for i in reversed(range(unet_depth)):
            # `n_blocks` at the same resolution
            out_channels_up = in_channels_up
            for _ in range(n_blocks):
                up.append(
                    UpBlockND(
                        in_channels=in_channels_up,
                        out_channels=out_channels_up,
                        dim=1,
                        has_attn=is_attn[i],
                        activation=activation_fn_name,
                        norm=norm,
                        n_groups=n_groups,
                    )
                )
            # Final block to reduce the number of channels
            out_channels_up = in_channels_up // channel_multiplier[i]
            up.append(UpBlockND(in_channels=in_channels_up, 
                                out_channels=out_channels_up, 
                                dim=1, 
                                has_attn=is_attn[i], 
                                activation=activation_fn_name, 
                                norm=norm,
                                n_groups=n_groups))
            
            in_channels_up = out_channels_up
            # Up sample at all resolutions except last
            if i > 0:
                up.append(UpsampleND(n_channels=in_channels_up, dim=1))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, latent_channels)
        else:
            self.norm = nn.Identity()

        if use1x1:
            self.final = nn.Conv1d(in_channels_up, out_size, kernel_size=1)
        else:
            self.final = nn.Conv1d(in_channels_up, out_size, kernel_size=(3,), padding=(1,))

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(
                "Only 3D tensors [batch, in_channels, grid_x] accepted for 1D UNet"
            )
        #add coord features
        if self.coord_features: 
            coord_feat = oned_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        
        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, UpsampleND):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)

        x = self.final(self.activation(self.norm(x)))

        return x

class UNet2D(nn.Module):
    """2D U-Net"""
    def __init__(self, 
                 in_size: int, 
                 out_size: int,
                 latent_channels: int, 
                 unet_depth: int, 
                 channel_multiplier: Union[Tuple[int, ...], List[int]], 
                 is_attn: Union[Tuple[bool, ...], List[bool]], 
                 mid_attn: bool, 
                 n_blocks: int, 
                 activation_fn_name: str,
                 norm: bool,
                 n_groups: int,
                 use1x1: bool,
                 coord_features: bool = True):
        super().__init__()

        self.activation = activation_func.get_activation(activation_fn_name)
        
        self.coord_features = coord_features
        if self.coord_features:
            in_size = in_size + 2
        
        # Project image into feature map
        if use1x1: #false by default
            self.image_proj = nn.Conv2d(in_size, latent_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv2d(in_size, latent_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels_down = in_channels_down = latent_channels
        # For each resolution
        for i in range(unet_depth):
            # Number of output channels at this resolution
            out_channels_down = in_channels_down * channel_multiplier[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlockND(
                        in_channels=in_channels_down,
                        out_channels=out_channels_down,
                        dim=2,
                        has_attn=is_attn[i],
                        activation=activation_fn_name,
                        norm=norm,
                        n_groups=n_groups,
                    )
                )
                in_channels_down = out_channels_down
            # Down sample at all resolutions except the last
            if i < unet_depth - 1:
                down.append(DownsampleND(n_channels=in_channels_down, dim=2))

        # Combine the set of modules
        self.down = nn.ModuleList(down)
        
        out_channels_mid = out_channels_down
        self.middle = MiddleBlockND(n_channels=out_channels_mid, 
                                    dim=2, 
                                    has_attn=mid_attn, 
                                    activation=activation_fn_name, 
                                    norm=norm,
                                    n_groups=n_groups)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels_up = out_channels_mid
        # For each resolution
        for i in reversed(range(unet_depth)):
            # `n_blocks` at the same resolution
            out_channels_up = in_channels_up
            for _ in range(n_blocks):
                up.append(
                    UpBlockND(
                        in_channels=in_channels_up,
                        out_channels=out_channels_up,
                        dim=2,
                        has_attn=is_attn[i],
                        activation=activation_fn_name,
                        norm=norm,
                        n_groups=n_groups,
                    )
                )
            # Final block to reduce the number of channels
            out_channels_up = in_channels_up // channel_multiplier[i]
            up.append(UpBlockND(in_channels=in_channels_up, 
                                out_channels=out_channels_up, 
                                dim=2, 
                                has_attn=is_attn[i], 
                                activation=activation_fn_name, 
                                norm=norm,
                                n_groups=n_groups))
            in_channels_up = out_channels_up
            # Up sample at all resolutions except last
            if i > 0:
                up.append(UpsampleND(n_channels=in_channels_up, dim=2))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, latent_channels)
        else:
            self.norm = nn.Identity()

        if use1x1:
            self.final = nn.Conv2d(in_channels_up, out_size, kernel_size=1)
        else:
            self.final = nn.Conv2d(in_channels_up, out_size, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(
                "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D UNet"
            )

        if self.coord_features: 
            coord_feat = twod_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, UpsampleND):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)

        x = self.final(self.activation(self.norm(x)))

        return x
    
class UNet3D(nn.Module):
    """3D U-Net"""
    def __init__(self, 
                 in_size: int, 
                 out_size: int,
                 latent_channels: int, 
                 unet_depth: int, 
                 channel_multiplier: Union[Tuple[int, ...], List[int]], 
                 is_attn: Union[Tuple[bool, ...], List[bool]], 
                 mid_attn: bool, 
                 n_blocks: int, 
                 activation_fn_name: str,
                 norm: bool, 
                 n_groups: int,
                 use1x1: bool,
                 coord_features: bool = True):
        super().__init__()

        self.activation = activation_func.get_activation(activation_fn_name)
        self.coord_features = coord_features
        # Add relative coordinate feature
        if self.coord_features:
            in_size = in_size + 3
        
        # Project image into feature map
        if use1x1: #false by default
            self.image_proj = nn.Conv3d(in_size, latent_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv3d(in_size, latent_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels_down = in_channels_down = latent_channels
        # For each resolution
        for i in range(unet_depth):
            # Number of output channels at this resolution
            out_channels_down = in_channels_down * channel_multiplier[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlockND(
                        in_channels=in_channels_down,
                        out_channels=out_channels_down,
                        dim=3,
                        has_attn=is_attn[i],
                        activation=activation_fn_name,
                        norm=norm,
                        n_groups=n_groups,
                    )
                )
                in_channels_down = out_channels_down
            # Down sample at all resolutions except the last
            if i < unet_depth - 1:
                down.append(DownsampleND(n_channels=in_channels_down, dim=3))

        # Combine the set of modules
        self.down = nn.ModuleList(down)
        
        out_channels_mid = out_channels_down
        self.middle = MiddleBlockND(n_channels=out_channels_mid, 
                                    dim=3, 
                                    has_attn=mid_attn, 
                                    activation=activation_fn_name, 
                                    norm=norm,
                                    n_groups=n_groups)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels_up = out_channels_mid
        # For each resolution
        for i in reversed(range(unet_depth)):
            # `n_blocks` at the same resolution
            out_channels_up = in_channels_up
            for _ in range(n_blocks):
                up.append(
                    UpBlockND(
                        in_channels=in_channels_up,
                        out_channels=out_channels_up,
                        dim=3,
                        has_attn=is_attn[i],
                        activation=activation_fn_name,
                        norm=norm,
                        n_groups=n_groups,
                    )
                )
            # Final block to reduce the number of channels
            out_channels_up = in_channels_up // channel_multiplier[i]
            up.append(UpBlockND(in_channels=in_channels_up, 
                                out_channels=out_channels_up, 
                                dim=3, 
                                has_attn=is_attn[i], 
                                activation=activation_fn_name, 
                                norm=norm,
                                n_groups=n_groups))
            
            in_channels_up = out_channels_up
            # Up sample at all resolutions except last
            if i > 0:
                up.append(UpsampleND(n_channels=in_channels_up, dim=3))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, latent_channels)
        else:
            self.norm = nn.Identity()

        if use1x1:
            self.final = nn.Conv3d(in_channels_up, out_size, kernel_size=1)
        else:
            self.final = nn.Conv3d(in_channels_up, out_size, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x: torch.Tensor):
        if x.dim() != 5:
            raise ValueError(
                "Only 5D tensors [batch, in_channels, grid_x, grid_y, grid_z] accepted for 3D ResNet"
            )
        
        #add feature map
        if self.coord_features: 
            coord_feat = threed_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, UpsampleND):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)

        x = self.final(self.activation(self.norm(x)))

        return x