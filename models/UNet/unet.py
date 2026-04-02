from typing import List, Tuple, Union, Optional
import torch
from torch import Tensor
import torch.nn as nn
from utils import activation_func
import torch.nn.functional as F

from transformers import PreTrainedModel
from .unet_utils import DownBlockND, UpBlockND, MiddleBlockND, DownsampleND, UpsampleND, UNetConfig
from utils.grid_utils import oned_meshgrid, twod_meshgrid, threed_meshgrid
from utils.model_utils import CustomNorm

class UNet(PreTrainedModel): 
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
        channel_multiplier (Union[Tuple[int, ...], List[int]]): Channel multipliers for each resolution level (default: (1, 2, 2, 4))
        is_attn (Union[Tuple[bool, ...], List[bool]]): Whether to use attention at each resolution (default: (False, False, False, False))
        mid_attn (bool): Whether to use attention in the middle block (default: False)
        n_blocks (int): Number of residual blocks per resolution (default: 2)
        norm: "group" or "layer" or "batch" or "none" (default: "layer")
        norm_layer_eps: Used to avoid zero-div error. float (default: 1e-5)
        use1x1 (bool): Whether to use 1x1 convolutions in initial/final layers (default: False)

    The architecture includes:
    - Multi-resolution processing with skip connections
    - Optional coordinate features for spatial awareness
    - Wide residual blocks with optional attention
    - Configurable normalization and activation functions
    - Flexible input/output sequence handling
    """
    #TODO: Do this for all models
    main_input_name = "input_data" #input_ids is written?
    conditioning_input_name = "conditioning_input_data"
    config_class = UNetConfig
    
    def __init__(self, config):
        
        super().__init__(config)
        super().post_init()

        self.config = config

        activation = activation_func.get_activation(config.activation_fn_name)
        if activation is None:
            raise NotImplementedError(f"Activation {config.activation_fn_name} not implemented")
        
        self.unet = self.build_UNet()(config=config, activation=activation)
       
    def build_UNet(self):
        """Get the appropriate upsampler based on the dimension."""
        if self.config.dimension == 1:
            return UNet1D
        elif self.config.dimension == 2:
            return UNet2D
        elif self.config.dimension == 3:
            return UNet3D
        else:
            raise NotImplementedError(f"UNet not implemented for dimension {self.config.dimension}")

    
    ### Main Forward function ###
    def forward(self, input_data: Tensor, **kwargs) -> Tensor:

        # waht is the shape of input? [6, 256, 256] ?
        
        if "conditioning_input_data" in kwargs:
            #NOTE: Conditioning data can be passed into a conv network before concatination with input_data.
            conditioning_input_data = kwargs["conditioning_input_data"]
            input_data = torch.cat([input_data, conditioning_input_data], dim=2)
        else:
            conditioning_input_data = None

        batch, input_seq, input_channels, *spatial = input_data.shape
        x = input_data.reshape(batch, input_seq * input_channels, *spatial)
        return self.unet(x, **kwargs)

#Unet based on the dimension
class UNet1D(PreTrainedModel):
    """1D U-Net"""
    def __init__(self, config, activation: nn.Module):
        super().__init__(config)

        # Number of resolutions (depth of unet)
        unet_depth = len(config.channel_multiplier)
        self.activation = activation
        
        # Project image into feature map
        if config.use1x1: #false by default
            self.image_proj = nn.Conv1d(config.in_size, config.latent_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv1d(config.in_size, config.latent_channels, kernel_size=(3, ), padding=(1, ))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels_down = in_channels_down = config.latent_channels
        # For each resolution
        for i in range(unet_depth):
            # Number of output channels at this resolution
            out_channels_down = in_channels_down * config.channel_multiplier[i]
            # Add `n_blocks`
            for _ in range(config.n_blocks):
                down.append(
                    DownBlockND(
                        config=config,
                        in_channels=in_channels_down,
                        out_channels=out_channels_down,
                        dim=1,
                        has_attn=config.is_attn[i],
                        activation=config.activation_fn_name,
                    )
                )
                in_channels_down = out_channels_down
            # Down sample at all resolutions except the last
            if i < unet_depth - 1:
                down.append(DownsampleND(n_channels=in_channels_down, dim=1))

        # Combine the set of modules
        self.down = nn.ModuleList(down)
        out_channels_mid=out_channels_down
        self.middle = MiddleBlockND(config=config,
                                    n_channels=out_channels_mid, 
                                    dim=1, 
                                    has_attn=config.mid_attn, 
                                    activation=config.activation_fn_name, 
                                    )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels_up = out_channels_mid
        # For each resolution
        for i in reversed(range(unet_depth)):
            # `n_blocks` at the same resolution
            out_channels_up = in_channels_up
            for _ in range(config.n_blocks):
                up.append(
                    UpBlockND(
                        config=config,
                        in_channels=in_channels_up,
                        out_channels=out_channels_up,
                        dim=1,
                        has_attn=config.is_attn[i],
                        activation=config.activation_fn_name,
                    )
                )
            # Final block to reduce the number of channels
            out_channels_up = in_channels_up // config.channel_multiplier[i]
            up.append(UpBlockND(config=config,
                                in_channels=in_channels_up, 
                                out_channels=out_channels_up, 
                                dim=1, 
                                has_attn=config.is_attn[i], 
                                activation=config.activation_fn_name, 
                                ))
            
            in_channels_up = out_channels_up
            # Up sample at all resolutions except last
            if i > 0:
                up.append(UpsampleND(n_channels=in_channels_up, dim=1))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        self.norm = CustomNorm(config=config, num_channels=config.latent_channels, array_length=3, channel_at_last_position=False)
        if config.use1x1:
            self.final = nn.Conv1d(in_channels_up, config.out_size, kernel_size=1)
        else:
            self.final = nn.Conv1d(in_channels_up, config.out_size, kernel_size=(3,), padding=(1,))

        # Overall downsampling factor for padding logic
        self._ds_factor = 2 ** (unet_depth - 1)

    def forward(self, x: torch.Tensor, **kwargs):
        if x.dim() != 3:
            raise ValueError(
                "Only 3D tensors [batch, in_channels, grid_x] accepted for 1D UNet"
            )
        
        # ---- Symmetric padding so length divisible by self._ds_factor ----
        L_orig = x.shape[-1]
        pad_L = (-L_orig) % self._ds_factor
        left = pad_L // 2
        right = pad_L - left
        if pad_L:
            x = F.pad(x, (left, right))

        # add coord features after padding
        if self.config.coord_features:
            coord_feat = oned_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        
        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x, **kwargs)
            h.append(x)

        x = self.middle(x, **kwargs)

        for m in self.up:
            if isinstance(m, UpsampleND):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, **kwargs)

        x = self.norm(x, **kwargs)
        x = self.activation(x)
        x = self.final(x)

        # Crop back to original length if padded
        if pad_L:
            x = x[..., left:left + L_orig]

        return x

class UNet2D(PreTrainedModel):
    """2D U-Net"""
    def __init__(self, config, activation: nn.Module):
        super().__init__(config)

        self.activation = activation
        # Number of resolutions (depth of unet)
        unet_depth = len(config.channel_multiplier)
        
        # Project image into feature map
        if config.use1x1: #false by default
            self.image_proj = nn.Conv2d(config.in_size, config.latent_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv2d(config.in_size, config.latent_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels_down = in_channels_down = config.latent_channels
        # For each resolution
        for i in range(unet_depth):
            # Number of output channels at this resolution
            out_channels_down = in_channels_down * config.channel_multiplier[i]
            # Add `n_blocks`
            for _ in range(config.n_blocks):
                down.append(
                    DownBlockND(
                        config=config,
                        in_channels=in_channels_down,
                        out_channels=out_channels_down,
                        dim=2,
                        has_attn=config.is_attn[i],
                        activation=config.activation_fn_name,
                    )
                )
                in_channels_down = out_channels_down
            # Down sample at all resolutions except the last
            if i < unet_depth - 1:
                down.append(DownsampleND(n_channels=in_channels_down, dim=2))

        # Combine the set of modules
        self.down = nn.ModuleList(down)
        
        out_channels_mid = out_channels_down
        self.middle = MiddleBlockND(config=config,
                                    n_channels=out_channels_mid, 
                                    dim=2, 
                                    has_attn=config.mid_attn, 
                                    activation=config.activation_fn_name, 
                                    )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels_up = out_channels_mid
        # For each resolution
        for i in reversed(range(unet_depth)):
            # `n_blocks` at the same resolution
            out_channels_up = in_channels_up
            for _ in range(config.n_blocks):
                up.append(
                    UpBlockND(
                        config=config,
                        in_channels=in_channels_up,
                        out_channels=out_channels_up,
                        dim=2,
                        has_attn=config.is_attn[i],
                        activation=config.activation_fn_name,
                    )
                )
            # Final block to reduce the number of channels
            out_channels_up = in_channels_up // config.channel_multiplier[i]
            up.append(UpBlockND(config=config,
                                in_channels=in_channels_up, 
                                out_channels=out_channels_up, 
                                dim=2, 
                                has_attn=config.is_attn[i], 
                                activation=config.activation_fn_name, 
                                ))
            in_channels_up = out_channels_up
            # Up sample at all resolutions except last
            if i > 0:
                up.append(UpsampleND(n_channels=in_channels_up, dim=2))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        self.norm = CustomNorm(config=config, num_channels=config.latent_channels, array_length=4, channel_at_last_position=False)
        if config.use1x1:
            self.final = nn.Conv2d(in_channels_up, config.out_size, kernel_size=1)
        else:
            self.final = nn.Conv2d(in_channels_up, config.out_size, kernel_size=(3, 3), padding=(1, 1))

        # Overall downsampling factor for padding logic
        self._ds_factor = 2 ** (unet_depth - 1)

    def forward(self, x: torch.Tensor, **kwargs):
        if x.dim() != 4:
            raise ValueError(
                "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D UNet"
            )

        # ---- Pad so H and W divisible by self._ds_factor ----
        H_orig, W_orig = x.shape[-2], x.shape[-1]
        pad_H = (-H_orig) % self._ds_factor
        pad_W = (-W_orig) % self._ds_factor
        top = pad_H // 2
        bottom = pad_H - top
        left = pad_W // 2
        right = pad_W - left
        if pad_H or pad_W:
            x = F.pad(x, (left, right, top, bottom))

        # add coord features after padding
        if self.config.coord_features:
            coord_feat = twod_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.image_proj(x)

        h = [x]
        for i,m in enumerate(self.down):
            x = m(x, **kwargs)
            h.append(x)

        x = self.middle(x, **kwargs)

        for i, m in enumerate(self.up):
            if isinstance(m, UpsampleND):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop() #Me: modifies the list by removing the last element.
                x = torch.cat((x, s), dim=1)
                x = m(x, **kwargs)

        x = self.norm(x, **kwargs)
        x = self.activation(x)
        x = self.final(x)

        # Crop back to original size if padded
        if pad_H or pad_W:
            x = x[..., top:top + H_orig, left:left + W_orig]

        return x
    
class UNet3D(PreTrainedModel):
    """3D U-Net"""
    def __init__(self, config, activation: nn.Module):
        super().__init__(config)

        self.activation = activation
        # Number of resolutions (depth of unet)
        unet_depth = len(config.channel_multiplier)
        
        # Project image into feature map
        if config.use1x1: #false by default
            self.image_proj = nn.Conv3d(config.in_size, config.latent_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv3d(config.in_size, config.latent_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels_down = in_channels_down = config.latent_channels
        # For each resolution
        for i in range(unet_depth):
            # Number of output channels at this resolution
            out_channels_down = in_channels_down * config.channel_multiplier[i]
            # Add `n_blocks`
            for _ in range(config.n_blocks):
                down.append(
                    DownBlockND(
                        config=config,
                        in_channels=in_channels_down,
                        out_channels=out_channels_down,
                        dim=3,
                        has_attn=config.is_attn[i],
                        activation=config.activation_fn_name,
                    )
                )
                in_channels_down = out_channels_down
            # Down sample at all resolutions except the last
            if i < unet_depth - 1:
                down.append(DownsampleND(n_channels=in_channels_down, dim=3))

        # Combine the set of modules
        self.down = nn.ModuleList(down)
        
        out_channels_mid = out_channels_down
        self.middle = MiddleBlockND(config=config,
                                    n_channels=out_channels_mid, 
                                    dim=3, 
                                    has_attn=config.mid_attn, 
                                    activation=config.activation_fn_name, 
                                    )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels_up = out_channels_mid
        # For each resolution
        for i in reversed(range(unet_depth)):
            # `n_blocks` at the same resolution
            out_channels_up = in_channels_up
            for _ in range(config.n_blocks):
                up.append(
                    UpBlockND(
                        config=config,
                        in_channels=in_channels_up,
                        out_channels=out_channels_up,
                        dim=3,
                        has_attn=config.is_attn[i],
                        activation=config.activation_fn_name,
                    )
                )
            # Final block to reduce the number of channels
            out_channels_up = in_channels_up // config.channel_multiplier[i]
            up.append(UpBlockND(config=config,
                                in_channels=in_channels_up, 
                                out_channels=out_channels_up, 
                                dim=3, 
                                has_attn=config.is_attn[i], 
                                activation=config.activation_fn_name, 
                                ))
            
            in_channels_up = out_channels_up
            # Up sample at all resolutions except last
            if i > 0:
                up.append(UpsampleND(n_channels=in_channels_up, dim=3))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        self.norm = CustomNorm(config=config, num_channels=config.latent_channels, array_length=5, channel_at_last_position=False)
        if config.use1x1:
            self.final = nn.Conv3d(in_channels_up, config.out_size, kernel_size=1)
        else:
            self.final = nn.Conv3d(in_channels_up, config.out_size, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # Overall downsampling factor for padding logic
        self._ds_factor = 2 ** (unet_depth - 1)

    def forward(self, x: torch.Tensor, **kwargs):
        if x.dim() != 5:
            raise ValueError(
                "Only 5D tensors [batch, in_channels, grid_x, grid_y, grid_z] accepted for 3D ResNet"
            )
        
        # ---- Pad so D, H, W divisible by self._ds_factor ----
        D_orig, H_orig, W_orig = x.shape[-3], x.shape[-2], x.shape[-1]
        pad_D = (-D_orig) % self._ds_factor
        pad_H = (-H_orig) % self._ds_factor
        pad_W = (-W_orig) % self._ds_factor

        front = pad_D // 2
        back = pad_D - front
        top = pad_H // 2
        bottom = pad_H - top
        left = pad_W // 2
        right = pad_W - left

        if pad_D or pad_H or pad_W:
            x = F.pad(x, (left, right, top, bottom, front, back))

        # add feature map after padding
        if self.config.coord_features:
            coord_feat = threed_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x, **kwargs)
            h.append(x)

        x = self.middle(x, **kwargs)

        for m in self.up:
            if isinstance(m, UpsampleND):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, **kwargs)

        #without the movedim, step by step
        x = self.norm(x, **kwargs)
        x = self.activation(x)
        x = self.final(x)

        # Crop back if padded
        if pad_D or pad_H or pad_W:
            x = x[..., front:front + D_orig, top:top + H_orig, left:left + W_orig]

        return x