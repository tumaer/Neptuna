
from typing import List, Tuple, Union, Optional
import torch
from torch import Tensor
import torch.nn as nn
from utils import activation_func

from .unet_utils import DownBlockND, UpBlockND, MiddleBlockND, DownsampleND, UpsampleND

class UNet(nn.Module): 
    """Modern U-Net architecture

    This is a modern U-Net architecture with wide-residual blocks and spatial attention blocks

    Args:
        n_input_scalar_components (int): Number of scalar components in the model
        n_input_vector_components (int): Number of vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        time_history (int): Number of time steps in the input
        time_future (int): Number of time steps in the output
        hidden_channels (int): Number of channels in the hidden layers
        activation (str): Activation function to use
        norm (bool): Whether to use normalization
        ch_mults (list): List of channel multipliers for each resolution
        is_attn (list): List of booleans indicating whether to use attention blocks
        mid_attn (bool): Whether to use attention block in the middle block
        n_blocks (int): Number of residual blocks in each resolution
        use1x1 (bool): Whether to use 1x1 convolutions in the initial and final layers
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        activation_fn_name: str="gelu",
        sequence_info: Optional[List[List[int]]] = [[1,1,1,1]],
        dimension: int = 2,
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
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
        n_resolutions = len(ch_mults)

        insize = in_channels*sequence_info[0][0]
        outsize = out_channels*sequence_info[0][1]
        
        self.unet = self.build_UNet()(insize, 
                                   outsize, 
                                   latent_channels, 
                                   n_resolutions, 
                                   ch_mults, 
                                   is_attn, 
                                   mid_attn, 
                                   n_blocks, 
                                   activation_fn_name, 
                                   norm, 
                                   use1x1)
       
    def build_UNet(self):
        """Get the appropriate upsampler based on the dimension."""
        if self.dimension == 1:
            return UNet1D
        elif self.dimension == 2:
            return UNet2D
        elif self.dimension == 3:
            return UNet3D
        else:
            raise NotImplementedError(f"Upsampler not implemented for dimension {self.dimension}")

    
    ### Main Forward function ###
    def forward(self, input_data: Tensor,
                labels: Tensor) -> Tensor:
        
        #assert input_data.dim() == 5 #for 2d
        #assert labels.dim() == 5
        
        #orig_shape = input_data.shape
        batch, input_seq, input_fields, *spatial = input_data.shape
        x=input_data.reshape(batch, input_seq * input_fields, *spatial)

        x = self.unet(x)

        batch, output_seq, output_fields, *spatial = labels.shape
        x = x.reshape(
            batch, output_seq, output_fields, *spatial)
        return x

#Unet based on the dimension
class UNet1D(nn.Module):
    """1D U-Net"""
    def __init__(self, 
                 insize, 
                 outsize,
                 hidden_channels, 
                 n_resolutions, 
                 ch_mults, 
                 is_attn, 
                 mid_attn, 
                 n_blocks, 
                 activation_fn_name,
                 norm, 
                 use1x1):
        super().__init__()

        self.activation = activation_func.get_activation(activation_fn_name)
        # Project image into feature map
        if use1x1: #false by default
            self.image_proj = nn.Conv1d(insize, hidden_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv1d(insize, hidden_channels, kernel_size=(3, ), padding=(1, ))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = hidden_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlockND(
                        in_channels,
                        out_channels,
                        dim=1,
                        has_attn=is_attn[i],
                        activation=activation_fn_name,
                        norm=norm,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(DownsampleND(in_channels, dim=1))

        # Combine the set of modules
        self.down = nn.ModuleList(down)
        
        self.middle = MiddleBlockND(out_channels, dim=1, has_attn=mid_attn, activation=activation_fn_name, norm=norm)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlockND(
                        in_channels,
                        out_channels,
                        dim=1,
                        has_attn=is_attn[i],
                        activation=activation_fn_name,
                        norm=norm,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlockND(in_channels, out_channels, dim=1, has_attn=is_attn[i], activation=activation_fn_name, norm=norm))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(UpsampleND(in_channels, dim=1))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, hidden_channels)
        else:
            self.norm = nn.Identity()

        if use1x1:
            self.final = nn.Conv1d(in_channels, outsize, kernel_size=1)
        else:
            self.final = nn.Conv1d(in_channels, outsize, kernel_size=(3,), padding=(1,))

    def forward(self, x: torch.Tensor):

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
                 insize, 
                 outsize,
                 hidden_channels, 
                 n_resolutions, 
                 ch_mults, 
                 is_attn, 
                 mid_attn, 
                 n_blocks, 
                 activation_fn_name,
                 norm, 
                 use1x1):
        super().__init__()

        self.activation = activation_func.get_activation(activation_fn_name)
        # Project image into feature map
        if use1x1: #false by default
            self.image_proj = nn.Conv2d(insize, hidden_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv2d(insize, hidden_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = hidden_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlockND(
                        in_channels,
                        out_channels,
                        dim=2,
                        has_attn=is_attn[i],
                        activation=activation_fn_name,
                        norm=norm,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(DownsampleND(in_channels, dim=2))

        # Combine the set of modules
        self.down = nn.ModuleList(down)
        
        self.middle = MiddleBlockND(out_channels, dim=2, has_attn=mid_attn, activation=activation_fn_name, norm=norm)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlockND(
                        in_channels,
                        out_channels,
                        dim=2,
                        has_attn=is_attn[i],
                        activation=activation_fn_name,
                        norm=norm,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlockND(in_channels, out_channels, dim=2, has_attn=is_attn[i], activation=activation_fn_name, norm=norm))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(UpsampleND(in_channels, dim=2))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, hidden_channels)
        else:
            self.norm = nn.Identity()

        if use1x1:
            self.final = nn.Conv2d(in_channels, outsize, kernel_size=1)
        else:
            self.final = nn.Conv2d(in_channels, outsize, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor):

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
                 insize, 
                 outsize,
                 hidden_channels, 
                 n_resolutions, 
                 ch_mults, 
                 is_attn, 
                 mid_attn, 
                 n_blocks, 
                 activation_fn_name,
                 norm, 
                 use1x1):
        super().__init__()

        self.activation = activation_func.get_activation(activation_fn_name)
        # Project image into feature map
        if use1x1: #false by default
            self.image_proj = nn.Conv3d(insize, hidden_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv3d(insize, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = hidden_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlockND(
                        in_channels,
                        out_channels,
                        dim=3,
                        has_attn=is_attn[i],
                        activation=activation_fn_name,
                        norm=norm,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(DownsampleND(in_channels, dim=3))

        # Combine the set of modules
        self.down = nn.ModuleList(down)
        
        self.middle = MiddleBlockND(out_channels, dim=3, has_attn=mid_attn, activation=activation_fn_name, norm=norm)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlockND(
                        in_channels,
                        out_channels,
                        dim=3,
                        has_attn=is_attn[i],
                        activation=activation_fn_name,
                        norm=norm,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlockND(in_channels, out_channels, dim=3, has_attn=is_attn[i], activation=activation_fn_name, norm=norm))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(UpsampleND(in_channels, dim=3))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, hidden_channels)
        else:
            self.norm = nn.Identity()

        if use1x1:
            self.final = nn.Conv3d(in_channels, outsize, kernel_size=1)
        else:
            self.final = nn.Conv3d(in_channels, outsize, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x: torch.Tensor):

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