
from typing import List, Tuple, Union, Optional
import torch
from torch import Tensor
import torch.nn as nn
from utils import activation_func
from models.unet.unet_utils import SpectralConv2d

class ResidualBlock2D(nn.Module):
    """Wide Residual Blocks used in modern Unet architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Activation function to use.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn_name: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        self.activation = activation_func.get_activation(activation_fn_name)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation_fn_name} not implemented")
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        # First convolution layer
        h = self.conv1(self.activation(self.norm1(x)))
        # Second convolution layer
        h = self.conv2(self.activation(self.norm2(h)))
        # Add the shortcut connection and return
        return h + self.shortcut(x)

class AttentionBlock2D(nn.Module):
    """Attention block This is similar to [transformer multi-head
    attention]

    Args:
        n_channels (int): the number of channels in the input
        n_heads (int): the number of heads in multi-head attention
        d_k: the number of dimensions in each head
        n_groups (int): the number of groups for [group normalization][torch.nn.GroupNorm].

    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: Optional[int] = None, n_groups: int = 1):
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res

class DownBlock2D(nn.Module):
    """Down block This combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the first half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (nn.Module): Activation function
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        self.res = ResidualBlock2D(in_channels, out_channels, activation_fn_name=activation, norm=norm)
        if has_attn:
            self.attn = AttentionBlock2D(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x

class UpBlock2D(nn.Module):
    """Up block that combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the second half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (str): Activation function
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock2D(in_channels + out_channels, out_channels, activation_fn_name=activation, norm=norm)
        if has_attn:
            self.attn = AttentionBlock2D(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x

class MiddleBlock2D(nn.Module):
    """Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        activation (str): Activation function to use. Defaults to "gelu".
        norm (bool, optional): Whether to use normalization. Defaults to False.
    """

    def __init__(self, n_channels: int, has_attn: bool = False, activation: str = "gelu", norm: bool = False):
        super().__init__()
        self.res1 = ResidualBlock2D(n_channels, n_channels, activation_fn_name=activation, norm=norm)
        self.attn = AttentionBlock2D(n_channels) if has_attn else nn.Identity() #for now it is identity
        self.res2 = ResidualBlock2D(n_channels, n_channels, activation_fn_name=activation, norm=norm)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x

class Upsample2D(nn.Module):
    r"""Scale up the feature map by $2 \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=n_channels, 
                                       out_channels=n_channels, 
                                       kernel_size=(4, 4), 
                                       stride=(2, 2), 
                                       padding=(1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)

class Downsample2D(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=n_channels, 
                              out_channels=n_channels, 
                              kernel_size=(3, 3), 
                              stride=(2, 2), 
                              padding=(1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)

class Unet(nn.Module): 
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
        hidden_channels: int,
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
        self.hidden_channels = hidden_channels

        self.activation = activation_func.get_activation(activation_fn_name)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation_fn_name} not implemented")
        # Number of resolutions (depth of unet)
        n_resolutions = len(ch_mults)

        insize = in_channels*sequence_info[0][0]
        outsize = out_channels*sequence_info[0][1]
        n_channels = hidden_channels
        
        self.unet = self.getUnet()(insize, 
                                   outsize, 
                                   n_channels, 
                                   n_resolutions, 
                                   ch_mults, 
                                   is_attn, 
                                   mid_attn, 
                                   n_blocks, 
                                   activation_fn_name, 
                                   norm, 
                                   use1x1)
       
    def getUnet(self):
        """Get the appropriate upsampler based on the dimension."""
        if self.dimension == 1:
            pass
        elif self.dimension == 2:
            return Unet2D
        elif self.dimension == 3:
            pass
        else:
            raise NotImplementedError(f"Upsampler not implemented for dimension {self.dimension}")

    
    ### Main Forward function ###
    def forward(self, input_data: Tensor,
                labels: Tensor) -> Tensor:
        
        assert input_data.dim() == 5 #for 2d
        assert labels.dim() == 5
        
        #orig_shape = input_data.shape
        batch, input_seq, input_fields, *spatial = input_data.shape
        x=input_data.reshape(batch, input_seq * input_fields, *spatial)

        x = self.unet(x)

        batch, output_seq, output_fields, *spatial = labels.shape
        x = x.reshape(
            batch, output_seq, output_fields, *spatial)
        return x

#Unet based on the dimension
class Unet2D(nn.Module):
    """2D U-Net"""
    def __init__(self, 
                 insize, 
                 outsize,
                 n_channels, 
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
            self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock2D(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation_fn_name,
                        norm=norm,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample2D(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)
        
        self.middle = MiddleBlock2D(out_channels, has_attn=mid_attn, activation=activation_fn_name, norm=norm)

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
                    UpBlock2D(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation_fn_name,
                        norm=norm,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock2D(in_channels, out_channels, has_attn=is_attn[i], activation=activation_fn_name, norm=norm))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample2D(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, n_channels)
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
            if isinstance(m, Upsample2D):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)

        x = self.final(self.activation(self.norm(x)))

        return x