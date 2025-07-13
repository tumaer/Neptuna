
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from utils.model_utils import PretrainedConfig
from utils import activation_func

class UNetConfig(PretrainedConfig):
    """
    Args:
        activation_fn_name (str): Name of the activation function to use. Default is "gelu".
        norm (bool): Whether to apply normalization. Default is False.
        n_groups (int): Number of groups for group normalization. Default is 1.
        channel_multiplier (Union[Tuple[int, ...], List[int]]): Multipliers for the number of channels at each stage of the UNet. Default is (1, 2, 2, 4).
        is_attn (Union[Tuple[bool, ...], List[bool]]): Flags indicating whether attention is applied at each stage. Default is (False, False, False, False).
        mid_attn (bool): Whether to apply attention in the middle block. Default is False.
        n_blocks (int): Number of blocks in each stage of the UNet. Default is 2.
        use1x1 (bool): Whether to use 1x1 convolutions. Default is False.
        **kwargs: Additional keyword arguments passed to the parent class.
        
    """

    model_type = "UNet"
    
    def __init__(
        self,
        activation_fn_name: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
        channel_multiplier: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        use1x1: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation_fn_name = activation_fn_name
        self.norm = norm
        self.n_groups = n_groups
        self.channel_multiplier = channel_multiplier
        self.is_attn = is_attn
        self.mid_attn = mid_attn
        self.n_blocks = n_blocks
        self.use1x1 = use1x1

class ResidualBlockND(nn.Module):
    """Residual Block for 1D, 2D, or 3D data.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dim (int): Dimensionality of the data. Should be 1, 2, or 3.
        activation_fn_name (str): Name of the activation function.
        norm (bool): Whether to use GroupNorm.
        n_groups (int): Number of groups in GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        activation_fn_name: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()

        # Get activation function
        self.activation = activation_func.get_activation(activation_fn_name)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation_fn_name} not implemented")

        # Select appropriate convolution class
        if dim == 1:
            Conv = nn.Conv1d
            kernel_size = (3,)
            padding = (1,)
            shortcut_kernel = (1,)
        elif dim == 2:
            Conv = nn.Conv2d
            kernel_size = (3, 3)
            padding = (1, 1)
            shortcut_kernel = (1, 1)
        elif dim == 3:
            Conv = nn.Conv3d
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)
            shortcut_kernel = (1, 1, 1)
        else:
            raise ValueError(f"Unsupported dimension: {dim}. Must be 1, 2, or 3.")

        # Convolution layers
        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = Conv(in_channels, out_channels, kernel_size=shortcut_kernel)
        else:
            self.shortcut = nn.Identity()

        # Normalization layers
        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.conv1(self.activation(self.norm1(x)))
        h = self.conv2(self.activation(self.norm2(h)))
        return h + self.shortcut(x)

class AttentionBlockND(nn.Module):
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
        batch_size, n_channels, *spatial_dims = x.shape  # Works for 1D, 2D, or 3D
        seq_len = int(torch.prod(torch.tensor(spatial_dims)))

        # Flatten spatial dims: [B, C, D1, D2, ...] -> [B, seq_len, C]
        x = x.view(batch_size, n_channels, seq_len).permute(0, 2, 1)  # [B, seq, C]

        # Project and split qkv: [B, seq, C] -> [B, seq, H, 3*d_k]
        qkv = self.projection(x).view(batch_size, seq_len, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Each: [B, seq, H, d_k]

        # Attention: [B, I, H, D] × [B, J, H, D] → [B, I, J, H]
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        attn = attn.softmax(dim=1)  # Softmax over queries (axis=1)

        # Output: [B, I, J, H] × [B, J, H, D] → [B, I, H, D]
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        res = res.view(batch_size, seq_len, self.n_heads * self.d_k)  # [B, seq, C']
        res = self.output(res)  # [B, seq, C]
        res += x  # Residual connection

        # Reshape back to original: [B, seq, C] -> [B, C, *spatial_dims]
        res = res.permute(0, 2, 1).view(batch_size, n_channels, *spatial_dims)
        return res

    # def forward(self, x: torch.Tensor):
    #     # Get shape
    #     batch_size, n_channels, height, width = x.shape
    #     # Change `x` to shape `[batch_size, seq, n_channels]`
    #     x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
    #     # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
    #     qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
    #     # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
    #     q, k, v = torch.chunk(qkv, 3, dim=-1)
    #     # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
    #     attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
    #     # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
    #     attn = attn.softmax(dim=1)
    #     # Multiply by values
    #     res = torch.einsum("bijh,bjhd->bihd", attn, v)
    #     # Reshape to `[batch_size, seq, n_heads * d_k]`
    #     res = res.view(batch_size, -1, self.n_heads * self.d_k)
    #     # Transform to `[batch_size, seq, n_channels]`
    #     res = self.output(res)

    #     # Add skip connection
    #     res += x

    #     # Change to shape `[batch_size, in_channels, height, width]`
    #     res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
    #     return res


class DownBlockND(nn.Module):
    """Generalized Down Block combining ResidualBlock and optional AttentionBlock for 1D, 2D, or 3D inputs.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dim (int): Dimensionality (1, 2, or 3).
        has_attn (bool): Whether to include attention.
        activation (str): Name of activation function.
        norm (bool): Whether to use GroupNorm.
        n_groups (int): Number of groups in GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        self.res = ResidualBlockND(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            activation_fn_name=activation,
            norm=norm,
            n_groups=n_groups,
        )

        if has_attn:
            self.attn = AttentionBlockND(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x

class UpBlockND(nn.Module):
    """Generalized Up Block combining ResidualBlock and optional AttentionBlock for 1D, 2D, or 3D inputs.

    These are used in the second half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels (from skip connection).
        out_channels (int): Number of output channels.
        dim (int): Dimensionality (1, 2, or 3).
        has_attn (bool): Whether to include attention.
        activation (str): Activation function.
        norm (bool): Whether to use GroupNorm.
        n_groups (int): Number of groups for normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        self.res = ResidualBlockND(
            in_channels=in_channels + out_channels,
            out_channels=out_channels,
            dim=dim,
            activation_fn_name=activation,
            norm=norm,
            n_groups=n_groups,
        )
        self.attn = AttentionBlockND(out_channels) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x

class MiddleBlockND(nn.Module):
    """Middle block

    Combines a `ResidualBlockND`, `AttentionBlock`, and another `ResidualBlockND`.
    Used at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        dim (int): Dimensionality (1, 2, or 3).
        has_attn (bool): Whether to use attention block. Defaults to False.
        activation (str): Activation function to use. Defaults to "gelu".
        norm (bool): Whether to use normalization. Defaults to False.
        n_groups (int): Number of groups for normalization. Defaults to 1.
    """

    def __init__(self, 
                 n_channels: int, 
                 dim: int, 
                 has_attn: bool = False, 
                 activation: str = "gelu", 
                 norm: bool = False, 
                 n_groups: int = 1):
        super().__init__()
        self.res1 = ResidualBlockND(dim=dim, in_channels=n_channels, out_channels=n_channels, activation_fn_name=activation, norm=norm, n_groups=n_groups)
        self.attn = AttentionBlockND(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlockND(dim=dim, in_channels=n_channels, out_channels=n_channels, activation_fn_name=activation, norm=norm, n_groups=n_groups)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x

class UpsampleND(nn.Module):
    r"""Scale up the feature map by $2 \times$ along all spatial dimensions.

    Uses kernel_size=(4,...), stride=(2,...), padding=(1,...)
    for 1D, 2D, and 3D transpose convolutions.

    Args:
        n_channels (int): Number of channels in the input and output.
        dim (int): Dimensionality of the data. Should be 1, 2, or 3.
    """

    def __init__(self, n_channels: int, dim: int):
        super().__init__()

        if dim == 1:
            self.conv = nn.ConvTranspose1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=(4,),
                stride=(2,),
                padding=(1,)
            )
        elif dim == 2:
            self.conv = nn.ConvTranspose2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1)
            )
        elif dim == 3:
            self.conv = nn.ConvTranspose3d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=(4, 4, 4),
                stride=(2, 2, 2),
                padding=(1, 1, 1)
            )
        else:
            raise ValueError(f"Unsupported dimension: {dim}. Must be 1, 2, or 3.")

    def forward(self, x: torch.Tensor):
        return self.conv(x)

class DownsampleND(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$ along all spatial dimensions.

    Uses kernel_size=(3,...), stride=(2,...), padding=(1,...)
    for 1D, 2D, and 3D convolutions.

    Args:
        n_channels (int): Number of channels in the input and output.
        dim (int): Dimensionality of the data. Should be 1, 2, or 3.
    """

    def __init__(self, n_channels: int, dim: int):
        super().__init__()

        if dim == 1:
            self.conv = nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=(3,),
                stride=(2,),
                padding=(1,)
            )
        elif dim == 2:
            self.conv = nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            )
        elif dim == 3:
            self.conv = nn.Conv3d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=(1, 1, 1)
            )
        else:
            raise ValueError(f"Unsupported dimension: {dim}. Must be 1, 2, or 3.")

    def forward(self, x: torch.Tensor):
        return self.conv(x)
