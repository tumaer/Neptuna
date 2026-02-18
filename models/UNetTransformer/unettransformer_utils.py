
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from utils.grid_utils import twod_meshgrid
from utils.model_utils import PretrainedConfig
from utils import activation_func
from utils.model_utils import CustomNorm
from transformers.models.swinv2.modeling_swinv2 import (
    Swinv2Attention,
    window_reverse,
    window_partition,
)
import itertools
import matplotlib.pyplot as plt

class UNetTransformerConfig(PretrainedConfig):
    """
    Args:
        activation_fn_name (str): Name of the activation function to use. Default is "gelu".
        channel_multiplier (Union[Tuple[int, ...], List[int]]): Multipliers for the number of channels at each stage of the UNet. Default is (1, 2, 2, 4).
        n_blocks (int): Number of blocks in each stage of the UNet. Default is 2.
        norm: "group" or "layer" or "batch" or "none" (default: "layer")
        norm_layer_eps: Used to avoid zero-div error. float (default: 1e-5)
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    model_type = "UNetTransformer"
    
    def __init__(
        self,
        activation_fn_name: str = "gelu",
        channel_multiplier: Union[Tuple[int, ...], List[int]] = (1, 2),
        n_blocks: int = 1,
        norm: str = "layer",
        norm_layer_eps: float = 1e-5,
        num_grids: int = 2,
        downsample_method: str = "average",
        window_size: int = 16,
        num_heads: int = 8,
        attention_type: str = "swin",
        attention_concat_all: bool = False,
        attention_concat_type: str = 'se',
        use_dca: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation_fn_name = activation_fn_name
        self.channel_multiplier = channel_multiplier
        self.n_blocks = n_blocks
        self.norm = norm
        self.norm_layer_eps = norm_layer_eps
        if num_grids != len(channel_multiplier):
            raise ValueError(f"num_grids ({num_grids}) must be equal to the length of channel_multiplier:({len(channel_multiplier)}).")
        self.num_grids = num_grids
        self.downsample_method = downsample_method
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = True
        self.attention_probs_dropout_prob = 0.0
        self.attention_type = attention_type
        if attention_concat_all and attention_concat_type is None:
            raise ValueError("attention_concat_type must be specified if attention_concat_all is True.")
        self.attention_concat_all = attention_concat_all
        self.attention_concat_type = attention_concat_type
        self.use_dca = use_dca

class ResidualBlockND(nn.Module):
    """Residual Block for 1D, 2D, or 3D data.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dim (int): Dimensionality of the data. Should be 1, 2, or 3.
        activation (str): Name of the activation function.
        norm_layer: "group" or "layer" or "batch" (default: "layer")
    """

    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        dim: int,
        shift: bool = False,
        activation: str = "gelu",
        input_resolution: Tuple[int, int] = [1, 1],
    ):
        super().__init__()

        # Get activation function
        self.activation = activation_func.get_activation(activation)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        if dim == 2:
            Conv = nn.Conv2d
            kernel_size = (3, 3)
            padding = (1, 1)
            shortcut_kernel = (1, 1)
        else:
            raise ValueError(f"Unsupported dimension: {dim}. Must be 2.")
        
        self.attention_type = config.attention_type

        # Convolution layers
        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        if config.attention_type is not None:
            if config.attention_type == "swin":
                self.attention = SwinAttention(config, out_channels, shift, input_resolution)
            elif config.attention_type == "local_window": # Only works on square input !!!
                self.attention = LocalWindowAttention(out_channels, out_channels // config.num_heads, num_heads=config.num_heads,
                    attn_ratio=1,
                    resolution=input_resolution[0],
                    window_resolution=config.window_size,
                    kernels=config.num_heads * [5])
            elif config.attention_type == "linattention":
                self.attention = LinformerAttention(out_channels, input_resolution[0], input_resolution[1], proj_dim=16) 
            elif config.attention_type == "sea":
                self.attention = Sea_Attention(out_channels, 6, config.num_heads)
            else:
                raise NotImplementedError(f"Attention type {config.attention_type} not implemented.")
            self.conv2 = Conv(out_channels, out_channels, kernel_size=(1, 1), padding=(0, 0))
        else:
            self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.shift = shift
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = Conv(in_channels, out_channels, kernel_size=shortcut_kernel)
        else:
            self.shortcut = nn.Identity()

        self.norm1 = CustomNorm(config=config, num_channels=in_channels, array_length=dim+2, channel_at_last_position=False)
        self.norm2 = CustomNorm(config=config, num_channels=out_channels, array_length=dim+2, channel_at_last_position=False)

    def forward(self, x: torch.Tensor, **kwargs):
        h = self.norm1(x, **kwargs)
        h = self.activation(h)
        h = self.conv1(h)
    
        if self.attention_type is not None:
            h = h + self.shortcut(x)
            y = h
            h = self.norm2(h, **kwargs)
            h = self.attention(h)
            h = y + self.conv2(h)
            return h
        else:
            h = self.norm2(h, **kwargs)
            h = self.activation(h)
            h = self.conv2(h)
            return h + self.shortcut(x)


class MiddleBlockND(nn.Module):
    """Middle block

    Combines a `ResidualBlockND`, and another `ResidualBlockND`.
    Used at the lowest resolution of the U-Net.

    Args:
        config (UNetConfig): Configuration for the UNet.
        n_channels (int): Number of channels in the input and output.
        dim (int): Dimensionality (2).
        activation (str): Activation function to use. Defaults to "gelu".
    """

    def __init__(self, 
                 config,
                 n_channels: int, 
                 dim: int, 
                 activation: str = "gelu",
                 input_resolution: Tuple[int, int] = None,
                 ):
        super().__init__()
        self.res1 = ResidualBlockND(config=config, dim=dim, in_channels=n_channels, out_channels=n_channels, activation=activation, shift=False, input_resolution=input_resolution) 
        self.res2 = ResidualBlockND(config=config, dim=dim, in_channels=n_channels, out_channels=n_channels, activation=activation, shift=True, input_resolution=input_resolution) 

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.res1(x, **kwargs)
        x = self.res2(x, **kwargs)
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

        if dim == 2:
            self.conv = nn.ConvTranspose2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1)
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

        if dim == 2:
            self.conv = nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            )
        else:
            raise ValueError(f"Unsupported dimension: {dim}. Must be 1, 2, or 3.")

    def forward(self, x: torch.Tensor, **kwargs):
        return self.conv(x)

import torch
import torch.nn as nn


class SEAttention(nn.Module):
    """
    SE as global channel self-attention. Single-head version
    Channels are treated as tokens.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_dim = max(channels // reduction, 8)

        # Token mixer (equivalent to FFN in transformers)
        self.token_mixer = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, channels, bias=False)
        )

        self.gate = nn.Sigmoid()

    def forward(self, x, **kwargs):
        """
        x: [B, C, H, W]
        """
        b, c, h, w = x.shape

        # Step 1: Tokenization (global pooling)
        tokens = x.mean(dim=(2, 3))  # [B, C]

        # Step 2: Global channel self-attention (dense mixing)
        attn = self.token_mixer(tokens)  # [B, C]

        # Step 3: Gating
        attn = self.gate(attn).view(b, c, 1, 1)

        # Step 4: Apply attention
        return x * attn



class MultiHeadSEAttention(nn.Module):
    """
    Multi-head SE transformer-style channel attention
    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(self, channels, num_heads, reduction=16):
        super().__init__()

        self.hidden_dim = max(channels // reduction, 8)

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Shared token projection
        self.fc1 = nn.Linear(channels, self.hidden_dim, bias=False)

        # Per-head channel mixing
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.head_dim, bias=False)
            )
            for _ in range(num_heads)
        ])

        # Output projection
        self.fc_out = nn.Linear(channels, channels, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x, **kwargs):
        """
        x: [B, C, H, W]
        """
        b, c, _, _ = x.shape

        # Tokenization: global pooling
        tokens = x.mean(dim=(2, 3))  # [B, C]

        # Shared embedding
        shared = self.fc1(tokens)    # [B, hidden_dim]

        # Multi-head channel attention
        head_outputs = [
            head(shared) for head in self.heads
        ]                             # list of [B, head_dim]

        attn = torch.cat(head_outputs, dim=1)  # [B, C]

        # Final projection + gating
        attn = self.fc_out(attn)
        attn = self.gate(attn).view(b, c, 1, 1)

        return x * attn


class ECAttention(nn.Module):
    """Constructs a ECA module.
    Taken from ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    Qilong Wang et. al
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1  # k_size must be odd
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, **kwargs):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
    

class MultiHeadECA(nn.Module):
    """
    Multi-Head Efficient Channel Attention (ECA)
    Adapted from ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    Qilong Wang et. al
    Args:
        channel (int): Number of input channels
        num_heads (int): Number of attention heads
        k_size (int): Kernel size for 1D convolution
    """

    def __init__(self, channels, num_heads=8, gamma=2, b=1):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1  # k_size must be odd

        # One ECA conv per head (each sees all channels)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=k_size,
                padding=k_size // 2,
                bias=False
            )
            for _ in range(num_heads)
        ])

        # Fuse heads (concat → projection)
        self.proj = nn.Conv1d(
            in_channels=num_heads,
            out_channels=1,
            kernel_size=1,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, **kwargs):
        """
        x: [B, C, H, W]
        """

        # Global average pooling
        y = self.avg_pool(x)                    # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)    # [B, 1, C]

        # Per-head channel attention
        head_outputs = []
        for conv in self.convs:
            h = conv(y)                        # [B, 1, C]
            head_outputs.append(h)

        # Concatenate heads
        y = torch.cat(head_outputs, dim=1)     # [B, num_heads, C]

        # Project back to single attention map
        y = self.proj(y)                       # [B, 1, C]

        # Restore shape
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]

        # Channel gating
        y = self.sigmoid(y)

        return x * y.expand_as(x)
    

class ConservativeDownsampling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_grids = config.num_grids

    def forward(self, input_data, **kwargs):
        if self.config.coord_features:
            coord_feat = twod_meshgrid(list(input_data.shape), input_data.device)
            x = torch.cat((input_data, coord_feat), dim=1) # [8, 3+2, 256/2^i, 64/2^i]
        inputs = [x]
        for i in range(1, self.num_grids):
            if f"grid_{i}" in kwargs:
                B, T, C, H, W = kwargs[f"grid_{i}"].shape
                if T != 1:
                    raise ValueError("Conservative downsampling only supports single-frame inputs for now.")
                grid = kwargs[f"grid_{i}"].reshape(B, C, H, W)
                if self.config.coord_features:
                    coord_feat = twod_meshgrid(list(grid.shape), grid.device)
                    x = torch.cat((grid, coord_feat), dim=1) # [8, 3+2, 256/2^i, 64/2^i]
                inputs.append(x)
            else:
                raise ValueError(f"Grid {i} not found in kwargs for conservative downsampling.")

        """for j, inp in enumerate(inputs):   
            for i in range(inputs[0][0].shape[0]):
                fig, ax = plt.subplots()
                arr = inp[0][i]

                im = ax.imshow(arr)
                fig.colorbar(im, ax=ax)

                fig.savefig(f"{j}_{i}.png", bbox_inches="tight")
                plt.close(fig)"""
        return inputs
    

class SwinAttention(nn.Module):
    ### Model taken from Poseidon: Efficient Foundation Models for PDEs. 
    # Github: https://github.com/camlab-ethz/poseidon?tab=readme-ov-file 
    # Paper: https://arxiv.org/abs/2405.19101
    def __init__(self, config, latent_dim, shift, input_resolution):
        super().__init__()
        self.config = config

        if not shift or input_resolution[0] <= config.window_size or input_resolution[1] <= config.window_size:
            self.shift_size = 0
        else:
            self.shift_size = config.window_size // 2

        self.attention = Swinv2Attention(
            config=config,
            dim=latent_dim,
            num_heads=config.num_heads,
            window_size=config.window_size,
        )
        self.window_size = config.window_size

        # Cache for attention masks
        self.attn_mask_cache = {}
        # Cache for padding calculations
        self.pad_cache = {}

    def get_attn_mask(self, height, width, dtype): # creates a spatial maks indicating window partitioning
        # Use cached attention mask when possible
        cache_key = (height, width, self.shift_size, self.window_size, dtype) # 32, 32, 0, 16
        if cache_key in self.attn_mask_cache: # {}
            return self.attn_mask_cache[cache_key]


        if self.shift_size > 0:
            # calculate attention mask for shifted window multihead self attention # maks out attention across different windows
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype) # 1, 32, 32, 1
            height_slices = ( # partition feature map into different regions
                slice(0, -self.window_size), # everything apart from lower window size
                slice(-self.window_size, -self.shift_size), # 8 between lower 8 and rest
                slice(-self.shift_size, None), # lower 8
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0 # label each sub-region of feature map with unique integer
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size) # 4, 16, 16, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # 4, 256
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # 4, 256, 256 # for each each window, compute pairwise difference between region labels
            # if two token same label -> 0 -> attend to each other # else (different labels) -> non 0 -> not attend
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0)) # keep 0, convert non-0 to large negative number to kill attention in softmax
        else:
            attn_mask = None
            
        # Cache the result
        self.attn_mask_cache[cache_key] = attn_mask
        return attn_mask
    
    def maybe_pad(self, hidden_states, height, width):
        # Use cached padding calculations when possible
        cache_key = (height, width, self.window_size) # 32, 32, 16
        if cache_key in self.pad_cache: # {}
            pad_values = self.pad_cache[cache_key]
            if pad_values[3] > 0 or pad_values[5] > 0:
                hidden_states = nn.functional.pad(hidden_states, pad_values)
            return hidden_states, pad_values

        # compute how much padding is needed
        pad_right = (self.window_size - width % self.window_size) % self.window_size # 0
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size # 0
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        
        # Cache the pad values
        self.pad_cache[cache_key] = pad_values
        
        if pad_right > 0 or pad_bottom > 0:
            hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values


    def forward(self, input, **kwargs):
        _, channels, height, width = input.shape
        input = torch.permute(input, (0, 2, 3, 1))
        input, _ = self.maybe_pad(input, height, width) # add padding to hidden_states if needed
        _, height_pad, width_pad, _ = input.shape 

        # Only apply cyclic shift if needed
        if self.shift_size > 0:
            shifted_input_data = torch.roll( # [16, 32, 32, 48] -> [16, 32, 32, 48]
                input, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2) # shifts: up and left for 8, dims: spatial dimensions (height and width)
            ) # roll: ((row-shift_size) mod H, (col-shift_size) mod W)
        else:
            shifted_input_data = input

        input_data_windows = window_partition(shifted_input_data, self.window_size) # divide input into windows
        input_data_windows = input_data_windows.view(
            -1, self.window_size * self.window_size, channels
        )# 512, 16, 16

        # Get attention mask (cached when possible)
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=input.dtype) # mask ensures that attention stays within shifted windows
        if attn_mask is not None:
            attn_mask = attn_mask.to(input_data_windows.device)
            
        attention_outputs = self.attention( # forward pass through Swinv2Attention
            input_data_windows, 
            attention_mask=attn_mask
        )
        attention_output = attention_outputs[0]

        # Reconstruct feature map
        attention_windows = attention_output.view(
            -1, self.window_size, self.window_size, channels
        )
        shifted_windows = window_reverse(
            attention_windows, self.window_size, height_pad, width_pad
        )
        
        # Reverse cyclic shift if needed
        if self.shift_size > 0:
            attention_windows = torch.roll( # shift back
                shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            attention_windows = shifted_windows

        attention = torch.permute(attention_windows, (0, 3, 1, 2))

        return attention


class LinformerAttention(nn.Module):
    def __init__(self, channels, width, height, proj_dim):
        """
        Taken from Linformer: Self-Attention with Linear Complexity (2020)
        Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma

        channels: number of input channels (C)
        width, height: spatial resolution (W, H)
        proj_dim: Linformer projection dimension (k)
        """
        super().__init__()

        self.channels = channels
        self.width = width
        self.height = height
        self.seq_len = width * height
        self.proj_dim = proj_dim

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)

        # Learned spatial projection matrices
        self.E_k = nn.Parameter(torch.randn(self.seq_len, proj_dim))
        self.E_v = nn.Parameter(torch.randn(self.seq_len, proj_dim))

        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x):
        """
        x: (B, C, W, H)
        """
        B, C, W, H = x.shape
        assert W == self.width and H == self.height

        # Flatten spatial dimensions
        x = x.view(B, C, W * H).transpose(1, 2)  # (B, N, C)

        # Q, K, V projections
        Q = self.q_proj(x)  # (B, N, C)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Project keys and values along spatial dimension
        K_proj = torch.einsum("bnc,nk->bkc", K, self.E_k)
        V_proj = torch.einsum("bnc,nk->bkc", V, self.E_v)

        # Attention
        attn_scores = torch.matmul(Q, K_proj.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(C)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, V_proj)
        out = self.out_proj(out)

        # Restore spatial layout
        out = out.transpose(1, 2).view(B, C, W, H)

        return out
    
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)


    
class CascadedGroupAttention(torch.nn.Module):
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim, resolution=resolution))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0: # add the previous output to the input
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1) # B, C/h, H, W
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # B, C/h, N
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale
                +
                (trainingab[i] if self.training else self.ab[i])
            )
            attn = attn.softmax(dim=-1) # BNN
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W) # BCHW
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x


class LocalWindowAttention(torch.nn.Module):
    r""" Local Window Attention. Taken from https://github.com/microsoft/Cream/blob/main/EfficientViT/classification/model/efficientvit.py

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution
        
        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                attn_ratio=attn_ratio, 
                                resolution=window_resolution,
                                kernels=kernels,)

    def forward(self, x):
        H = W = self.resolution
        B, C, H_, W_ = x.shape
        # Only check this for classifcation models
        assert H == H_ and W == W_, 'input feature has wrong size, expect {}, got {}'.format((H, W), (H_, W_))
               
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                       C).transpose(2, 3).reshape(B, pH, pW, C)
            if padding:
                x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)
        return x
    

class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()
        
        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]))

    def forward(self, x):
        B, C, N = x.shape
        x = x + torch.nn.functional.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        
        return x
    
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
    
class Sea_Attention(torch.nn.Module):
    "Taken from: Wan, Q., Huang, Z., Lu, J., Yu, G. and Zhang, L., 2025. "
    "SeaFormer++: Squeeze-enhanced axial transformer for mobile visual recognition. "
    "International Journal of Computer Vision, 133(6), pp.3645-3666."
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=2,
                 activation=nn.GELU):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1)
        self.to_k = Conv2d_BN(dim, nh_kd, 1)
        self.to_v = Conv2d_BN(dim, self.dh, 1)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0))
        self.proj_encode_row = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.proj_encode_column = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        
        self.dwconv = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh)
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1)
        self.sigmoid = h_sigmoid()

    def forward(self, x):  
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # detail enhance
        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        # squeeze axial attention
        ## squeeze row
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        ## squeeze column
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)
        
        xx = self.sigmoid(xx) * qkv
        return xx
    