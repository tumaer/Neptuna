from models.model_utils import cfd_PretrainedConfig
from transformers.models.swinv2.modeling_swinv2 import (
    Swinv2Attention,
    Swinv2DropPath,
    Swinv2Intermediate,
    Swinv2Output,
    window_reverse,
    window_partition,
)
from transformers.utils import ModelOutput
from dataclasses import dataclass
import torch
from torch import nn
from typing import Optional, Tuple
import math
import collections

@dataclass
class ScOTOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class ScOTConfig(cfd_PretrainedConfig):
    """https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/configuration_swinv2.py"""

    model_type = "swinv2"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        resolution_x=224,
        resolution_y=224,
        patch_size=4,
        in_channels=3,
        out_channels=1,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        skip_connections=[True, True, True],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        residual_model="convnext",  # "convnext" or "resnet"
        use_conditioning=False,
        input_steps=4,
        output_steps=3,
        output_hidden_states=False,
        output_attentions=False,
        coord_features=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.skip_connections = skip_connections
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.use_conditioning = use_conditioning
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        # we set the hidden_size attribute in order to make Swinv2 work with VisionEncoderDecoderModel
        # this indicates the channel dimension after the last stage of the model
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1)) # actually not used but recomputed as num_features
        self.pretrained_window_sizes = (0, 0, 0, 0)
        self.out_channels = out_channels
        self.residual_model = residual_model
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.coord_features = coord_features


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, time):
        return super().forward(x)


class ConditionalLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps # small constant to avoid division by zero
        # instead of using nn.Parameter like in LayerNorm, weight and bias are learned linear functions of time (-> they vary with time)
        self.weight = nn.Linear(1, dim)
        self.bias = nn.Linear(1, dim)

    def forward(self, x, time):
        # x: [16, 1024, 48]
        # compute mean and variance of input over last dimension (like in LayerNorm)
        mean = x.mean(dim=-1, keepdim=True) # [16, 1024, 1]
        var = (x**2).mean(dim=-1, keepdim=True) - mean**2 # [16, 1024, 1]
        # Normalize input x (zero mean, unit variance)
        x = (x - mean) / (var + self.eps).sqrt()
        time = time.reshape(-1, 1).type_as(x) # [16, 1]
        weight = self.weight(time).unsqueeze(1) #[16, 1, 48]
        bias = self.bias(time).unsqueeze(1) # [16, 1, 48]
        if x.dim() == 4:
            weight = weight.unsqueeze(1)
            bias = bias.unsqueeze(1)
        return weight * x + bias


class ConvNeXtBlock(nn.Module):
    r"""Taken from: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, config, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d( # dim = 48
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        if config.use_conditioning:
            layer_norm = ConditionalLayerNorm
        else:
            layer_norm = LayerNorm
        self.norm = layer_norm(dim, eps=config.layer_norm_eps)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim # 48 -> 192
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim) # 192 -> 48
        self.weight = (#[48]
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )  # was gamma before
        self.drop_path = Swinv2DropPath(drop_path) if drop_path > 0.0 else nn.Identity() # Identity

    def forward(self, x, time):
        batch_size, sequence_length, hidden_size = x.shape # 16, 1024, 48
        #! assumes square images
        input_dim = math.floor(sequence_length**0.5) #32

        input = x # [16, 1024, 48]
        x = x.reshape(batch_size, input_dim, input_dim, hidden_size) # [16, 32, 32, 48]
        x = x.permute(0, 3, 1, 2) # [16, 48, 32, 32]
        x = self.dwconv(x) # depth-wise Conv2d -> [16, 48, 32, 32]
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C) # [16, 32, 32, 48]
        x = self.norm(x, time) # (Conditional-)LayerNorm
        x = self.pwconv1(x) # Linear 48 -> 192 (1x1 conv) # [16, 32, 32, 192]
        x = self.act(x) # GeLU
        x = self.pwconv2(x) # Linear 192 -> 48 (1x1 conv) # [16, 32, 32, 48]
        if self.weight is not None:
            x = self.weight * x
        x = x.reshape(batch_size, sequence_length, hidden_size) # [16, 1024, 48]

        x = input + self.drop_path(x) # is identity now ; drop_path: randomly drops entire residual paths during training per sample
        return x


class ResNetBlock(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        kernel_size = 3
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad) # 48 -> 48
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad)
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x, time):
        batch_size, sequence_length, hidden_size = x.shape
        #! assumes square images
        input_dim = math.floor(sequence_length**0.5) # 32

        input = x # [16, 1024, 48]
        x = x.reshape(batch_size, input_dim, input_dim, hidden_size) # [16, 32, 32, 48]
        x = x.permute(0, 3, 1, 2) # [16, 48, 32, 32]
        x = self.conv1(x) # [16, 48, 32, 32]
        x = self.bn1(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv2(x) # [16, 48, 32, 32]
        x = self.bn2(x)
        x = x.permute(0, 2, 3, 1) # [16, 32, 32, 48]
        x = x.reshape(batch_size, sequence_length, hidden_size) # [16, 1024, 48]
        x = x + input  # [16, 1024, 48]
        return x


class ScOTPatchEmbeddings(nn.Module):
    """
    This class turns `input_data` of shape `(batch_size, in_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        resolution_x, resolution_y, patch_size = config.resolution_x, config.resolution_y, config.patch_size # 128, 4
        in_channels, hidden_size = config.in_channels, config.embed_dim # 4, 48
        self.input_steps = config.input_steps
        resolution = (resolution_x, resolution_y)
        self.coord_features = config.coord_features
        
        patch_size = ( # (4, 4)
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (resolution[1] // patch_size[1]) * (
            resolution[0] // patch_size[0]
        )# 1024 = (128 / 4) * (128 / 4)
        self.resolution = resolution
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = num_patches
        self.grid_size = ( # number of patches in each dimension of the image
            resolution[0] // patch_size[0],
            resolution[1] // patch_size[1],
        ) # (32, 32) = 128 / 4

        self.projection = nn.Conv2d( 
            in_channels * self.input_steps + (2 if self.coord_features else 0), hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def maybe_pad(self, input_data, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            input_data = nn.functional.pad(input_data, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            input_data = nn.functional.pad(input_data, pad_values)
        return input_data

    def forward(
        self, input_data: Optional[torch.FloatTensor]
    ) -> Tuple[torch.Tensor, Tuple[int]]:
        _, in_channels, height, width = input_data.shape # 4, 128, 128
        if in_channels != (self.in_channels * self.input_steps) + (2 if self.coord_features else 0):
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # pad the input to be divisible by self.patch_size, if needed (zero padding)
        input_data = self.maybe_pad(input_data, height, width)
        embeddings = self.projection(input_data) # pixel values: 16, 4, 128, 128 (see drawing ipad)
        _, _, height, width = embeddings.shape # 16, 48, 32, 32
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        # flatten(2) -> 32 * 32 = 1024 -> [16, 48, 1024]
        # transpose(1, 2) -> [16, 1024, 48]

        return embeddings, output_dimensions


class ScOTEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.patch_embeddings = ScOTPatchEmbeddings(config) # use convolution to calculate patch embeddings
        num_patches = self.patch_embeddings.num_patches # 1024
        self.patch_grid = self.patch_embeddings.grid_size # (32, 32)
        self.mask_token = ( # None
            nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            if use_mask_token
            else None
        )
        if config.use_absolute_embeddings: # absolute position information into the patch embeddings (spatial structure of trajectory)
            self.position_embeddings = nn.Parameter( # zeros of shape (1: broadcast the same positional embeddings across all inputs in the batch, number of patches, dimensionality of each token)
                torch.zeros(1, num_patches, config.embed_dim)
            )
        else:
            self.position_embeddings = None
        if config.use_conditioning:
            layer_norm = ConditionalLayerNorm
        else:
            layer_norm = LayerNorm
        self.norm = layer_norm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_data: Optional[torch.FloatTensor], # [16, 4, 128, 128]
        bool_masked_pos: Optional[torch.BoolTensor] = None, # None
        time: Optional[torch.FloatTensor] = None, # [16]
    ) -> Tuple[torch.Tensor]:
        embeddings, output_dimensions = self.patch_embeddings(input_data) # patch embeddings: [16, 4, 128, 128] -> [16, 1024, 48]
        embeddings = self.norm(embeddings, time) # [16, 1024, 48]
        batch_size, seq_len, _ = embeddings.size()

        if bool_masked_pos is not None: # None # Boolean masked positions: Indicates which patches are masked (1) and which aren't (0)
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.position_embeddings is not None: # None # adds positional information to patch embeddings: spatial structure of the patch (in the trajectory) (compare with position of pixels in image)
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions # [16, 1024, 48], (32, 32)


class ScOTLayer(nn.Module):
    def __init__(
        self,
        config,
        dim, # feature dimension 48 (or according to EncodeStage [48, 96, 192, 384])
        input_resolution, # spatial size 32 (or according to EncodeStage [32, 16, 8, 4])
        num_heads, # number of attention heads 3 (or according to EncodeStage [3, 6, 12, 24]
        drop_path=0.0, # 0
        shift_size=0, # according if # of ScOTLayer even / odd: 0 or 8
        pretrained_window_size=0, #0
    ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward # 0
        self.shift_size = shift_size # [0,0]
        self.window_size = config.window_size # 16
        self.input_resolution = input_resolution # 16 (changes!)
        self.set_shift_and_window_size(input_resolution) # sets and checks shift and window size based on input resolution
        self.attention = Swinv2Attention(
            config=config,
            dim=dim, # 48 (changes!)
            num_heads=num_heads, # 3 (changes!)
            window_size=self.window_size, #16
            pretrained_window_size=(
                pretrained_window_size
                if isinstance(pretrained_window_size, collections.abc.Iterable)
                else (pretrained_window_size, pretrained_window_size)
            ),
        )
        if config.use_conditioning:
            layer_norm = ConditionalLayerNorm
        else:
            layer_norm = LayerNorm
        self.layernorm_before = layer_norm(dim, eps=config.layer_norm_eps)
        self.drop_path = Swinv2DropPath(drop_path) if drop_path > 0.0 else nn.Identity() # 0 -> Identity
        self.intermediate = Swinv2Intermediate(config, dim) # Linear, activation
        self.output = Swinv2Output(config, dim) # Linear, activation
        self.layernorm_after = layer_norm(dim, eps=config.layer_norm_eps)
        
        # Cache for attention masks
        self.attn_mask_cache = {}
        # Cache for padding calculations
        self.pad_cache = {}

    def set_shift_and_window_size(self, input_resolution):
        target_window_size = ( #(16, 16)
            self.window_size
            if isinstance(self.window_size, collections.abc.Iterable)
            else (self.window_size, self.window_size)
        )
        target_shift_size = ( # (0, 0)
            self.shift_size
            if isinstance(self.shift_size, collections.abc.Iterable)
            else (self.shift_size, self.shift_size)
        )
        window_dim_x = ( # 32
            input_resolution[0].item()
            if torch.is_tensor(input_resolution[0])
            else input_resolution[0]
        )
        window_dim_y = ( 
            input_resolution[1].item()
            if torch.is_tensor(input_resolution[1])
            else input_resolution[1]
        )
        window_dim = min(window_dim_x, window_dim_y)
        self.window_size = (
            window_dim if window_dim <= target_window_size[0] else target_window_size[0]
        )
        self.shift_size = ( # 0
            0
            if input_resolution
            <= (
                self.window_size
                if isinstance(self.window_size, collections.abc.Iterable)
                else (self.window_size, self.window_size)
            )
            else target_shift_size[0]
        )

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

    def forward(
        self,
        hidden_states: torch.Tensor, # [16, 1024, 48]
        input_dimensions: Tuple[int, int], # (32, 32)
        time: torch.Tensor, # [16]
        head_mask: Optional[torch.FloatTensor] = None, # None
        output_attentions: Optional[bool] = False,# False
        always_partition: Optional[bool] = False, # False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not always_partition: # not False
            self.set_shift_and_window_size(input_dimensions) # set shift and window size parameters
            
        height, width = input_dimensions # 32, 32
        batch_size, seq_len, channels = hidden_states.size() # 16, 48, 1024
        shortcut = hidden_states # [16, 48, 1024]

        # Reshape and pad in one step if needed
        hidden_states = hidden_states.view(batch_size, height, width, channels) # [16, 32, 32, 48]
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width) # add padding to hidden_states if needed
        _, height_pad, width_pad, _ = hidden_states.shape # 32, 32
        
        # Only apply cyclic shift if needed
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll( # [16, 32, 32, 48] -> [16, 32, 32, 48]
                hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2) # shifts: up and left for 8, dims: spatial dimensions (height and width)
            ) # roll: ((row-shift_size) mod H, (col-shift_size) mod W)
        else:
            shifted_hidden_states = hidden_states # 16, 32, 32, 48

        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size) # divide input into windows # 64, 16, 16, 48
        hidden_states_windows = hidden_states_windows.view(
            -1, self.window_size * self.window_size, channels
        )# 64, 256, 48
        
        # Get attention mask (cached when possible)
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype) # mask ensures that attention stays within shifted windows
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)
            
        attention_outputs = self.attention( # forward pass through Swinv2Attention
            hidden_states_windows, # 64, 256, 48
            attn_mask, # None
            head_mask, # None
            output_attentions=output_attentions, # False
        )
        attention_output = attention_outputs[0] # 64, 256, 48
        
        # Reconstruct feature map
        attention_windows = attention_output.view(
            -1, self.window_size, self.window_size, channels
        ) # 64, 16, 16, 48
        shifted_windows = window_reverse(
            attention_windows, self.window_size, height_pad, width_pad
        )# 16, 32, 32, 48
        
        # Reverse cyclic shift if needed
        if self.shift_size > 0:
            attention_windows = torch.roll( # shift back
                shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            attention_windows = shifted_windows
            
        # Handle padding if necessary
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded: # False
            attention_windows = attention_windows[:, :height, :width, :].contiguous()
            
        attention_windows = attention_windows.view(batch_size, height * width, channels)
        
        hidden_states = shortcut + self.drop_path(self.layernorm_before(attention_windows, time))
        
        residual = hidden_states
        layer_output = self.output(self.intermediate(hidden_states))
        layer_output = residual + self.drop_path(self.layernorm_after(layer_output, time))
        
        layer_outputs = (
            (layer_output, attention_outputs[1])
            if output_attentions
            else (layer_output,)
        )
        return layer_outputs


class ScOTPatchRecovery(nn.Module):
    """https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py"""

    def __init__(self, config):
        super().__init__()
        resolution_x, resolution_y, patch_size = config.resolution_x, config.resolution_y, config.patch_size # 128, 4
        out_channels, hidden_size = ( # 4, 48
            config.out_channels,
            config.embed_dim,
        )
        resolution = (resolution_x, resolution_y)
        patch_size = ( # 4, 4
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (resolution[0] // patch_size[0]) * ( # 1024
            resolution[1] // patch_size[1]
        )
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.resolution = resolution
        self.out_channels = out_channels
        self.output_steps = config.output_steps
        self.grid_size = ( # 32, 32
            resolution[0] // patch_size[0],
            resolution[1] // patch_size[1],
        )

        self.projection = nn.ConvTranspose2d(
            in_channels=hidden_size, # 48
            out_channels=out_channels * self.output_steps, # 4
            kernel_size=patch_size, # (4, 4)
            stride=patch_size, # (4, 4)
        )
        # the following is not done in Pangu
        self.mixup = nn.Conv2d(
            out_channels * self.output_steps, # 48
            out_channels * self.output_steps, # 48
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )

    def maybe_crop(self, input_data, height, width):
        if input_data.shape[2] > height:
            input_data = input_data[:, :, :height, :]
        if input_data.shape[3] > width:
            input_data = input_data[:, :, :, :width]
        return input_data

    def forward(self, hidden_states): # [16, 1024, 48]
        hidden_states = hidden_states.transpose(1, 2) # [16, 48, 1024]
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], hidden_states.shape[1], *self.grid_size
        ) # [16, 48, 32, 32]

        output = self.projection(hidden_states)
        output = self.maybe_crop(output, self.resolution[0], self.resolution[1]) # check if last two dimensions have the expected dim, otherwise crop
        return self.mixup(output)


class ScOTPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(
        self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = LayerNorm
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1) # False
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature

    def forward(
        self,
        input_feature: torch.Tensor,
        input_dimensions: Tuple[int, int],
        time: torch.Tensor,
    ) -> torch.Tensor:
        height, width = input_dimensions #32, 32
        # `dim` is height * width
        batch_size, dim, in_channels = input_feature.shape # 16, 1024, 48

        input_feature = input_feature.view(batch_size, height, width, in_channels) # 16, 32, 32, 48
        # pad input to be divisible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, height, width) # here: no padding needed
        # [batch_size, height/2, width/2, in_channels]
        # splitting into 4 groups: (even rows, even cols), (odd rows, even cols), (even rows, odd cols), (odd rows, odd cols)
        input_feature_0 = input_feature[:, 0::2, 0::2, :] # 16, 16, 16, 48  #0::2: starting at 0, taking every second element
        # [batch_size, height/2, width/2, in_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :] # 16, 16, 16, 48
        # [batch_size, height/2, width/2, in_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :] # 16, 16, 16, 48
        # [batch_size, height/2, width/2, in_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :] # 16, 16, 16, 48
        # [batch_size, height/2 * width/2, 4*in_channels]
        input_feature = torch.cat(
            [input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1
        ) # 16, 16, 16, 192 (48*4)
        input_feature = input_feature.view(
            batch_size, -1, 4 * in_channels # 16, 256, 192
        )  # [batch_size, height/2 * width/2, 4*C]

        input_feature = self.reduction(input_feature) # 16, 256, 96 # 4 * dim -> 2 * dim
        input_feature = self.norm(input_feature, time) # 16, 256, 96

        return input_feature


class ScOTPatchUnmerging(nn.Module):
    def __init__(
        self,
        input_resolution: Tuple[int],
        dim: int,
        norm_layer: nn.Module = LayerNorm,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution # (4, 4)
        self.dim = dim # 384
        self.upsample = nn.Linear(dim, 2 * dim, bias=False) # 384 -> 768
        self.mixup = nn.Linear(dim // 2, dim // 2, bias=False) # 192 -> 192
        self.norm = norm_layer(dim // 2) # 192

    def maybe_crop(self, input_feature, height, width):
        height_in, width_in = input_feature.shape[1], input_feature.shape[2] # 8, 8
        if height_in > height:
            input_feature = input_feature[:, :height, :, :]
        if width_in > width:
            input_feature = input_feature[:, :, :width, :]
        return input_feature

    def forward(
        self,
        input_feature: torch.Tensor, # 16, 16, 384
        output_dimensions: Tuple[int, int], # (8, 8)
        time: torch.Tensor, # 16
    ) -> torch.Tensor:
        output_height, output_width = output_dimensions # 8, 8
        batch_size, seq_len, hidden_size = input_feature.shape # 16, 16, 384
        r = output_height / output_width # r is proportion factor
        input_height = math.floor((seq_len * r) ** 0.5) # To calculate new height (double old one) -> sqrt(4 * area * proportion factor)
        input_width = math.floor((seq_len / r) ** 0.5)
        input_feature = self.upsample(input_feature) # Linear: [16, 16, 384] -> [16, 16, 768]
        input_feature = input_feature.reshape(
            batch_size, input_height, input_width, 2, 2, hidden_size // 2
        ) # [16, 4, 4, 2, 2, 192]
        input_feature = input_feature.permute(0, 1, 3, 2, 4, 5) # [16, 4, 2, 4, 2, 192
        input_feature = input_feature.reshape(
            batch_size, 2 * input_height, 2 * input_width, hidden_size // 2
        ) # [16, 8, 8, 192]

        input_feature = self.maybe_crop(input_feature, output_height, output_width) # crop in case the feature does not have the specified dim
        input_feature = input_feature.reshape(batch_size, -1, hidden_size // 2) # [16, 64, 192]

        input_feature = self.norm(input_feature, time) # LayerNorm
        return self.mixup(input_feature) # Linear: [16, 64, 192] -> [16, 64, 192]
