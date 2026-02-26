from utils.model_utils import PretrainedConfig
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
from typing import List, Optional, Tuple
import math
import collections
from utils.model_utils import CustomNorm
from transformers.pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer

@dataclass
class ScOTOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class ScOTConfig(PretrainedConfig):
    """
    Configuration class for the SCOT model. https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/configuration_swinv2.py

    Args:
        patch_size (int): Size of the patches to be extracted. Default is 4.
        depths (List[int]): Number of layers in each stage of the model. Default is [2, 2, 6, 2].
        num_heads (List[int]): Number of attention heads in each stage. Default is [3, 6, 12, 24].
        skip_connections (List[bool]): Whether to use skip connections in each stage. Default is [True, True, True].
        window_size (int): Size of the attention window. Default is 7.
        mlp_ratio (float): Ratio of the hidden size in the MLP block. Default is 4.0.
        qkv_bias (bool): Whether to use bias in the query, key, and value projections. Default is True.
        hidden_dropout_prob (float): Dropout probability for hidden layers. Default is 0.0.
        attention_probs_dropout_prob (float): Dropout probability for attention probabilities. Default is 0.0.
        drop_path_rate (float): Drop path rate for stochastic depth. Default is 0.1.
        hidden_act (str): Activation function to use. Default is "gelu".
        use_absolute_embeddings (bool): Whether to use absolute positional embeddings. Default is False.
        initializer_range (float): Range of the initializer for weights. Default is 0.02.
        norm_layer_eps (float): Epsilon value for layer normalization. Default is 1e-5.
        residual_model (str): Type of residual model to use ("convnext" or "resnet"). Default is "convnext".
        use_conditioning (bool): Whether to use conditioning in the model. Default is False.
        **kwargs: Additional keyword arguments passed to the parent class."""

    model_type = "swinv2"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        patch_size: int = 4,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        skip_connections: List[bool] = [True, True, True],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        drop_path_rate: float = 0.1,
        hidden_act: str = "gelu",
        use_absolute_embeddings: bool = False,
        initializer_range: float = 0.02,
        residual_model: str = "convnext",  # "convnext" or "resnet"
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.patch_size = patch_size
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
        self.initializer_range = initializer_range
        # we set the hidden_size attribute in order to make Swinv2 work with VisionEncoderDecoderModel
        # this indicates the channel dimension after the last stage of the model
        self.hidden_size = int(self.latent_channels * 2 ** (len(depths) - 1)) # actually not used but recomputed as num_features
        self.pretrained_window_sizes = (0, 0, 0, 0)
        self.residual_model = residual_model
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions


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

    def __init__(self, config, input_resolution, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d( # dim = 48
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv

        self.input_resolution = input_resolution
        self.norm = CustomNorm(config=config, num_channels=dim, array_length=4, channel_at_last_position=True)

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

    def forward(self, x, **kwargs):
        batch_size, sequence_length, hidden_size = x.shape # 16, 1024, 48

        input = x # [16, 1024, 48]
        x = x.reshape(batch_size, self.input_resolution[0], self.input_resolution[1], hidden_size) # [16, 32, 32, 48]
        x = x.permute(0, 3, 1, 2) # [16, 48, 32, 32]
        x = self.dwconv(x) # depth-wise Conv2d -> [16, 48, 32, 32]
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C) # [16, 32, 32, 48]
        x = self.norm(x, **kwargs) # (Conditional-)LayerNorm
        x = self.pwconv1(x) # Linear 48 -> 192 (1x1 conv) # [16, 32, 32, 192]
        x = self.act(x) # GeLU
        x = self.pwconv2(x) # Linear 192 -> 48 (1x1 conv) # [16, 32, 32, 48]
        if self.weight is not None:
            x = self.weight * x
        x = x.reshape(batch_size, sequence_length, hidden_size) # [16, 1024, 48]

        x = input + self.drop_path(x) # is identity now ; drop_path: randomly drops entire residual paths during training per sample
        return x


class ResNetBlock(nn.Module):
    def __init__(self, config, input_resolution, dim):
        super().__init__()
        kernel_size = 3
        self.input_resolution = input_resolution
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad) # 48 -> 48
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad)
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x, **kwargs):
        batch_size, sequence_length, hidden_size = x.shape

        input = x # [16, 1024, 48]
        x = x.reshape(batch_size, self.input_resolution[0], self.input_resolution[1], hidden_size) # [16, 32, 32, 48]
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
        resolution, patch_size = config.grid_resolution, config.patch_size # 128, 4
        in_channels, hidden_size = config.in_channels, config.latent_channels # 4, 48
        self.coord_features = config.coord_features
        self.config = config
        
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
            in_channels * config.sequence_info[0] + (2 if self.coord_features else 0), hidden_size, kernel_size=patch_size, stride=patch_size
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
        if in_channels != (self.in_channels * self.config.sequence_info[0]) + (2 if self.coord_features else 0):
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
            nn.Parameter(torch.zeros(1, 1, config.latent_channels))
            if use_mask_token
            else None
        )
        if config.use_absolute_embeddings: # absolute position information into the patch embeddings (spatial structure of trajectory)
            self.position_embeddings = nn.Parameter( # zeros of shape (1: broadcast the same positional embeddings across all inputs in the batch, number of patches, dimensionality of each token)
                torch.zeros(1, num_patches, config.latent_channels)
            )
        else:
            self.position_embeddings = None


        self.norm = CustomNorm(config=config, num_channels=config.latent_channels, array_length=3, channel_at_last_position=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_data: Optional[torch.FloatTensor], # [16, 4, 128, 128]
        bool_masked_pos: Optional[torch.BoolTensor] = None, # None
        **kwargs
    ) -> Tuple[torch.Tensor]:
        embeddings, output_dimensions = self.patch_embeddings(input_data) # patch embeddings: [16, 4, 128, 128] -> [16, 1024, 48]
        embeddings = self.norm(embeddings, **kwargs) # [16, 1024, 48]
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


#################################
     # coming from RoPE for 2d    
#################################
    
# so the story goes like this:
# 1. we need to pair (divide to half) the elements in the direction of hidden size for each token. 
#    this function gives the flattened meshgrid of pairs in 2 direction: pos_x and pos_y
#    This is the extened 2d version for "pos" vector in original RoPE.

def init_t_xy(end_x: int, end_y: int, zero_center=False):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    
    return t_x, t_y

# 2. then we need omega_x and omega_y because eventually we want phi = pos_x * omega_x + pos_y * omega_y. make head_dim divisible by 4. head_dim = hidden_dim//num_heads
# "angle" a frequency vector pointing in an arbitrary 2D direction.
# If α=0 you get purely-x (axial); if α=π/2 purely-y; anything else is “diagonal / mixed”.
# considering angle you can generate head_dim / 4 omegas and then duplicate using sin/cos angle create omega_x and omega_y

def init_random_2d_freqs(head_dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    theta = theta
    mag = 1 / (theta ** (torch.arange(0, head_dim, 4)[: (head_dim // 4)].float() / head_dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0) # shape (2, num_heads, head_dim//2)
    return freqs

# 3. here we create the rotation Matrix for using complex number structure with sin and cos. cis = cos + i sin

def compute_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor):
    N = t_x.shape[0]
    # No float 16 for this range
    # with torch.cuda.amp.autocast(enabled=False):
    with torch.amp.autocast('cuda',enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y) #shape (num_head, N, head_dim//2)
        
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    # assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


#################################
    # coming from HuggingFace
#################################


class Swinv2SelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, pretrained_window_size=[0, 0]):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )
        self.pretrained_window_size = pretrained_window_size
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        # mlp to generate continuous relative position bias
        self.continuous_position_bias_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )

        t_x, t_y = init_t_xy(end_x=self.window_size[1], end_y=self.window_size[0]) # TODO: is the window a tuple?
        self.register_buffer('rope_t_x', t_x)
        self.register_buffer('rope_t_y', t_y)


        freqs = init_random_2d_freqs(
            head_dim=self.attention_head_size, num_heads=self.num_attention_heads, theta=10.0, rotate=True
        )
        self.rope_freqs = nn.Parameter(freqs, requires_grad=True)

        # get relative_coords_table
        # relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.int64).float()
        # relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.int64).float()
        # relative_coords_table = (
        #     torch.stack(meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
        #     .permute(1, 2, 0)
        #     .contiguous()
        #     .unsqueeze(0)
        # )  # [1, 2*window_height - 1, 2*window_width - 1, 2]
        # if pretrained_window_size[0] > 0:
        #     relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
        #     relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        # elif window_size > 1:
        #     relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        #     relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        # relative_coords_table *= 8  # normalize to -8, 8
        # relative_coords_table = (
        #     torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        # )
        # # set to same dtype as mlp weight
        # relative_coords_table = relative_coords_table.to(next(self.continuous_position_bias_mlp.parameters()).dtype)
        # self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # # get pair-wise relative position index for each token inside the window
        # coords_h = torch.arange(self.window_size[0])
        # coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))
        # coords_flatten = torch.flatten(coords, 1)
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # relative_coords[:, :, 0] += self.window_size[0] - 1
        # relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # relative_position_index = relative_coords.sum(-1)
        # self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        batch_size, dim, num_channels = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # apply RoPE
        
        freqs_cis = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y)
        query_layer, key_layer = apply_rotary_emb(query_layer, key_layer, freqs_cis)
   

        # cosine attention
        attention_scores = nn.functional.normalize(query_layer, dim=-1) @ nn.functional.normalize(
            key_layer, dim=-1
        ).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attention_scores = attention_scores * logit_scale
        # relative_position_bias_table = self.continuous_position_bias_mlp(self.relative_coords_table).view(
        #     -1, self.num_attention_heads
        # )
        # [window_height*window_width,window_height*window_width,num_attention_heads]
        # relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        # )
        # [num_attention_heads,window_height*window_width,window_height*window_width]
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        # attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in Swinv2Model forward() function)
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            ) + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinSelfOutput with Swin->Swinv2
class Swinv2SelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class Swinv2Attention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, pretrained_window_size=0):
        super().__init__()
        self.self = Swinv2SelfAttention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            pretrained_window_size=pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else (pretrained_window_size, pretrained_window_size),
        )
        self.output = Swinv2SelfOutput(config, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


#################################


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
        self.config = config
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
        

        self.layernorm_before = CustomNorm(config=config, num_channels=dim, array_length=3, channel_at_last_position=True)
        self.drop_path = Swinv2DropPath(drop_path) if drop_path > 0.0 else nn.Identity() # 0 -> Identity
        self.intermediate = Swinv2Intermediate(config, dim) # Linear, activation
        self.output = Swinv2Output(config, dim) # Linear, activation
        self.layernorm_after = CustomNorm(config=config, num_channels=dim, array_length=3, channel_at_last_position=True)
        
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

        # check if depth works with new window size
        if self.window_size // 2 ** (len(self.config.depths) - 1) < 1:
            raise ValueError(f"Depths ({self.config.depths}) of network too large for dataset resolution ({self.config.grid_resolution}) in combination with specified patch_size ({self.config.patch_size}). Increase / decrease either depths or patch_size.")

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
        head_mask: Optional[torch.FloatTensor] = None, # None
        output_attentions: Optional[bool] = False,# False
        always_partition: Optional[bool] = False, # False
        **kwargs
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
        
        hidden_states = shortcut + self.drop_path(self.layernorm_before(attention_windows, **kwargs))
        
        residual = hidden_states
        layer_output = self.output(self.intermediate(hidden_states))
        layer_output = residual + self.drop_path(self.layernorm_after(layer_output, **kwargs))
        
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
        resolution, patch_size = config.grid_resolution, config.patch_size # 128, 4
        out_channels, hidden_size = ( # 4, 48
            config.out_channels,
            config.latent_channels,
        )
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
        self.grid_size = ( # 32, 32
            resolution[0] // patch_size[0],
            resolution[1] // patch_size[1],
        )

        self.projection = nn.ConvTranspose2d(
            in_channels=hidden_size, # 48
            out_channels=out_channels * config.sequence_info[1], # 4
            kernel_size=patch_size, # (4, 4)
            stride=patch_size, # (4, 4)
        )
        # the following is not done in Pangu
        self.mixup = nn.Conv2d(
            out_channels * config.sequence_info[1], # 48
            out_channels * config.sequence_info[1], # 48
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
        self, config, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = CustomNorm
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(config=config, num_channels=2 * dim, array_length=3, channel_at_last_position=True)

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
        **kwargs
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
        input_feature = self.norm(input_feature, **kwargs) # 16, 256, 96

        return input_feature


class ScOTPatchUnmerging(nn.Module):
    def __init__(
        self,
        config,
        input_resolution: Tuple[int],
        dim: int,
        norm_layer: nn.Module = CustomNorm,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution # (4, 4)
        self.dim = dim # 384
        self.upsample = nn.Linear(dim, 2 * dim, bias=False) # 384 -> 768
        self.mixup = nn.Linear(dim // 2, dim // 2, bias=False) # 192 -> 192
        self.norm = norm_layer(config=config, num_channels=dim // 2, array_length=3, channel_at_last_position=True) # 192

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
        **kwargs
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

        input_feature = self.norm(input_feature, **kwargs) # LayerNorm
        return self.mixup(input_feature) # Linear: [16, 64, 192] -> [16, 64, 192]
