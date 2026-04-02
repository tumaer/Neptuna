"""
Sample Model for Attention-Koopman (AK) on time-dependent 2D Fluid Data. 
The whole outline is to enncode the batch of [T, X, Y] data into latent space by 3D attention
and then evolve the spatial representation by an approximate Koopman operator/Koopman-inspired operator 
which is conditioned on the attenstioned data along the time dimension

The order of Modules:

Encoder > Propagator > Decoder
1. Encoder & Decoder: based on ScOT3D
2. Propagator: based on KNO or DeepKoopmanTransform (?)

This file includes only Encoder
"""

import math
import torch
from torch import nn
from utils.model_utils import PretrainedConfig
from typing import List, Optional, Tuple
from transformers.pytorch_utils import meshgrid
from transformers.models.swinv2.modeling_swinv2 import (
    Swinv2DropPath,
    Swinv2Intermediate,
    Swinv2Output,
    Swinv2EncoderOutput

)


class AK_Config(PretrainedConfig):

    model_type = 'ak'
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
            self,
            patch_size=(2, 4, 4),
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            skip_connections=[True, True, True, True],
            window_size=(2, 7, 7),
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            # use_absolute_embeddings=False, # Depracting 
            initializer_range=0.02, # a default to initialize the model weights
            layer_norm_eps=1e-5,
            p=1,  # for loss: 1 for l1, 2 for l2
            channel_slice_list_normalized_loss=None,  # if None will fall back to absolute loss otherwise normalized loss with split channels
                                                      # TODO: how channel loss weighting take place currently?
            residual_model="resnet",  # "convnext" or "resnet"; Currently only ResNet
            learn_residual=False,  # learn the residual for time-dependent problems TODO: have I used it?
            output_hidden_states: bool = False,
            output_attentions: bool = False,
            **kwargs,
            ):
        

        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.depths = depths
        self.num_heads = num_heads
        self.skip_connections = skip_connections
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale  
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # self.use_absolute_embeddings = use_absolute_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.p = p
        self.channel_slice_list_normalized_loss = channel_slice_list_normalized_loss
        self.residual_model = residual_model
        self.learn_residual = learn_residual
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions


class AK_Embeddings(nn.Module):
    """
    This class lifts `input_data` of shape `(batch_size, in_channels, time_frames, height, width) [B, C_in, T, H, W]` 
    into the initial `hidden_states` of shape `(batch_size, hidden_size, time_frames, patched_height, patched_width)`
    to a higher dimension to be consumed by a Transformer. 
    TODO: How [B, C_in, T, X, Y] is used then?
    TODO: how mask_token is useful? 
            self.mask_token = ( # None
            nn.Parameter(torch.zeros(1, 1, config.latent_channels))
            if use_mask_token
            else None
        )
    TODO: check dropout addition
    """


    def __init__(self, config) -> None:
        super().__init__()


        self.config = config
        self.patch_size = config.patch_size 
        self.in_channels, self.hidden_size = config.in_channels, config.latent_channels # by latent I mean embed dim here, 
                                                                              # these two are coming from data_config and  model_config respectively. 
                                                                              # TODO: sanity check these two
        
        self.coord_features = config.coord_features # coming from data_config, not a learned feature like positional embedding, 
                                                    # but a fixed feature that is concatenated to the input data to give the model a sense of coordinates in the trajectory.
        

        seq_in = config.sequence_info[0] # coming from data_config

        self.resolution = [seq_in, config.grid_resolution[0], config.grid_resolution[1]]

        # Sanity check for 3D patch size
        if len(self.patch_size) != 3:
            raise ValueError(f'Patch size must be a tuple of length 3 for 3D inputs. Got {self.patch_size}')
        
        
        self.num_patches = (self.resolution[0] // self.patch_size[0]) * (self.resolution[1] // self.patch_size[1]) * (self.resolution[2] // self.patch_size[2])
        # not presice, only for init is used!


        self.grid_size = ( # number of patches in each dimension of the input
            self.resolution[0] // self.patch_size[0],
            self.resolution[1] // self.patch_size[1],
            self.resolution[2] // self.patch_size[2]
        ) 

        # input with shape (batch_size, in_channels, time_frames, height, width) to (batch_size, hidden_size, time_frames_prime, height_prime, width_prime) 
        # based on Pytorch Conv3D https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html

        self.lift = nn.Conv3d(
            in_channels=self.in_channels + (2 if self.coord_features else 0), 
            out_channels=self.hidden_size, 
            kernel_size=self.patch_size, 
            stride=self.patch_size)
        
        self.norm = nn.LayerNorm(self.hidden_size)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def maybe_pad(self, input_data, T, H, W):
        #  B, C, T, H, W = input_data.shape
        pad_Tr = T % self.patch_size[0]
        pad_Hr = H % self.patch_size[1]
        pad_Wr = W % self.patch_size[2] 
        pad_Tl = pad_Hl = pad_Wl = 0
        pad_values = (pad_Wl, pad_Wr, pad_Hl, pad_Hr, pad_Tl, pad_Tr) 
        # (pad_last_dim_left, pad_last_dim_right, pad_2nd_last_left, pad_2nd_last_right, ...)

        if any(pad != 0 for pad in pad_values):
            input_data = nn.functional.pad(input_data, pad_values)

        return input_data


    def forward(self, input_data):

        # B, T, C, H, W >> B, C, T, H, W ; > change the the shape from Trainer dataloader so later it can be liftable by Conv3d
        input_data = input_data.permute(0, 2, 1, 3, 4) 
        _, in_channels, T,  H, W = input_data.shape 


        if in_channels != self.in_channels + (2 if self.coord_features else 0):
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        # pad the input to be divisible by self.patch_size, if needed (zero padding)
        input_data = self.maybe_pad(input_data, T, H, W)

        embeddings = self.lift(input_data)    

        _, _, T_patch, H_patch, W_patch = embeddings.shape

        embeddings_dimensions = (T_patch, H_patch, W_patch)

        embeddings = embeddings.flatten(2).transpose(1, 2) # Batch, Num_patches, Hidden_size >> prepare for norm

        embeddings = self.norm(embeddings) # Batch, Num_patches, Hidden_size

        # embeddings = self.dropout(embeddings) # Batch, Num_patches, Hidden_size

        return embeddings, embeddings_dimensions


class AK_Merging(nn.Module):
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
    
    def __init__(self, 
             
                dim: int, 
                norm_layer: nn.Module = nn.LayerNorm
                ):
        super().__init__()


        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        
    def maybe_pad(self, input_feature, H, W): # we won't pad the T dim. It is only patched in the Embedding layer. 
        should_pad = (H % 2 == 1) or (W % 2 == 1) # False
        if should_pad:
            pad_values = (0, 0, 0, W % 2, 0, H % 2, 0, 0)
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature
    
    def forward(
        self,
        input_feature: torch.Tensor,
        input_dimensions: Tuple[int, int],
        **kwargs
    ) -> torch.Tensor:
        """
        Forward function.
        Input feature, tensor size (B, T, H, W, C).
        """


        T, H, W = input_dimensions 
        # `dim` is height * width
        # batch_size, in_channels, T, H, W = input_feature.shape 
        batch_size, L, in_channels = input_feature.shape 

        assert L == T * H * W, f"L={L} != T*H*W={T*H*W}"

        input_feature = input_feature.reshape(batch_size, T, H, W, in_channels) 
        # pad input to be divisible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, H, W) # here: no padding needed
        # [batch_size, T , H/2, W/2, in_channels] T is already patched.
        # splitting into 4 groups: (even rows, even cols), (odd rows, even cols), (even rows, odd cols), (odd rows, odd cols)
        input_feature_0 = input_feature[:, :, 0::2, 0::2, :]   #0::2: starting at 0, taking every second element
        # [batch_size, T, H/2, W/2, in_channels]
        input_feature_1 = input_feature[:, :, 1::2, 0::2, :] 
        # [batch_size, T, H/2, W/2, in_channels]
        input_feature_2 = input_feature[:, :, 0::2, 1::2, :] 
        # [batch_size, T, H/2, W/2, in_channels]
        input_feature_3 = input_feature[:, :, 1::2, 1::2, :]
        # [batch_size, T, H/2, W/2, 4*in_channels]
        input_feature = torch.cat(
            [input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1
        )
    

        input_feature = input_feature.reshape(
            batch_size, -1, 4 * in_channels 
        )  # [batch_size, T * H/2 * W/2, 4*C]


        # NOTE: It seems first norm and then reduction is more stable for Merging!
        input_feature = self.norm(input_feature) 
        input_feature = self.reduction(input_feature) #  4 * dim -> 2 * dim

        # NOTE: input feature shape leaving downsampling is (batch_size, T * H/2 * W/2, 2*in_channels)

        return input_feature


class AK_Attention3D(nn.Module):

    """
    Attention Module including time as the third dimension.
    So by 3d window partition and creating a 3d meshgrid and a 3d mask, we enforce 3d attention. Everything else stays same. 
    TODO: Extend to 4D to account fro Channel dimension as well
    TODO: Add prune heads functionality
    """

    def __init__ (self,
                  config,
                  dim,
                  input_resolution: Tuple[int],
                  num_heads,
                  shift_size: Tuple[int],
                  window_size: Tuple[int]
                  ):
        super().__init__()

        self.config = config
        self.dim = dim
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.window_size = window_size

        #### part1 similar to Class ScOTSelfAttenstion3D from HuggingFace
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )
        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.attention_head_size * num_heads # extra!

        self.window_size = config.window_size # (Wd,WH,WW) TODO: sanity check addition

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1)))) # taken from SwinV2 in Huggingface

        # positional bias: (based on SwinV2)
        # 1. all possible normalized and log-transformed* relative positions for 3D window
        # 2. An MLP to generate continuous relative position bias
        # 3. create index tensor
        # 4. "during forward pass", get the relative position bias for each head using the index tensor

        # 1. 
        # 3Dwindow size: (Wt, Wh, Ww)
        relative_coords_t = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.int64).float()
        relative_coords_h = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.int64).float()
        relative_coords_w = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.int64).float()
        relative_coords_table = (
            torch.stack(meshgrid([relative_coords_t, relative_coords_h, relative_coords_w], indexing="ij"))
            .permute(1, 2, 3, 0)
            .contiguous()
            .unsqueeze(0)
        )  # shape: [1, 2*Wt - 1, 2*Wh - 1, 2*Ww - 1, 3]


        # normalize to -1, 1
        relative_coords_table[..., 0] /= (self.window_size[0] - 1)
        relative_coords_table[..., 1] /= (self.window_size[1] - 1)
        relative_coords_table[..., 2] /= (self.window_size[2] - 1)


        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        ) 
        # This log-scaled transform compresses large relative distances while preserving fine detail for nearby tokens, 
        # enabling Swin v2 to learn stable, resolution-agnostic relative position bias.


        # 2.
        self.continuous_position_bias_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )

        # set to same dtype as mlp weight
        relative_coords_table = relative_coords_table.to(next(self.continuous_position_bias_mlp.parameters()).dtype)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)


        # 3. get pair-wise relative position index for each token inside the window
        # 3.A: create absolute coordinate for each token inside the window
        coords_t = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(meshgrid([coords_t, coords_h, coords_w], indexing="ij")) # shape: [3, Wt, Wh, Ww]

        # 3.B: compuet the relative coordinate
        coords_flatten = torch.flatten(coords, 1) # shape: [3, Wt*Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # shape: [3, Wt*Wh*Ww, Wt*Wh*Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # shape: [Wt*Wh*Ww, Wt*Wh*Ww, 3]

        # 3.C: shift to start from 0 for indexing
        relative_coords[:, :, 0] += self.window_size[0] - 1  
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        # 3.D: combine to a single index (from 3D to 1D - example: 25 elements in 2D*)
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)


        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)




        #### part2 similar to Class ScOTSelfOutput3D from HuggingFace
        self.dense = nn.Linear(dim, dim)
        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        
        """
        input features with shape of (num_windows*B, N, C)
        """
        
        B_, N, C = hidden_states.shape # B_: num_windows * batch_size, N: Wt*Wh*Ww, C: hidden size (dim)

        device = hidden_states.device
        dtype  = hidden_states.dtype

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        # shapes: (B_, num_heads, N, head_size)


        # cosine attention: Q · K^T 
        attention_scores = \
        nn.functional.normalize(query_layer, dim=-1) @ \
        nn.functional.normalize(key_layer, dim=-1).transpose(-2, -1).contiguous() # shape: (B_, num_heads, N, N) 

        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp() # shape: (num_heads, 1, 1)

        # scale the attention scores. Instead of dividing by sqrt(d), we multiply by a learnable logit_scale parameter
        attention_scores = attention_scores * logit_scale # shape: (B_, num_heads, N, N)


        # step 4 from Init
        relative_coords_table = self.relative_coords_table.to(device=device)
        relative_position_index = self.relative_position_index.to(device=device)

        relative_position_bias_table = self.continuous_position_bias_mlp(relative_coords_table).view(
            -1, self.num_attention_heads
        ) # shape: [(2*Wt-1)*(2*Wh-1)*(2*Ww-1), num_heads]

        
        relative_position_bias = relative_position_bias_table[relative_position_index.view(-1)].view(
            N, N, -1
        ) # shape: [Wt*Wh*Ww, Wt*Wh*Ww, num_heads]

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        # shape: [num_heads, Wt*Wh*Ww, Wt*Wh*Ww]

        # final attention score with relative position bias, shape: (B_, num_heads, N, N)
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)



        if attention_mask is not None: # TODO: check
            # Apply the attention mask is (precomputed for all layers in Swinv2Model forward() function)
            attention_mask = attention_mask.to(device=device, dtype=dtype)
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                B_ // mask_shape, mask_shape, self.num_attention_heads, N, N
            ) 
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, N, N)

        #  Normalize the attention scores to probabilities. (the meaningful part), shape: (B_, num_heads, N, N)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # Apply dropout , shape: (B_, num_heads, N, N)
        attention_probs = self.dropout(attention_probs)

        # Attention @ Value > weighted sum of the values based on the attention probabilities
        output = (attention_probs @ value_layer).transpose(1, 2).reshape(B_, N, C) # shape: (B_, N, C)

        # OutputClass
        output = self.dense(output)
        output = self.dropout(output)

        outputs =  (output,)

        return outputs


def window_partition_3d(input, window_size): # normally gets imported from Huggingface SwinV2 for 2D
    """
    This function is responsible to create the windows and give a total shape of 
    [window count, each window element count, C or dim]
    
    Args:
        x: (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size*window_size, C)
    """

    B, T, H, W, C = input.shape
    input = input.reshape(
        B, 
        T // window_size[0], window_size[0], 
        H // window_size[1], window_size[1], 
        W // window_size[2], window_size[2], 
        C)
    windows = input.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, window_size[0]*window_size[1]*window_size[2], C)
    return windows
     

def window_reverse_3d(windows, window_size, T, H, W):
    """
    This function is responsible to reverse the windows back to the original shape. input shape is nW*B, tW*hW*wW, C
    Returns:
        x: (B, T, H, W, C)
    """
    C = windows.shape[-1]
    windows = windows.reshape(
        -1, 
        T // window_size[0], 
        H // window_size[1], 
        W // window_size[2], 
        window_size[0], 
        window_size[1], 
        window_size[2], 
        C
        )
    windows = windows.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(-1, T, H, W, C)

    return windows


class AK_Block(nn.Module):

    """
    The main feauture extraction block*
    Here we Window the input and apply SW-MSA or W-MSA based on the shift size.
    Args:   
        config: AK_Config
        dim (int): Number of hidden channels
        input_resolution (Tuple[int]): Input resolution
        num_heads (int): Number of attention heads
        shift_size (Tuple[int]): Shift size for SW-MSA

    """


    def __init__ (self,
                  config,
                  dim,
                  input_resolution: Tuple[int],
                  num_heads,
                  shift_size: Tuple[int],
                  drop_path=0.0, 
                  ):
        super().__init__()

        self.config = config
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.set_shift_and_window_size_3d()

        # the extractor which is a 3D Attention
        self.attn = AK_Attention3D(
            config=self.config, 
            dim=self.dim, 
            input_resolution=self.input_resolution, 
            num_heads=self.num_heads, 
            shift_size=self.shift_size, 
            window_size=self.window_size,
    )
        
        self.norm_before = nn.LayerNorm(self.dim)
        self.drop_path = Swinv2DropPath(drop_path) if drop_path > 0.0 else nn.Identity() # 0 -> Identity
        self.intermediate = Swinv2Intermediate(config, dim)
        self.output = Swinv2Output(config, dim)
        self.norm_after = nn.LayerNorm(self.dim)

        self.attn_mask_cache = {}

     
    def set_shift_and_window_size_3d(self):
       
        use_window_size = list(self.config.window_size)
        hidden_state_shape = list(self.input_resolution)
        if self.shift_size is not None:
            use_shift_size = list(self.shift_size)
        for i in range(len(hidden_state_shape)):
            if hidden_state_shape[i] <= use_window_size[i]:
                use_window_size[i] = hidden_state_shape[i]
                if self.shift_size is not None:
                    use_shift_size[i] = 0

        if self.shift_size is None:
            self.window_size = tuple(use_window_size)
        else:
            self.window_size = tuple(use_window_size)
            self.shift_size = tuple(use_shift_size)


    def get_attn_mask_3d(self, T, H, W, window_size, shift_size, dtype, device):

        """
        creats a mask and window it according to window size and shift size
        """

        cache_key = (T, H, W, window_size, shift_size, dtype, str(device)) 
        if cache_key in self.attn_mask_cache: # {}
            # return self.attn_mask_cache[cache_key]
            cached = self.attn_mask_cache[cache_key]
            return None if cached is None else cached.to(device=device, dtype=dtype)

        if any(i > 0 for i in shift_size):
            img_mask = torch.zeros((1, T, H, W, 1), dtype=dtype, device=device)  # 1 T H W 1

            T_slices = (
                slice(0, -window_size[0]),
                slice(-window_size[0], -shift_size[0]),
                slice(-shift_size[0], None)
            )
            H_slices = (
                slice(0, -window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None)
            )
            W_slices = (
                slice(0, -window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None)
            )

            count = 0

            for t in T_slices:
                for h in H_slices:
                    for w in W_slices:
                        img_mask[:, t, h, w, :] = count
                        count += 1


            mask_windows = window_partition_3d(img_mask, window_size) # nW, ws[0]*ws[1]*ws[2], 1
            mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn_mask = attn_mask.to(device=device, dtype=dtype)

        else:
            attn_mask = None



        self.attn_mask_cache[cache_key] = attn_mask

        return attn_mask


    def maybe_pad(self, hidden_states, T, H, W):
        #  B, T, H, W, C = hidden_states.shape
        pad_Tr = (self.window_size[0] - T % self.window_size[0]) % self.window_size[0]
        pad_Hr = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1] 
        pad_Wr = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2] 
        pad_Tl = pad_Hl = pad_Wl = 0
        pad_values = (0, 0, pad_Wl, pad_Wr, pad_Hl, pad_Hr, pad_Tl, pad_Tr ) 
        # (pad_last_dim_left, pad_last_dim_right, pad_2nd_last_left, pad_2nd_last_right, ...)
        hidden_states = nn.functional.pad(hidden_states, pad_values)

        return hidden_states, pad_values


    def forward(
        self,
        hidden_states: torch.Tensor, # embedded output
        input_dimensions: Tuple[int, int], # (32, 32)
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        
        
        shortcut = hidden_states # B, T*H*W, C
        
        B, _, C = hidden_states.shape
        T, H, W = input_dimensions

        # self.window_size, self.shift_size coming from self.set_shift_and_window_size_3d

        # Layer norm before attention
        hidden_states = self.norm_before(hidden_states)

        hidden_states = hidden_states.reshape(B, T, H, W, C) # shape: (B, T, H, W, C)

        # pad the input if needed # first place I need 3d data
        hidden_states, pad_values = self.maybe_pad(hidden_states, T, H, W)
        _, Tp, Hp, Wp, _ = hidden_states.shape

        # cyclic shift in-between (W-MSA and SW-MSA)
        if any(i > 0 for i in self.shift_size):
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(-4, -3, -2))
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition_3d(shifted_hidden_states, self.window_size) # shape: (num_windows*B, window_size*window_size*window_size, C)

        # Attention mask for SW-MSA
        attn_mask = self.get_attn_mask_3d(Tp, Hp, Wp, self.window_size, self.shift_size, hidden_states.dtype, hidden_states.device) # shape: (1, window_size*window_size*window_size, C)

        # Apply attention
        attn_outputs = self.attn(
            hidden_states=hidden_states_windows,
            attention_mask=attn_mask
        )
        attn_output = attn_outputs[0] # shape: (B_, N, C)


        # reconstruct and merge windows
        attn_output_reverse = window_reverse_3d(attn_output, self.window_size, Tp, Hp, Wp) # shape: (B, Tp, Hp, Wp, C)

        # reverse cyclic shift if needed
        if any(i > 0 for i in self.shift_size):
        
            attn_windows = torch.roll(attn_output_reverse, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(-4, -3, -2))
        else:
            attn_windows = attn_output_reverse

        # remove padding if needed
        was_padded = any(pad_values[i] > 0 for i in range(len(pad_values)))
        if was_padded:
            attn_windows = attn_windows[
                :,
                : T,
                : H,
                : W,
                :
            ]

        # Residual connection 1
        attn_windows = attn_windows.reshape(B, T*H*W, C)
        output = shortcut + self.drop_path(attn_windows)

        # Layer norm + MLP + Drop_path after attention + Residual connection 2
        output = output + self.drop_path(self.output(self.intermediate(self.norm_after(output))))
        # shape is B, -1, C

        outputs = (
            (output,)
        )


        return outputs


class AK_EncoderStage(nn.Module):

    def __init__(
            self,
            config,
            dim,
            input_resolution,
            depth,
            num_heads,
            drop_path,
            downsample
    ):
        super().__init__()
        self.config = config
        self.dim = dim

        self.blocks = nn.ModuleList(
            [
                AK_Block(
                    config=config,
                    dim=dim, 
                    input_resolution=input_resolution, # not needed probably
                    num_heads=num_heads,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else (config.window_size[0] // 2, config.window_size[1] // 2, config.window_size[2] // 2), # tuple or tensor
                    drop_path=drop_path[i]
                ) for i in range(depth)
            ]
        )


        # PatchMerging
        if downsample is not None: # AK_hMerging in every stage except last one

            self.downsample = downsample(
                dim=dim, norm_layer=nn.LayerNorm
            )
        else:
            self.downsample = None

    def forward(
            self,
            hidden_states,
            input_dimensions: Tuple[int, int],
            **kwargs
    ):
        T, H, W = input_dimensions
        # inputs = hidden_states

        for i, block in enumerate(self.blocks):

            block_output = block(
                hidden_states=hidden_states,
                input_dimensions=input_dimensions,
                **kwargs
            )

            hidden_states = block_output[0] # shape: B, T*H*W, C


        hidden_states_before_downsampling = hidden_states

        if self.downsample is not None:
            

            hidden_states = self.downsample(
                # hidden_states_before_downsampling + inputs, # Check the existence of this addition: not originally
                hidden_states_before_downsampling,
                input_dimensions=input_dimensions, # when you keep the shape as B, N, C, this is useful
                **kwargs
            )


            Hds, Wds = H // 2, W // 2
            output_dimensions = (T, H, W, T, Hds, Wds) 

        else:
 
            output_dimensions = (T, H, W, T, H, W)


        stage_outputs = (
            hidden_states, 
            hidden_states_before_downsampling, 
            output_dimensions)


        return stage_outputs


class AK_Encoder(nn.Module):

    def __init__(
            self,
            config,
            grid_size,
    ):
        super().__init__()

        self.config = config

        self.num_layers = len(config.depths) 
        

        # drop path rate 
        self.drop_rates_encode_decode = torch.linspace( 
            0, config.drop_path_rate, 2 * sum(config.depths) 
        ) 
        dpr = [ 
            x.item()
            for x in self.drop_rates_encode_decode[: self.drop_rates_encode_decode.shape[0] // 2] 
        ]


        self.stages = nn.ModuleList(
            [
                AK_EncoderStage(
                    config=config,
                    dim=int(config.latent_channels * 2**i), #important
                    input_resolution=(
                        grid_size[0], # will stay the same after Embedding 
                        grid_size[1] // (2**i),
                        grid_size[2] // (2**i),
                    ),
                    depth=config.depths[i], 
                    num_heads=config.num_heads[i], 
                    drop_path=dpr[ 
                        sum(config.depths[:i]) : sum(config.depths[: i + 1]) 
                    ], # TODO: what is this?!
                    downsample=(
                        AK_Merging if (i < self.num_layers - 1) else None 
                    )

                ) for i in range(self.num_layers)
            ]
        )

    def forward(
            self,
            hidden_states,
            input_dimensions: Tuple[int, int],
            # head_mask: Optional[List[torch.FloatTensor]] = None, # TODO: study the use and results.
            output_hidden_states: Optional[bool] = True,
            # output_hidden_states_before_downsample: Optional[bool] = True, # Prefer to add by default
            **kwargs
    ):
        
        if output_hidden_states:

            all_hidden_states = ()
            all_reshaped_hidden_states = ()

        else:

            all_hidden_states = None
            all_reshaped_hidden_states = None



        if output_hidden_states: # in this model we need explicit 3D patches.

            all_hidden_states += (hidden_states,) # B, Num_patches, hidden_size
            B, _, C = hidden_states.shape

            reshaped_hidden_states = hidden_states.reshape(B, *input_dimensions, C)
            reshaped_hidden_states = reshaped_hidden_states.permute(0, 4, 1, 2, 3) # TODO: why we need the reshaped one to have C at the second?
                                                                                   # it seems the output class by HuggingFace has this format, but still!
            all_reshaped_hidden_states += (reshaped_hidden_states,)


        for i, stage in enumerate(self.stages):


            stage_outputs = stage(
                hidden_states=hidden_states,
                input_dimensions=input_dimensions,
                **kwargs
            )


            hidden_states = stage_outputs[0] # shape: B, T*H//2*W//2, C_prime 
            hidden_states_before_downsampling = stage_outputs[1] # shape: B, T*H*W, C
            output_dimensions = stage_outputs[2]
            input_dimensions = (output_dimensions[-3], output_dimensions[-2], output_dimensions[-1]) 
            

            if output_hidden_states:

                all_hidden_states += (hidden_states_before_downsampling,)


                B, _, C = hidden_states_before_downsampling.shape
                # reshaped_hidden_states = hidden_states_before_downsampling.reshape(B, output_dimensions[0], output_dimensions[1], output_dimensions[2], C)
                # reshaped_hidden_states = hidden_states_before_downsampling.reshape(B, *input_dimensions, C)
                reshaped_hidden_states = hidden_states_before_downsampling.reshape(B, output_dimensions[0], output_dimensions[1], output_dimensions[2], C)
                reshaped_hidden_states = reshaped_hidden_states.permute(0, 4, 1, 2, 3)
                all_reshaped_hidden_states += (reshaped_hidden_states,)




        return Swinv2EncoderOutput( # Just a return data class
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


