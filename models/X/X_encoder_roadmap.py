"""
This "SWin-based Encoder" extracts and encodes Information received from Embedder.
Encode includes Attention mechanism. Atten block in simple order is:
LN, MHSA, Residual + LN, MLP, Residual.
"""

import math
import collections
from torch import nn
import torch
from typing import List, Optional, Tuple
from transformers.models.swinv2.modeling_swinv2 import (
    Swinv2DropPath,
    Swinv2Intermediate,
    Swinv2Output,
    # Swinv2EncoderOutput
    Swinv2Attention,
    window_reverse,
    window_partition,

)
from transformers.pytorch_utils import meshgrid

####################################################
#            Joint_SpaceTime (Encoder_ST)           >> Swin 3D
####################################################

class Attention_ST(nn.Module):

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

class Merging_ST(nn.Module):

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

class Block_ST(nn.Module):

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
        self.set_shift_and_window_size_3d(input_resolution)

        # the extractor which is a 3D Attention
        self.attn = Attention_ST(
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

     
    def set_shift_and_window_size_3d(self, input_resolution): # TODO How have I written this?
       
        use_window_size = list(self.config.window_size)
        hidden_state_shape = list(input_resolution)
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
        
        B, C, _ = hidden_states.shape
        T, H, W = input_dimensions
        
        hidden_states = hidden_states.transpose(1, 2)

        shortcut = hidden_states # B, T*H*W, C
        
    
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

class EncoderStage_ST(nn.Module):

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

        self.blocks = nn.ModuleList(
            [
                Block_ST(
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
        if downsample is not None: # Merging in every stage except last one

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

class Encoder_ST(nn.Module):

    def __init__(self, config, grid_size):
        super().__init__()

        self.num_stages = len(config.depths) 
        
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
                EncoderStage_ST(
                    config=config,
                    dim=int(config.latent_features * 2**i), #important
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
                        Merging_ST if (i < self.num_stages - 1) else None 
                    )

                ) for i in range(self.num_stages)
            ]
        )

    def forward(
            self,
            hidden_states,
            input_dimensions: Tuple[int, int],
            output_hidden_states: Optional[bool] = True,
            **kwargs
    ):
        
        if output_hidden_states:

            all_hidden_states = ()
            all_reshaped_hidden_states = ()

        else:

            all_hidden_states = None
            all_reshaped_hidden_states = None



        if output_hidden_states: # 

            all_hidden_states += (hidden_states,) # B, C_embedded, Num_patches
            B, C, _ = hidden_states.shape

            reshaped_hidden_states = hidden_states.reshape(B, C, *input_dimensions)
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
                reshaped_hidden_states = hidden_states_before_downsampling.reshape(B, output_dimensions[0], output_dimensions[1], output_dimensions[2], C)
                reshaped_hidden_states = reshaped_hidden_states.permute(0, 4, 1, 2, 3)
                all_reshaped_hidden_states += (reshaped_hidden_states,)




        return { # Just a return data class
            hidden_states,
            all_hidden_states,
            all_reshaped_hidden_states,
        }


####################################################
#            Split_Encode (Encoder_D)
####################################################

class Merging_2D(nn.Module):
    def __init__(
        self, config, input_resolution: Tuple[int], dim: int, norm_layer: nn.LayerNorm
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

class Block_2D(nn.Module):

    def __init__(self,
                  config,
                  dim,
                  input_resolution: Tuple[int],
                  num_heads,
                  shift_size: Tuple[int],
                  drop_path=0.0,
                  pretrained_window_size=0 
                  ):
        super().__init__()

        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution
        self.set_shift_and_window_size(input_resolution)
        self.attention = Swinv2Attention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            pretrained_window_size=(
                pretrained_window_size
                if isinstance(pretrained_window_size, collections.abc.Iterable)
                else (pretrained_window_size, pretrained_window_size)
            ),
        )
        
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.drop_path = Swinv2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.intermediate = Swinv2Intermediate(config, dim)
        self.output = Swinv2Output(config, dim)
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        
        # Cache for attention masks
        self.attn_mask_cache = {}

        pass

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

    def maybe_pad(self, hidden_states, X, Y):

        # compute how much padding is needed
        pad_XR = (self.window_size - X % self.window_size) % self.window_size # 0
        pad_YR = (self.window_size - Y % self.window_size) % self.window_size # 0
        pad_XL = pad_YL = 0
        pad_values = (pad_YL, pad_YR, pad_XL, pad_XR, 0, 0)
        
        hidden_states = nn.functional.pad(hidden_states, pad_values)

        return hidden_states, pad_values

    def get_attn_mask(self, height, width, dtype, device):
        # Use cached attention mask when possible
        cache_key = (height, width, self.shift_size, self.window_size, dtype, str(device)) 
        if cache_key in self.attn_mask_cache: # {}
            return self.attn_mask_cache[cache_key]

        if self.shift_size > 0:
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype, device=device) 
            height_slices = ( 
                slice(0, -self.window_size), 
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None), 
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0 
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size) 
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size) 
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) 
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0)) 
        else:
            attn_mask = None
            
        # Cache the result
        self.attn_mask_cache[cache_key] = attn_mask
        return attn_mask

    def forward(self, x, input_dimensions, **kwargs):

        B, C, T, _ = x.shape
        _, _, _, X, Y = input_dimensions

        shortcut = x # TODO change and align shape

        x = x.permute(0, 2, 1, 3).reshape(B*T, C, X, Y)

        # 1. pad
        x_padded, pad_values = self.maybe_pad(x, X, Y)
        _, Xp, Yp, _ = x_padded.shape

        # 2. cyclic shif if needed
        if self.shift_size > 0:
            x_shifted = torch.roll( 
                x_padded, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            ) # roll: ((row-shift_size) mod H, (col-shift_size) mod W)
        else:
            x_shifted = x_padded 

        # 3. partition windows
        x_windows = window_partition(x_shifted, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) # shape B*T, num_windows, C

        # 4. get attention mask for SW-MSA
        attention_mask = self.get_attn_mask(Xp, Yp, x_windows.dtype, x_windows.device) # shape: (1, num_windows, num_windows)
        if attention_mask is not None:
            attention_mask = attention_mask.to(x_windows.device)

        # 5. apply attention
        attention_outputs = self.attention(x_windows, attention_mask=attention_mask, head_mask=None, output_attention=False)

        attention_output = attention_outputs[0]

        # 4. reverse windows
        attention_windows = attention_output.view(
            -1, self.window_size, self.window_size, C
        )
        shifted_windows = window_reverse(
            attention_windows, self.window_size, Xp, Yp  # TODO: check if this is correct, I think it should be Xp, Yp
        )
        
        # 3. Reverse cyclic shift if needed
        if self.shift_size > 0:
            attention_windows = torch.roll( 
                shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            attention_windows = shifted_windows

        # 2. remove padding if needed
        was_padded = any(pad != 0 for pad in pad_values)
        if was_padded: 
            attention_windows = attention_windows[:, :X, :Y, :].contiguous()
            
        attention_windows = attention_windows.view(B*T, X*Y, C)
        
        hidden_states = shortcut + self.drop_path(self.layernorm_before(attention_windows, **kwargs))
        
        residual = hidden_states
        layer_output = self.output(self.intermediate(hidden_states))
        layer_output = residual + self.drop_path(self.layernorm_after(layer_output, **kwargs))
        
        outputs = (
             (layer_output,)
        )


        return outputs

class EncoderStage_2D(nn.Module):

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

        window_size = ( # (16, 16)
            config.window_size
            if isinstance(config.window_size, collections.abc.Iterable)
            else (config.window_size, config.window_size)
        )

        self.blocks = nn.ModuleList(
            [
                Block_2D(
                    config=config,
                    dim=dim, 
                    input_resolution=input_resolution, # not needed probably
                    num_heads=num_heads,
                    shift_size=( 
                        [0, 0]
                        if (i % 2 == 0)
                        else [window_size[0] // 2, window_size[1] // 2]
                    ),drop_path=drop_path[i]
                ) for i in range(depth)
            ]
        )


        # PatchMerging
        if downsample is not None: # Merging in every stage except last one

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
        H, W = input_dimensions
        # inputs = hidden_states

        for i, block in enumerate(self.blocks):

            block_output = block(
                hidden_states=hidden_states,
                input_dimensions=input_dimensions,
                **kwargs
            )

            hidden_states = block_output[0] 


        hidden_states_before_downsampling = hidden_states

        if self.downsample is not None:
            

            hidden_states = self.downsample(
                # hidden_states_before_downsampling + inputs, # Check the existence of this addition: not originally
                hidden_states_before_downsampling,
                input_dimensions=input_dimensions, # when you keep the shape as B, N, C, this is useful
                **kwargs
            )

            # height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2

            Hds, Wds = H // 2, W // 2
            output_dimensions = (H, W, Hds, Wds) 

        else:
 
            output_dimensions = (H, W, H, W)


        stage_outputs = (
            hidden_states, 
            hidden_states_before_downsampling, 
            output_dimensions)


        return stage_outputs

class Encoder_2D(nn.Module):

    def __init__(self, config, grid_size):
        super().__init__()

        self.num_stages = len(config.depths) 
        
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
                EncoderStage_2D(
                    config=config,
                    dim=int(config.latent_features * 2**i), #important
                    input_resolution=(
                        grid_size[1] // (2**i),
                        grid_size[2] // (2**i),
                    ),
                    depth=config.depths[i], 
                    num_heads=config.num_heads[i], 
                    drop_path=dpr[ 
                        sum(config.depths[:i]) : sum(config.depths[: i + 1]) 
                    ], # TODO: what is this?!
                    downsample=(
                        Merging_2D if (i < self.num_stages - 1) else None 
                    )

                ) for i in range(self.num_stages)
            ]
        )

    def forward(
            self,
            hidden_states,
            input_dimensions: Tuple[int, int],
            output_hidden_states: Optional[bool] = True,
            **kwargs
    ):
        
        if output_hidden_states:

            all_hidden_states = ()
            all_reshaped_hidden_states = ()

        else:

            all_hidden_states = None
            all_reshaped_hidden_states = None



        if output_hidden_states: # 

            all_hidden_states += (hidden_states,) # B, C_embedded, Num_patches
            B, C, _ = hidden_states.shape

            reshaped_hidden_states = hidden_states.reshape(B, C, *input_dimensions)
            all_reshaped_hidden_states += (reshaped_hidden_states,)


        for i, stage in enumerate(self.stages):


            stage_outputs = stage(
                hidden_states=hidden_states,
                input_dimensions=input_dimensions,
                **kwargs
            )


            hidden_states = stage_outputs[0] # shape: B, H//2*W//2, C_prime 
            hidden_states_before_downsampling = stage_outputs[1] # shape: B, H*W, C
            output_dimensions = stage_outputs[2]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1]) 
            

            if output_hidden_states:

                all_hidden_states += (hidden_states_before_downsampling,)


                B, _, C = hidden_states_before_downsampling.shape
                reshaped_hidden_states = hidden_states_before_downsampling.reshape(B, output_dimensions[0], output_dimensions[1], C)
                reshaped_hidden_states = reshaped_hidden_states.permute(0, 3, 1, 2)
                all_reshaped_hidden_states += (reshaped_hidden_states,)




        return { # Just a return data class
            hidden_states, # shape B', C', Patches = B*T, C_embedd, Xp*Yp
            all_hidden_states,
            all_reshaped_hidden_states,
        }

class Encoder_1D(nn.Module):
    
        def __init__(self, config):
            super().__init__()
    
            pass
    
        def forward(self, x):
    
            pass

class Encoder_SpaceTime(nn.Module):

    def __init__ (self, config, grid_size):
        super().__init__()

        self.encoder_space = Encoder_2D(config, grid_size)
        self.encoder_time = Encoder_1D(config, grid_size)

    def forward(self, x, input_dimentions):

        # x shape B, C, T, X*Y
        # inside 2d encoder for space change shape to B*T, C, X, Y  
        # output B*T, C_embedded, Xp*Yp
        # inside 1d encoder for time change shape to B*Xp*Yp, C_embedded, T

        x_encode_space = self.encoder_space(x, input_dimentions)
        x_encode_time = self.encoder_time(x_encode_space[0])

        # the out put of whole encoder part will go to prop in shape B", C", T

        # this shape in prop will advance in time so makes sense to keep the shape.



        

        

####################################################
#            Encoder_Split_Attention
####################################################

class SelfAttention_2D(nn.Module): # >> Normal SwinAttention

    def __init__(self, config, dim, num_heads):

        super().__init__()

        pass

    def forward(self, x):

        pass

class SelfAttention_1D(nn.Module): # >> adapt yourself

    def __init__(self, config, dim, num_heads):

        super().__init__()

        pass

    def forward(self, x):

        pass


class FactorizedAttention(nn.Module):
    
    def __init__(self, config, dim, num_heads):

        self.attention_space = SelfAttention_2D(config, dim, num_heads)
        self.attention_time = SelfAttention_1D(config, dim, num_heads)

        # MLP layer here

    def forward(self, x):

        pass


class Block_F(nn.Module):

    def __init__(self, config, dim, num_heads, drop_path):

        super().__init__()

        self.attention = FactorizedAttention(config, dim, num_heads)

    def forward(self, x):

        pass

class EncoderStage_F(nn.Module):
    pass


class Encoder_F(nn.Module):
    pass






####################################################
#            Encoder_Split_DotProduct
####################################################

def FacDotPro():
    pass

class HeadFactorizedAttention(nn.Module):
    def __init__(self, config, dim, num_heads):

        super().__init__()

        pass

    def forward(self, x):

        pass


class Block_H(nn.Module):

    def __init__(self, config, dim, num_heads, drop_path):

        super().__init__()

        self.attention = HeadFactorizedAttention(config, dim, num_heads)

    def forward(self, x):

        pass


class EncoderStage_H(nn.Module):
    pass

class Encoder_H(nn.Module):
    pass






####################################################
#            Main Encoder
####################################################

class X_Encode(nn.Module):

    def __init__(self, config):

        super().__init__()

        pass

    def forward(self, x):

        pass