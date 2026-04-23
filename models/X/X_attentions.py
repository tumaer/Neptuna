import math
import collections
from typing import List, Optional, Tuple
import torch
from torch import nn
from transformers.pytorch_utils import meshgrid
from transformers.models.swinv2.modeling_swinv2 import (
    Swinv2Attention,
    Swinv2DropPath
)




class Attention_1D(nn.Module):

    """
    Full Seq Attention + Relative Position Bias (RPB) (based on Swin V2 and HuggingFace's implementation of it)
    TODO try APE
    """

    def __init__(self, config, dim):

        super().__init__()

        self.config = config
        seq_len = math.ceil(config.sequence_info[0]/config.patch_time)
        num_heads = config.num_heads_time
        self.num_heads = num_heads


        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )
        
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = dim


        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        
        # mlp to generate continuous relative position bias
        self.continuous_position_bias_mlp = nn.Sequential(
            nn.Linear(1, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )


        ####################################################
        # happens in forward 
        ####################################################
        # relative_coords_table: shape [1, 2*seq_len - 1, 1]
        # relative_coords_t = torch.arange(-(seq_len - 1), seq_len, dtype=torch.float32)
        # relative_coords_table = relative_coords_t.view(1, 2 * seq_len - 1, 1)

        # # normalize
        # relative_coords_table /= (seq_len - 1)

        # # map roughly to [-8, 8]
        # relative_coords_table *= 8

        # # log-scaled transform, same spirit as Swin V2
        # relative_coords_table = (
        #     torch.sign(relative_coords_table)
        #     * torch.log2(torch.abs(relative_coords_table) + 1.0)
        #     / math.log2(8)
        # )

        # relative_coords_table = relative_coords_table.to(
        #     next(self.continuous_position_bias_mlp.parameters()).dtype
        # )
        # self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # # pairwise relative position index: shape [seq_len, seq_len]
        # coords_t = torch.arange(seq_len)
        # relative_coords = coords_t[:, None] - coords_t[None, :]   # [T, T]
        # relative_position_index = relative_coords + (seq_len - 1) # shift to [0, 2T-2]
        # self.register_buffer("relative_position_index", relative_position_index, persistent=False)
        ######################################################


        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)

        self.dense = nn.Linear(self.all_head_size, self.all_head_size)
        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):

        # hidden_states: [B, seq_len , C]
        B, seq_len , C = hidden_states.shape

        device = hidden_states.device
        dtype  = hidden_states.dtype

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        # shapes: (B, num_heads, seq_len, head_size)

        # cosine attention
        attention_scores = nn.functional.normalize(query_layer, dim=-1) @ nn.functional.normalize(
            key_layer, dim=-1
        ).transpose(-2, -1) # (B, num_heads, seq_len, seq_len)

        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attention_scores = attention_scores * logit_scale


        ################################################

        # relative_coords_table: shape [1, 2*seq_len - 1, 1]
        relative_coords_t = torch.arange(-(seq_len - 1), seq_len, dtype=torch.float32)
        relative_coords_table = relative_coords_t.view(1, 2 * seq_len - 1, 1)

        # normalize
        relative_coords_table /= (seq_len - 1)

        # map roughly to [-8, 8]
        relative_coords_table *= 8

        # log-scaled transform, same spirit as Swin V2
        relative_coords_table = (
            torch.sign(relative_coords_table)
            * torch.log2(torch.abs(relative_coords_table) + 1.0)
            / math.log2(8)
        )

        relative_coords_table = relative_coords_table.to(
            next(self.continuous_position_bias_mlp.parameters()).dtype
        )

        coords_t = torch.arange(seq_len)
        relative_coords = coords_t[:, None] - coords_t[None, :]   # [T, T]
        relative_position_index = relative_coords + (seq_len - 1) # shift to [0, 2T-2]

        ################################################

        relative_coords_table = relative_coords_table.to(device=device)
        relative_position_index = relative_position_index.to(device=device)

        # add position bias
        relative_position_bias_table = self.continuous_position_bias_mlp(relative_coords_table).view(
            -1, self.num_heads
        )
    
        relative_position_bias = relative_position_bias_table[relative_position_index.view(-1)].view(
            seq_len, seq_len, -1
        )
        
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, seq_len, seq_len
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        #  Normalize the attention scores to probabilities. (the meaningful part), shape: (B_, num_heads, N, N)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # Apply dropout , shape: (B_, num_heads, N, N)
        attention_probs = self.dropout1(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Attention @ Value > weighted sum of the values based on the attention probabilities
        output = (attention_probs @ value_layer).transpose(1, 2).reshape(B, seq_len, C) 

        # OutputClass
        output = self.dense(output)
        output = self.dropout2(output)

        outputs =  (output,) # NOTE for output attentions
        
        return outputs


class Attention_2D(nn.Module):

    """
    Swin V2 Attention
    """

    def __init__ (self, config,
                  dim, input_resolution: Tuple[int],
                  num_heads, shift_size: Tuple[int], window_size: Tuple[int]
                  ):
        super().__init__()

        self.config = config
        self.all_head_size = dim
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )

        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
    

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1)))) # taken from SwinV2 in Huggingface

        self.continuous_position_bias_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )

       
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.int64).float()
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.int64).float()
        relative_coords_table = (
            torch.stack(meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # [1, 2*window[0] - 1, 2*window[1] - 1, 2]


        # normalize to -1, 1
        relative_coords_table[..., 0] /= (self.window_size[0] - 1)
        relative_coords_table[..., 1] /= (self.window_size[1] - 1)
        

        relative_coords_table *= 8  # normalize to -8, 8

        relative_coords_table = (
            torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        ) 

        # set to same dtype as mlp weight
        relative_coords_table = relative_coords_table.to(next(self.continuous_position_bias_mlp.parameters()).dtype)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)


        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)


        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob) # >> NOTE this is drop OUT
        self.dense = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob) # >> NOTE this is drop OUT

        
      


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        """
        input features with shape of (num_windows*B, N, C)
        """
        
        B_, N, C = hidden_states.shape # B_: num_windows * batch_size, N: Wh*Ww, C: hidden size 

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


        relative_coords_table = self.relative_coords_table.to(device=device)
        relative_position_index = self.relative_position_index.to(device=device)

        relative_position_bias_table = self.continuous_position_bias_mlp(relative_coords_table).view(
            -1, self.num_heads
        ) 

        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        # final attention score with relative position bias, shape: (B_, num_heads, N, N)
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)


        if attention_mask is not None: # TODO check
            # Apply the attention mask is (precomputed for all layers in Swinv2Model forward() function)
            attention_mask = attention_mask.to(device=device, dtype=dtype)
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                B_ // mask_shape, mask_shape, self.num_heads, N, N
            ) 
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_heads, N, N)

        #  Normalize the attention scores to probabilities. (the meaningful part), shape: (B_, num_heads, N, N)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # Apply dropout , shape: (B_, num_heads, N, N)
        attention_probs = self.dropout1(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Attention @ Value > weighted sum of the values based on the attention probabilities
        output = (attention_probs @ value_layer).transpose(1, 2).reshape(B_, N, C) # shape: (B_, N, C)

        # OutputClass
        output = self.dense(output)
        output = self.dropout2(output)

        outputs =  (output,)

        return outputs


class Attention_3D(nn.Module):

    """
    Adapted Swin V2 for 3D
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
        self.all_head_size = dim
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.window_size = config.window_size # (Wd,WH,WW) TODO: modify later >> to get a full 3d window 

        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1)))) 

        self.continuous_position_bias_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )

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
       

        # set to same dtype as mlp weight
        relative_coords_table = relative_coords_table.to(next(self.continuous_position_bias_mlp.parameters()).dtype)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)


        # get pair-wise relative position index for each token inside the window
        coords_t = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(meshgrid([coords_t, coords_h, coords_w], indexing="ij")) # shape: [3, Wt, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1) # shape: [3, Wt*Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # shape: [3, Wt*Wh*Ww, Wt*Wh*Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # shape: [Wt*Wh*Ww, Wt*Wh*Ww, 3]
        relative_coords[:, :, 0] += self.window_size[0] - 1  
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        # combine to a single index (from 3D to 1D - example: 25 elements in 2D*)
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)


        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
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
        attention_probs = self.dropout1(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Attention @ Value > weighted sum of the values based on the attention probabilities
        output = (attention_probs @ value_layer).transpose(1, 2).reshape(B_, N, C) # shape: (B_, N, C)

        # OutputClass
        output = self.dense(output)
        output = self.dropout2(output)

        outputs =  (output,)

        return outputs


class FactorizedAttention(nn.Module):

    def __init__(self, config, dim, temporal_len, input_resolution: Tuple[int], num_heads, shift_size: Tuple[int], window_size: Tuple[int]):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attention_space = Attention_2D(config=config, 
                                            dim=dim, 
                                            num_heads=num_heads,
                                            input_resolution=input_resolution,
                                            shift_size=shift_size,
                                            window_size=window_size)
        
        self.norm2 = nn.LayerNorm(dim)
        
        self.attention_time = Attention_1D(config=config,
                                           dim=dim,
                                           seq_len=temporal_len,
                                           num_heads=num_heads)
        
        self.norm3 = nn.LayerNorm(dim)
        
        self.MLP = nn.Linear(dim, dim)

    def forward(self, x):

        # shape x ?

        residual1 = x
        x = self.norm1(x)
        x = self.attention_space(x)[0] + residual1
        
        residual2 = x
        x = self.norm2(x)
        x = self.attention_time(x)[0] + residual2

        residual3 = x
        x = self.norm3(x)
        x = self.MLP(x) + residual3

        return x


class HeadFactorizedAttention(nn.Module):

    def __init__(self, config, dim):
        
        self.norm1 = nn.LayerNorm(dim)
        self.attention = self.headfactorized_attention()
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.norm2 = nn.LayerNorm(dim)
        self.dense = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)

    def headfactorized_attention(self, Q, K, V):

        # Q, K, V shapes [B, temporal, spatial, num_head, head_size]

        # Normalize the query with the square of its depth.
        Q = Q / nn.sqrt(Q.shape[-1]).type(Q.dtype)

        # Split heads for each axial attention dimension.
        num_attn_dims = 2 # temporal and spatial
        if Q.shape[-2] % num_attn_dims != 0:
            raise ValueError(f'In head-axial dot-product attention, number of 'f'heads ({Q.shape[-2]}) should be divisible by number '
                     f'of attention dimensions ({num_attn_dims})!')
        

        Q = torch.split(Q, num_attn_dims, dim=-2)
        K = torch.split(K, num_attn_dims, dim=-2)
        V = torch.split(V, num_attn_dims, dim=-2)
        # queries, keys, and values are each a list with two arrays (since
        # we have two dims, t and hw) that are made by spliting heads:
        # [(bs, t, hw, h//2, c), (bs, t, hw, h//2, c)].
        
        prefix_str = 'abcd'
        outputs = []
        for i, (q, k, v) in enumerate(zip(Q, K, V)):

            # Shape of query, key, and value: [bs, t, hw, h//2, c].

            axis = i + 1  # to account for the batch dim

            batch_dims = prefix_str[:axis]

            einsum_str = f'{batch_dims}x...z,{batch_dims}y...z->{batch_dims}x...y'
            # For axis=1 einsum_str (q,k->a): ax...z,ay...z->ax...y
            # For axis=2 einsum_str (q,k->a): abx...z,aby...z->abx...y

            attention_scores = torch.einsum(einsum_str, q, k)
            # For axis=1 (attention over t): attn_logits.shape: [bs, t, hw, h//2, t]
            # For axis=2 (attention over hw): attn_logits.shape: [bs, t, hw, h//2, hw]


            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # dropout - more similar to Swin than ViViT
            attention_probs = self.dropout1(attention_probs)


            einsum_str = f'{batch_dims}x...y,{batch_dims}y...z->{batch_dims}x...z'
            # For axis=1 einsum_str (a,v->o): ax...y,ay...z->ax...z
            # For axis=2 einsum_str (a,v->o): abx...y,aby...z->abx...z

            outputs.append(
                torch.einsum(einsum_str, attention_probs, v))

        # Output is list with two arrays [(bs, t, hw, h//2, c), (bs, t, hw, h//2, c)]
        # concatinate the heads.
            
        output = torch.concatenate(outputs, dim=-2)

        return output

        
    def forward(self, x):


        # shape x ?

        x = self.norm1(x)
        x = self.attention(x)
        # drop out?
        x = self.norm2(x)
        x = self.dense(x)
        output = self.dropout2(x)

        return output

