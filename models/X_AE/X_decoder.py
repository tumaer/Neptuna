"""
This decoder Module is by default a mirrored Swin-based encoder. 
The additional option is to deactivate attention layers. (TODO)
Also, including symmetrical time blocks in decoder can be tested.
"""

import torch
from torch import nn
from typing import List, Optional, Tuple
from models.X.X_attentions import Attention_1D, Attention_2D
from transformers.models.swinv2.modeling_swinv2 import (
    Swinv2DropPath,
    Swinv2Intermediate,
    Swinv2Output,
    Swinv2Attention,
    window_reverse,
    window_partition,

)

class X_UnMerging(nn.Module): # TODO verify with source code

    def __init__(
        self,
        config,
        input_resolution: Tuple[int],
        dim: int,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution 
        self.dim = dim 
        self.upsample = nn.Linear(dim, 2 * dim, bias=False) 
        self.mixup = nn.Linear(dim // 2, dim // 2, bias=False) 
        self.norm = nn.LayerNorm(dim//2) 

    def maybe_crop(self, input_feature, H, W):
        H_in, W_in = input_feature.shape[2], input_feature.shape[3] 
        if H_in > H:
            input_feature = input_feature[:, :H, :, :]
        if W_in > W:
            input_feature = input_feature[:, :, :W, :]
        return input_feature

    def forward(
        self,
        input_feature: torch.Tensor, 
        output_dimensions: Tuple[int, int], 
        **kwargs
    ) -> torch.Tensor:
        
        H, W = output_dimensions 
  
        batch_size, _, in_channels= input_feature.shape 

        input_feature = self.upsample(input_feature) 

        input_feature = input_feature.reshape(
            batch_size, H//2, W//2, 2, 2, in_channels //2
        )
        input_feature = input_feature.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, H, W, in_channels //2) 

        input_feature = self.maybe_crop(input_feature, H, W) # crop in case the feature does not have the specified dim
        input_feature = input_feature.reshape(batch_size, -1, in_channels // 2) 

        input_feature = self.mixup(self.norm(input_feature)) 



        return input_feature

class TimeBlock(nn.Module):

    def __init__(self, config, patch_dimensions, dpr):

        super().__init__()

        self.patch_dimensions = patch_dimensions
        self.stages = len(config.depths)
        self.hidden_features_last = config.latent_channels * 2 ** (self.stages-1)
        self.num_time_blocks = config.num_time_blocks
        # TODO add assert for existence of time_blocks when encode_type is encoder. 

        

        self.norm1 = nn.LayerNorm(self.hidden_features_last)
        self.attention = Attention_1D(config=config,
                                          dim=self.hidden_features_last
                                          ) 
        
        self.norm2 = nn.LayerNorm(self.hidden_features_last)
        self.MLP = nn.Sequential(
                nn.Linear(self.hidden_features_last, int(config.mlp_ratio * self.hidden_features_last)),
                nn.GELU(),
                nn.Linear(int(config.mlp_ratio * self.hidden_features_last), self.hidden_features_last)
            )      
        
        self.drop_path = Swinv2DropPath(dpr) if dpr > 0. else nn.Identity()

        # NOTE you can try the ordering of SwinV2 

    def forward(self, input):
        
        # input shape: B*num_t_patch, num_xy_patch, self.hidden_features * 2 ** (self.stages-1)
        

        # TODO add sanity check for shape. entry below should have shape [B*num_xy_patch, num_t_patch, self.hidden_features_last]

        residual1 = input
        x = self.norm1(input)
        x = self.attention(x)
        x = self.drop_path(x[0])
        x = x + residual1
    

        residual2 = x
        x = self.norm2(x)
        x = self.MLP(x)
        x = self.drop_path(x)
        x = x + residual2

        return x

class X_DecoderBlock(nn.Module):

    def __init__(self, config, dim, patch_dimensions, num_heads, dpr, shift_size):

        super().__init__()

        self.window_size = config.window_size
        self.shift_size = shift_size
        new_window_size, new_shift_size = self.set_window_and_shift_size(patch_dimensions)

        self.window_size=new_window_size
        self.shift_size=new_shift_size

        # TODO add Attention_3D here later into an if_condition

        self.attention = Attention_2D(config=config, # >> or just import SwinAttention! 
                                      dim=dim,
                                      input_resolution=patch_dimensions,
                                      num_heads=num_heads,
                                      shift_size=self.shift_size,
                                      window_size=self.window_size)
        
        self.norm1 = nn.LayerNorm(dim)
        
        # coming from original Swin
        self.drop_path = Swinv2DropPath(dpr) if dpr > 0.0 else nn.Identity() # >> NOTE this is drop PATH

        self.MLP = Swinv2Intermediate(config=config, dim=dim)

        self.output = Swinv2Output(config=config, dim=dim)

        self.norm2 = nn.LayerNorm(dim)

        # ordering comes from original Swin, but I can change it later if needed.

        self.attn_mask_cache = {}


    def set_window_and_shift_size(self, patch_dimensions):
        
        window_size = [r if r <= w else w for r, w in zip(patch_dimensions, self.window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(patch_dimensions, window_size, self.shift_size)]
        return window_size, shift_size
    

    def get_attention_mask(self, X, Y, dtype, device):

        WS, SZ = self.window_size[0], self.shift_size[0]

        cache_key = (X, Y, SZ, WS, dtype, str(device)) 
        if cache_key in self.attn_mask_cache: # {}
            return self.attn_mask_cache[cache_key]

        if SZ > 0:
            img_mask = torch.zeros((1, X, Y, 1), dtype=dtype, device=device) 
            X_slices = ( 
                slice(0, -WS), 
                slice(-WS, -SZ), 
                slice(-SZ, None),
            )
            Y_slices = (
                slice(0, -WS),
                slice(-WS, -SZ),
                slice(-SZ, None),
            )
            count = 0 # label each sub-region of feature map with unique integer
            for X_slice in X_slices:
                for Y_slice in Y_slices:
                    img_mask[:, X_slice, Y_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, WS) 
            mask_windows = mask_windows.view(-1, WS * WS) 
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) 
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0)) 
        else:
            attn_mask = None
            
        # Cache the result
        self.attn_mask_cache[cache_key] = attn_mask

        return attn_mask


    def maybe_pad(self, input, X, Y):
        
        # shape : B*T, X, Y, C 

        pad_Xr = (self.window_size[0] - X % self.window_size[0]) % self.window_size[0]
        pad_Yr = (self.window_size[1] - Y % self.window_size[1]) % self.window_size[1] 
        pad_Xl = pad_Yl = 0
        pad_values = (0, 0, pad_Yl, pad_Yr, pad_Xl, pad_Xr) 
        # (pad_last_dim_left, pad_last_dim_right, pad_2nd_last_left, pad_2nd_last_right, ...)
        input = nn.functional.pad(input, pad_values)

        return input, pad_values


    def forward(self, input, input_dimensions, **kwargs):

        hidden_states = input
        shortcut = hidden_states
        
        B, _, C = hidden_states.shape
        X, Y = input_dimensions 
        
        hidden_states = hidden_states.reshape(B, X, Y, C) # shape: (B', X, Y, C), B'=B*T

        hidden_states, pad_values = self.maybe_pad(hidden_states, X, Y)
        _, Xp, Yp, _ = hidden_states.shape

        # cyclic shift in-between (W-MSA and SW-MSA)
        if any(i > 0 for i in self.shift_size):
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # partition windows NOTE assumes square window
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size[0]) # shape: (B'*nW, xW, yW, C)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size[0] * self.window_size[0], C) # shape: (B'*nW, xW*yW, C)

        # Attention mask for SW-MSA
        attn_mask = self.get_attention_mask(Xp, Yp, hidden_states.dtype, hidden_states.device) # shape: (1, xW*yW, C)

        # Apply attention
        attn_outputs = self.attention(
            hidden_states=hidden_states_windows,
            attention_mask=attn_mask
        )
        attn_output = attn_outputs[0] # shape: (B_, N, C)


        # reconstruct and merge windows
        attn_output_reverse = window_reverse(attn_output, self.window_size[0], Xp, Yp) # shape: (B, X, Y, C)

        # reverse cyclic shift if needed
        if any(i > 0 for i in self.shift_size):
        
            attn_windows = torch.roll(attn_output_reverse, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            attn_windows = attn_output_reverse

        # remove padding if needed
        was_padded = any(pad_values[i] > 0 for i in range(len(pad_values)))
        if was_padded:
            attn_windows = attn_windows[
                :,
                : X,
                : Y,
                :
            ]

        # Residual connection 1
        attn_windows = attn_windows.reshape(B, X*Y, C)

        # Attention output + Layer norm + Drop_path + Residual connection 1
        output = self.norm1(attn_windows) 
        output = shortcut + self.drop_path(output)

        # MLP + Layer norm + Drop_path after attention + Residual connection 2
        output = self.output(self.MLP(output))
        output = output + self.drop_path(self.norm2(output))
    

        outputs = (
            (output,)
        )


        return outputs

class X_DecdoerStage(nn.Module):

    def __init__(self, config, dim, patch_dimensions, depth, num_heads, dpr, upsample):
        super().__init__()

        self.blocks = nn.ModuleList([

                X_DecoderBlock(config=config,
                               dim=dim,
                               patch_dimensions=patch_dimensions,
                               num_heads=num_heads,
                               dpr=dpr[i],
                               shift_size=[0, 0] if (i % 2 == 0) else (config.window_size[0]//2, config.window_size[1]//2)
                               )

                for i in reversed(range(depth))

        ])

        if upsample is not None:

            self.upsample = upsample(config=config, input_resolution=patch_dimensions,  dim=dim, norm_layer=nn.LayerNorm)
        else:

            self.upsample = None

    def forward(self, input, input_dimensions, **kwargs):

        # shape of input: B*T, C, XY
        
        X, Y = input_dimensions 

        for block in self.blocks:

            block_output = block(
                          input=input, 
                          input_dimensions=input_dimensions, 
                          **kwargs)
            
            input = block_output[0]

        hidden_state_before_ds = input

        if self.upsample is not None:

            input = self.upsample(input, output_dimensions=(2*input_dimensions[0], 2*input_dimensions[1]))

            output_dimensions = (X, Y, X*2, Y*2)

        else:

            output_dimensions = (X, Y, X, Y)


        stage_output = (input, hidden_state_before_ds, output_dimensions)   

        
        return stage_output

class X_Decoder(nn.Module):

    def __init__(self, config, patch_dimensions):

        super().__init__()

        self.config = config    
        
        self.num_stages = len(config.depths)
        self.split_type = config.split_type 
        self.num_time_blocks = config.num_time_blocks
        self.hidden_features = config.latent_channels
        self.patch_dimensions = patch_dimensions  # [num_t_patch, num_x_patch, num_y_patch]
        self.dpr = [x.item() for x in torch.linspace(0, config.dpr_time, self.num_time_blocks)]


        self.dpr_encode_decode = torch.linspace( 
            0, config.dpr_space, 2 * sum(config.depths) 
        ) 
        self.dpr_ED = [ 
            x.item()
            for x in self.dpr_encode_decode[self.dpr_encode_decode.shape[0] // 2: ] 
        ]


        if self.split_type=="encoder":

            self.time_blocks = nn.ModuleList([

                TimeBlock(config=config, 
                          patch_dimensions=patch_dimensions, 
                          dpr=self.dpr[i])
                for i in reversed(range(self.num_time_blocks))
            ])


        self.stages = nn.ModuleList([

            X_DecdoerStage(config=config,
                           dim=int(self.hidden_features * 2 ** i),
                           patch_dimensions= (patch_dimensions[1]//(2**i), patch_dimensions[2]//(2**i)),
                           depth=config.depths[i],
                           num_heads=config.num_heads[i],
                           dpr=self.dpr_ED[sum(config.depths[i+1:]): sum(config.depths[i:])],
                           upsample=X_UnMerging if i != 0 else None
                           )

            for i in reversed(range(self.num_stages))
        ])

        


            

    def forward(self, input, input_dimensions, skip_states, output_hidden_states: Optional[bool] = True, **kwargs):

        # input_dimensions: num_t_patch, num_x_patch, num_y_patch >> coming from prop
        # input shape: [B*num_xy_patch, num_t_patch, C_embedd] TODO
       

        B, seq_len, C = input.shape
        num_x_patch, num_y_patch = input_dimensions

        
        # do NOT care of hidden_states getting saved in decoder
        # if output_hidden_states:

        #     all_hidden_states = ()
        #     all_reshaped_hidden_states = ()

        # else:

        #     all_hidden_states = None
        #     all_reshaped_hidden_states = None


        # if output_hidden_states:  

        #     all_hidden_states += (input,) 

        #     reshaped_hidden_states = input.reshape(B, C, T, input_dimensions[3], input_dimensions[4])
        #     all_reshaped_hidden_states += (reshaped_hidden_states,)



        # input = input.permute(0, 2, 3, 4, 1).reshape(B*T, X*Y, C)


        # for cases: feature - condition - STjoint - STFattn - STFhead it goes single encoder
        # for case: two encoders we need two for loops like below.


        if self.split_type=="encoder":
            

            for i, time_block in enumerate(self.time_blocks):

                if i!=0 and skip_states[len(skip_states) - i] is not None:

                    input = input + skip_states[len(skip_states) - i]

                time_block_output = time_block(input=input)
                input = time_block_output

                # if output_hidden_states:

                #     all_hidden_states += (time_block_output,)
                #     reshaped_hidden_states = time_block_output.reshape(Batch //num_t_patch, num_x_patch, num_y_patch, num_t_patch, hidden_features_last)
                #     reshaped_hidden_states = reshaped_hidden_states.permute(0, 4, 3, 1, 2)
                #     all_reshaped_hidden_states += (reshaped_hidden_states,)
            
        input = input.reshape(-1, num_x_patch, num_y_patch, seq_len, C).permute(0, 3, 1, 2, 4)
        input = input.reshape(-1, num_x_patch*num_y_patch, C)
        
        for i, stage in enumerate(self.stages):

            input = input + skip_states[len(skip_states) - len(self.time_blocks) - i]

            stage_outputs = stage(
                input=input,
                input_dimensions=input_dimensions,
                **kwargs
            )


            input = stage_outputs[0]  
            # hidden_state_before_ds = stage_outputs[1] 
            output_dimensions = stage_outputs[2]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1]) # TODO only spatial dims
            

            # if output_hidden_states:

            #     all_hidden_states += (hidden_state_before_ds,)


            #     B_, _, C = hidden_state_before_ds.shape
            #     reshaped_hidden_states = hidden_state_before_ds.reshape(B, T, output_dimensions[1], output_dimensions[1], C)
            #     reshaped_hidden_states = reshaped_hidden_states.permute(0, 4, 1, 2, 3)
            #     all_reshaped_hidden_states += (reshaped_hidden_states,)

        
        
        decoded = input


        return decoded
    
        # the same entry to encoder is reconstructed in reverse fashion.
        # ready to go to patch recovery

                

                

            
                


