"""

This Embedder receives data from DataLoader in shape [B, T, C, X, Y].

Here, the code prepares it different ways to pass to Encoder.

"""

 
import math
import torch.nn as nn
from utils.model_utils import PretrainedConfig

 

def to_2tuple(x):

    if isinstance(x, tuple):
        return x

    return (x, x)

 

def to_3tuple(t, x):

    if isinstance(x, tuple):
        x = x[0]

    return (t, x[0], x[1])

 
class X_Config(PretrainedConfig):

    model_type = 'x'
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
            self,
            latent_time=None,
            patch_space=None,
            patch_time=None,
            depths=None,
            num_heads=None,
            skip_connections=None,
            skip_connections_time=None,
            window_size=None,
            mlp_ratio=None,
            num_time_blocks=None,
            num_heads_time=None,
            embed_type=None,
            encode_type=None,
            split_type=None,
            dpr_space=None,
            dpr_time=None,
            qkv_bias=True,
            qk_scale=None,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02, # a default to initialize the model weights
            layer_norm_eps=1e-5,
            p=1,  # for loss: 1 for l1, 2 for l2
            channel_slice_list_normalized_loss=None,  # TODO if None will fall back to absolute loss otherwise normalized loss with split channels
            residual_model="resnet",  # "convnext" or "resnet"; Currently only ResNet
            learn_residual=False,  # learn the residual for time-dependent problems
            output_hidden_states: bool = False,
            output_attentions: bool = False,
            **kwargs,
            ):
        

        super().__init__(**kwargs)

        self.latent_time = latent_time
        self.patch_space = patch_space
        self.patch_time = patch_time
        self.depths = depths
        self.num_heads = num_heads
        self.skip_connections = skip_connections
        self.skip_connections_time = skip_connections_time
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale  
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.p = p
        self.channel_slice_list_normalized_loss = channel_slice_list_normalized_loss
        self.residual_model = residual_model
        self.learn_residual = learn_residual
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.num_time_blocks=num_time_blocks
        self.embed_type = embed_type
        self.encode_type = encode_type
        self.split_type = split_type
        self.dpr_space = dpr_space
        self.dpr_time = dpr_time
        self.num_heads_time = num_heads_time


class X_Embedder(nn.Module):

 

    def __init__(self, config):

 

        super().__init__()


        self.C_embedd = config.latent_channels
        self.norm = nn.LayerNorm(self.C_embedd)
        self.T_embedd = config.latent_time
        self.T_in = math.ceil(config.sequence_info[0]/config.patch_time)
        self.patch = [config.patch_time, config.patch_space[0], config.patch_space[1]]
        self.resolution =[config.sequence_info[0], config.grid_resolution[0], config.grid_resolution[1]]
        
        self.patch_dimensions = (
            self.resolution[0] // self.patch[0],
            self.resolution[1] // self.patch[1],
            self.resolution[2] // self.patch[2]

        ) # tuple or list?


        self.coord_features = config.coord_features
        # True OR Flase; it adds non-learnable geometry-based features to Channel dim. Assumes 2D data in space.

 

        self.embed_type = config.embed_type
        # 3 types; 1. changes in time as feature: T*C __ feature
        #          2. timestep as regulator separate feature: t* __ condition
        #          3. times-space embedding together: T, C __ separate >> needs extra encode_type arg.


        self.encode_type = config.encode_type
        # 2 types; 1. time-space joint __ joint
        #          2. time-space split __ split


        self.split_type = config.split_type
        # 3 types; 1. in Encoder __ encoder
        #          2. in Self-Attention __ attention
        #          3. in dot-production __ dotproduction

 

        if self.embed_type=="separate" and self.encode_type is None:

            raise ValueError("For embed_type=separate, encode_type muse be either 'joint' or 'split'.")

 

 

        if self.embed_type=="feature":

            self.in_features = (config.in_channels * config.sequence_info[0]) + (2 if self.coord_features else 0)

            self.embed = nn.Conv2d(in_channels=self.in_features,

                                   out_channels=config.hidden_features,

                                   kernel_size=to_2tuple(config.patch_space),

                                   stride=to_2tuple(config.patch_space))
            
            #  then output will keep shape B, C*T, X, Y. One encoder waits afterwards

 

        elif self.embed_type=="condition":

            pass

 

        elif self.embed_type=="separate":
            self.in_features = (config.in_channels) + (2 if self.coord_features else 0)

            self.embed = nn.Conv3d(in_channels=self.in_features,

                                       out_channels=config.latent_channels,

                                       kernel_size=to_3tuple(config.patch_time, config.patch_space),

                                       stride=to_3tuple(config.patch_time,config.patch_space))
            
            # self.embed_t = nn.Conv3d(in_channels=self.T_in,

            #                            out_channels=self.T_embedd,

            #                            kernel_size=(1, 1, 1),

            #                            stride=(1, 1, 1))

 

            # if self.encode_type=="joint":

            #     self.embed = nn.Conv3d(in_channels=self.in_features,

            #                            out_channels=config.hidden_features,

            #                            kernel_size=to_3tuple(config.patch_space, config.patch_time),

            #                            stride=to_3tuple(config.patch_space, config.patch_time))
                
                #  then output will reshape from B, C, T, X, Y to B, C, T*X*Y . One encoder waits afterwards

 

 

            # elif self.encode_type=="split":

            #     if self.split_type in {"encoder", "dotproduction"}:

            #         self.embed = nn.Conv3d(in_channels=self.in_features,

            #                                out_channels=config.hidden_features,

            #                                kernel_size=to_3tuple(config.patch_space, config.patch_time),

            #                                stride=to_3tuple(config.patch_space, config.patch_time))
                    
                    #  then output will reshape from B, C, T, X, Y to B, C, T, X*Y. Two encoders are there afterwards.


                # elif self.encode_type=="attention":

                #     self.embed = nn.Conv3d(in_channels=self.in_features,

                #                            out_channels=config.hidden_features,

                #                            kernel_size=to_3tuple(config.patch_space, config.patch_time),

                #                            stride=to_3tuple(config.patch_space, config.patch_time))
                    
                    #  then output will reshape from B, C, T, X, Y to B, C, T*X*Y. One encoder with two self-attention inside is there afterwards.
                    

                # elif self.encode_type=="dotproduction":

                #     self.embed = nn.Conv3d(in_channels=self.in_features,

                #                            out_channels=config.hidden_features,

                #                            kernel_size=to_3tuple(config.patch_space, config.patch_time),

                #                            stride=to_3tuple(config.patch_space, config.patch_time))
                    
                #     #  then output will reshape from B, C, T, X, Y to B, C, T, X*Y. One encoder with one self-attention inside is there afterwards.
            

            # TODO Shoild I add LayerNorm after embedding? Or add it in encoder?

    def maybe_pad_2d(self, input):
        
        B, F, X, Y = input.shape

        pad_XR = X % self.patch_space
        pad_YR = Y % self.patch_space
        pad_XL = pad_YL = 0
        pad_values = (pad_YL, pad_YR, pad_XL, pad_XR)

        if any(pad != 0 for pad in pad_values):
            input = nn.functional.pad(input, pad_values)

        return input

    def maybe_pad_3d(self, input):
        
        B, C, T, X, Y = input.shape

        pad_TR = T % self.patch[0]
        pad_XR = X % self.patch[1]
        pad_YR = Y % self.patch[2]
        pad_XL = pad_YL = pad_TL = 0
        pad_values = (pad_YL, pad_YR, pad_XL, pad_XR, pad_TL, pad_TR)

        if any(pad != 0 for pad in pad_values):
            input = nn.functional.pad(input, pad_values)

        return input
 
    def forward(self, input):

        B, T, C, X, Y = input.shape
        # coming from dataloader in shape [B, T, C, X, Y]

        if self.embed_type=="feature":
            x = input.reshape(B, T*C, X, Y)
            x = self.maybe_pad_2d(x)
        else:
            x = input.permute(0, 2, 1, 3, 4) # from [B, T, C, X, Y] to [B, C, T, X, Y]
            x = self.maybe_pad_3d(x)



        x_embedded = self.embed(x) # either a nn.Con2d or a nn.Conv3d
        x_embedded = self.norm(x_embedded)
        # x_embedded = x_embedded.permute(0, 2, 1, 3, 4)
        # x_t_embedded = self.embed_t(x_embedded)
        # x_embedded = x_t_embedded.permute(0, 2, 1, 3, 4)
        embedded_dims = x_embedded.shape



        # if self.embed_type=="separate": 

        #     if self.encode_type=="joint":
        #         x_embedded = x_embedded.reshape(B, self.C_embedd, -1) # out [B, C_embedd, T_embedd*X_embedd*Y_embedd]

        #     elif self.encode_type=="split":

        #         if self.split_type in {"encoder", "dotproduction"}:
        #             x_embedded = x_embedded.reshape(B, self.C_embedd, T_embedd, -1) # out [B, C_embedd, T_embedd, X_embedd*Y_embedd]

        #         elif self.split_type=="attention":
        #             x_embedded = x_embedded.reshape(B, self.C_embedd, -1) # out [B, C_embedd, T_embedd*X_embedd*Y_embedd]

        
        # TODO give [B, C_embedd, T_embedd*X_embedd*Y_embedd] for all.
        # TODO give only patch dim, not everything



        return x_embedded, embedded_dims



 