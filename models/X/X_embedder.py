"""

This Embedder receives data from DataLoader in shape [B, T, C, X, Y].

Here, the code prepares it different ways to pass to Encoder.

"""

 

import torch.nn as nn

 

def to_2tuple(x):

    if isinstance(x, tuple):
        return x

    return (x, x)

 

def to_3tuple(x, t):

    if isinstance(x, tuple):
        x = x[0]

    return (t, x, x)

 

class X_embed(nn.Module):

 

    def __init__(self, config):

 

        super().__init__()


        self.C_embedd = config.hidden_features
        self.patch_space = config.patch_space # Currently available for single integer
        self.patch_time = config.patch_time
        


        self.coord_features = config.coord_feature
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

                                       out_channels=config.hidden_features,

                                       kernel_size=to_3tuple(config.patch_space, config.patch_time),

                                       stride=to_3tuple(config.patch_space, config.patch_time))

 

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

        pad_XR = X % self.patch_space
        pad_YR = Y % self.patch_space
        pad_TR = T % self.patch_time
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



        x_embedded = self.embed(input) # either a nn.Con2d or a nn.Conv3d
        T_embedd = x_embedded.shape[2]
        embedded_dims = x_embedded.shape



        if self.embed_type=="separate": 

            if self.encode_type=="joint":
                x_embedded = x_embedded.reshape(B, self.C_embedd, -1) # out [B, C_embedd, T_embedd*X_embedd*Y_embedd]

            elif self.encode_type=="split":

                if self.split_type in {"encoder", "dotproduction"}:
                    x_embedded = x_embedded.reshape(B, self.C_embedd, T_embedd, -1) # out [B, C_embedd, T_embedd, X_embedd*Y_embedd]

                elif self.split_type=="attention":
                    x_embedded = x_embedded.reshape(B, self.C_embedd, -1) # out [B, C_embedd, T_embedd*X_embedd*Y_embedd]



        return x_embedded, embedded_dims



 