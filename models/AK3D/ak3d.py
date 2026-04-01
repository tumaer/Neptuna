"""
This is complete file for AK model. it connects Encoder, Propagator and Decoder together. 
It also contains the forward function for the whole model.

"""

import torch
from torch import nn
import torch.nn.init as init
from transformers import PreTrainedModel
from typing import List, Optional
from utils.grid_utils import twod_meshgrid_3d

from models.AK3D.ak3d_encoder import AK_Config, AK_Embeddings, AK_Encoder
from models.AK3D.ak3d_decoder import AK_Decoder, AK_Projection
from models.AK3D.ak3d_prop import AK_Prop



class ResNetBlock(nn.Module):
    def __init__(self, config, input_resolution, dim):
        super().__init__()
        kernel_size = 3
        self.input_resolution = input_resolution
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad) 
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad)
        self.bn1 = nn.BatchNorm3d(dim)
        self.bn2 = nn.BatchNorm3d(dim)

        # self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad) 
        # self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad)
        # self.bn1 = nn.BatchNorm2d(dim)
        # self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x, **kwargs):
        
        input = x 
        # x = x.reshape(batch_size, self.input_resolution[0], self.input_resolution[1], self.input_resolution[2], hidden_size) 
        # x = x.permute(0, 4, 1, 2, 3).contiguous() 
        x = self.conv1(x) 
        x = self.bn1(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # x = x.permute(0, 2, 3, 4, 1).contiguous() 
        # x = x.reshape(batch_size, sequence_length, hidden_size) 
        x = x + input  
        return x

class AK3D(PreTrainedModel):

    main_input_name = "input_data"
    config_class = AK_Config

    def __init__(
            self,
            config,
    ):
        super().__init__(config)



        self.config = config
        self.num_layers_encoder = len(config.depths)
        self.num_layers_decoder = len(config.depths)

        self.lift = AK_Embeddings(config)
        self.encoder = AK_Encoder(config, self.lift.grid_size)
        self.propagator = AK_Prop(config)
        self.decoder = AK_Decoder(config, self.lift.grid_size)
        self.projection = AK_Projection(config)

        res_model = ResNetBlock 


        self.residual_blocks = nn.ModuleList(
            [
                (
                    nn.ModuleList(
                        [
                            res_model(config, (self.lift.grid_size[0] // 2 ** i, self.lift.grid_size[1] // 2 ** i), config.latent_channels * 2**(i))
                            for _ in range(depth)
                        ]
                    )
                    if depth > 0
                    else nn.ModuleList([nn.Identity()])
                )
                for i, depth in enumerate(config.skip_connections)
            ]
        )

        self.post_init()


    def _init_weights(self, module):

        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)): 
            # TODO: Should I check other init methods for Conv layers? like Kaiming or Xavier?
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)

        # LayerNorm
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                init.ones_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)

        # BatchNorm3d
        elif isinstance(module, nn.BatchNorm3d):
            # only if affine=True (otherwise weight/bias are None) which is True in our case
            if module.weight is not None:
                init.ones_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)

        # else: do nothing (keep PyTorch defaults)


    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings


    def forward(
        self,
        input_data: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        

        # input from dataloader B, T_in, C, H, W

       
        if self.config.coord_features: 

            # NOTE:must be added. not the same as abs po embedding. that is learnable param but this is fixed. 
            # check if you need 2d or 3d meshgrid. Here it is a 2d meshgrid for 3d data

            coord_feat = twod_meshgrid_3d(list(input_data.shape), input_data.device)
            input_data = torch.cat((input_data, coord_feat), dim=2)

        # input after coord B, T_in, C+2, H, W
            
        # ====================================================
        # Embedding/Lift
        # ====================================================

        embedding_output, embedd_dimensions = self.lift( 
            input_data, 
            **kwargs
        )

        # output from embedding: B, T_patch*H_patch*W_patch, hidden_size ; input_dimensions: Tp, Hp, Wp

        # ====================================================
        # Encoder
        # ====================================================

        encoder_outputs = self.encoder( 
            embedding_output,
            input_dimensions=embedd_dimensions, # Tp, Hp, Wp
            output_hidden_states=True,
            **kwargs
        )

        # ====================================================
        # Skip connections with ResNet blocks
        # ====================================================

        skip_states = list(encoder_outputs[-1][1:]) 
        # The reshaped outputs of Encoder Stage, ignore Embedding and these outputs are the ones before PatchMerging.
        # skip_states = list(encoder_outputs[1][1:])
        
        for i in range(len(skip_states)):
            for block in self.residual_blocks[i]: # 2  blocks (last skip layer: identity)
                if isinstance(block, nn.Identity):
                    skip_states[i] = block(skip_states[i])
                else: # is not Identity
                    skip_states[i] = block(skip_states[i], **kwargs)


        # ====================================================
        # Propagator
        # ====================================================
                    
        prop_output = self.propagator(
            encoder_outputs[0], 
            **kwargs
        )
                    

        # ====================================================
        # Decoder
        # ====================================================

        input_dim_t = skip_states[-1].shape[-3]
        input_dim_x = skip_states[-1].shape[-2]
        input_dim_y = skip_states[-1].shape[-1]

        decoder_output = self.decoder(
            prop_output,
            (input_dim_t, input_dim_x, input_dim_y),
            skip_states=skip_states[:-1],
            output_hidden_states=True,
            **kwargs
        )


        # ====================================================
        # Projection
        # ====================================================

        sequence_output = decoder_output[0] # 8, 27, 3, 64, 64 ; update >>> B, T*H*W, C
        dims = tuple(decoder_output[-1][-1].shape[1:-1])
        assert dims == embedd_dimensions, f"Decoder output dimensions {dims} do not match embedding dimensions {embedd_dimensions}"
        prediction = self.projection(sequence_output, embedd_dimensions) 

        # B, T_out, c_OUT, H, W final prediction shape
        # TODO: IS this the right shape for prediction? yes


        return prediction