import math
import torch
from torch import nn
import torch.nn.init as init
from transformers import PreTrainedModel
from typing import List, Optional
from models.X import X_embedder, X_encoder, X_propagator, X_decoder, X_recovery
from models.X.X_utils import ResNetBlock
from utils.grid_utils import twod_meshgrid_3d

class X(PreTrainedModel):

    main_input_name = "input_data"
    config_class = X_embedder.X_Config # TODO add

    def __init__(
            self,
            config,
    ):
        super().__init__(config)

        self.config = config
        self.T_out = math.ceil(self.config.sequence_info[1]/self.config.patch_time)
        self.T_in = math.ceil(self.config.sequence_info[0]/self.config.patch_time)
        self.num_prop = config.latent_time
        self.dim_t = config.latent_channels * 2 ** (len(config.depths)-1)

        self.embedder = X_embedder.X_Embedder(config)
        self.encoder = X_encoder.X_Encoder(config, self.embedder.patch_dimensions)
        self.one_step_propagator = X_propagator.X_Processor(config)
        self.decoder = X_decoder.X_Decoder(config, self.embedder.patch_dimensions)
        self.recovery = X_recovery.X_Recovery(config)
# 
        skip_connections = ResNetBlock

        self.residual_blocks_space = nn.ModuleList(
            [
                (
                    nn.ModuleList(
                        [
                            skip_connections(config=config, 
                                             input_resolution=(self.embedder.patch_dimensions[1] // 2 ** i, self.embedder.patch_dimensions[2] // 2 ** i), 
                                             dim=config.latent_channels * 2**(i))
                            for _ in range(depth)
                        ]
                    )
                    if depth > 0
                    else nn.ModuleList([nn.Identity()])
                )
                for i, depth in enumerate(config.skip_connections) # edit here
            ]
        )

        self.residual_blocks_time = nn.ModuleList(

            [

                skip_connections(config=config,
                                 dim=self.dim_t,
                                 time_res=True)
                for _ in range(config.skip_connections_time)
            ]


        )



        self.post_init()

    def forward(
            self,
            input_data,
            **kwargs
    ):
        
        if self.config.coord_features: 
                                                                                                          
            coord_feat = twod_meshgrid_3d(list(input_data.shape), input_data.device)
            input_data = torch.cat((input_data, coord_feat), dim=2)

        embedded_data, embedded_dims = self.embedder(input_data)
        encoder_output = self.encoder(input=embedded_data, input_dimensions=embedded_dims) # TODO give seq_len data from embedding already. 
                                                                                   # from the moment we have input + input_dim give compact version

        time_in = encoder_output[1]
        
        all_t_out = []
        
        for t in range(0, self.num_prop):

            time_in = self.one_step_propagator(time_in, counter=t)
            all_t_out.append(time_in)


        skip_states = list(encoder_output[2][1:]) 
        # The reshaped outputs of Encoder Stage, ignore Embedding and these outputs are the ones before PatchMerging.
        # Also, it includes the outputs of TimeBlocks.
        # skip_states = list(encoder_outputs[1][1:])
        
        for i in range(len(self.residual_blocks_space)):
            for block in self.residual_blocks_space[i]: # 2  blocks (last skip layer: identity)
                if isinstance(block, nn.Identity):
                    skip_states[i] = block(skip_states[i])
                else: # is not Identity
                    skip_states[i] = block(skip_states[i], **kwargs)

        for i, block in enumerate(self.residual_blocks_time): 
            skip_states[i+len(self.residual_blocks_space)] = block(skip_states[i+len(self.residual_blocks_space)], **kwargs)


        # input_dim_t = encoder_output[3][len(self.residual_blocks_space)].shape[2]
        input_dim_x = encoder_output[3][len(self.residual_blocks_space)].shape[3]
        input_dim_y = encoder_output[3][len(self.residual_blocks_space)].shape[4]

        decoder_output = self.decoder(
            all_t_out[-1],
            # torch.concatenate(all_t_out, dim=1), 
            (input_dim_x, input_dim_y),
            skip_states=skip_states[:-1],
            output_hidden_states=True,
            **kwargs
        )

        recovered = self.recovery(decoder_output)



        
        return recovered
    
    # output has
    # last_hidden_state_after_spatial_attention: (B*num_t_patch, num_x_patch*num_y_patch, C)
    # last_hidden_state_after_temporal_attention: (B*num_x_patch*num_y_patch, num_t_patch, C)
    # all_hidden_states: (num_stages + num_time_blocks) * [corresponding shape above]
    # all_reshaped_hidden_states: (num_stages + num_time_blocks) * [B, C, num_t_patch, num_x_patch, num_y_patch]

    # all_t_out is a list of predicted timesteps. EACH shape (B*num_x_patch*num_y_patch, 1, C) >> 

    # decoder_output has the same shape as encoder entry