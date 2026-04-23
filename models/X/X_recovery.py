import math
import torch
from torch import nn

class X_Recovery(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.T_in = math.ceil(config.sequence_info[0]/config.patch_time)
        self.T_out = math.ceil(config.sequence_info[1]/config.patch_time)
        self.t_out_real = config.sequence_info[1] 

        self.out_channels, self.C_embedd = config.out_channels, config.latent_channels 
        

        self.patch = [config.patch_time, config.patch_space[0], config.patch_space[1]]
        self.resolution_out =[config.sequence_info[1], config.grid_resolution[0], config.grid_resolution[1]]

        self.T_patch_out = math.ceil(self.resolution_out[0]/self.patch[0])
        self.X_patch = math.ceil(self.resolution_out[1]/self.patch[1])
        self.Y_patch = math.ceil(self.resolution_out[2]/self.patch[2])
        self.T_embed = config.latent_time

   
        self.conv_t = nn.ConvTranspose3d(
            in_channels=self.T_embed, 
            out_channels=self.T_patch_out, 
            kernel_size=1,
            stride=1
        ) 
       

        self.projection = nn.ConvTranspose3d(
            in_channels=self.C_embedd,
            out_channels=self.out_channels, 
            kernel_size=(self.patch[0], self.patch[1], self.patch[2]), 
            stride=(self.patch[0], self.patch[1], self.patch[2])
        )


        # the following is not done in Pangu # copied from Poseidon
        self.mixup = nn.Conv3d(
            in_channels=self.out_channels , 
            out_channels=self.out_channels ,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )

    def maybe_crop_3d(self, input_data, resolution):
        T = resolution[0]
        X = resolution[1]
        Y = resolution[2]
        if input_data.shape[2] > T:
            input_data = input_data[:, :, :T, :, :]
        if input_data.shape[3] > X:
            input_data = input_data[:, :, :, :X, :]
        if input_data.shape[4] > Y:
            input_data = input_data[:, :, :, :, :Y]
        return input_data



    def forward(self, hidden_states): 
        
        
        # Hidden_state shape : B, C, T, H, W : 8, 27, 3, 64, 64 >> Output shape 8, 1, 1, 256, 256 (out_channel =1 and out_seq_len=1)
        # hidden_states = hidden_states.transpose(1, 2) # [16, 48, 1024]
        # hidden_states = hidden_states.reshape(
        #     hidden_states.shape[0], hidden_states.shape[1], *self.grid_size
        # ) # [16, 48, 32, 32]


        # hidden_states shape: B*T_embedd, X_patch*Y_patch, C_embedd
        B_, seq_len, C = hidden_states.shape

        hidden_states = hidden_states.reshape(-1, self.T_embed, self.X_patch, self.Y_patch, C).permute(0, 1, 4, 2, 3) # B, T_embedd, C_embedd, X_patch, Y_patch

        hidden_states = self.conv_t(hidden_states) # B, T_patch, C_embedd, X_patch, Y_patch

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4) # B, C_embedd, T_patch, X_patch, Y_patch

        output = self.projection(hidden_states) # B, C_out, T, X, Y ?

        output = self.maybe_crop_3d(output, self.resolution_out) # check if last two dimensions have the expected dim, otherwise crop

        output = self.mixup(output)

        output = output.permute(0, 2, 1, 3, 4)

        return output