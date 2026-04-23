"""
This Module will receive encoded data from Encoder, while features in space and time are extracted,
Now it will propagate in time. according to requested output timesteps, it will propagate.

Idea 1. generate all in once >> NOTE doing this first
Idea 2. generate one step at a time, and feed it back to the input for next step prediction.

Sources: KNO, XODE papers
"""
import math
import torch
from torch import nn

# Koopman 1D structure
class Koopman_Operator(nn.Module):

    def __init__(self,
                # config,
                dim, 
                modes = 16
        ):
        super().__init__()


        self.dim = dim # number of windows N in [B, N, C]
        self.scale = (1 / (dim * dim))
        self.modes = modes
        self.koopman_matrix = nn.Parameter(self.scale * torch.rand(dim, dim, self.modes, dtype=torch.cfloat))


    # Complex multiplication
    def time_marching(self, input, weights):
        # (batch, t, x), (t, t+1, x) -> (batch, t+1, x)
        return torch.einsum("btx,tfx->bfx", input, weights)
    

    def forward(self, x):
        # x.shape [B, _N, latent_dim]
        # Fourier Transform
        x_ft = torch.fft.rfft(x) # [B, _N, latent_dim//2+1]
        # Koopman Operator Time Marching
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device = x.device)
        out_ft[:, :, :self.modes] = self.time_marching(x_ft[:, :, :self.modes], self.koopman_matrix)
        #Inverse Fourier Transform
        x = torch.fft.irfft(out_ft) # shape [B, _N, latent_dim]

        return x

class X_Processor(nn.Module):

    def __init__(self, config, modes=16, decompose=4, linear_type=True, normalization=False): 
        # keep normalization False, since in the main code it ends up in BatchNorm which we do NOT do here. 
        # there are spatial patches in B dim when coming to propagator.

        super().__init__()

        # t_in = math.ceil(config.sequence_info[0] / config.patch_time)
        t_in = config.latent_time

        ###########################################################################

        # which one is correct? at each prop, we go one step ahead. so I should choose t_out = 1 for

        # t_out = math.ceil(config.sequence_info[1] / config.patch_time) 
        t_out = 1
        # t_out = config.latent_time >> TODO try

        #############################################################################


        op_size = config.operator_size # coming from paper as Hyperparameter
        self.decompose = decompose
        self.modes = modes
        self.linear_type = linear_type
        self.normalization = normalization

        self.lift = nn.Linear(t_in, op_size)
        # self.lift2 = nn.Linear(t_out, op_size)

        self.koopman_layer = Koopman_Operator(dim=op_size, modes = self.modes)

        self.w0 = nn.Conv1d(op_size, op_size, 1)

        self.project = nn.Linear(op_size, t_in)

        
        
    
    def forward(self, x, counter):

        # shape x : B, T, C

        # similar to FNO, we have two lines as KNO paper suggests: (coming from their code)

        # 1. Reconstruct; I need T to expand. so I permute and then pass to nn.Linear


        x_reconstruct = x.permute(0, 2, 1) # [B, C, T]
        x_reconstruct = self.lift(x_reconstruct)
        x_reconstruct = torch.tanh(x_reconstruct)
        x_reconstruct = self.project(x_reconstruct)


        # 2. Predict

        x = self.lift(x.permute(0, 2, 1))
        x = torch.tanh(x) 
        # in original KNO, after this tanh, a permute happens which brings feature dim to last position;
        # I need to permute again that C goes last and pass to koopman layers.
        
        x = x.permute(0, 2, 1) # B, T, C

        x_w = x
        for _ in range(self.decompose):

            x1 = self.koopman_layer(x)

            if self.linear_type:
                x = x + x1
            else:
                x = torch.tanh(x + x1)


            if self.normalization: # always off for my case
                x = torch.tanh(self.norm_layer(self.w0(x_w)) + x)
            else:
                x = torch.tanh(self.w0(x_w) + x)

        # permute back to pass to another nn.Linear
                
        x = x.permute(0, 2, 1)

        x = self.project(x) # I assume shape is B, C, T

        x = x.permute(0, 2, 1) # final shape B, T, C so it can comeback to loop safely


        return x

        # this forward only prop one time step. Inside the forward function of the full model, 
        # create a loop: for t in range(0, T_out): and run as much as you need. 
        # I assume T_out = math.ceil(config.sequence_info[1]/config_patch_time)