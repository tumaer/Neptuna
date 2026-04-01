"""
This file is a Koopman propagator for the AK3D model. 
The AK3DProp class implements the necessary methods to propagate the state of the AK3D model using an approximate of a Koopman operator. 
The class includes methods for initializing the propagator, computing the Koopman operator, and propagating the state over time. 

the input from last stage of encoder comes here with shape : B, T*H//2*W//2, latent_dim
I think I need attention outputs! I want to use attention output in time dim to propagate the state. 
two types: 3D attention + Koopman propagator or Factorized attention + Koopman-conditioned propagator.
output can be multiout or a single step prediction at a time.
"""



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
        x = torch.fft.irfft(out_ft, n=x.size(-1)) # shape [B, _N, latent_dim]
        return x


class AK_Prop(nn.Module):
    def __init__(self, 
                config,
                modes = 16, 
                decompose = 4, 
                linear_type = True, 
                normalization = False):
        super().__init__()
       
        # self.dim = config.latent_channels * 2 **(len(config.depths) - 1) # should be the same as the output of the encoder
        last_Tp = config.sequence_info[0] // config.patch_size[0] + config.sequence_info[0] % config.patch_size[0]
        last_Hp = (config.grid_resolution[0] // config.patch_size[1] +  config.grid_resolution[0] % config.patch_size[1]) // 2 ** (len(config.depths) - 1) 
        last_Wp = (config.grid_resolution[1] // config.patch_size[2] +  config.grid_resolution[1] % config.patch_size[2]) // 2 ** (len(config.depths) - 1)
        self.dim = last_Tp * last_Hp * last_Wp

        self.decompose = decompose
        self.modes = modes

        self.koopman_layer = Koopman_Operator(dim=self.dim, modes = self.modes)

        self.w0 = nn.Conv1d(self.dim, self.dim, 1)

        self.linear_type = linear_type # If this variable is False, activate function will be worked after Koopman Matrix
        
        self.normalization = normalization

        if self.normalization:
            self.norm_layer = torch.nn.BatchNorm2d(self.dim)


    def forward(self, x):
        
        x = torch.tanh(x)
        shortcut = x

        x_w = x

        for i in range(self.decompose):

            x1 = self.koopman_layer(x) # Koopman Operator

            if self.linear_type:
                x = x + x1
            else:
                x = torch.tanh(x + x1)

        if self.normalization:
            x = torch.tanh(self.norm_layer(self.w0(x_w)) + x)
        else:
            x = torch.tanh(self.w0(x_w) + x)


        # x = x.permute(0, 2, 1)

        out = x + shortcut

        return out

        # return x, x_reconstruct

# Koopman 2D structure
class Koopman_Operator2D(nn.Module):
    def __init__(self, 
                dim, 
                modes_x, 
                modes_y
                ):
        super().__init__()

        self.dim = dim # latent dimension at the end of encoder
        self.scale = (1 / (dim * dim))
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.koopman_matrix = nn.Parameter(self.scale * torch.rand(dim, dim, self.modes_x, self.modes_y, dtype=torch.cfloat))

    # Complex multiplication
    def time_marching(self, input, weights):
        # (batch, t, x,y ), (t, t+1, x,y) -> (batch, t+1, x,y)
        return torch.einsum("btxy,tfxy->bfxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Fourier Transform
        x_ft = torch.fft.rfft2(x)
        # Koopman Operator Time Marching
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device = x.device)
        out_ft[:, :, :self.modes_x, :self.modes_y] = self.time_marching(x_ft[:, :, :self.modes_x, :self.modes_y], self.koopman_matrix)
        out_ft[:, :, -self.modes_x:, :self.modes_y] = self.time_marching(x_ft[:, :, -self.modes_x:, :self.modes_y], self.koopman_matrix)
        #Inverse Fourier Transform
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


