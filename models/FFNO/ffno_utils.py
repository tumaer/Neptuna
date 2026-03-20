from types import SimpleNamespace
from typing import List, Sequence, Union

import torch
import torch.nn as nn

from utils.model_utils import CustomNorm, PretrainedConfig


class FFNOConfig(PretrainedConfig):
	"""Factorized Fourier neural operator (FFNO) model config."""

	model_type = "FFNO"

	def __init__(
		self,
		num_fno_modes: Union[int, List[int]] = 16,
		num_fno_layers: int = 4,
		factor: int = 4,
		n_ff_layers: int = 2,
		layer_norm: bool = True,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.num_fno_modes = num_fno_modes
		self.num_fno_layers = num_fno_layers
		self.factor = factor
		self.n_ff_layers = n_ff_layers
		self.layer_norm = layer_norm


def _expand_modes(num_modes: Union[int, Sequence[int]], dimension: int) -> List[int]:
	if isinstance(num_modes, int):
		return [num_modes] * dimension
	modes = list(num_modes)
	if len(modes) != dimension:
		raise ValueError(
			f"Expected {dimension} Fourier modes, got {len(modes)}: {modes}"
		)
	return modes


def build_source_args(config: FFNOConfig) -> SimpleNamespace:
	return SimpleNamespace(
		modes=_expand_modes(config.num_fno_modes, config.dimension),
		width=config.latent_channels,
		in_dim=config.in_size,
		out_dim=config.out_size,
		num_chemical=config.out_channels,
		n_layers=config.num_fno_layers,
		factor=config.factor,
		n_ff_layers=config.n_ff_layers,
		layer_norm=config.layer_norm,
	)


class FeedForward(nn.Module):
    def __init__(self, dim, factor, n_layers, layer_norm, config, array_length):
        super().__init__()
        if layer_norm:
            self.norm = CustomNorm(
                config=config,
                num_channels=dim,
                array_length=array_length,
                channel_at_last_position=True,
            )
        else:
            self.norm = nn.Identity()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
            ))

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x, **kwargs)
        return x


class SpectralConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes1, factor=4, n_ff_layers=2, layer_norm=True, config=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes1 = n_modes1

        self.fourier_weight = nn.ParameterList([])

        weight1 = torch.FloatTensor(in_dim, out_dim, n_modes1, 2)
        param1 = nn.Parameter(weight1)
        nn.init.xavier_normal_(param1)
        self.fourier_weight.append(param1)

        if config is None:
            raise ValueError("FFNO SpectralConv1d requires config for CustomNorm")
        self.backcast_ff = FeedForward(out_dim, factor, n_ff_layers, layer_norm, config=config, array_length=3)

    def forward(self, x, **kwargs):
        x = self.forward_fourier(x)
        b = self.backcast_ff(x, **kwargs)
        return b

    def forward_fourier(self, x):
        x = x.permute(0, 2, 1)
        B, _, M = x.shape

        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft = x_ft.new_zeros(B, self.out_dim, M // 2 + 1)

        out_ft[:, :, :self.n_modes1] = torch.einsum(
            "bix,iox->box",
            x_ft[:, :, :self.n_modes1],
            torch.view_as_complex(self.fourier_weight[0]),
        )

        xx = torch.fft.irfft(out_ft, n=M, dim=-1, norm='ortho')
        x = xx.permute(0, 2, 1)
        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes1, n_modes2, factor=4,
                 n_ff_layers=2, layer_norm=True, config=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes1 = n_modes1
        self.n_modes2 = n_modes2

        self.fourier_weight = nn.ParameterList([])

        weight1 = torch.FloatTensor(in_dim, out_dim, n_modes2, 2)
        param1 = nn.Parameter(weight1)
        nn.init.xavier_normal_(param1)
        self.fourier_weight.append(param1)

        weight2 = torch.FloatTensor(in_dim, out_dim, n_modes1, 2)
        param2 = nn.Parameter(weight2)
        nn.init.xavier_normal_(param2)
        self.fourier_weight.append(param2)

        if config is None:
            raise ValueError("FFNO SpectralConv2d requires config for CustomNorm")
        self.backcast_ff = FeedForward(out_dim, factor, n_ff_layers, layer_norm, config=config, array_length=4)

    def forward(self, x, **kwargs):
        x = self.forward_fourier(x)
        b = self.backcast_ff(x, **kwargs)
        return b

    def forward_fourier(self, x):
        x = x.permute(0, 3, 1, 2)
        B, I, M, N = x.shape
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft = x_fty.new_zeros(B, self.out_dim, M, N//2+1)

        out_ft[:, :, :, :self.n_modes2] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty[:, :, :, :self.n_modes2],
                torch.view_as_complex(self.fourier_weight[0]))

        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        out_ft = x_ftx.new_zeros(B, self.out_dim, M//2+1, N)
        out_ft[:, :, :self.n_modes1, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx[:, :, :self.n_modes1, :],
                torch.view_as_complex(self.fourier_weight[1]))

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')
        x = xx + xy
        x = x.permute(0, 2, 3, 1)
        return x

class SpectralConv3d(nn.Module):
    def __init__(self, in_dim, out_dim, modes1, modes2, modes3, factor=4,
                 n_ff_layers=2, layer_norm=True, config=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes1 = modes1
        self.n_modes2 = modes2
        self.n_modes3 = modes3
        n_modes = [modes1, modes2, modes3]
  
        self.fourier_weight = nn.ParameterList([])
        for i in range(3):
            weight = torch.FloatTensor(in_dim, out_dim, n_modes[i], 2)
            param = nn.Parameter(weight)
            nn.init.xavier_normal_(param)
            self.fourier_weight.append(param)

        if config is None:
            raise ValueError("FFNO SpectralConv3d requires config for CustomNorm")
        self.backcast_ff = FeedForward(out_dim, factor, n_ff_layers, layer_norm, config=config, array_length=5)

    def forward(self, x, **kwargs):
        x = self.forward_fourier(x)
        b = self.backcast_ff(x, **kwargs)
        return b

    def forward_fourier(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        B, I, M, N, Z = x.shape

        x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')
        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N, Z)
        out_ft[:, :, :self.n_modes1, :, :] = torch.einsum(
                "bixyz,iox->boxyz", x_ftx[:, :, :self.n_modes1, :, :],
                torch.view_as_complex(self.fourier_weight[0]))
        xx = torch.fft.irfft(out_ft, n=M, dim=-3, norm='ortho')

        x_fty = torch.fft.rfft(x, dim=-2, norm='ortho')
        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1, Z)
        out_ft[:, :, :, :self.n_modes2, :] = torch.einsum(
                "bixyz,ioy->boxyz", x_fty[:, :, :, :self.n_modes2, :],
                torch.view_as_complex(self.fourier_weight[1]))
        xy = torch.fft.irfft(out_ft, n=N, dim=-2, norm='ortho')
        
        x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft = x_ftz.new_zeros(B, I, M, N, Z // 2 + 1)
        out_ft[:, :, :, :, :self.n_modes3] = torch.einsum(
                "bixyz,ioz->boxyz", x_ftz[:, :, :, :, :self.n_modes3],
                torch.view_as_complex(self.fourier_weight[2]))
        xz = torch.fft.irfft(out_ft, n=Z, dim=-1, norm='ortho')

        x = xx + xy + xz
        x = x.permute(0, 2, 3, 4, 1)
        return x