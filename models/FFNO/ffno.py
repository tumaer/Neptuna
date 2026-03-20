import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel

from utils.grid_utils import oned_meshgrid, twod_meshgrid, threed_meshgrid
from .ffno_utils import FFNOConfig, SpectralConv1d, SpectralConv2d, SpectralConv3d


class FFNO(PreTrainedModel):
	"""Factorized Fourier neural operator (FFNO) model.
	
	Adapted from the REALM repository (https://github.com/deepflame-ai/REALM)
	"""

	main_input_name = "input_data"
	conditioning_input_name = "conditioning_input_data"
	config_class = FFNOConfig

	def __init__(self, config) -> None:
		super().__init__(config)
		self.config = config
		self.ffno = self.build_FFNO()(config=config)

	def build_FFNO(self):
		if self.config.dimension == 1:
			return FFNO1D
		elif self.config.dimension == 2:
			return FFNO2D
		elif self.config.dimension == 3:
			return FFNO3D
		else:
			raise NotImplementedError(
				"Invalid dimensionality. Only 1D, 2D and 3D FFNO implemented"
			)

	def _build_coords(self, x: Tensor) -> Tensor:
		if not self.config.coord_features:
			shape = [1, 0, *x.shape[2:]]
			return x.new_zeros(shape)

		if self.config.dimension == 1:
			shape_1d = [x.shape[0], x.shape[1], x.shape[-1]]
			coord_1d = oned_meshgrid(shape_1d, x.device)
			return coord_1d[:1]
		elif self.config.dimension == 2:
			return twod_meshgrid(list(x.shape), x.device)[:1]
		elif self.config.dimension == 3:
			return threed_meshgrid(list(x.shape), x.device)[:1]
		else:
			raise NotImplementedError

	def forward(self, input_data: Tensor, **kwargs) -> Tensor:
		if "conditioning_input_data" in kwargs:
			conditioning_input_data = kwargs["conditioning_input_data"]
			input_data = torch.cat([input_data, conditioning_input_data], dim=2)

		batch, input_seq, input_channels, *spatial = input_data.shape
		x = input_data.reshape(batch, input_seq * input_channels, *spatial)

		coords = self._build_coords(x)
		return self.ffno(x, coords, **kwargs)


class FFNO1D(PreTrainedModel):
	"""1D FFNO."""

	def __init__(self, config) -> None:
		super().__init__(config)

		self.width = config.latent_channels
		self.in_dim = config.in_size
		self.out_dim = config.out_size
		self.n_layers = config.num_fno_layers

		if isinstance(config.num_fno_modes, int):
			self.modes1 = config.num_fno_modes
		else:
			self.modes1 = config.num_fno_modes[0]

		self.in_proj = nn.Linear(self.in_dim, self.width)
		self.spectral_layers = nn.ModuleList([])
		for _ in range(self.n_layers):
			self.spectral_layers.append(
				SpectralConv1d(
					in_dim=self.width,
					out_dim=self.width,
					n_modes1=self.modes1,
					factor=config.factor,
					n_ff_layers=config.n_ff_layers,
					layer_norm=config.layer_norm,
					config=config,
				)
			)

		self.out = nn.Sequential(
			nn.Linear(self.width, 128),
			nn.GELU(),
			nn.Linear(128, self.out_dim),
		)

	def forward(self, x, coords, **kwargs):
		batch_size = x.shape[0]
		x = torch.cat((x, coords.repeat(batch_size, 1, 1)), dim=1)
		x = self.in_proj(x.permute(0, 2, 1))
		x = F.gelu(x)
		for i in range(self.n_layers):
			layer = self.spectral_layers[i]
			b = layer(x, **kwargs)
			x = x + b
		x = self.out(x).permute(0, 2, 1)
		return x


class FFNO2D(PreTrainedModel):
	"""2D FFNO."""

	def __init__(self, config) -> None:
		super().__init__(config)

		if isinstance(config.num_fno_modes, int):
			self.modes1 = self.modes2 = config.num_fno_modes
		else:
			self.modes1, self.modes2 = config.num_fno_modes

		self.width = config.latent_channels
		self.in_dim = config.in_size
		self.out_dim = config.out_size
		self.n_layers = config.num_fno_layers

		self.in_proj = nn.Linear(self.in_dim, self.width)

		self.spectral_layers = nn.ModuleList([])
		for _ in range(self.n_layers):
			self.spectral_layers.append(
				SpectralConv2d(
					in_dim=self.width,
					out_dim=self.width,
					n_modes1=self.modes1,
					n_modes2=self.modes2,
					factor=config.factor,
					n_ff_layers=config.n_ff_layers,
					layer_norm=config.layer_norm,
					config=config,
				)
			)
		self.out = nn.Sequential(
			nn.Linear(self.width, 128),
			nn.GELU(),
			nn.Linear(128, self.out_dim),
		)

	def forward(self, x, coords, **kwargs):
		batch_size = x.shape[0]
		x = torch.cat((x, coords.repeat(batch_size, 1, 1, 1)), dim=1)
		x = self.in_proj(x.permute(0, 2, 3, 1))
		x = F.gelu(x)
		for i in range(self.n_layers):
			layer = self.spectral_layers[i]
			b = layer(x, **kwargs)
			x = x + b
		x = self.out(x).permute(0, 3, 1, 2)
		return x


class FFNO3D(PreTrainedModel):
	"""3D FFNO."""

	def __init__(self, config) -> None:
		super().__init__(config)

		if isinstance(config.num_fno_modes, int):
			self.modes1 = self.modes2 = self.modes3 = config.num_fno_modes
		else:
			self.modes1 = config.num_fno_modes[0]
			self.modes2 = config.num_fno_modes[1]
			self.modes3 = config.num_fno_modes[2]

		self.width = config.latent_channels
		self.in_dim = config.in_size
		self.out_dim = config.out_size
		self.n_layers = config.num_fno_layers

		self.in_proj = nn.Linear(self.in_dim, self.width)

		self.spectral_layers = nn.ModuleList([])
		for _ in range(self.n_layers):
			self.spectral_layers.append(
				SpectralConv3d(
					self.width,
					self.width,
					self.modes1,
					self.modes2,
					self.modes3,
					factor=config.factor,
					n_ff_layers=config.n_ff_layers,
					layer_norm=config.layer_norm,
					config=config,
				)
			)
		self.out = nn.Sequential(
			nn.Linear(self.width, 128),
			nn.GELU(),
			nn.Linear(128, self.out_dim),
		)

	def forward(self, x, coords, **kwargs):
		batch_size = x.shape[0]
		if coords.shape[0] != batch_size:
			x = torch.cat((x, coords.repeat(batch_size, 1, 1, 1, 1)), dim=1)
		else:
			x = torch.cat((x, coords), dim=1)
		x = self.in_proj(x.permute(0, 2, 3, 4, 1))
		x = F.gelu(x)
		for i in range(self.n_layers):
			layer = self.spectral_layers[i]
			b = layer(x, **kwargs)
			x = x + b
		x = self.out(x).permute(0, 4, 1, 2, 3)
		return x
