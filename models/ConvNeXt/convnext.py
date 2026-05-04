"""
Copied and adpated from https://github.com/facebookresearch/ConvNeXt
"""

import torch
from torch import Tensor
from transformers import PreTrainedModel

from utils.grid_utils import oned_meshgrid, threed_meshgrid, twod_meshgrid
from .convnext_utils import (
	ConvNeXtConfig,
	ConvNeXt1D,
	ConvNeXt2D,
	ConvNeXt3D,
	build_source_args,
)


class ConvNeXt(PreTrainedModel):
	"""ConvNeXt model ("A ConvNet for the 2020s", Liu et al. 2022)
	
	Adapted from the REALM repository (https://github.com/deepflame-ai/REALM)
	"""

	main_input_name = "input_data"
	conditioning_input_name = "conditioning_input_data"
	config_class = ConvNeXtConfig

	def __init__(self, config) -> None:
		super().__init__(config)
		self.config = config

		source_args = build_source_args(config)
		self.convnext = self.build_convnext()(source_args, config)

	def build_convnext(self):
		if self.config.dimension == 1:
			return ConvNeXt1D
		elif self.config.dimension == 2:
			return ConvNeXt2D
		elif self.config.dimension == 3:
			return ConvNeXt3D
		else:
			raise NotImplementedError(
				"Invalid dimensionality. Only 1D, 2D, 3D ConvNeXt implemented"
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

		if self.config.dimension == 1:
			coords = self._build_coords(x)
			out = self.convnext(x, coords, **kwargs)
			return out

		coords = self._build_coords(x)
		return self.convnext(x, coords, **kwargs)

