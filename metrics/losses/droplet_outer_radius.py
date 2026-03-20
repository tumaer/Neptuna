from typing import Dict, List, Optional, Tuple, Union, Literal

import torch
from torch import nn

from ..loss_framework import LossComponent, WeightSchedule, NormalizationHelper


class DropletOuterRadius(LossComponent):
	"""
	Droplet outer-radius error metric derived from density.
	  1) Extract density field.
	  2) Slice a configurable boundary edge (west/east/south/north).
	  3) Compute `axis_diff = axis[..., :1] - axis[..., :-1]`.
	  4) Find the last positive/negative block start index along the remaining axis.
	  5) Convert to normalized radius by dividing by axis length.

	Supports L1 and L2-style reductions over the resulting radius difference.
	"""

	def __init__(
		self,
		norm_helper: NormalizationHelper,
		weight: Union[float, WeightSchedule] = 1.0,
		name: Optional[str] = None,
		data_dim: int = None,
		field_names: List[str] = None,
		density_key: str = "Density",
		grid_resolution: Optional[Union[int, float, List[Union[int, float]], Tuple[Union[int, float], ...]]] = None,
		edge: Literal["west", "east", "south", "north"] = "west",
		direction: Literal["positive", "negative", "postive"] = "positive",
		error_mode: Literal["l1", "l2"] = "l2",
		clamp_negative_predictions: bool = True,
		epsilon: float = 1e-8,
	):
		super().__init__(
			weight=weight,
			name=name or "droplet_outer_radius",
			data_dim=data_dim,
			field_names=field_names,
			norm_helper=norm_helper,
		)

		self.density_key = density_key
		self.grid_resolution = grid_resolution
		self.edge = edge
		self.error_mode = error_mode
		self.clamp_negative_predictions = clamp_negative_predictions
		self.epsilon = epsilon
		self.direction =  direction

		if self.error_mode not in ("l1", "l2"):
			raise ValueError(f"DropletOuterRadius: unsupported error_mode '{error_mode}'. Use 'l1' or 'l2'.")

		if self.edge not in ("west", "east", "south", "north"):
			raise ValueError(
				f"DropletOuterRadius: unsupported edge '{self.edge}'. "
				"Use one of: west, east, south, north."
			)

		if self.direction not in ("positive", "negative"):
			raise ValueError(
				f"DropletOuterRadius: unsupported direction '{direction}'. "
				"Use 'positive' or 'negative'."
			)

		if self.field_names is not None and self.density_key not in self.field_names:
			raise ValueError(
				f"DropletOuterRadius: density_key '{self.density_key}' not found in field_names={self.field_names}."
			)

	@staticmethod
	def _idx_finder(x: torch.Tensor, direction: str = "positive") -> torch.Tensor:
		"""
		Find the last directional block-start index along the last dimension.

		Returns -1 when no positive block exists.
		"""
		if direction == "positive":
			mask = x > 0
		elif direction == "negative":
			mask = x < 0
		else:
			raise ValueError(f"DropletOuterRadius: invalid direction '{direction}'.")

		prev = torch.cat(
			[torch.zeros_like(mask[..., :1], dtype=torch.bool), ~mask[..., :-1]],
			dim=-1,
		)
		block_starts = mask & prev

		idx_last = block_starts.shape[-1] - 1 - torch.argmax(
			block_starts.flip(dims=[-1]).to(torch.int64),
			dim=-1,
		)

		has_block = block_starts.any(dim=-1)
		minus_one = torch.full_like(idx_last, -1)
		return torch.where(has_block, idx_last, minus_one)

	def _extract_density(self, tensor: torch.Tensor) -> torch.Tensor:
		"""Extract density field with shape (B, T, 1, H, W)."""
		if self.field_names is None:
			raise ValueError("DropletOuterRadius requires `field_names` to be set.")

		if tensor.ndim != 5:
			raise ValueError(
				f"DropletOuterRadius expects tensor shape (B, T, C, H, W). Got {tuple(tensor.shape)}."
			)

		try:
			density_idx = self.field_names.index(self.density_key)
		except ValueError as exc:
			raise ValueError(
				f"DropletOuterRadius: density_key '{self.density_key}' not found in field_names={self.field_names}."
			) from exc

		return tensor[:, :, density_idx:density_idx + 1, :, :]

	def _get_density_channel_index(self) -> Optional[int]:
		"""Return density channel index in `field_names`, if available."""
		if self.field_names is None:
			return None
		try:
			return self.field_names.index(self.density_key)
		except ValueError:
			return None

	def _get_density_loss_weight(self, elem_shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
		"""
		Build broadcastable weight tensor for elem shape (B, T, 1),
		using only the density channel weight (if channel weights exist).
		"""
		if len(elem_shape) != 3:
			raise ValueError(
				f"DropletOuterRadius: expected elem shape (B, T, 1), got {tuple(elem_shape)}."
			)

		weight = torch.tensor(self.weight_schedule.base_weight, device=device, dtype=dtype).view(1, 1, 1)

		if self.weight_schedule.timestep_weights is not None:
			t = self.weight_schedule.timestep_weights.to(device=device, dtype=dtype).view(1, -1, 1)
			weight = weight * t

		if self.weight_schedule.channel_weights is not None:
			density_idx = self._get_density_channel_index()
			if density_idx is not None:
				cw = self.weight_schedule.channel_weights.to(device=device, dtype=dtype)
				if density_idx < 0 or density_idx >= cw.numel():
					raise ValueError(
						f"DropletOuterRadius: density channel index {density_idx} out of range "
						f"for channel_weights of length {cw.numel()}."
					)
				weight = weight * cw[density_idx].view(1, 1, 1)

		return weight

	def _outer_radius_from_density(self, rho: torch.Tensor) -> torch.Tensor:
		"""
		Compute outer radius proxy from density.

		Input shape:  (B, T, 1, H, W)
		Output shape: (B, T, 1)
		"""
		if rho.ndim != 5 or rho.shape[2] != 1:
			raise ValueError(
				f"DropletOuterRadius expects rho shape (B, T, 1, H, W). Got {tuple(rho.shape)}."
			)

		# Edge convention matches IntegralConservationRMSE:
		# west/east = x-min/x-max, south/north = y-min/y-max.
		if self.edge == "west":
			axis_slice = rho[..., 0]      # x-min -> (B, T, 1, H)
		elif self.edge == "east":
			axis_slice = rho[..., -1]     # x-max -> (B, T, 1, H)
		elif self.edge == "south":
			axis_slice = rho[..., 0, :]   # y-min -> (B, T, 1, W)
		else:  # north
			axis_slice = rho[..., -1, :]  # y-max -> (B, T, 1, W)

		axis_diff = axis_slice[..., :1] - axis_slice[..., :-1]  # (B, T, 1, H-1)

		idx = self._idx_finder(axis_diff, direction=self.direction).to(dtype=rho.dtype)

		# Infer normalization scale directly from tensor geometry
		axis_len = float(axis_slice.shape[-1])
		if axis_len <= 0:
			raise ValueError(
				f"DropletOuterRadius: inferred axis length must be > 0, got {axis_len}."
			)

		return idx / axis_len

	def forward(
		self,
		model: nn.Module,
		predictions: torch.Tensor,
		labels: torch.Tensor,
		input_frames: Optional[torch.Tensor],
		return_detailed: bool = False,
		keep_bc_dims: bool = False,
		preserve_component_grads: bool = False,
	) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
		if predictions.shape != labels.shape:
			raise ValueError(
				f"DropletOuterRadius: predictions and labels must have same shape. "
				f"Got {tuple(predictions.shape)} vs {tuple(labels.shape)}."
			)

		# Match source metric behavior: compute on denormalized density.
		pred_fields = self.norm_helper.denormalize_to_fields(predictions)
		true_fields = self.norm_helper.denormalize_to_fields(labels)

		if self.density_key not in pred_fields or self.density_key not in true_fields:
			raise ValueError(
				f"DropletOuterRadius: missing density field '{self.density_key}' in denormalized tensors."
			)

		pred_rho = pred_fields[self.density_key].unsqueeze(2)  # (B, T, 1, H, W)
		true_rho = true_fields[self.density_key].unsqueeze(2)  # (B, T, 1, H, W)

		if self.clamp_negative_predictions:
			pred_rho = torch.clamp_min(pred_rho, 0.0)

		pred_radius = self._outer_radius_from_density(pred_rho)
		true_radius = self._outer_radius_from_density(true_rho)

		diff = pred_radius - true_radius  # (B, T, 1)

		if self.error_mode == "l1":
			elem = torch.abs(diff)
		else:
			elem = diff ** 2

		weight_tensor = self._get_density_loss_weight(
			elem.shape,
			device=elem.device,
			dtype=elem.dtype,
		)
		weighted = elem * weight_tensor

		# Keep batch/channel dims for rollout metrics
		if keep_bc_dims:
			reduce_dims = [1]  # reduce time only -> (B, 1)
		else:
			reduce_dims = list(range(1, weighted.ndim))  # per-sample scalar

		per_sample = weighted.mean(dim=reduce_dims)
		if self.error_mode == "l2":
			per_sample = torch.sqrt(per_sample + self.epsilon)

		total_loss = per_sample if keep_bc_dims else per_sample.mean()

		if not return_detailed:
			return total_loss

		weighted_for_detailed = weighted if preserve_component_grads else weighted.detach()
		detailed: Dict[str, torch.Tensor] = {}

		# Per timestep: reduce batch + channel
		per_timestep = weighted_for_detailed.mean(dim=(0, 2))
		if self.error_mode == "l2":
			per_timestep = torch.sqrt(per_timestep + self.epsilon)
		detailed["per_timestep"] = per_timestep if preserve_component_grads else per_timestep.detach()

		# Per channel: reduce batch + time (channel dimension is size 1)
		per_channel = weighted_for_detailed.mean(dim=(0, 1))
		if self.error_mode == "l2":
			per_channel = torch.sqrt(per_channel + self.epsilon)
		detailed["per_channel"] = per_channel if preserve_component_grads else per_channel.detach()

		# Optional diagnostics
		detailed["mean_pred_radius"] = pred_radius.mean().detach()
		detailed["mean_true_radius"] = true_radius.mean().detach()

		return total_loss, detailed
