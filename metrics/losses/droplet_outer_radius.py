from typing import Dict, List, Optional, Tuple, Union, Literal

import torch
from torch import nn

from ..loss_framework import LossComponent, WeightSchedule, NormalizationHelper


class DropletOuterRadius(LossComponent):
	"""
	Droplet outer-radius error metric derived from density.
	  1) Extract density field.
	  2) Slice a configurable boundary edge (west/east/south/north).
	  3) Compute local density jumps along the edge axis.
	  4) Find the two strongest jump points and keep the farthest point.
	  5) Convert to normalized radius by dividing by axis length.

	Supports MAE and RMSE-style reductions over the resulting radius difference.
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
		metric_mode: Literal["mae", "rmse"] = "rmse",
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
		self.metric_mode = str(metric_mode).lower()
		self.clamp_negative_predictions = clamp_negative_predictions
		self.epsilon = epsilon

		if self.metric_mode not in ("mae", "rmse"):
			raise ValueError(
				f"DropletOuterRadius: unsupported metric_mode '{self.metric_mode}'. "
				"Use 'mae' or 'rmse'."
			)

		if self.edge not in ("west", "east", "south", "north"):
			raise ValueError(
				f"DropletOuterRadius: unsupported edge '{self.edge}'. "
				"Use one of: west, east, south, north."
			)

		if self.field_names is not None and self.density_key not in self.field_names:
			raise ValueError(
				f"DropletOuterRadius: density_key '{self.density_key}' not found in field_names={self.field_names}."
			)

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

		axis_len = axis_slice.shape[-1]
		if axis_len <= 0:
			raise ValueError(
				f"DropletOuterRadius: inferred axis length must be > 0, got {axis_len}."
			)
		if axis_len == 1:
			# Single-point axis has no jump; radius defaults to edge index 0.
			idx = torch.zeros_like(axis_slice[..., 0], dtype=rho.dtype)
			return idx / float(axis_len)

		# Compute local jump magnitudes between adjacent samples.
		axis_jump = torch.abs(axis_slice[..., 1:] - axis_slice[..., :-1])  # (B, T, 1, L-1)
		k = min(2, axis_jump.shape[-1])
		topk_indices = torch.topk(axis_jump, k=k, dim=-1, largest=True).indices

		# Convert jump index i (between i and i+1) to point index i+1, then take farthest point.
		candidate_points = topk_indices + 1
		idx = candidate_points.max(dim=-1).values.to(dtype=rho.dtype)

		return idx / float(axis_len)

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

		if self.metric_mode == "mae":
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

		per_sample = weighted.mean(dim=reduce_dims)  #per_sample -> (B,)
		if self.metric_mode == "rmse":
			per_sample = torch.sqrt(per_sample)

		total_loss = per_sample if keep_bc_dims else per_sample.mean()

		if not return_detailed:
			return total_loss

		weighted_for_detailed = weighted if preserve_component_grads else weighted.detach()
		detailed: Dict[str, torch.Tensor] = {}

		# Per timestep: reduce batch + channel
		per_timestep = weighted_for_detailed.mean(dim=(0, 2))
		if self.metric_mode == "rmse":
			per_timestep = torch.sqrt(per_timestep)
		detailed["per_timestep"] = per_timestep if preserve_component_grads else per_timestep.detach()

		# Per channel: reduce batch + time (channel dimension is size 1)
		per_channel = weighted_for_detailed.mean(dim=(0, 1))
		if self.metric_mode == "rmse":
			per_channel = torch.sqrt(per_channel)
		detailed["per_channel"] = per_channel if preserve_component_grads else per_channel.detach()

		# Optional diagnostics
		detailed["mean_pred_radius"] = pred_radius.mean().detach()
		detailed["mean_true_radius"] = true_radius.mean().detach()

		return total_loss, detailed
