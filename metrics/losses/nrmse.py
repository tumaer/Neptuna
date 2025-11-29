from typing import Optional, List, Dict, Union, Tuple

import torch
import torch.nn as nn

from ..training_metrics import LossComponent, WeightSchedule


class NRMSE(LossComponent):
    """
    Normalized RMSE:
        sqrt( <|u - v|^2> / (<|u|^2> + eps) )
    where <> is an average over the same dimensions as the loss reduction.
    """
    def __init__(
        self,
        weight: Union[float, WeightSchedule] = 1.0,
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        norm_stats: Dict[str, Dict[str, float]] = None,
        reduction: str = 'mean',
        epsilon: float = 1e-8
    ):
        super().__init__(
            weight=weight,
            name=name,
            data_dim=data_dim,
            field_names=field_names,
            norm_stats=norm_stats
        )
        self.reduction = reduction
        self.epsilon = epsilon

    def _apply_reduction(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        elif self.reduction == 'none':
            return x
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        return_detailed: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # u = labels, v = predictions
        diff = predictions - labels
        sq_error = diff ** 2

        # Get weight tensor with proper broadcasting
        weight_tensor = self.weight_schedule.get_weight(sq_error.shape).to(predictions.device)

        # Weighted squared error
        weighted_sq_error = sq_error * weight_tensor

        # Numerator: <|u - v|^2>
        num = self._apply_reduction(weighted_sq_error)

        # Denominator: <|u|^2>
        sq_u = labels ** 2
        weighted_sq_u = sq_u * weight_tensor
        denom = self._apply_reduction(weighted_sq_u)

        nrmse = torch.sqrt(num / (denom + self.epsilon) + self.epsilon)

        if not return_detailed:
            return nrmse

        detailed: Dict[str, torch.Tensor] = {}

        if labels.ndim >= 2:
            # Per-timestep
            dims_to_reduce = [0] + list(range(2, labels.ndim))
            num_t = weighted_sq_error.mean(dim=dims_to_reduce)
            denom_t = (weighted_sq_u).mean(dim=dims_to_reduce)
            detailed['per_timestep'] = torch.sqrt(
                num_t / (denom_t + self.epsilon) + self.epsilon
            ).detach()

        if labels.ndim >= 3:
            # Per-channel
            dims_to_reduce = [0, 1] + list(range(3, labels.ndim))
            num_c = weighted_sq_error.mean(dim=dims_to_reduce)
            denom_c = (weighted_sq_u).mean(dim=dims_to_reduce)
            detailed['per_channel'] = torch.sqrt(
                num_c / (denom_c + self.epsilon) + self.epsilon
            ).detach()

        return nrmse, detailed