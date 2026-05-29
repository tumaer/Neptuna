from typing import Dict, List, Optional
import torch

from ..loss_weighting_strategies import LossWeightingStrategyBase


class LearningRateAnnealing(LossWeightingStrategyBase):
    """
    Learning-rate annealing loss reweighting from:
    Wang et al. (2020) "Understanding and mitigating gradient pathologies in physics-informed neural networks".
    """

    def __init__(
        self,
        update_frequency: int = 1,
        use_gradients: bool = True,
        reference_key: Optional[str] = None,
        alpha: float = 0.9,
        min_weight: float = 0.0,
        max_weight: float = 1e6,
        epsilon: float = 1e-12,
        freeze_reference_weight: bool = True,
        # Aggregation across steps within an epoch:
        # - "last"  : use last recorded step (closest to paper's "current iterate")
        # - "mean"  : use mean over steps
        # - "max"   : use max over steps
        epoch_agg: str = "last",
    ):
        super().__init__(update_frequency=update_frequency, use_gradients=use_gradients)
        self.reference_key = reference_key
        self.alpha = float(alpha)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.epsilon = float(epsilon)
        self.freeze_reference_weight = bool(freeze_reference_weight)
        self.epoch_agg = str(epoch_agg).lower().strip()
        if self.epoch_agg not in ("last", "mean", "max"):
            raise ValueError(f"epoch_agg must be one of ('last','mean','max'), got: {epoch_agg}")

    def _clip(self, x: float) -> float:
        return max(self.min_weight, min(self.max_weight, x))

    def _as_tensor(self, xs: List[float]) -> torch.Tensor:
        if not xs:
            return torch.tensor([], dtype=torch.float32)
        return torch.tensor(xs, dtype=torch.float32)

    def _agg_epoch(self, xs: List[float]) -> float:
        """
        Aggregate a list of per-step scalars into one per-epoch scalar.
        """
        if not xs:
            return 0.0
        if self.epoch_agg == "last":
            return float(xs[-1])
        t = self._as_tensor(xs)
        if t.numel() == 0:
            return 0.0
        if self.epoch_agg == "mean":
            return float(t.mean())
        # self.epoch_agg == "max"
        return float(t.max())

    def step(
        self,
        epoch: int,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_stats_history: Optional[Dict[str, Dict[str, List[float]]]] = None,
    ) -> Optional[Dict[str, Dict]]:
        self.current_epoch = epoch

        # Save loss history
        for component_name, losses in loss_history.items():
            self.history.setdefault(component_name, []).append(losses.copy())

        # Cache grad stats history (nested) for analysis/inspection
        if grad_stats_history is not None:
            for component_name, stats_dict in grad_stats_history.items():
                if component_name not in self.grad_stats_history:
                    self.grad_stats_history[component_name] = {}
                if isinstance(stats_dict, dict):
                    for stat_name, stat_values in stats_dict.items():
                        self.grad_stats_history[component_name].setdefault(stat_name, []).append(stat_values.copy())

        # Flatten the needed per-loss-key histories for this epoch
        # Expect grad_stats_history like:
        # { loss_key: {"norm":[...], "max":[...], "mean_abs":[...], ...}, ... }
        grad_meanabs_history = None
        grad_max_history = None
        if grad_stats_history:
            grad_meanabs_history = {}
            grad_max_history = {}
            for loss_key, stats in grad_stats_history.items():
                if not isinstance(stats, dict):
                    continue

                # Prefer "mean_abs" if present; fall back to "norm"
                if "mean_abs" in stats and isinstance(stats["mean_abs"], list):
                    grad_meanabs_history[loss_key] = stats["mean_abs"]
                elif "norm" in stats and isinstance(stats["norm"], list):
                    grad_meanabs_history[loss_key] = stats["norm"]

                if "max" in stats and isinstance(stats["max"], list):
                    grad_max_history[loss_key] = stats["max"]

        if self.should_update(epoch):
            return self.compute_new_weights(
                loss_history=loss_history,
                current_weights=current_weights,
                grad_meanabs_history=grad_meanabs_history,
                grad_max_history=grad_max_history,
            )

        return None


    def _choose_reference_key(self, keys_available: List[str]) -> Optional[str]:
        """
        Pick the reference loss key used in the numerator.
        Preference order:
          1) self.reference_key if provided and present
          2) common PINN names if present
          3) first available key
        """
        if self.reference_key is not None and self.reference_key in keys_available:
            return self.reference_key

        for cand in ("Lr", "residual", "pde_residual", "physics", "PDE", "PDEResidual"):
            if cand in keys_available:
                return cand

        return keys_available[0] if keys_available else None

    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_meanabs_history: Optional[Dict[str, List[float]]] = None,
        grad_max_history: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Dict]:
        """
        Paper-faithful under per-epoch constraints:

        Numerator ~ max_theta |∇ L_ref|  ==> use grad_max_history[ref_key] aggregated per epoch
        Denominator ~ mean_theta |∇ L_i| ==> use grad_meanabs_history[loss_key] aggregated per epoch
        """
        if not grad_meanabs_history or not isinstance(grad_meanabs_history, dict):
            return {k: v.copy() for k, v in current_weights.items()}

        if not grad_max_history or not isinstance(grad_max_history, dict):
            # Without "max" stats we cannot match the paper numerator; safest is no-op.
            return {k: v.copy() for k, v in current_weights.items()}

        keys_available = [k for k in grad_meanabs_history.keys() if k in grad_max_history]
        ref_key = self._choose_reference_key(keys_available)
        if ref_key is None or ref_key not in grad_max_history:
            return {k: v.copy() for k, v in current_weights.items()}

        # Unweight the reference gradient max values
        ref_weight = self._get_previous_weight(ref_key, current_weights)
        if ref_weight <= self.epsilon:
            return {k: v.copy() for k, v in current_weights.items()}
        
        grad_max_ref_weighted = grad_max_history.get(ref_key, [])
        grad_max_ref_unweighted = [g / ref_weight for g in grad_max_ref_weighted]
        
        # Paper numerator: max_theta |∇ L_ref| (for the "current iterate")
        # Per-epoch approximation: aggregate stepwise max-theta stats within the epoch.
        g_ref_max = self._agg_epoch(grad_max_ref_unweighted)
        if g_ref_max <= self.epsilon:
            return {k: v.copy() for k, v in current_weights.items()}

        new_weight_scalars: Dict[str, float] = {}

        for loss_key, g_meanabs_steps in grad_meanabs_history.items():
            prev_w = self._get_previous_weight(loss_key, current_weights)

            if self.freeze_reference_weight and loss_key == ref_key:
                new_weight_scalars[loss_key] = prev_w
                continue

            # Unweight the gradient mean absolute values
            if prev_w <= self.epsilon:
                new_weight_scalars[loss_key] = prev_w
                continue
            
            g_meanabs_unweighted = [g / prev_w for g in g_meanabs_steps]

            # Paper denominator: mean_theta |∇ L_i|
            g_i_meanabs = self._agg_epoch(g_meanabs_unweighted)
            denom = max(float(g_i_meanabs), self.epsilon)

            w_hat = float(g_ref_max / denom)
            w_new = (1.0 - self.alpha) * float(prev_w) + self.alpha * float(w_hat)
            new_weight_scalars[loss_key] = self._clip(w_new)

        return self._reconstruct_weight_dict(new_weight_scalars, current_weights)
