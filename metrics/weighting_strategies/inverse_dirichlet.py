from typing import Dict, List, Optional
import torch

from ..loss_weighting_strategies import LossWeightingStrategyBase


class InverseDirichlet(LossWeightingStrategyBase):
    """
    Inverse-Dirichlet weighting from:
    Maddu et al. (2021) "Inverse-Dirichlet Weighting Enables Reliable Training of Physics Informed Neural Networks".
    """

    def __init__(
        self,
        update_frequency: int = 1,
        use_gradients: bool = True,
        alpha: float = 0.5,
        min_weight: float = 1e-6,
        max_weight: float = 1e6,
        epsilon: float = 1e-12,
        variance_proxy: str = "second_moment",  # or "variance"
        freeze_reference_weight: bool = False,
        reference_key: Optional[str] = None,
    ):
        super().__init__(update_frequency=update_frequency, use_gradients=use_gradients)
        self.alpha = float(alpha)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.epsilon = float(epsilon)

        if variance_proxy not in ("second_moment", "variance"):
            raise ValueError("variance_proxy must be 'second_moment' or 'variance'")
        self.variance_proxy = variance_proxy

        self.freeze_reference_weight = bool(freeze_reference_weight)
        self.reference_key = reference_key

    # ---- Overrides ---------------------------------------------------------

    def step(
        self,
        epoch: int,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_stats_history: Optional[Dict[str, List[float]]] = None,
    ) -> Optional[Dict[str, Dict]]:
        """
        Override because the provided base class does not forward grad_norm_history
        into compute_new_weights().
        """
        self.current_epoch = epoch

        for component_name, losses in loss_history.items():
            self.history.setdefault(component_name, []).append(losses.copy())

        # Cache full grad stats history (nested) for analysis
        if grad_stats_history is not None:
            for component_name, stats_dict in grad_stats_history.items():
                if component_name not in self.grad_stats_history:
                    self.grad_stats_history[component_name] = {}
                if isinstance(stats_dict, dict):
                    for stat_name, stat_values in stats_dict.items():
                        self.grad_stats_history[component_name].setdefault(stat_name, []).append(stat_values.copy())

        # Extract norms for GradNorm algorithm
        grad_var_history = None
        if grad_stats_history:
            grad_var_history = {
                k: v.get("var", [])
                for k, v in grad_stats_history.items()
                if isinstance(v, dict)
            }

        if self.should_update(epoch):
            return self.compute_new_weights(loss_history, current_weights, grad_var_history)

        return None

    # ---- Helpers -----------------------------------------------------------

    def _clip(self, x: float) -> float:
        return max(self.min_weight, min(self.max_weight, x))

    def _recover_unweighted_losses(self, losses: List[float], weight: float) -> List[float]:
        """Recover unweighted losses from weighted loss history."""
        if not losses:
            return []
        denom = max(float(weight), self.epsilon)
        return [float(l) / denom for l in losses]

    def _moment_stats(self, xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"mean": 0.0, "second": 0.0, "var": 0.0}
        t = torch.tensor(xs, dtype=torch.float32)
        mean = float(t.mean())
        second = float((t * t).mean())
        var = float(max(0.0, second - mean * mean))
        return {"mean": mean, "second": second, "var": var}


    def _choose_reference_key(self, keys: List[str]) -> Optional[str]:
        if self.reference_key is not None and self.reference_key in keys:
            return self.reference_key
        for cand in ("Lr", "residual", "pde_residual", "physics", "PDE", "PDEResidual"):
            if cand in keys:
                return cand
        return None

    # ---- Main algorithm ----------------------------------------------------

    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],            # unused, kept for signature compat
        current_weights: Dict[str, Dict],
        grad_var_history: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Dict]:
        # Need gradient-variance history
        if not grad_var_history:
            return {k: v.copy() for k, v in current_weights.items()}

        keys = [k for k, vs in grad_var_history.items() if isinstance(vs, list) and len(vs) > 0]
        if not keys:
            return {k: v.copy() for k, v in current_weights.items()}

        # 1) Aggregate batchwise var estimates into one V_k per epoch
        V: Dict[str, float] = {}
        for k in keys:
            t = torch.tensor(grad_var_history[k], dtype=torch.float32)

            # Option A (default): mean over batches
            Vk = float(t.mean())

            # Option B (robust): median over batches
            # Vk = float(t.median())

            V[k] = max(Vk, self.epsilon)

        # 2) numerator is max over tasks
        Vmax = max(V.values())
        if Vmax <= self.epsilon:
            return {k: v.copy() for k, v in current_weights.items()}

        # Optional extension from your previous version
        ref = self._choose_reference_key(keys)

        # 3) Eq. (8) + EMA
        new_weight_scalars: Dict[str, float] = {}
        for k in keys:
            prev_w = self._get_previous_weight(k, current_weights)

            if self.freeze_reference_weight and ref is not None and k == ref:
                new_weight_scalars[k] = float(prev_w)
                continue

            w_hat = float(Vmax / V[k])
            w_new = self.alpha * float(prev_w) + (1.0 - self.alpha) * w_hat
            new_weight_scalars[k] = self._clip(float(w_new))

        # 4) Rebuild your hierarchical weight structure
        return self._reconstruct_weight_dict(new_weight_scalars, current_weights)