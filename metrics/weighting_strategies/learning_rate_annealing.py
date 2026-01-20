from typing import Dict, List, Optional
import torch

from ..loss_weighting_strategies import LossWeightingStrategyBase


class LearningRateAnnealing(LossWeightingStrategyBase):
    """
    Learning-rate annealing loss reweighting from:
    Wang et al. (2020) "Understanding and mitigating gradient pathologies in physics-informed neural networks".

    Core update (Algorithm 1 / Eqs. 40-41):
        lambda_hat_i = max_theta |∇_θ L_ref|  /  mean_theta |∇_θ L_i|
        lambda_i     = (1 - alpha) * lambda_i + alpha * lambda_hat_i

    Notes for this codebase:
    - We assume `grad_norm_history[loss_key]` is a list of *scalar* gradient magnitudes collected over the epoch
      (e.g., per-step grad L2-norm, or mean(|grad|) over parameters). We only need *relative* magnitudes.
    - One loss term is treated as the reference term (paper uses PDE residual L_r with fixed weight 1.0).
    - All loss keys (including hierarchical keys like 'X/channel_0' or 'X/sub') are treated as independent terms.
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
    ):
        super().__init__(update_frequency=update_frequency, use_gradients=use_gradients)
        self.reference_key = reference_key
        self.alpha = float(alpha)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.epsilon = float(epsilon)
        self.freeze_reference_weight = bool(freeze_reference_weight)

    def _clip(self, x: float) -> float:
        return max(self.min_weight, min(self.max_weight, x))

    def _safe_mean(self, xs: List[float]) -> float:
        if not xs:
            return 0.0
        t = torch.tensor(xs, dtype=torch.float32)
        return float(t.mean())

    def _safe_max(self, xs: List[float]) -> float:
        if not xs:
            return 0.0
        t = torch.tensor(xs, dtype=torch.float32)
        return float(t.max())

    def _get_previous_weight(self, loss_key: str, current_weights: Dict[str, Dict]) -> float:
        """
        Extract the previous scalar weight for a flat loss_key from the hierarchical weight dict.
        Mirrors the pattern used in BalancedResidualDecayRate.
        """
        base_name, sub_name, channel_idx = self._parse_hierarchical_key(loss_key)

        if base_name not in current_weights:
            return 1.0

        config = current_weights[base_name]

        # Base component
        if sub_name is None and channel_idx is None:
            return float(config.get("base_weight", 1.0))

        # Per-channel
        if channel_idx is not None:
            if "channel_weights" in config:
                cw = config["channel_weights"]
                if channel_idx < len(cw):
                    return float(cw[channel_idx])
            return 1.0

        # Per-subcomponent
        if sub_name is not None:
            if "component_weights" in config:
                return float(config["component_weights"].get(sub_name, 1.0))
            return 1.0

        return 1.0

    def _reconstruct_weight_dict(
        self,
        new_weight_scalars: Dict[str, float],
        current_weights: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """
        Reconstruct hierarchical dict (base_weight / channel_weights / component_weights)
        from a flat mapping loss_key -> scalar.
        """
        new_weights: Dict[str, Dict] = {}

        for base_name, config in current_weights.items():
            new_config = config.copy()

            # Base weight
            if base_name in new_weight_scalars:
                new_config["base_weight"] = float(new_weight_scalars[base_name])

            # Channel weights
            if "channel_weights" in config:
                channel_weights = config["channel_weights"].clone()
                for ch_idx in range(len(channel_weights)):
                    ch_key = f"{base_name}/channel_{ch_idx}"
                    if ch_key in new_weight_scalars:
                        channel_weights[ch_idx] = float(new_weight_scalars[ch_key])
                new_config["channel_weights"] = channel_weights

            # Component weights
            if "component_weights" in config:
                component_weights = config["component_weights"].copy()
                for sub_name in list(component_weights.keys()):
                    comp_key = f"{base_name}/{sub_name}"
                    if comp_key in new_weight_scalars:
                        component_weights[sub_name] = float(new_weight_scalars[comp_key])
                new_config["component_weights"] = component_weights

            new_weights[base_name] = new_config

        return new_weights

    def _choose_reference_key(
        self,
        grad_norm_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
    ) -> Optional[str]:
        """
        Pick the reference loss key used in the numerator. Preference order:
        1) self.reference_key if provided and present
        2) common PINN names if present
        3) first key that exists in grad_norm_history
        """
        if self.reference_key is not None and self.reference_key in grad_norm_history:
            return self.reference_key

        for cand in ("Lr", "residual", "pde_residual", "physics", "PDE", "PDEResidual"):
            if cand in grad_norm_history:
                return cand

        # Fall back to first gradient key
        for k in grad_norm_history.keys():
            return k

        return None

    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_norm_history: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Dict]:
        # Algorithm relies on gradients.
        if grad_norm_history is None or not grad_norm_history:
            return {k: v.copy() for k, v in current_weights.items()}

        ref_key = self._choose_reference_key(grad_norm_history, current_weights)
        if ref_key is None or ref_key not in grad_norm_history:
            return {k: v.copy() for k, v in current_weights.items()}

        # Numerator: max gradient magnitude of reference term over the epoch (Eq. 40)
        g_ref_max = self._safe_max(grad_norm_history.get(ref_key, []))
        if g_ref_max <= self.epsilon:
            # If reference grads are ~0, don't change anything.
            return {k: v.copy() for k, v in current_weights.items()}

        # Compute new scalar weights for every loss key we have gradients for.
        new_weight_scalars: Dict[str, float] = {}

        for loss_key, g_list in grad_norm_history.items():
            prev_w = self._get_previous_weight(loss_key, current_weights)

            # Optionally keep reference weight fixed (paper keeps L_r unweighted).
            if self.freeze_reference_weight and loss_key == ref_key:
                new_weight_scalars[loss_key] = prev_w
                continue

            g_i_mean = self._safe_mean(g_list)  # denominator uses mean(|grad|) (Eq. 40)
            denom = max(g_i_mean, self.epsilon)
            w_hat = float(g_ref_max / denom)

            # Moving-average update (Eq. 41): lambda <- (1-alpha)*lambda + alpha*lambda_hat
            w_new = (1.0 - self.alpha) * float(prev_w) + self.alpha * float(w_hat)
            new_weight_scalars[loss_key] = self._clip(w_new)

        # Preserve any weights that don't have gradients recorded this epoch.
        # (e.g., if only some terms were logged)
        # We do this by simply reconstructing from the partial updates; everything else stays as-is.
        return self._reconstruct_weight_dict(new_weight_scalars, current_weights)
