from typing import Dict, List, Optional
import torch

from ..loss_weighting_strategies import LossWeightingStrategyBase


class InverseDirichlet(LossWeightingStrategyBase):
    """
    Inverse-Dirichlet weighting from:
    Maddu et al. (2021) "Inverse-Dirichlet Weighting Enables Reliable Training of Physics Informed Neural Networks".
    See Eq. (8): :contentReference[oaicite:0]{index=0}

        λ̂_k(τ) = max_t Var[∇_{θ_sh} L_t(τ)] / Var[∇_{θ_sh} L_k(τ)]
        λ_k(τ+1) = α λ_k(τ) + (1-α) λ̂_k(τ)

    Practical note for this codebase:
    - The paper's Var[·] is the variance over *gradient vector components* at a given training step.
    - Here we typically only have per-step scalar gradient magnitudes in `grad_norm_history[loss_key]`.
      We therefore use a proxy for the "Dirichlet energy / gradient variance scale" computed from the
      per-step scalars over the epoch.

    Proxy choices (see `variance_proxy`):
    - "second_moment" (default): E[g^2]  (robust, matches the paper's Adam-friendly note that the inverse
      of averaged squared gradients is efficient).
    - "variance": Var[g] = E[g^2] - (E[g])^2  (may be closer in spirit, but can be tiny if g is stable).

    Inputs expected:
      grad_norm_history = { loss_key: [g_step1, g_step2, ...], ... }
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
        grad_norm_history: Optional[Dict[str, List[float]]] = None,
    ) -> Optional[Dict[str, Dict]]:
        """
        Override because the provided base class does not forward grad_norm_history
        into compute_new_weights().
        """
        self.current_epoch = epoch

        for component_name, losses in loss_history.items():
            self.history.setdefault(component_name, []).append(losses.copy())

        if grad_norm_history is not None:
            for component_name, grads in grad_norm_history.items():
                self.grad_norm_history.setdefault(component_name, []).append(grads.copy())

        if self.should_update(epoch):
            return self.compute_new_weights(loss_history, current_weights, grad_norm_history)

        return None

    # ---- Helpers -----------------------------------------------------------

    def _clip(self, x: float) -> float:
        return max(self.min_weight, min(self.max_weight, x))

    def _moment_stats(self, xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"mean": 0.0, "second": 0.0, "var": 0.0}
        t = torch.tensor(xs, dtype=torch.float32)
        mean = float(t.mean())
        second = float((t * t).mean())
        var = float(max(0.0, second - mean * mean))
        return {"mean": mean, "second": second, "var": var}

    def _get_previous_weight(self, loss_key: str, current_weights: Dict[str, Dict]) -> float:
        base_name, sub_name, channel_idx = self._parse_hierarchical_key(loss_key)

        if base_name not in current_weights:
            return 1.0

        cfg = current_weights[base_name]

        # Base component
        if sub_name is None and channel_idx is None:
            return float(cfg.get("base_weight", 1.0))

        # Per-channel
        if channel_idx is not None:
            if "channel_weights" in cfg:
                cw = cfg["channel_weights"]
                if channel_idx < len(cw):
                    return float(cw[channel_idx])
            return 1.0

        # Per-subcomponent
        if sub_name is not None:
            if "component_weights" in cfg:
                return float(cfg["component_weights"].get(sub_name, 1.0))
            return 1.0

        return 1.0

    def _reconstruct_weight_dict(
        self,
        new_weight_scalars: Dict[str, float],
        current_weights: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        new_weights: Dict[str, Dict] = {}

        for base_name, cfg in current_weights.items():
            new_cfg = cfg.copy()

            # Base weight
            if base_name in new_weight_scalars:
                new_cfg["base_weight"] = float(new_weight_scalars[base_name])

            # Channel weights
            if "channel_weights" in cfg:
                channel_weights = cfg["channel_weights"].clone()
                for ch_idx in range(len(channel_weights)):
                    key = f"{base_name}/channel_{ch_idx}"
                    if key in new_weight_scalars:
                        channel_weights[ch_idx] = float(new_weight_scalars[key])
                new_cfg["channel_weights"] = channel_weights

            # Component weights
            if "component_weights" in cfg:
                component_weights = cfg["component_weights"].copy()
                for sub_name in list(component_weights.keys()):
                    key = f"{base_name}/{sub_name}"
                    if key in new_weight_scalars:
                        component_weights[sub_name] = float(new_weight_scalars[key])
                new_cfg["component_weights"] = component_weights

            new_weights[base_name] = new_cfg

        return new_weights

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
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_norm_history: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Dict]:
        # Requires gradient statistics
        if grad_norm_history is None or not grad_norm_history:
            return {k: v.copy() for k, v in current_weights.items()}

        # Consider keys for which we have gradients (loss_history not strictly required here)
        keys = list(grad_norm_history.keys())
        if len(keys) == 0:
            return {k: v.copy() for k, v in current_weights.items()}

        # Compute per-term "variance scale" proxy from per-step scalar gradient magnitudes
        scale: Dict[str, float] = {}
        for k in keys:
            stats = self._moment_stats(grad_norm_history.get(k, []))
            if self.variance_proxy == "second_moment":
                scale[k] = max(stats["second"], self.epsilon)  # E[g^2]
            else:
                scale[k] = max(stats["var"], self.epsilon)     # Var[g]

        # Reference: maximum variance proxy across tasks (Eq. 8 numerator)
        max_scale = max(scale.values()) if scale else 0.0
        if max_scale <= self.epsilon:
            return {k: v.copy() for k, v in current_weights.items()}

        # Optional "freeze" term (not required by the paper)
        ref = self._choose_reference_key(keys)

        # Compute λ̂_k and apply EMA update (Eq. 8)
        new_weight_scalars: Dict[str, float] = {}
        for k in keys:
            prev_w = self._get_previous_weight(k, current_weights)

            if self.freeze_reference_weight and ref is not None and k == ref:
                new_weight_scalars[k] = float(prev_w)
                continue

            w_hat = float(max_scale / max(scale[k], self.epsilon))
            w_new = self.alpha * float(prev_w) + (1.0 - self.alpha) * float(w_hat)
            new_weight_scalars[k] = self._clip(float(w_new))

        # Reconstruct hierarchical structure; keys not updated remain unchanged
        return self._reconstruct_weight_dict(new_weight_scalars, current_weights)
