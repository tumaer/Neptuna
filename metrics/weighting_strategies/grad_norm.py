from typing import Dict, List, Optional, Tuple
import torch

from ..loss_weighting_strategies import LossWeightingStrategyBase


class GradNorm(LossWeightingStrategyBase):
    """
    GradNorm (Chen et al., 2018), adapted to an epoch-level.
    """

    def __init__(
        self,
        update_frequency: int = 1,
        use_gradients: bool = True,
        alpha: float = 1.5,
        weight_lr: float = 0.025,
        min_weight: float = 1e-6,
        max_weight: float = 1e6,
        epsilon: float = 1e-12,
        grad_norms_are_weighted: bool = False,
        freeze_reference_weight: bool = False,
        reference_key: Optional[str] = None,
        renormalize_over_all_controlled_keys: bool = True,
        use_ema_stats: bool = False,
        ema_beta: float = 0.9,
        use_smooth_l1: bool = False,  # if True, use smooth |.|; if False, pure L1 like paper
    ):
        super().__init__(update_frequency=update_frequency, use_gradients=use_gradients)
        self.alpha = float(alpha)
        self.weight_lr = float(weight_lr)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.epsilon = float(epsilon)

        self.grad_norms_are_weighted = bool(grad_norms_are_weighted)
        self.freeze_reference_weight = bool(freeze_reference_weight)
        self.reference_key = reference_key

        self.renormalize_over_all_controlled_keys = bool(renormalize_over_all_controlled_keys)

        self.use_ema_stats = bool(use_ema_stats)
        self.ema_beta = float(ema_beta)
        self.use_smooth_l1 = bool(use_smooth_l1)

        # L_i(0) cache
        self.initial_loss: Dict[str, float] = {}

        # Optional EMA state
        self._ema_loss: Dict[str, float] = {}
        self._ema_grad: Dict[str, float] = {}

    # ---- Overrides ---------------------------------------------------------

    def step(
        self,
        epoch: int,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_norm_history: Optional[Dict[str, List[float]]] = None,
    ) -> Optional[Dict[str, Dict]]:
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

    def _mean(self, xs: List[float]) -> float:
        if not xs:
            return 0.0
        return float(torch.tensor(xs, dtype=torch.float32).mean().item())

    def _parse_hierarchical_key(self, loss_key: str) -> Tuple[str, Optional[str], Optional[int]]:
        if "/channel_" in loss_key:
            base, ch = loss_key.split("/channel_", 1)
            try:
                idx = int(ch)
            except ValueError:
                idx = None
            return base, None, idx

        if "/" in loss_key:
            base, sub = loss_key.split("/", 1)
            return base, sub, None

        return loss_key, None, None

    def _get_previous_weight(self, loss_key: str, current_weights: Dict[str, Dict]) -> float:
        base_name, sub_name, channel_idx = self._parse_hierarchical_key(loss_key)

        if base_name not in current_weights:
            return 1.0

        cfg = current_weights[base_name]

        if sub_name is None and channel_idx is None:
            return float(cfg.get("base_weight", 1.0))

        if channel_idx is not None:
            if "channel_weights" in cfg:
                cw = cfg["channel_weights"]
                if channel_idx < len(cw):
                    return float(cw[channel_idx])
            return 1.0

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

            if base_name in new_weight_scalars:
                new_cfg["base_weight"] = float(new_weight_scalars[base_name])

            if "channel_weights" in cfg:
                channel_weights = cfg["channel_weights"].clone()
                for ch_idx in range(len(channel_weights)):
                    key = f"{base_name}/channel_{ch_idx}"
                    if key in new_weight_scalars:
                        channel_weights[ch_idx] = float(new_weight_scalars[key])
                new_cfg["channel_weights"] = channel_weights

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

    def _ema_update(self, store: Dict[str, float], k: str, x: float) -> float:
        """EMA over epoch means; returns updated EMA value."""
        if k not in store:
            store[k] = float(x)
        else:
            store[k] = float(self.ema_beta * store[k] + (1.0 - self.ema_beta) * float(x))
        return store[k]

    # ---- Main algorithm ----------------------------------------------------

    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_norm_history: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Dict]:
        if grad_norm_history is None or not grad_norm_history:
            return {k: v.copy() for k, v in current_weights.items()}

        # Only terms where we have both loss and grad stats
        keys = [k for k in grad_norm_history.keys() if k in loss_history]
        if len(keys) < 2:
            return {k: v.copy() for k, v in current_weights.items()}

        # --- Epoch stats (means, optionally EMA-smoothed) --------------------
        L_mean: Dict[str, float] = {}
        g_mean: Dict[str, float] = {}
        prev_w: Dict[str, float] = {}

        for k in keys:
            prev_w[k] = self._get_previous_weight(k, current_weights)

            Li = self._mean(loss_history.get(k, []))
            gi = self._mean(grad_norm_history.get(k, []))

            if self.use_ema_stats:
                Li = self._ema_update(self._ema_loss, k, Li)
                gi = self._ema_update(self._ema_grad, k, gi)

            L_mean[k] = max(float(Li), 0.0)
            g_mean[k] = max(float(gi), 0.0)

            if k not in self.initial_loss:
                self.initial_loss[k] = max(L_mean[k], self.epsilon)

        # --- r_i = (L_i/L_i0) / mean_j(L_j/L_j0) -----------------------------
        L_tilde = {k: L_mean[k] / max(self.initial_loss.get(k, self.epsilon), self.epsilon) for k in keys}
        mean_L_tilde = sum(L_tilde.values()) / float(len(keys))
        if mean_L_tilde <= self.epsilon:
            return {k: v.copy() for k, v in current_weights.items()}
        r_i = {k: (L_tilde[k] / mean_L_tilde) for k in keys}

        # --- Build G_i and G_avg --------------------------------------------
        if self.grad_norms_are_weighted:
            # grad_norm_history already provides G_i ≈ ||∇_W(w_i L_i)||.
            G_i = {k: g_mean[k] for k in keys}
        else:
            # grad_norm_history provides g_i ≈ ||∇_W L_i||, so G_i ≈ w_i * g_i
            G_i = {k: prev_w[k] * g_mean[k] for k in keys}

        G_avg = sum(G_i.values()) / float(len(keys))
        if G_avg <= self.epsilon:
            return {k: v.copy() for k, v in current_weights.items()}

        # Targets: G*_i = G_avg * r_i^alpha
        G_star = {k: float(G_avg * (r_i[k] ** self.alpha)) for k in keys}

        # Optional freeze reference (non-paper)
        ref = self._choose_reference_key(keys) if self.freeze_reference_weight else None

        device = torch.device("cpu")
        w_t = {}
        for k in keys:
            w_t[k] = torch.tensor(float(prev_w[k]), device=device, dtype=torch.float32, requires_grad=True)

        # Choose g_i constant for the modeled G_i(w)
        if not self.grad_norms_are_weighted:
            g_const = {k: torch.tensor(max(g_mean[k], self.epsilon), device=device, dtype=torch.float32) for k in keys}
        else:
            g_const = {}
            for k in keys:
                denom = max(prev_w[k], self.epsilon)
                g_est = g_mean[k] / denom
                g_const[k] = torch.tensor(max(g_est, self.epsilon), device=device, dtype=torch.float32)

        # Build target constants
        target_t = {
            k: torch.tensor(float(G_star[k]), device=device, dtype=torch.float32)  # constant
            for k in keys
        }

        # L_grad
        L_grad = torch.zeros((), device=device, dtype=torch.float32)
        for k in keys:
            if ref is not None and k == ref:
                continue  # frozen; don't include it in optimization (optional choice)

            Gi_model = w_t[k] * g_const[k]
            diff = Gi_model - target_t[k]

            if self.use_smooth_l1:
                L_grad = L_grad + torch.sqrt(diff * diff + self.epsilon)
            else:
                L_grad = L_grad + torch.abs(diff)

        # Compute gradients
        L_grad.backward()

        # Gradient step on w
        new_w_scalars: Dict[str, float] = {}
        for k in keys:
            w0 = float(prev_w[k])

            if ref is not None and k == ref:
                new_w_scalars[k] = self._clip(w0)
                continue

            grad = w_t[k].grad
            if grad is None:
                new_w_scalars[k] = self._clip(w0)
                continue

            w_new = float(w0 - self.weight_lr * float(grad.item()))
            new_w_scalars[k] = self._clip(w_new)

        # --- Renormalize so sum_i w_i = T (paper Alg. 1 spirit) --------------
        if self.renormalize_over_all_controlled_keys:
            renorm_keys = keys
        else:
            renorm_keys = list(new_w_scalars.keys())

        T = float(len(renorm_keys))
        s = sum(float(new_w_scalars[k]) for k in renorm_keys)
        if s > self.epsilon:
            scale = T / s
            for k in renorm_keys:
                new_w_scalars[k] = self._clip(float(new_w_scalars[k]) * scale)

        return self._reconstruct_weight_dict(new_w_scalars, current_weights)
