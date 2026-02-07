import numpy as np
import torch
from typing import Iterable, Dict, Optional

from metrics.loss_framework import LossComponent, WeightSchedule, CompositeLoss
from metrics.loss_registry import get_loss_entry

def compute_metrics_for_n_rollouts(
    preds,
    targets,
    outputs_per_rollout: int = 1,
    metrics: Iterable[str] = ("l1", "l2"),
    include_per_timestep: bool = False,
    compute_metrics=None,
    loss_metric: Optional[LossComponent] = None,
) -> Dict[str, Dict[str, np.ndarray]]:

    # --- normalization & checks ---
    if isinstance(preds, np.ndarray):
        preds_arr = preds
    else:
        preds_arr = np.asarray(preds)

    if isinstance(targets, np.ndarray):
        targets_arr = targets
    else:
        targets_arr = np.asarray(targets)

    if preds_arr.shape != targets_arr.shape:
        raise ValueError(
            "Predictions and targets must have the same shape. "
            f"Got preds.shape={preds_arr.shape}, targets.shape={targets_arr.shape}."
        )

    if preds_arr.ndim < 3:
        raise ValueError("Expected input of at least 3 dims when flattened (B, T, C, …).")
    if outputs_per_rollout is None or outputs_per_rollout < 1:
        raise ValueError("outputs_per_rollout must be a positive integer.")

    total_steps = preds_arr.shape[1]
    if total_steps % outputs_per_rollout != 0:
        raise ValueError(
            f"Time dimension (T={total_steps}) is not divisible by outputs_per_rollout={outputs_per_rollout}."
        )
    num_rollouts = total_steps // outputs_per_rollout

    # grouped_shape = (
    #     preds_arr.shape[0],
    #     num_rollouts,
    #     outputs_per_rollout,
    #     preds_arr.shape[2],
    #     *preds_arr.shape[3:],
    # )
    # grouped_preds = preds_arr.reshape(grouped_shape)
    # grouped_targets = targets_arr.reshape(grouped_shape)

    # ------------------------------------------------------------------
    # Compute metrics using LossComponent / CompositeLoss
    # ------------------------------------------------------------------
    device = torch.device("cpu")

    loss_metric = loss_metric.to(device)

    preds_tensor = torch.as_tensor(preds_arr, dtype=torch.float32, device=device)
    targets_tensor = torch.as_tensor(targets_arr, dtype=torch.float32, device=device)

    B, T_flat, C = preds_tensor.shape[:3]

    def _compute_for_single_component(comp: LossComponent) -> Dict[str, Dict[str, np.ndarray]]:
        metric_key_name = comp.name if getattr(comp, "name", None) else "loss_error"
        metric_name_to_values: Dict[str, Dict[str, np.ndarray]] = {
            metric_key_name: {}
        }
        out = metric_name_to_values[metric_key_name]

        ws_top: Optional[WeightSchedule] = getattr(comp, "weight_schedule", None)
        component_weights_top = getattr(ws_top, "component_weights", None) if ws_top is not None else None
        use_component_weights_top = (
            isinstance(component_weights_top, dict) and len(component_weights_top) > 0
        )
        component_names_top = (
            list(component_weights_top.keys()) if use_component_weights_top else None
        )
        if use_component_weights_top:
            out["component_names"] = list(component_names_top)

        # --------------------------------------------------
        # Helper: true per-sample evaluation by sliding over B
        # --------------------------------------------------
        def eval_loss_batchwise(preds_slice: torch.Tensor, targets_slice: torch.Tensor):
            """
            Evaluate comp on each sample independently:

            preds_slice:   (B, T', C, ...)
            targets_slice: (B, T', C, ...)

            Returns:
              per_sample_channel: (B, C)  per-sample, per-channel scalar loss
              per_sample_overall: (B,)    per-sample scalar loss
            """
            bsz = preds_slice.shape[0]
            loss_key = comp.__class__.__name__

            # Overall per-sample
            per_sample_overall = torch.zeros(bsz, device=device)
            try:
                loss_registry_entry = get_loss_entry(loss_key)
            except ValueError:
                loss_registry_entry = {}

            # Per-channel via WeightSchedule, if available
            ws: Optional[WeightSchedule] = getattr(comp, "weight_schedule", None)
            original_channel_weights = getattr(ws, "channel_weights", None) if ws is not None else None
            component_weights = getattr(ws, "component_weights", None) if ws is not None else None
            use_component_weights = loss_registry_entry.get("sub_components", False)
            component_names = list(component_weights.keys()) if use_component_weights else None
            channel_aggregation = loss_registry_entry.get("channel_aggregation", "linear")


            reduction = getattr(comp, "reduction", "mean")

            use_channel_weights = ws is not None and original_channel_weights is not None
            per_sample_channel = None

            with torch.no_grad():
                # overall scalar per sample/channel (vectorized)
                total_loss = comp(
                    model=None,
                    predictions=preds_slice,
                    labels=targets_slice,
                    return_detailed=False,
                    keep_bc_dims=True
                )
                total_loss = (
                    total_loss.detach()
                    if torch.is_tensor(total_loss)
                    else torch.tensor(total_loss, device=device)
                )

                if use_component_weights:
                    if total_loss.ndim != 2 or total_loss.shape[1] != len(component_names):
                        raise ValueError(
                            "Component-wise loss output shape mismatch. "
                            f"Expected (B, {len(component_names)}), got {tuple(total_loss.shape)}."
                        )
                    per_sample_channel = total_loss
                    per_sample_overall = total_loss.mean(dim=1)

                # Channel aggregation to get overall per-sample loss
                if total_loss.ndim >= 2 and not use_component_weights:
                    if channel_aggregation == "linear":
                        if reduction == "sum":
                            per_sample_overall = total_loss.sum(dim=1)
                        else:
                            per_sample_overall = total_loss.mean(dim=1)
                    elif channel_aggregation == "sqrt":
                        channel_sum = (total_loss ** 2).sum(dim=1)
                        if reduction == "mean":
                            channel_sum = channel_sum / total_loss.shape[1]
                        per_sample_overall = torch.sqrt(channel_sum)
                    else:
                        raise ValueError(f"Unsupported channel_aggregation: {channel_aggregation}")

                # per-channel via keep_bc_dims outputs
                if use_channel_weights and not use_component_weights:
                    per_sample_channel = total_loss

            # If we could not decompose channels, broadcast overall
            if per_sample_channel is None:
                metric_dim = len(component_names) if use_component_weights else C
                per_sample_channel = per_sample_overall[:, None].expand(-1, metric_dim)

            return per_sample_channel, per_sample_overall

        # -----------------------------------------------------
        # A) Per-rollout-step stats, shape -> (R, C+1)
        # -----------------------------------------------------
        metric_dim = len(component_names_top) if use_component_weights_top else C

        per_sample_channel_metric = torch.zeros(B, num_rollouts, metric_dim, device=device)
        per_sample_overall_metric = torch.zeros(B, num_rollouts, device=device)

        for r in range(num_rollouts):
            t_start = r * outputs_per_rollout
            t_end = (r + 1) * outputs_per_rollout

            step_preds = preds_tensor[:, t_start:t_end, ...]
            step_targets = targets_tensor[:, t_start:t_end, ...]

            ch_vals, overall_vals = eval_loss_batchwise(step_preds, step_targets)
            # ch_vals: (B, metric_dim), overall_vals: (B,)
            per_sample_channel_metric[:, r, :] = ch_vals
            per_sample_overall_metric[:, r] = overall_vals

        # std over batch, as in legacy implementation
        per_step_channel_mean = per_sample_channel_metric.mean(dim=0)            # (N,R,C) --> (R, C)
        per_step_channel_std = per_sample_channel_metric.std(dim=0, unbiased=False) # (N,R,C) --> (R, C)
        per_step_overall_mean = per_sample_overall_metric.mean(dim=0)           # (N,R) --> (R,)
        per_step_overall_std = per_sample_overall_metric.std(dim=0, unbiased=False) # (N,R) --> (R,)

        per_step_mean = torch.cat(
            [per_step_channel_mean, per_step_overall_mean[:, None]], dim=-1
        )
        per_step_std = torch.cat(
            [per_step_channel_std, per_step_overall_std[:, None]], dim=-1
        )

        out["per_rollout_step_mean"] = per_step_mean.cpu().numpy()
        out["per_rollout_step_std"] = per_step_std.cpu().numpy()

        # cumulative over rollout steps (R), N is the number of windows, C is the number of channels
        cumulative_channel_per_sample = torch.cumsum(per_sample_channel_metric, dim=1)  # (N, R, C) --> (N, R, C)  (cumulative sum over rollout steps)
        cumulative_overall_per_sample = torch.cumsum(per_sample_overall_metric, dim=1)  # (N, R) --> (N, R)  (cumulative sum over rollout steps)

        cumulative_channel_mean = cumulative_channel_per_sample.mean(dim=0)             # (N, R, C) --> (R, C)
        cumulative_channel_std = cumulative_channel_per_sample.std(dim=0, unbiased=False) # (N, R, C) --> (R, C)
        cumulative_overall_mean = cumulative_overall_per_sample.mean(dim=0)             # (N, R) --> (R,)
        cumulative_overall_std = cumulative_overall_per_sample.std(dim=0, unbiased=False) # (N, R) --> (R,)

        cumulative_mean = torch.cat(
            [cumulative_channel_mean, cumulative_overall_mean[:, None]], dim=-1
        )
        cumulative_std = torch.cat(
            [cumulative_channel_std, cumulative_overall_std[:, None]], dim=-1
        )

        out["cumulative_rollout_step_mean"] = cumulative_mean.cpu().numpy()
        out["cumulative_rollout_step_std"] = cumulative_std.cpu().numpy()

        # -----------------------------------------------------
        # B) Optional per-timestep metrics, shape -> (T_flat, C+1)
        # -----------------------------------------------------
        if include_per_timestep:
            per_sample_channel_t = torch.zeros(B, T_flat, metric_dim, device=device)
            per_sample_overall_t = torch.zeros(B, T_flat, device=device)

            for t in range(T_flat):
                step_preds = preds_tensor[:, t : t + 1, ...]
                step_targets = targets_tensor[:, t : t + 1, ...]

                ch_vals, overall_vals = eval_loss_batchwise(step_preds, step_targets)
                per_sample_channel_t[:, t, :] = ch_vals      # (B, C)
                per_sample_overall_t[:, t] = overall_vals    # (B,)

            per_t_channel_mean = per_sample_channel_t.mean(dim=0)             # (N, T_flat, C) --> (T_flat, C)
            per_t_channel_std = per_sample_channel_t.std(dim=0, unbiased=False) # (N, T_flat, C) --> (T_flat, C)
            per_t_overall_mean = per_sample_overall_t.mean(dim=0)            # (N, T_flat) --> (T_flat,)
            per_t_overall_std = per_sample_overall_t.std(dim=0, unbiased=False) # (N, T_flat) --> (T_flat,)

            per_timestep_mean = torch.cat(
                [per_t_channel_mean, per_t_overall_mean[:, None]], dim=-1
            )
            per_timestep_std = torch.cat(
                [per_t_channel_std, per_t_overall_std[:, None]], dim=-1
            )

            out["per_timestep_mean"] = per_timestep_mean.cpu().numpy()  # (T_flat, C+1)
            out["per_timestep_std"] = per_timestep_std.cpu().numpy()  # (T_flat, C+1)

            cumulative_channel_per_sample_t = torch.cumsum(per_sample_channel_t, dim=1)  # (N, T_flat, C) --> (N, T_flat, C)  (cumulative sum over timesteps)
            cumulative_overall_per_sample_t = torch.cumsum(per_sample_overall_t, dim=1)  # (N, T_flat) --> (N, T_flat)  (cumulative sum over timesteps)

            cumulative_t_channel_mean = cumulative_channel_per_sample_t.mean(dim=0)      # (N, T_flat, C) --> (T_flat, C)
            cumulative_t_channel_std = cumulative_channel_per_sample_t.std(dim=0, unbiased=False) # (N, T_flat, C) --> (T_flat, C)
            cumulative_t_overall_mean = cumulative_overall_per_sample_t.mean(dim=0)      # (N, T_flat) --> (T_flat,)
            cumulative_t_overall_std = cumulative_overall_per_sample_t.std(dim=0, unbiased=False) # (N, T_flat) --> (T_flat,)

            cumulative_timestep_mean = torch.cat(
                [cumulative_t_channel_mean, cumulative_t_overall_mean[:, None]], dim=-1
            )
            cumulative_timestep_std = torch.cat(
                [cumulative_t_channel_std, cumulative_t_overall_std[:, None]], dim=-1
            )

            out["cumulative_timestep_mean"] = cumulative_timestep_mean.cpu().numpy()  # (T_flat, C+1)
            out["cumulative_timestep_std"] = cumulative_timestep_std.cpu().numpy()  # (T_flat, C+1)

        if use_component_weights_top:
            out["names"] = list(component_names_top)
        else:
            out["names"] = None

        return metric_name_to_values

    # Decide LossComponent vs CompositeLoss
    if isinstance(loss_metric, CompositeLoss):
        all_metrics: Dict[str, Dict[str, np.ndarray]] = {}
        for comp in loss_metric.loss_components:
            comp_metrics = _compute_for_single_component(comp)
            all_metrics.update(comp_metrics)
        return all_metrics
    else:
        return _compute_for_single_component(loss_metric)
