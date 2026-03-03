import numpy as np
import torch
from typing import Iterable, Dict, Optional
from collections.abc import Mapping
import torch.distributed as dist

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
    device: Optional[str] = None,
    input_frames: Optional[np.ndarray] = None,
    metric_batch_size: Optional[int] = None,
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
    if loss_metric is None:
        raise ValueError("loss_metric must be provided for rollout metric computation.")

    # Use configured device, or auto-select: cuda:0 / xpu:0 if available, else cpu
    if device is not None and str(device).strip():
        dev = torch.device(str(device).strip())
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        dev = torch.device("xpu:0")
    elif torch.cuda.is_available():
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")

    loss_metric = loss_metric.to(dev)

    B, T_flat, C = preds_arr.shape[:3]
    if metric_batch_size is None:
        batch_size_for_metrics = B
    else:
        batch_size_for_metrics = int(metric_batch_size)
        if batch_size_for_metrics < 1:
            raise ValueError("metric_batch_size must be >= 1 when provided.")

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
        def eval_loss_batchwise(
            preds_slice: torch.Tensor,
            targets_slice: torch.Tensor,
            input_frames: Optional[torch.Tensor],
        ):
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
            per_sample_overall = torch.zeros(bsz, device=dev)
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
                    input_frames=input_frames,
                    keep_bc_dims=True
                )
                total_loss = (
                    total_loss.detach()
                    if torch.is_tensor(total_loss)
                    else torch.tensor(total_loss, device=dev)
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
        # A) Per-rollout-step stats, shape -> (R, metric_dim+1)
        # -----------------------------------------------------
        metric_dim = len(component_names_top) if use_component_weights_top else C

        per_step_channel_sum = torch.zeros(num_rollouts, metric_dim, device=dev)
        per_step_channel_sumsq = torch.zeros(num_rollouts, metric_dim, device=dev)
        per_step_overall_sum = torch.zeros(num_rollouts, device=dev)
        per_step_overall_sumsq = torch.zeros(num_rollouts, device=dev)

        cumulative_rollout_channel_sum = torch.zeros(num_rollouts, metric_dim, device=dev)
        cumulative_rollout_channel_sumsq = torch.zeros(num_rollouts, metric_dim, device=dev)
        cumulative_rollout_overall_sum = torch.zeros(num_rollouts, device=dev)
        cumulative_rollout_overall_sumsq = torch.zeros(num_rollouts, device=dev)

        if include_per_timestep:
            per_t_channel_sum = torch.zeros(T_flat, metric_dim, device=dev)
            per_t_channel_sumsq = torch.zeros(T_flat, metric_dim, device=dev)
            per_t_overall_sum = torch.zeros(T_flat, device=dev)
            per_t_overall_sumsq = torch.zeros(T_flat, device=dev)

            cumulative_t_channel_sum = torch.zeros(T_flat, metric_dim, device=dev)
            cumulative_t_channel_sumsq = torch.zeros(T_flat, metric_dim, device=dev)
            cumulative_t_overall_sum = torch.zeros(T_flat, device=dev)
            cumulative_t_overall_sumsq = torch.zeros(T_flat, device=dev)

        num_samples = 0

        for b_start in range(0, B, batch_size_for_metrics):
            b_end = min(B, b_start + batch_size_for_metrics)
            bsz = b_end - b_start
            num_samples += bsz

            preds_batch = torch.as_tensor(preds_arr[b_start:b_end], dtype=torch.float32, device=dev)
            targets_batch = torch.as_tensor(targets_arr[b_start:b_end], dtype=torch.float32, device=dev)
            input_frames_batch = (
                torch.as_tensor(input_frames[b_start:b_end], dtype=torch.float32, device=dev)
                if input_frames is not None
                else None
            )

            per_sample_channel_metric = torch.zeros(bsz, num_rollouts, metric_dim, device=dev)
            per_sample_overall_metric = torch.zeros(bsz, num_rollouts, device=dev)

            for r in range(num_rollouts):
                t_start = r * outputs_per_rollout
                t_end = (r + 1) * outputs_per_rollout

                step_preds = preds_batch[:, t_start:t_end, ...]
                step_targets = targets_batch[:, t_start:t_end, ...]
                if t_start > 0:
                    prev_frame = preds_batch[:, t_start - 1:t_start, ...]
                else:
                    prev_frame = input_frames_batch

                ch_vals, overall_vals = eval_loss_batchwise(step_preds, step_targets, prev_frame)
                per_sample_channel_metric[:, r, :] = ch_vals
                per_sample_overall_metric[:, r] = overall_vals

            per_step_channel_sum += per_sample_channel_metric.sum(dim=0)
            per_step_channel_sumsq += (per_sample_channel_metric ** 2).sum(dim=0)
            per_step_overall_sum += per_sample_overall_metric.sum(dim=0)
            per_step_overall_sumsq += (per_sample_overall_metric ** 2).sum(dim=0)

            cumulative_channel_per_sample = torch.cumsum(per_sample_channel_metric, dim=1)
            cumulative_overall_per_sample = torch.cumsum(per_sample_overall_metric, dim=1)

            cumulative_rollout_channel_sum += cumulative_channel_per_sample.sum(dim=0)
            cumulative_rollout_channel_sumsq += (cumulative_channel_per_sample ** 2).sum(dim=0)
            cumulative_rollout_overall_sum += cumulative_overall_per_sample.sum(dim=0)
            cumulative_rollout_overall_sumsq += (cumulative_overall_per_sample ** 2).sum(dim=0)

            if include_per_timestep:
                per_sample_channel_t = torch.zeros(bsz, T_flat, metric_dim, device=dev)
                per_sample_overall_t = torch.zeros(bsz, T_flat, device=dev)

                for t in range(T_flat):
                    step_preds = preds_batch[:, t : t + 1, ...]
                    step_targets = targets_batch[:, t : t + 1, ...]
                    if t > 0:
                        prev_frame = preds_batch[:, t - 1:t, ...]
                    else:
                        prev_frame = input_frames_batch

                    ch_vals, overall_vals = eval_loss_batchwise(step_preds, step_targets, prev_frame)
                    per_sample_channel_t[:, t, :] = ch_vals
                    per_sample_overall_t[:, t] = overall_vals

                per_t_channel_sum += per_sample_channel_t.sum(dim=0)
                per_t_channel_sumsq += (per_sample_channel_t ** 2).sum(dim=0)
                per_t_overall_sum += per_sample_overall_t.sum(dim=0)
                per_t_overall_sumsq += (per_sample_overall_t ** 2).sum(dim=0)

                cumulative_channel_per_sample_t = torch.cumsum(per_sample_channel_t, dim=1)
                cumulative_overall_per_sample_t = torch.cumsum(per_sample_overall_t, dim=1)

                cumulative_t_channel_sum += cumulative_channel_per_sample_t.sum(dim=0)
                cumulative_t_channel_sumsq += (cumulative_channel_per_sample_t ** 2).sum(dim=0)
                cumulative_t_overall_sum += cumulative_overall_per_sample_t.sum(dim=0)
                cumulative_t_overall_sumsq += (cumulative_overall_per_sample_t ** 2).sum(dim=0)

            del preds_batch, targets_batch, input_frames_batch
            del per_sample_channel_metric, per_sample_overall_metric
            if include_per_timestep:
                del per_sample_channel_t, per_sample_overall_t

        if num_samples == 0:
            raise ValueError("No samples available for metric aggregation.")

        num_samples_f = float(num_samples)

        per_step_channel_mean = per_step_channel_sum / num_samples_f
        per_step_channel_var = torch.clamp(per_step_channel_sumsq / num_samples_f - per_step_channel_mean ** 2, min=0.0)
        per_step_channel_std = torch.sqrt(per_step_channel_var)
        per_step_overall_mean = per_step_overall_sum / num_samples_f
        per_step_overall_var = torch.clamp(per_step_overall_sumsq / num_samples_f - per_step_overall_mean ** 2, min=0.0)
        per_step_overall_std = torch.sqrt(per_step_overall_var)

        per_step_mean = torch.cat([per_step_channel_mean, per_step_overall_mean[:, None]], dim=-1)
        per_step_std = torch.cat([per_step_channel_std, per_step_overall_std[:, None]], dim=-1)

        out["per_rollout_step_mean"] = per_step_mean.cpu().numpy()
        out["per_rollout_step_std"] = per_step_std.cpu().numpy()

        cumulative_channel_mean = cumulative_rollout_channel_sum / num_samples_f
        cumulative_channel_var = torch.clamp(
            cumulative_rollout_channel_sumsq / num_samples_f - cumulative_channel_mean ** 2,
            min=0.0,
        )
        cumulative_channel_std = torch.sqrt(cumulative_channel_var)
        cumulative_overall_mean = cumulative_rollout_overall_sum / num_samples_f
        cumulative_overall_var = torch.clamp(
            cumulative_rollout_overall_sumsq / num_samples_f - cumulative_overall_mean ** 2,
            min=0.0,
        )
        cumulative_overall_std = torch.sqrt(cumulative_overall_var)

        cumulative_mean = torch.cat([cumulative_channel_mean, cumulative_overall_mean[:, None]], dim=-1)
        cumulative_std = torch.cat([cumulative_channel_std, cumulative_overall_std[:, None]], dim=-1)

        out["cumulative_rollout_step_mean"] = cumulative_mean.cpu().numpy()
        out["cumulative_rollout_step_std"] = cumulative_std.cpu().numpy()

        # -----------------------------------------------------
        # B) Optional per-timestep metrics, shape -> (T_flat, metric_dim+1)
        # -----------------------------------------------------
        if include_per_timestep:
            per_t_channel_mean = per_t_channel_sum / num_samples_f
            per_t_channel_var = torch.clamp(per_t_channel_sumsq / num_samples_f - per_t_channel_mean ** 2, min=0.0)
            per_t_channel_std = torch.sqrt(per_t_channel_var)
            per_t_overall_mean = per_t_overall_sum / num_samples_f
            per_t_overall_var = torch.clamp(per_t_overall_sumsq / num_samples_f - per_t_overall_mean ** 2, min=0.0)
            per_t_overall_std = torch.sqrt(per_t_overall_var)

            per_timestep_mean = torch.cat([per_t_channel_mean, per_t_overall_mean[:, None]], dim=-1)
            per_timestep_std = torch.cat([per_t_channel_std, per_t_overall_std[:, None]], dim=-1)

            out["per_timestep_mean"] = per_timestep_mean.cpu().numpy()
            out["per_timestep_std"] = per_timestep_std.cpu().numpy()

            cumulative_t_channel_mean = cumulative_t_channel_sum / num_samples_f
            cumulative_t_channel_var = torch.clamp(
                cumulative_t_channel_sumsq / num_samples_f - cumulative_t_channel_mean ** 2,
                min=0.0,
            )
            cumulative_t_channel_std = torch.sqrt(cumulative_t_channel_var)
            cumulative_t_overall_mean = cumulative_t_overall_sum / num_samples_f
            cumulative_t_overall_var = torch.clamp(
                cumulative_t_overall_sumsq / num_samples_f - cumulative_t_overall_mean ** 2,
                min=0.0,
            )
            cumulative_t_overall_std = torch.sqrt(cumulative_t_overall_var)

            cumulative_timestep_mean = torch.cat(
                [cumulative_t_channel_mean, cumulative_t_overall_mean[:, None]], dim=-1
            )
            cumulative_timestep_std = torch.cat(
                [cumulative_t_channel_std, cumulative_t_overall_std[:, None]], dim=-1
            )

            out["cumulative_timestep_mean"] = cumulative_timestep_mean.cpu().numpy()
            out["cumulative_timestep_std"] = cumulative_timestep_std.cpu().numpy()

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


class StreamingRolloutMetrics:
    """
    Streaming variant of `compute_metrics_for_n_rollouts` designed for
    `batch_eval_metrics=True` execution inside `Trainer.inference_loop`.

    It aggregates first and second moments per rollout/timestep over batches,
    then computes final mean/std only at `compute_result=True`.
    """

    def __init__(
        self,
        loss_metric: LossComponent,
        include_per_timestep: bool = True,
        device: Optional[str] = None,
        metric_batch_size: Optional[int] = None,
        base_metrics=None,
    ):
        if loss_metric is None:
            raise ValueError("loss_metric must be provided for StreamingRolloutMetrics.")

        self.loss_metric = loss_metric
        self.include_per_timestep = bool(include_per_timestep)
        self.device = device
        self.metric_batch_size = metric_batch_size
        self.base_metrics = base_metrics
        self.reset()

    def reset(self):
        self._num_samples = 0.0
        self._sum = {}
        self._sumsq = {}
        self._names = {}

    def _infer_input_frames(self, maybe_inputs):
        if maybe_inputs is None:
            return None
        if isinstance(maybe_inputs, Mapping):
            for key in ("input_data", "inputs", "input_ids"):
                if key in maybe_inputs:
                    return maybe_inputs[key]
            return None
        return maybe_inputs

    def _to_numpy(self, x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().float().cpu().numpy()
        return np.asarray(x)

    def _trim_last_step_for_remainder(self, preds, labels, input_frames, compute_result, eval_pred):
        """
        Match Accelerate's remainder semantics on the last logical step.
        """
        if not compute_result:
            return preds, labels, input_frames

        trainer = getattr(self.base_metrics, "trainer", None)
        if trainer is None:
            return preds, labels, input_frames

        gs = trainer.accelerator.gradient_state
        remainder = gs.remainder
        if remainder <= 0:
            return preds, labels, input_frames

        local_bs = preds.shape[0]
        rank = trainer.accelerator.process_index
        start = rank * local_bs
        valid_local = max(0, min(local_bs, remainder - start))

        preds = preds[:valid_local]
        labels = labels[:valid_local]
        if input_frames is not None:
            input_frames = input_frames[:valid_local]
        return preds, labels, input_frames

    def _accumulate_metric_block(self, metric_name, metric_values, batch_n: int):
        if metric_name not in self._sum:
            self._sum[metric_name] = {}
            self._sumsq[metric_name] = {}

        for summary_kind, arr in metric_values.items():
            if summary_kind in ("names", "component_names"):
                self._names[(metric_name, summary_kind)] = arr
                continue

            arr_np = np.asarray(arr, dtype=np.float64)
            if arr_np.ndim != 2:
                continue

            mean = arr_np
            std = metric_values.get(summary_kind.replace("_mean", "_std"), None)
            if summary_kind.endswith("_std"):
                continue

            if std is None:
                continue

            std_np = np.asarray(std, dtype=np.float64)
            if std_np.shape != mean.shape:
                continue

            batch_sum = batch_n * mean
            batch_sumsq = batch_n * (std_np ** 2 + mean ** 2)

            if summary_kind not in self._sum[metric_name]:
                self._sum[metric_name][summary_kind] = batch_sum
                self._sumsq[metric_name][summary_kind] = batch_sumsq
            else:
                self._sum[metric_name][summary_kind] += batch_sum
                self._sumsq[metric_name][summary_kind] += batch_sumsq

    def __call__(self, eval_pred, compute_result: bool = False):
        base_out = {}
        if self.base_metrics is not None:
            base_out = self.base_metrics(eval_pred, compute_result=compute_result) or {}

        preds = eval_pred.predictions
        labels = eval_pred.label_ids

        preds_np = self._to_numpy(preds)
        labels_np = self._to_numpy(labels)
        input_frames = self._infer_input_frames(getattr(eval_pred, "inputs", None))
        input_frames_np = self._to_numpy(input_frames)

        if preds_np is None or labels_np is None:
            return base_out

        # Expect logits shape (B, R, T, C, ...) and labels shape (B, R*T, C, ...)
        if preds_np.ndim >= 5:
            b, r, t, c = preds_np.shape[:4]
            extra = preds_np.shape[4:]
            outputs_per_rollout = t
            preds_flat = preds_np.reshape(b, r * t, c, *extra)
        elif preds_np.ndim >= 3:
            outputs_per_rollout = 1
            preds_flat = preds_np
        else:
            return base_out

        labels_flat = labels_np

        preds_flat, labels_flat, input_frames_np = self._trim_last_step_for_remainder(
            preds_flat, labels_flat, input_frames_np, compute_result, eval_pred
        )

        if preds_flat.shape[0] == 0:
            return base_out if not compute_result else base_out

        batch_metrics = compute_metrics_for_n_rollouts(
            preds=preds_flat,
            targets=labels_flat,
            outputs_per_rollout=outputs_per_rollout,
            include_per_timestep=self.include_per_timestep,
            loss_metric=self.loss_metric,
            device=self.device,
            input_frames=input_frames_np,
            metric_batch_size=None,
        )

        batch_n = int(preds_flat.shape[0])
        self._num_samples += float(batch_n)

        for metric_name, metric_values in batch_metrics.items():
            if isinstance(metric_values, dict):
                self._accumulate_metric_block(metric_name, metric_values, batch_n=batch_n)

        if not compute_result:
            return base_out

        if self._num_samples <= 0:
            out = dict(base_out)
            self.reset()
            return out

        # Reduce across distributed ranks.
        reduce_device = None
        try:
            if torch.is_tensor(eval_pred.predictions):
                reduce_device = eval_pred.predictions.device
        except Exception:
            reduce_device = None
        if reduce_device is None:
            reduce_device = torch.device("cpu")

        n_tensor = torch.tensor(self._num_samples, dtype=torch.float64, device=reduce_device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
        global_n = float(n_tensor.item())

        final_rollout_metrics = {}
        for metric_name, metric_sum_dict in self._sum.items():
            final_rollout_metrics[metric_name] = {}
            for summary_kind, local_sum in metric_sum_dict.items():
                local_sumsq = self._sumsq[metric_name][summary_kind]

                sum_tensor = torch.as_tensor(local_sum, dtype=torch.float64, device=reduce_device)
                sumsq_tensor = torch.as_tensor(local_sumsq, dtype=torch.float64, device=reduce_device)

                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(sumsq_tensor, op=dist.ReduceOp.SUM)

                mean = sum_tensor / global_n
                var = torch.clamp(sumsq_tensor / global_n - mean ** 2, min=0.0)
                std = torch.sqrt(var)

                final_rollout_metrics[metric_name][summary_kind] = mean.cpu().numpy()
                std_key = summary_kind.replace("_mean", "_std")
                final_rollout_metrics[metric_name][std_key] = std.cpu().numpy()

            names_val = self._names.get((metric_name, "names"), None)
            comp_names_val = self._names.get((metric_name, "component_names"), None)
            final_rollout_metrics[metric_name]["names"] = names_val
            if comp_names_val is not None:
                final_rollout_metrics[metric_name]["component_names"] = comp_names_val

        out = dict(base_out)
        # Keep base streaming metrics and rollout metrics in separate namespaces
        # to avoid key collisions (e.g. scalar "l2" vs rollout dict "l2").
        out["rollout_metrics"] = final_rollout_metrics
        self.reset()
        return out
