import numpy as np

def l1_error(preds, targets):
    """
    Compute the L1 error (Mean Absolute Error) between predictions and targets.

    This function calculates the Mean Absolute Error (MAE) across all dimensions
    of the input arrays. The inputs are reshaped to flatten spatial dimensions
    while preserving the batch structure for efficient computation.

    Parameters
    ----------
    preds : numpy.ndarray
        Predicted values with shape (num_samples, (num_eval_rollouts_plus_one*
        label_seq_length), num_channels, *spatial_dims).
    targets : numpy.ndarray
        Ground truth values with the same shape as preds.

    Returns
    -------
    float
        The L1 error (MAE) computed as the mean of absolute differences
        between predictions and targets across all elements.

    Notes
    -----
    The L1 error is computed as:
    L1 = mean(|preds - targets|)

    Both input arrays are reshaped to (num_samples, flattened_features, -1)
    where flattened_features combines the temporal, channel, and sequence
    dimensions for efficient computation.
    """

    diff = preds - targets
    l1_error = np.mean(np.abs(diff))
    return l1_error


def l2_error(preds, targets):
    """
    Compute the L2 error (Root Mean Squared Error) between predictions and targets.

    This function calculates the Root Mean Squared Error (RMSE) across all
    dimensions of the input arrays. The inputs are reshaped to flatten spatial
    dimensions while preserving the batch structure for efficient computation.

    Parameters
    ----------
    preds : numpy.ndarray
        Predicted values with shape (num_samples, (num_eval_rollouts_plus_one*
        label_seq_length), num_channels, *spatial_dims).
    targets : numpy.ndarray
        Ground truth values with the same shape as preds.

    Returns
    -------
    float
        The L2 error (RMSE) computed as the square root of the mean squared
        differences between predictions and targets across all elements.

    Notes
    -----
    The L2 error is computed as:
    L2 = sqrt(mean((preds - targets)^2))

    Both input arrays are reshaped to (num_samples, flattened_features, -1)
    where flattened_features combines the temporal, channel, and sequence
    dimensions for efficient computation.
    """
    diff = preds - targets
    l2_error = np.mean((diff) ** 2) ** 0.5
    return l2_error


def compute_metrics_for_n_rollouts(preds, targets, outputs_per_rollout=1, metrics=("l1", "l2")):
    """Compute specified error metrics per rollout step.

    Parameters
    ----------
    preds, targets : np.ndarray
        Either:
        - Flattened arrays of identical shapes ``(B, R*T_out, C, *spatial)`` with
          ``outputs_per_rollout=T_out``, or
        - Grouped arrays of identical shapes ``(B, R, T_out, C, *spatial)``.
    outputs_per_rollout : int, optional
        Number of outputs produced per rollout step (``T_out``). Defaults to 1.
        Used to regroup flattened inputs so that metrics are computed per rollout step (R),
        aggregating across the ``T_out`` outputs of that step.
    metrics : tuple | list
        Iterable of metric identifiers to compute. Supported strings:
        - "l1": mean absolute error (MAE)
        - "l2": root-mean-squared error (RMSE)
        Custom metrics can be added by extending the `_METRIC_IMPL` dict.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        For each metric name, returns a dictionary with keys:
        - "per_step_mean":   array of shape ``(R, C+1)`` (per-channel and overall)
        - "per_step_std":    array of shape ``(R, C+1)`` (per-channel and overall)
        - "cumulative_mean": array of shape ``(R, C+1)`` cumulative over rollout steps
        - "cumulative_std":  array of shape ``(R, C+1)`` cumulative over rollout steps
    """

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

    # # Regroup to (B, R, T_out, C, *spatial)
    # if preds_arr.ndim >= 5:
    #     # Assume already grouped as (B, R, T_out, C, *spatial)
    #     grouped_preds = preds_arr
    #     grouped_targets = targets_arr
    # else:
    #     # Expect flattened as (B, R*T_out, C, *spatial)
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
    # Reshape to (B, R, T_out, C, *spatial)
    grouped_shape = (
        preds_arr.shape[0],
        num_rollouts,
        outputs_per_rollout,
        preds_arr.shape[2],
        *preds_arr.shape[3:],
    )
    grouped_preds = preds_arr.reshape(grouped_shape)
    grouped_targets = targets_arr.reshape(grouped_shape)

    difference = grouped_preds - grouped_targets  # (B, R, T_out, C, *spatial)

    # Internal metric implementation
    def _mae(values):
        return np.abs(values)

    def _mse(values):
        return values ** 2

    _METRIC_IMPL = {
        "l1": ("l1_error", _mae, False),  # (public name, elementwise_transform, use_rmse)
        "l2": ("l2_error", _mse, True),   # compute RMSE (sqrt(mean of squared error))
    }

    metric_name_to_values = {}

    # Validate dims and prepare reduction axes
    if difference.ndim < 4:
        raise ValueError("Expected grouped input of at least 4 dims (B, R, T_out, C, …).")

    # Axes:
    # 0=batch, 1=rollout step (R), 2=outputs per rollout (T_out), 3=channel (C), 4+=spatial
    spatial_axes_start = 4
    spatial_reduction_axes = tuple(range(spatial_axes_start, difference.ndim))

    # For per-channel metrics, reduce over T_out + spatial (keep channel)
    per_channel_reduction_axes = (2,) + spatial_reduction_axes
    # For overall metrics, reduce over T_out + channel + spatial
    overall_reduction_axes = (2, 3) + spatial_reduction_axes

    for metric_key in metrics:
        if metric_key not in _METRIC_IMPL:
            raise ValueError(
                f"Unknown metric '{metric_key}'. Available: {sorted(_METRIC_IMPL)}"
            )
        metric_key_name, elementwise_transform, use_rmse = _METRIC_IMPL[metric_key]

        elementwise_error = elementwise_transform(difference)  # (B, R, T_out, C, *spatial)

        # ------------------------------
        # Per-channel per-sample metric
        # ------------------------------
        # Mean over T_out and spatial -> (B, R, C)
        per_sample_channel_mean = np.mean(elementwise_error, axis=per_channel_reduction_axes)
        # For RMSE, take sqrt BEFORE batch aggregation to get per-sample RMSE
        if use_rmse:
            per_sample_channel_metric = np.sqrt(per_sample_channel_mean)
        else:
            per_sample_channel_metric = per_sample_channel_mean

        # ------------------------------
        # Overall per-sample metric
        # ------------------------------
        # Mean over T_out, channel and spatial -> (B, R)
        per_sample_overall_mean = np.mean(elementwise_error, axis=overall_reduction_axes)
        if use_rmse:
            per_sample_overall_metric = np.sqrt(per_sample_overall_mean)
        else:
            per_sample_overall_metric = per_sample_overall_mean

        # --------------------------------------
        # Per-step mean and std across batch (R)
        # --------------------------------------
        # Channel-wise: (B, R, C) -> mean/std over B -> (R, C)
        per_step_channel_mean = per_sample_channel_metric.mean(axis=0)
        per_step_channel_std = per_sample_channel_metric.std(axis=0, ddof=0)

        # Overall: (B, R) -> mean/std over B -> (R,)
        per_step_overall_mean = per_sample_overall_metric.mean(axis=0)
        per_step_overall_std = per_sample_overall_metric.std(axis=0, ddof=0)

        # Stack channel-wise with overall -> (R, C+1)
        per_step_mean = np.concatenate(
            [per_step_channel_mean, per_step_overall_mean[:, None]], axis=-1
        )
        per_step_std = np.concatenate(
            [per_step_channel_std, per_step_overall_std[:, None]], axis=-1
        )

        # --------------------------------------
        # Cumulative over rollout steps (axis=1 -> R)
        # --------------------------------------
        # Channel-wise cumulative per-sample: (B, R, C)
        cumulative_channel_per_sample = np.cumsum(per_sample_channel_metric, axis=1)
        # Overall cumulative per-sample: (B, R)
        cumulative_overall_per_sample = np.cumsum(per_sample_overall_metric, axis=1)

        # Mean/std across batch -> (R, C) and (R,)
        cumulative_channel_mean = cumulative_channel_per_sample.mean(axis=0)
        cumulative_channel_std = cumulative_channel_per_sample.std(axis=0, ddof=0)

        cumulative_overall_mean = cumulative_overall_per_sample.mean(axis=0)
        cumulative_overall_std = cumulative_overall_per_sample.std(axis=0, ddof=0)

        cumulative_mean = np.concatenate(
            [cumulative_channel_mean, cumulative_overall_mean[:, None]], axis=-1
        )
        cumulative_std = np.concatenate(
            [cumulative_channel_std, cumulative_overall_std[:, None]], axis=-1
        )

        metric_name_to_values[metric_key_name] = {
            "per_step_mean": per_step_mean,
            "per_step_std": per_step_std,
            "cumulative_mean": cumulative_mean,
            "cumulative_std": cumulative_std,
        }

    return metric_name_to_values
