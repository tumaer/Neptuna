import numpy as np
from utils.compute_stats import re_normalize_data

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

def center_of_mass_from_density(rho: np.ndarray) -> np.ndarray:
    """
    Compute 1D center of mass along x from a 2D density field.

    Parameters
    ----------
    rho : np.ndarray
        Array of shape (B, T, 1, H, W) containing density values.

    Returns
    -------
    np.ndarray
        Array of shape (B, T, 1) with x center of mass coordinates
        normalized to [0, 1].
    """
    if not isinstance(rho, np.ndarray):
        rho = np.asarray(rho)
    if rho.ndim != 5 or rho.shape[2] != 1:
        raise ValueError(f"rho must have shape (B, T, 1, H, W). Got {rho.shape}.")

    _, _, _, H, W = rho.shape
    # Coordinates in [0, 1]
    xs = np.linspace(0.0, 1.0, W, dtype=rho.dtype)
    X = np.broadcast_to(xs[None, None, None, None, :], (1, 1, 1, H, W))  # (1,1,1,H,W)

    eps = np.array(1e-12, dtype=rho.dtype)
    total_rho = np.sum(rho, axis=(3, 4)) + eps  # (B, T, 1)
    com_x = np.sum(rho * X, axis=(3, 4)) / total_rho  # (B, T, 1)
    return com_x  # (B, T, 1)

def _grouped_center_of_mass_from_density(rho_grouped: np.ndarray) -> np.ndarray:
    # rho_grouped: (B, R, T_out, 1, H, W) -> returns (B, R, T_out, 1) for x
    if rho_grouped.ndim != 6 or rho_grouped.shape[3] != 1:
        raise ValueError(f"Expected grouped rho of shape (B, R, T, 1, H, W). Got {rho_grouped.shape}.")
    _, _, _, _, H, W = rho_grouped.shape
    xs = np.linspace(0.0, 1.0, W, dtype=rho_grouped.dtype)
    X = np.broadcast_to(xs[None, None, None, None, None, :], (1, 1, 1, 1, H, W))  # (1,1,1,1,H,W)
    # Densities are already non-negative; compute COM directly
    mass = rho_grouped
    eps = np.array(1e-12, dtype=rho_grouped.dtype)
    total_mass = np.sum(mass, axis=(4, 5))  # (B, R, T_out, 1)
    com_x_num = np.sum(mass * X, axis=(4, 5))  # (B, R, T_out, 1)
    com_x = com_x_num / (total_mass + eps)  # (B, R, T_out, 1)
    return com_x  # (B, R, T_out, 1)


def idx_finder(x):
    mask = x > 0           # shape [B, T, C, X], boolean

    # Shift mask right by one to compare with previous element
    prev = np.concatenate(
            [np.zeros_like(mask[..., :1], dtype=bool),  ~mask[..., :-1]],
            axis=-1
        )

    # block_starts[b,t,c,i] = True when a new positive block begins at index i
    block_starts = mask & prev   # shape [B, T, C, X]

        # Now find the *last* block start index along axis -1
        # If no positive block exists, last occurrence = -1
    idx_last = block_starts.shape[-1] - 1 - np.argmax(
            block_starts[..., ::-1], axis=-1
        )

        # For slices with no positive values, block_starts is all False,
        # argmax returns 0 → idx_last = X-1, which is wrong.
        # Fix this by masking out cases where there was no block.
    has_block = block_starts.any(axis=-1)  # shape [B,T,C]
    idx_last = np.where(has_block, idx_last, -1)  # -1 = "no block found"

    return idx_last  # shape [B,T,C]


def compute_metrics_for_n_rollouts(
    preds,
    targets,
    outputs_per_rollout=1,
    metrics=("l1", "l2"),
    include_per_timestep: bool = False,
    norm_stats: dict = None,
    norm_strategy: str = None,
    out_channel_names: list = None,
    denormalize_for_physics: bool = True,
    dataset_name: str = None,
    grid_resolution: int = None,
    channel_names: list = None,
    metric_mode: str = "both",
    relative_epsilon: float = 1e-12,
):


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
    flat_difference = preds_arr - targets_arr      # (B, R*T_out, C, *spatial)
    
    if "Density" in channel_names:
        ############################################################
        #specific for rho
        preds_rho = preds_arr[:, :, 0:1, :, :]
        targets_rho = targets_arr[:, :, 0:1, :, :]
        
        grouped_preds_rho = grouped_preds[:, :, :, 0:1, :, :]
        grouped_targets_rho = grouped_targets[:, :, :, 0:1, :, :]
        # Optionally re-normalize rho for physics-based metrics (COM, gradients)
        if (
            denormalize_for_physics
            and norm_stats is not None
            and norm_strategy is not None
            and out_channel_names is not None
            and len(out_channel_names) > 0
        ):
            preds_rho = re_normalize_data(preds_rho, norm_stats["Density"], norm_strategy)
            preds_rho[preds_rho < 0] = 0
            targets_rho = re_normalize_data(targets_rho, norm_stats["Density"], norm_strategy)

            grouped_preds_rho = re_normalize_data(grouped_preds_rho, norm_stats["Density"], norm_strategy)
            grouped_preds_rho[grouped_preds_rho < 0] = 0
            grouped_targets_rho = re_normalize_data(grouped_targets_rho, norm_stats["Density"], norm_strategy)

        difference_rho = grouped_preds_rho - grouped_targets_rho
        flat_difference_rho = preds_rho - targets_rho

        ############################################################
        #specific for grad_rho
        preds_grad_rho_x = np.gradient(preds_rho, axis=3)
        preds_grad_rho_y = np.gradient(preds_rho, axis=4)
        preds_rho_grad = np.sqrt(preds_grad_rho_x**2 + preds_grad_rho_y**2)

        grouped_preds_grad_rho_x = np.gradient(grouped_preds_rho, axis=4)
        grouped_preds_grad_rho_y = np.gradient(grouped_preds_rho, axis=5)
        grouped_preds_grad_rho = np.sqrt(grouped_preds_grad_rho_x**2 + grouped_preds_grad_rho_y**2)

        targets_rho_grad_x = np.gradient(targets_rho, axis=3)
        targets_rho_grad_y = np.gradient(targets_rho, axis=4)
        targets_rho_grad = np.sqrt(targets_rho_grad_x**2 + targets_rho_grad_y**2)

        grouped_targets_grad_rho_x = np.gradient(grouped_targets_rho, axis=4)
        grouped_targets_grad_rho_y = np.gradient(grouped_targets_rho, axis=5)
        grouped_targets_grad_rho = np.sqrt(grouped_targets_grad_rho_x**2 + grouped_targets_grad_rho_y**2)

        difference_grad_rho = grouped_preds_grad_rho - grouped_targets_grad_rho
        flat_difference_grad_rho = preds_rho_grad - targets_rho_grad
        ############################################################

        ############################################################
        # Droplet X-Center of mass metrics from rho
        ############################################################
        if dataset_name == "Aerobreakup":
            grouped_com_preds = _grouped_center_of_mass_from_density(grouped_preds_rho)
            grouped_com_preds = grouped_com_preds[:, :, :, :, None, None]
            grouped_com_targets = _grouped_center_of_mass_from_density(grouped_targets_rho)
            grouped_com_targets = grouped_com_targets[:, :, :, :, None, None]
            difference_com = grouped_com_preds - grouped_com_targets  # (B, R, T_out, 1, 1, 1)

            flat_com_preds = center_of_mass_from_density(preds_rho)  # (B, T_flat, 1)
            flat_com_targets = center_of_mass_from_density(targets_rho)  # (B, T_flat, 1)
            flat_com_preds = flat_com_preds[:, :, :, None, None]
            flat_com_targets = flat_com_targets[:, :, :, None, None]
            flat_difference_com = flat_com_preds - flat_com_targets  # (B, T_flat, 1, 1, 1)
        ############################################################

        ############################################################
        # Droplet-Outer radius from rho
        ############################################################
        if dataset_name == "LaserDroplet":
            preds_axis = preds_rho[..., 0] #extract the first column of the rho field
            preds_axis_diff = preds_axis[..., :1] - preds_axis[..., :-1]
            preds_rad_idx = idx_finder(preds_axis_diff)/grid_resolution[0] # [B, Rollout, C]
            preds_rad_idx = preds_rad_idx[:, :, :, None, None]

            grouped_preds_axis = grouped_preds_rho[..., 0]
            grouped_preds_axis_diff = grouped_preds_axis[..., :1] - grouped_preds_axis[..., :-1]
            grouped_preds_rad_idx = idx_finder(grouped_preds_axis_diff)/grid_resolution[0] # [grouped, B, Rollout, C]
            grouped_preds_rad_idx = grouped_preds_rad_idx[:, :, :, :, None, None]
    
            targets_axis = targets_rho[..., 0]
            targets_axis_diff = targets_axis[..., :1] - targets_axis[..., :-1]
            targets_rad_idx = idx_finder(targets_axis_diff)/grid_resolution[0] # [B, Rollout, C]
            targets_rad_idx = targets_rad_idx[:, :, :, None, None]

            grouped_targets_axis = grouped_targets_rho[..., 0]
            grouped_targets_axis_diff = grouped_targets_axis[..., :1] - grouped_targets_axis[..., :-1]
            grouped_targets_rad_idx = idx_finder(grouped_targets_axis_diff)/grid_resolution[0] # [Grouped, B, Rollout, C]
            grouped_targets_rad_idx = grouped_targets_rad_idx[:, :, :, :, None, None]

            difference_outer_radius = grouped_preds_rad_idx - grouped_targets_rad_idx
            flat_difference_outer_radius = preds_rad_idx - targets_rad_idx
    
    #Kinetic Energy using U,V and density
    if "Velocity_X" in channel_names and "Velocity_Y" in channel_names and "Density" in channel_names:
        preds_vel_x = preds_arr[:, :, 2:3, :, :]
        preds_vel_y = preds_arr[:, :, 3:4, :, :]
        targets_vel_x = targets_arr[:, :, 2:3, :, :]
        targets_vel_y = targets_arr[:, :, 3:4, :, :]

        grouped_preds_vel_x = grouped_preds[:, :, :, 2:3, :, :]
        grouped_preds_vel_y = grouped_preds[:, :, :, 3:4, :, :]
        grouped_targets_vel_x = grouped_targets[:, :, :, 2:3, :, :]
        grouped_targets_vel_y = grouped_targets[:, :, :, 3:4, :, :]

        if (
            denormalize_for_physics
            and norm_stats is not None
            and norm_strategy is not None
            and out_channel_names is not None
            and len(out_channel_names) > 0
        ):
            preds_vel_x = re_normalize_data(preds_vel_x, norm_stats["Velocity_X"], norm_strategy)
            preds_vel_y = re_normalize_data(preds_vel_y, norm_stats["Velocity_Y"], norm_strategy)
            targets_vel_x = re_normalize_data(targets_vel_x, norm_stats["Velocity_X"], norm_strategy)
            targets_vel_y = re_normalize_data(targets_vel_y, norm_stats["Velocity_Y"], norm_strategy)

            grouped_preds_vel_x = re_normalize_data(grouped_preds_vel_x, norm_stats["Velocity_X"], norm_strategy)
            grouped_preds_vel_y = re_normalize_data(grouped_preds_vel_y, norm_stats["Velocity_Y"], norm_strategy)
            grouped_targets_vel_x = re_normalize_data(grouped_targets_vel_x, norm_stats["Velocity_X"], norm_strategy)
            grouped_targets_vel_y = re_normalize_data(grouped_targets_vel_y, norm_stats["Velocity_Y"], norm_strategy)

        if dataset_name == "LaserDroplet":
            preds_KE = (0.5 * ( preds_vel_x**2 + preds_vel_y**2 ) * preds_rho * (1.25e-07)**2).sum(axis=(3,4))
            preds_KE = preds_KE[:, :, :, None, None]
            targets_KE = (0.5 * ( targets_vel_x**2 + targets_vel_y**2 ) * targets_rho * (1.25e-07)**2).sum(axis=(3,4))
            targets_KE = targets_KE[:, :, :, None, None]
            
            grouped_preds_KE = (0.5 * ( grouped_preds_vel_x**2 + grouped_preds_vel_y**2 ) * grouped_preds_rho * (1.25e-07)**2 ).sum(axis=(4,5))
            grouped_preds_KE = grouped_preds_KE[:, :, :, :, None, None]
            grouped_targets_KE = (0.5 * ( grouped_targets_vel_x**2 + grouped_targets_vel_y**2 ) * grouped_targets_rho * (1.25e-07)**2  ).sum(axis=(4,5))
            grouped_targets_KE = grouped_targets_KE[:, :, :, :, None, None]

        if dataset_name == "Aerobreakup":
            preds_KE = (0.5 * ( preds_vel_x**2 + preds_vel_y**2 ) * preds_rho * (1.171875000e-05)**2).sum(axis=(3,4))
            preds_KE = preds_KE[:, :, :, None, None]
            targets_KE = (0.5 * ( targets_vel_x**2 + targets_vel_y**2 ) * targets_rho * (1.171875000e-05)**2 ).sum(axis=(3,4))
            targets_KE = targets_KE[:, :, :, None, None]
            
            grouped_preds_KE = (0.5 * ( grouped_preds_vel_x**2 + grouped_preds_vel_y**2 ) * grouped_preds_rho * (1.171875000e-05)**2 ).sum(axis=(4,5))
            grouped_preds_KE = grouped_preds_KE[:, :, :, :, None, None]
            grouped_targets_KE = (0.5 * ( grouped_targets_vel_x**2 + grouped_targets_vel_y**2 ) * grouped_targets_rho * (1.171875000e-05)**2  ).sum(axis=(4,5))
            grouped_targets_KE = grouped_targets_KE[:, :, :, :, None, None]

        difference_KE = grouped_preds_KE - grouped_targets_KE
        flat_difference_KE = preds_KE - targets_KE
    
    #Vorticity production using U,V
    if "Velocity_X" in channel_names and "Velocity_Y" in channel_names:
        preds_vel_x = preds_arr[:, :, 2:3, :, :]
        preds_vel_y = preds_arr[:, :, 3:4, :, :]
        targets_vel_x = targets_arr[:, :, 2:3, :, :]
        targets_vel_y = targets_arr[:, :, 3:4, :, :]

        grouped_preds_vel_x = grouped_preds[:, :, :, 2:3, :, :]
        grouped_preds_vel_y = grouped_preds[:, :, :, 3:4, :, :]
        grouped_targets_vel_x = grouped_targets[:, :, :, 2:3, :, :]
        grouped_targets_vel_y = grouped_targets[:, :, :, 3:4, :, :]

        if (
            denormalize_for_physics
            and norm_stats is not None
            and norm_strategy is not None
            and out_channel_names is not None
            and len(out_channel_names) > 0
        ):
            preds_vel_x = re_normalize_data(preds_vel_x, norm_stats["Velocity_X"], norm_strategy)
            preds_vel_y = re_normalize_data(preds_vel_y, norm_stats["Velocity_Y"], norm_strategy)
            targets_vel_x = re_normalize_data(targets_vel_x, norm_stats["Velocity_X"], norm_strategy)
            targets_vel_y = re_normalize_data(targets_vel_y, norm_stats["Velocity_Y"], norm_strategy)

            grouped_preds_vel_x = re_normalize_data(grouped_preds_vel_x, norm_stats["Velocity_X"], norm_strategy)
            grouped_preds_vel_y = re_normalize_data(grouped_preds_vel_y, norm_stats["Velocity_Y"], norm_strategy)
            grouped_targets_vel_x = re_normalize_data(grouped_targets_vel_x, norm_stats["Velocity_X"], norm_strategy)
            grouped_targets_vel_y = re_normalize_data(grouped_targets_vel_y, norm_stats["Velocity_Y"], norm_strategy)

        dpreds_u_dy, dpreds_u_dx = np.gradient(preds_vel_x, axis=(-2, -1))
        dpreds_v_dy, dpreds_v_dx = np.gradient(preds_vel_y, axis=(-2, -1))
        preds_omega = dpreds_v_dx - dpreds_u_dy
        preds_VP = (preds_omega * preds_omega).mean(axis=(3,4))
        preds_VP = preds_VP[:, :, :, None, None]

        dtargets_u_dy, dtargets_u_dx = np.gradient(targets_vel_x, axis=(-2, -1))
        dtargets_v_dy, dtargets_v_dx = np.gradient(targets_vel_y, axis=(-2, -1))
        targets_omega = dtargets_v_dx - dtargets_u_dy
        targets_VP = (targets_omega * targets_omega).mean(axis=(3,4))
        targets_VP = targets_VP[:, :, :, None, None]

        # Grouped vorticity production (VP)
        grouped_dpreds_u_dy, grouped_dpreds_u_dx = np.gradient(grouped_preds_vel_x, axis=(4, 5))
        grouped_dpreds_v_dy, grouped_dpreds_v_dx = np.gradient(grouped_preds_vel_y, axis=(4, 5))
        grouped_preds_omega = grouped_dpreds_v_dx - grouped_dpreds_u_dy
        grouped_preds_VP = (grouped_preds_omega * grouped_preds_omega).mean(axis=(4, 5))
        grouped_preds_VP = grouped_preds_VP[:, :, :, :, None, None]

        grouped_dtargets_u_dy, grouped_dtargets_u_dx = np.gradient(grouped_targets_vel_x, axis=(4, 5))
        grouped_dtargets_v_dy, grouped_dtargets_v_dx = np.gradient(grouped_targets_vel_y, axis=(4, 5))
        grouped_targets_omega = grouped_dtargets_v_dx - grouped_dtargets_u_dy
        grouped_targets_VP = (grouped_targets_omega * grouped_targets_omega).mean(axis=(4, 5))
        grouped_targets_VP = grouped_targets_VP[:, :, :, :, None, None]

        difference_VP = grouped_preds_VP - grouped_targets_VP
        flat_difference_VP = preds_VP - targets_VP

    ############################################################
    # Map logical difference pairs so metrics can choose which to use
    diff_pairs = {
        "default": (difference, flat_difference),
        "rho": (difference_rho, flat_difference_rho) if "Density" in channel_names else None,
        "grad_rho": (difference_grad_rho, flat_difference_grad_rho) if "Density" in channel_names else None,
        "x_com": (difference_com, flat_difference_com) if dataset_name == "Aerobreakup" and "Density" in channel_names else None,
        "outer_radius": (difference_outer_radius, flat_difference_outer_radius) if dataset_name == "LaserDroplet" and "Density" in channel_names else None,
        "KE": (difference_KE, flat_difference_KE) if "Velocity_X" in channel_names and "Velocity_Y" in channel_names and "Density" in channel_names else None,
        "VP": (difference_VP, flat_difference_VP) if "Velocity_X" in channel_names and "Velocity_Y" in channel_names else None,
    }
    # Map the corresponding targets, needed for relative error normalization
    target_pairs = {
        "default": (grouped_targets, targets_arr),
        "rho": (grouped_targets_rho, targets_rho) if "Density" in channel_names else None,
        "grad_rho": (grouped_targets_grad_rho, targets_rho_grad) if "Density" in channel_names else None,
        "x_com": (grouped_com_targets, flat_com_targets) if dataset_name == "Aerobreakup" and "Density" in channel_names else None,
        "outer_radius": (grouped_targets_rad_idx, targets_rad_idx) if dataset_name == "LaserDroplet" and "Density" in channel_names else None,
        "KE": (grouped_targets_KE, targets_KE) if "Velocity_X" in channel_names and "Velocity_Y" in channel_names and "Density" in channel_names else None,
        "VP": (grouped_targets_VP, targets_VP) if "Velocity_X" in channel_names and "Velocity_Y" in channel_names else None,
    }
    # Internal metric implementation
    def _mae(values):
        return np.abs(values)

    def _mse(values):
        return values ** 2

    # Validate metric_mode
    if metric_mode not in ("absolute", "relative", "both"):
        raise ValueError(f"metric_mode must be one of 'absolute', 'relative', 'both'. Got: {metric_mode}")

    # Build metric dictionary conditionally based on available channels/dataset
    _METRIC_IMPL = {
        "l1": ("l1_error", _mae, False, "default"),
        "l2": ("l2_error", _mse, True, "default"),
    }

    if "Density" in channel_names:
        _METRIC_IMPL.update({
            "l1_rho": ("l1_rho_error", _mae, False, "rho"),
            "l2_rho": ("l2_rho_error", _mse, True, "rho"),
            "l1_grad_rho": ("l1_grad_rho_error", _mae, False, "grad_rho"),
            "l2_grad_rho": ("l2_grad_rho_error", _mse, True, "grad_rho"),
        })
        if dataset_name == "Aerobreakup":
            _METRIC_IMPL.update({
                "l1_x_com": ("l1_x_com_error", _mae, False, "x_com"),
                "l2_x_com": ("l2_x_com_error", _mse, True, "x_com"),
            })
        if dataset_name == "LaserDroplet":
            _METRIC_IMPL.update({
                "l1_drop_outer_rad": ("l1_drop_outer_rad_error", _mae, False, "outer_radius"),
                "l2_drop_outer_rad": ("l2_drop_outer_rad_error", _mse, True, "outer_radius"),
            })

    if "Velocity_X" in channel_names and "Velocity_Y" in channel_names:
        _METRIC_IMPL.update({
            "l1_VP": ("l1_VP_error", _mae, False, "VP"),
            "l2_VP": ("l2_VP_error", _mse, True, "VP"),
        })
        if "Density" in channel_names:  # KE needs U, V, and rho
            _METRIC_IMPL.update({
                "l1_KE": ("l1_KE_error", _mae, False, "KE"),
                "l2_KE": ("l2_KE_error", _mse, True, "KE"),
            })

    # Optionally filter the requested metrics to only those available
    metrics = [m for m in metrics if m in _METRIC_IMPL]

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
        metric_key_name, elementwise_transform, use_rmse, diff_pair_key = _METRIC_IMPL[metric_key]
        grouped_diff, flat_diff = diff_pairs[diff_pair_key]
        grouped_target, flat_target = target_pairs[diff_pair_key]

        # Absolute metrics
        if metric_mode in ("absolute", "both"):
            elem_err_abs = elementwise_transform(grouped_diff)  # (B, R, T_out, C, *spatial)

            # ------------------------------
            # Per-channel per-sample metric
            # ------------------------------
            # Mean over T_out and spatial -> (B, R, C)
            per_sample_channel_mean = np.mean(elem_err_abs, axis=per_channel_reduction_axes)
            # For RMSE, take sqrt BEFORE batch aggregation to get per-sample RMSE
            if use_rmse:
                per_sample_channel_metric = np.sqrt(per_sample_channel_mean)
            else:
                per_sample_channel_metric = per_sample_channel_mean

            # ------------------------------
            # Overall per-sample metric
            # ------------------------------
            # Mean over T_out, channel and spatial -> (B, R)
            per_sample_overall_mean = np.mean(elem_err_abs, axis=overall_reduction_axes)
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
            per_step_mean = np.concatenate([per_step_channel_mean, per_step_overall_mean[:, None]], axis=-1)
            per_step_std = np.concatenate([per_step_channel_std, per_step_overall_std[:, None]], axis=-1)

            # --------------------------------------
            # Cumulative over rollout steps (axis=1 -> R)
            # --------------------------------------
            # Channel-wise cumulative per-sample: (B, R, C)
            cumulative_channel_per_sample = np.cumsum(per_sample_channel_metric, axis=1)
            # Overall cumulative per-sample: (B, R)
            cumulative_overall_per_sample = np.cumsum(per_sample_overall_metric, axis=1)
            # Mean/std across batch -> (R, C) and (R,)
            cumulative_channel_mean = cumulative_channel_per_sample.mean(axis=0) # (R, C)
            cumulative_channel_std = cumulative_channel_per_sample.std(axis=0, ddof=0) # (R, C)
            cumulative_overall_mean = cumulative_overall_per_sample.mean(axis=0) # (R,)
            cumulative_overall_std = cumulative_overall_per_sample.std(axis=0, ddof=0) # (R,)
            # Stack channel-wise with overall -> (R, C+1)
            cumulative_mean = np.concatenate([cumulative_channel_mean, cumulative_overall_mean[:, None]], axis=-1)
            cumulative_std = np.concatenate([cumulative_channel_std, cumulative_overall_std[:, None]], axis=-1)

            abs_result = {
                "per_rollout_step_mean": per_step_mean,
                "per_rollout_step_std": per_step_std,
                "cumulative_rollout_step_mean": cumulative_mean,
                "cumulative_rollout_step_std": cumulative_std,
            }

            # ------------------------------------------------------------------
            # Optional: Per-timestep metrics using flattened arrays (B, R*T_out, C,…)
            # T_flat = R*T_out
            # Here the output_sequence_length is always 1, only the stride shows the jump between timesteps
            # ------------------------------------------------------------------
            if include_per_timestep:
                elem_err_flat = elementwise_transform(flat_diff)  # (B, T_flat, C, *spatial)

                # Spatial axes start at dim=3 for flattened arrays
                spatial_axes_start_flat = 3
                spatial_reduction_axes_flat = tuple(range(spatial_axes_start_flat, elem_err_flat.ndim))

                # Per-sample per-timestep per-channel mean over spatial -> (B, T_flat, C)
                per_sample_channel_mean_flat = np.mean(elem_err_flat, axis=spatial_reduction_axes_flat)
                if use_rmse:
                    per_sample_channel_metric_flat = np.sqrt(per_sample_channel_mean_flat)
                else:
                    per_sample_channel_metric_flat = per_sample_channel_mean_flat

                # Overall per-sample per-timestep mean over spatial+channel -> (B, T_flat)
                overall_reduction_axes_flat = (2,) + spatial_reduction_axes_flat
                per_sample_overall_mean_flat = np.mean(elem_err_flat, axis=overall_reduction_axes_flat)
                if use_rmse:
                    per_sample_overall_metric_flat = np.sqrt(per_sample_overall_mean_flat)
                else:
                    per_sample_overall_metric_flat = per_sample_overall_mean_flat

                # Aggregate across batch resulting in shapes -> (T_flat, C) and (T_flat,)
                per_timestep_channel_mean = per_sample_channel_metric_flat.mean(axis=0)
                per_timestep_channel_std = per_sample_channel_metric_flat.std(axis=0, ddof=0)
                per_timestep_overall_mean = per_sample_overall_metric_flat.mean(axis=0)
                per_timestep_overall_std = per_sample_overall_metric_flat.std(axis=0, ddof=0)

                per_timestep_mean = np.concatenate(
                    [per_timestep_channel_mean, per_timestep_overall_mean[:, None]], axis=-1
                )
                per_timestep_std = np.concatenate(
                    [per_timestep_channel_std, per_timestep_overall_std[:, None]], axis=-1
                )

                # Cumulative over timesteps (time axis=1 -> T_flat)
                cumulative_channel_per_sample_flat = np.cumsum(per_sample_channel_metric_flat, axis=1) # (B, T_flat, C)
                cumulative_overall_per_sample_flat = np.cumsum(per_sample_overall_metric_flat, axis=1) # (B, T_flat)

                cumulative_timestep_channel_mean = cumulative_channel_per_sample_flat.mean(axis=0) # (T_flat, C)
                cumulative_timestep_channel_std = cumulative_channel_per_sample_flat.std(axis=0, ddof=0) # (T_flat, C)
                cumulative_timestep_overall_mean = cumulative_overall_per_sample_flat.mean(axis=0) # (T_flat,)
                cumulative_timestep_overall_std = cumulative_overall_per_sample_flat.std(axis=0, ddof=0) # (T_flat,)

                cumulative_timestep_mean = np.concatenate(
                    [cumulative_timestep_channel_mean, cumulative_timestep_overall_mean[:, None]], axis=-1
                )
                cumulative_timestep_std = np.concatenate(
                    [cumulative_timestep_channel_std, cumulative_timestep_overall_std[:, None]], axis=-1
                )

                abs_result.update({
                    "per_timestep_mean": per_timestep_mean,
                    "per_timestep_std": per_timestep_std,
                    "cumulative_timestep_mean": cumulative_timestep_mean,
                    "cumulative_timestep_std": cumulative_timestep_std,
                })

            metric_name_to_values[metric_key_name] = abs_result

        # Relative metrics
        if metric_mode in ("relative", "both"):
            # Compute relative error as ratio of norms:
            # L1-relative: sum(|Y_true - Y_pred|) / (sum(|Y_true|) + eps)
            # L2-relative: sqrt(sum((Y_true - Y_pred)^2) / (sum(Y_true^2) + eps))
            eps_group = np.array(relative_epsilon, dtype=grouped_target.dtype)
            # ------------------------------
            # Per-channel per-sample metric
            # ------------------------------
            if not use_rmse:
                # L1 norm ratios over T_out + spatial dims
                num_pc = np.sum(np.abs(grouped_diff), axis=per_channel_reduction_axes)
                den_pc = np.sum(np.abs(grouped_target), axis=per_channel_reduction_axes) + eps_group
                per_sample_channel_metric_rel = num_pc / den_pc
                # Overall (reduce also over channel)
                num_overall = np.sum(np.abs(grouped_diff), axis=overall_reduction_axes)
                den_overall = np.sum(np.abs(grouped_target), axis=overall_reduction_axes) + eps_group
                per_sample_overall_metric_rel = num_overall / den_overall
            else:
                # L2 norm ratios over T_out + spatial dims
                num_pc_sq = np.sum(grouped_diff ** 2, axis=per_channel_reduction_axes)
                den_pc_sq = np.sum(grouped_target ** 2, axis=per_channel_reduction_axes) + eps_group
                per_sample_channel_metric_rel = np.sqrt(num_pc_sq / den_pc_sq)
                # Overall (reduce also over channel)
                num_overall_sq = np.sum(grouped_diff ** 2, axis=overall_reduction_axes)
                den_overall_sq = np.sum(grouped_target ** 2, axis=overall_reduction_axes) + eps_group
                per_sample_overall_metric_rel = np.sqrt(num_overall_sq / den_overall_sq)

            # --------------------------------------
            # Per-step mean and std across batch (R)
            # --------------------------------------
            # Channel-wise: (B, R, C) -> mean/std over B -> (R, C)
            per_step_channel_mean = per_sample_channel_metric_rel.mean(axis=0)
            per_step_channel_std = per_sample_channel_metric_rel.std(axis=0, ddof=0)
            # Overall: (B, R) -> mean/std over B -> (R,)
            per_step_overall_mean = per_sample_overall_metric_rel.mean(axis=0)
            per_step_overall_std = per_sample_overall_metric_rel.std(axis=0, ddof=0)
            # Stack channel-wise with overall -> (R, C+1)
            per_step_mean = np.concatenate([per_step_channel_mean, per_step_overall_mean[:, None]], axis=-1)
            per_step_std = np.concatenate([per_step_channel_std, per_step_overall_std[:, None]], axis=-1)

            # --------------------------------------
            # Cumulative over rollout steps (axis=1 -> R)
            # --------------------------------------
            # Channel-wise cumulative per-sample: (B, R, C)
            cumulative_channel_per_sample = np.cumsum(per_sample_channel_metric_rel, axis=1)
            # Overall cumulative per-sample: (B, R)
            cumulative_overall_per_sample = np.cumsum(per_sample_overall_metric_rel, axis=1)
            # Mean/std across batch -> (R, C) and (R,)
            cumulative_channel_mean = cumulative_channel_per_sample.mean(axis=0) # (R, C)
            cumulative_channel_std = cumulative_channel_per_sample.std(axis=0, ddof=0) # (R, C)
            cumulative_overall_mean = cumulative_overall_per_sample.mean(axis=0) # (R,)
            cumulative_overall_std = cumulative_overall_per_sample.std(axis=0, ddof=0) # (R,)
            # Stack channel-wise with overall -> (R, C+1)
            cumulative_mean = np.concatenate([cumulative_channel_mean, cumulative_overall_mean[:, None]], axis=-1)
            cumulative_std = np.concatenate([cumulative_channel_std, cumulative_overall_std[:, None]], axis=-1)

            rel_result = {
                "per_rollout_step_mean": per_step_mean,
                "per_rollout_step_std": per_step_std,
                "cumulative_rollout_step_mean": cumulative_mean,
                "cumulative_rollout_step_std": cumulative_std,
            }

            # ------------------------------------------------------------------
            # Optional: Per-timestep metrics using flattened arrays (B, R*T_out, C,…)
            # T_flat = R*T_out
            # Here the output_sequence_length is always 1, only the stride shows the jump between timesteps
            # ------------------------------------------------------------------
            if include_per_timestep:
                eps_flat = np.array(relative_epsilon, dtype=flat_target.dtype)
                # Spatial axes start at dim=3 for flattened arrays
                spatial_axes_start_flat = 3
                spatial_reduction_axes_flat = tuple(range(spatial_axes_start_flat, flat_diff.ndim))
                if not use_rmse:
                    # L1 norm ratios over spatial dims
                    num_pc_flat = np.sum(np.abs(flat_diff), axis=spatial_reduction_axes_flat)  # (B, T_flat, C)
                    den_pc_flat = np.sum(np.abs(flat_target), axis=spatial_reduction_axes_flat) + eps_flat  # (B, T_flat, C)
                    per_sample_channel_metric_flat_rel = num_pc_flat / den_pc_flat
                    # Overall: reduce over channel+spatial
                    overall_reduction_axes_flat = (2,) + spatial_reduction_axes_flat
                    num_overall_flat = np.sum(np.abs(flat_diff), axis=overall_reduction_axes_flat)  # (B, T_flat)
                    den_overall_flat = np.sum(np.abs(flat_target), axis=overall_reduction_axes_flat) + eps_flat  # (B, T_flat)
                    per_sample_overall_metric_flat_rel = num_overall_flat / den_overall_flat
                else:
                    # L2 norm ratios over spatial dims
                    num_pc_flat_sq = np.sum(flat_diff ** 2, axis=spatial_reduction_axes_flat)  # (B, T_flat, C)
                    den_pc_flat_sq = np.sum(flat_target ** 2, axis=spatial_reduction_axes_flat) + eps_flat  # (B, T_flat, C)
                    per_sample_channel_metric_flat_rel = np.sqrt(num_pc_flat_sq / den_pc_flat_sq)
                    # Overall: reduce over channel+spatial
                    overall_reduction_axes_flat = (2,) + spatial_reduction_axes_flat
                    num_overall_flat_sq = np.sum(flat_diff ** 2, axis=overall_reduction_axes_flat)  # (B, T_flat)
                    den_overall_flat_sq = np.sum(flat_target ** 2, axis=overall_reduction_axes_flat) + eps_flat  # (B, T_flat)
                    per_sample_overall_metric_flat_rel = np.sqrt(num_overall_flat_sq / den_overall_flat_sq)

                # Aggregate across batch resulting in shapes -> (T_flat, C) and (T_flat,)
                per_timestep_channel_mean = per_sample_channel_metric_flat_rel.mean(axis=0)
                per_timestep_channel_std = per_sample_channel_metric_flat_rel.std(axis=0, ddof=0)
                per_timestep_overall_mean = per_sample_overall_metric_flat_rel.mean(axis=0)
                per_timestep_overall_std = per_sample_overall_metric_flat_rel.std(axis=0, ddof=0)

                per_timestep_mean = np.concatenate([per_timestep_channel_mean, per_timestep_overall_mean[:, None]], axis=-1)
                per_timestep_std = np.concatenate([per_timestep_channel_std, per_timestep_overall_std[:, None]], axis=-1)

                # Cumulative over timesteps (time axis=1 -> T_flat)
                cumulative_channel_per_sample_flat = np.cumsum(per_sample_channel_metric_flat_rel, axis=1) # (B, T_flat, C)
                cumulative_overall_per_sample_flat = np.cumsum(per_sample_overall_metric_flat_rel, axis=1) # (B, T_flat)

                cumulative_timestep_channel_mean = cumulative_channel_per_sample_flat.mean(axis=0) # (T_flat, C)
                cumulative_timestep_channel_std = cumulative_channel_per_sample_flat.std(axis=0, ddof=0) # (T_flat, C)
                cumulative_timestep_overall_mean = cumulative_overall_per_sample_flat.mean(axis=0) # (T_flat,)
                cumulative_timestep_overall_std = cumulative_overall_per_sample_flat.std(axis=0, ddof=0) # (T_flat,)

                cumulative_timestep_mean = np.concatenate(
                    [cumulative_timestep_channel_mean, cumulative_timestep_overall_mean[:, None]], axis=-1
                )
                cumulative_timestep_std = np.concatenate(
                    [cumulative_timestep_channel_std, cumulative_timestep_overall_std[:, None]], axis=-1
                )

                rel_result.update({
                    "per_timestep_mean": per_timestep_mean,
                    "per_timestep_std": per_timestep_std,
                    "cumulative_timestep_mean": cumulative_timestep_mean,
                    "cumulative_timestep_std": cumulative_timestep_std,
                })

            metric_name_to_values["rel_" + metric_key_name] = rel_result

    return metric_name_to_values