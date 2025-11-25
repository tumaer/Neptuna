import numpy as np
import torch
from typing import Dict, List, Optional, Union, Tuple

def compute_loss_metrics_for_n_rollouts(
    preds: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    loss_fn,  # CompositeLoss instance
    outputs_per_rollout: int = 1,
    include_per_timestep: bool = False,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute loss metrics per rollout step using the new loss framework.

    Parameters
    ----------
    preds : np.ndarray or torch.Tensor
        Predictions with shape ``(B, R*T_out, C, *spatial)`` or ``(B, R, T_out, C, *spatial)``.
    targets : np.ndarray or torch.Tensor
        Ground truth with the same shape as preds.
    loss_fn : CompositeLoss
        The composite loss function instance from fetch_loss_metric().
    outputs_per_rollout : int, optional
        Number of outputs produced per rollout step (``T_out``). Defaults to 1.
    include_per_timestep : bool, optional
        When True, compute per-timestep metrics using flattened arrays.
    device : str, optional
        Device to run computations on. Defaults to 'cuda' if available.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        For each loss component (and 'total'), returns a dictionary with keys:
        - "per_rollout_step_mean":   array of shape ``(R, C+1)`` (per-channel and overall)
        - "per_rollout_step_std":    array of shape ``(R, C+1)`` (per-channel and overall)
        - "cumulative_rollout_step_mean": array of shape ``(R, C+1)`` cumulative over rollout steps
        - "cumulative_rollout_step_std":  array of shape ``(R, C+1)`` cumulative over rollout steps
        If include_per_timestep=True, also includes:
        - "per_timestep_mean":   array of shape ``(R*T_out, C+1)``
        - "per_timestep_std":    array of shape ``(R*T_out, C+1)``
        - "cumulative_timestep_mean": array of shape ``(R*T_out, C+1)``
        - "cumulative_timestep_std":  array of shape ``(R*T_out, C+1)``
    """
    # Convert to torch tensors if needed
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).float()
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets).float()
    
    preds = preds.to(device)
    targets = targets.to(device)
    
    if preds.shape != targets.shape:
        raise ValueError(
            f"Predictions and targets must have the same shape. "
            f"Got preds.shape={preds.shape}, targets.shape={targets.shape}."
        )
    
    if preds.ndim < 3:
        raise ValueError("Expected input of at least 3 dims (B, T, C, …).")
    
    total_steps = preds.shape[1]
    if total_steps % outputs_per_rollout != 0:
        raise ValueError(
            f"Time dimension (T={total_steps}) is not divisible by outputs_per_rollout={outputs_per_rollout}."
        )
    
    num_rollouts = total_steps // outputs_per_rollout
    batch_size = preds.shape[0]
    num_channels = preds.shape[2]
    
    # Reshape to (B, R, T_out, C, *spatial)
    grouped_shape = (
        batch_size,
        num_rollouts,
        outputs_per_rollout,
        num_channels,
        *preds.shape[3:],
    )
    grouped_preds = preds.reshape(grouped_shape)
    grouped_targets = targets.reshape(grouped_shape)
    
    metric_results = {}
    
    # -------------------------------------------------------------------------
    # Per-rollout metrics
    # -------------------------------------------------------------------------
    # For each rollout step r, compute loss on slices [:, r, :, :, ...]
    # which have shape (B, T_out, C, *spatial)
    
    # Storage for per-sample per-rollout losses
    # Shape will be (B, R) for total, (B, R, num_components) for detailed
    model = None  # Loss functions don't need the model for evaluation
    
    # Get loss component names
    component_names = [comp.name for comp in loss_fn.loss_components]
    
    # Initialize storage: per_sample_losses[component_name] = (B, R)
    per_sample_losses = {name: [] for name in component_names}
    per_sample_losses['total'] = []
    
    # Also store per-channel losses if components support detailed breakdown
    per_sample_channel_losses = {name: [] for name in component_names}
    
    # Process each rollout step
    for r in range(num_rollouts):
        # Extract slice for this rollout: (B, T_out, C, *spatial)
        preds_r = grouped_preds[:, r, :, :, ...]
        targets_r = grouped_targets[:, r, :, :, ...]
        
        # Compute loss with detailed breakdown
        with torch.no_grad():
            total_loss, detailed = loss_fn(model, preds_r, targets_r, return_detailed=True)
        
        # Extract per-sample losses for each component
        for comp_name in component_names:
            comp_total = detailed[comp_name]['total']  # Should be scalar or (B,)
            
            if comp_total.ndim == 0:
                # Scalar loss - replicate for all samples
                comp_total = comp_total.unsqueeze(0).expand(batch_size)
            
            per_sample_losses[comp_name].append(comp_total.cpu().numpy())
            
            # Try to get per-channel breakdown if available
            comp_detailed = detailed[comp_name]
            if 'per_channel' in comp_detailed:
                per_channel = comp_detailed['per_channel']  # (C,) or (B, C)
                if per_channel.ndim == 1:
                    per_channel = per_channel.unsqueeze(0).expand(batch_size, -1)
                per_sample_channel_losses[comp_name].append(per_channel.cpu().numpy())
        
        # Store total loss
        if total_loss.ndim == 0:
            total_loss = total_loss.unsqueeze(0).expand(batch_size)
        per_sample_losses['total'].append(total_loss.cpu().numpy())
    
    # Stack into (B, R)
    for key in per_sample_losses:
        per_sample_losses[key] = np.stack(per_sample_losses[key], axis=1)  # (B, R)
    
    # Process per-channel losses if available
    has_channel_breakdown = {}
    for key in per_sample_channel_losses:
        if per_sample_channel_losses[key]:
            per_sample_channel_losses[key] = np.stack(per_sample_channel_losses[key], axis=1)  # (B, R, C)
            has_channel_breakdown[key] = True
        else:
            has_channel_breakdown[key] = False
    
    # Compute statistics for each component
    for comp_name in ['total'] + component_names:
        per_sample_overall = per_sample_losses[comp_name]  # (B, R)
        
        # Check if we have per-channel breakdown
        if has_channel_breakdown.get(comp_name, False):
            per_sample_channel = per_sample_channel_losses[comp_name]  # (B, R, C)
            
            # Per-step stats: (R, C)
            per_step_channel_mean = per_sample_channel.mean(axis=0)
            per_step_channel_std = per_sample_channel.std(axis=0, ddof=0)
            
            # Per-step overall: (R,)
            per_step_overall_mean = per_sample_overall.mean(axis=0)
            per_step_overall_std = per_sample_overall.std(axis=0, ddof=0)
            
            # Concatenate: (R, C+1)
            per_step_mean = np.concatenate(
                [per_step_channel_mean, per_step_overall_mean[:, None]], axis=-1
            )
            per_step_std = np.concatenate(
                [per_step_channel_std, per_step_overall_std[:, None]], axis=-1
            )
            
            # Cumulative
            cumulative_channel = np.cumsum(per_sample_channel, axis=1)  # (B, R, C)
            cumulative_overall = np.cumsum(per_sample_overall, axis=1)  # (B, R)
            
            cumulative_channel_mean = cumulative_channel.mean(axis=0)
            cumulative_channel_std = cumulative_channel.std(axis=0, ddof=0)
            cumulative_overall_mean = cumulative_overall.mean(axis=0)
            cumulative_overall_std = cumulative_overall.std(axis=0, ddof=0)
            
            cumulative_mean = np.concatenate(
                [cumulative_channel_mean, cumulative_overall_mean[:, None]], axis=-1
            )
            cumulative_std = np.concatenate(
                [cumulative_channel_std, cumulative_overall_std[:, None]], axis=-1
            )
        else:
            # Only overall metric available - create dummy channel dimension
            per_step_mean = per_sample_overall.mean(axis=0)[:, None]  # (R, 1)
            per_step_std = per_sample_overall.std(axis=0, ddof=0)[:, None]  # (R, 1)
            
            cumulative_overall = np.cumsum(per_sample_overall, axis=1)  # (B, R)
            cumulative_mean = cumulative_overall.mean(axis=0)[:, None]  # (R, 1)
            cumulative_std = cumulative_overall.std(axis=0, ddof=0)[:, None]  # (R, 1)
        
        metric_results[comp_name] = {
            "per_rollout_step_mean": per_step_mean,
            "per_rollout_step_std": per_step_std,
            "cumulative_rollout_step_mean": cumulative_mean,
            "cumulative_rollout_step_std": cumulative_std,
        }
    
    # -------------------------------------------------------------------------
    # Per-timestep metrics (optional)
    # -------------------------------------------------------------------------
    if include_per_timestep:
        # Use flattened arrays: (B, R*T_out, C, *spatial)
        total_timesteps = num_rollouts * outputs_per_rollout
        
        per_sample_timestep_losses = {name: [] for name in component_names}
        per_sample_timestep_losses['total'] = []
        per_sample_timestep_channel_losses = {name: [] for name in component_names}
        
        for t in range(total_timesteps):
            # Extract single timestep: (B, 1, C, *spatial)
            preds_t = preds[:, t:t+1, :, ...]
            targets_t = targets[:, t:t+1, :, ...]
            
            with torch.no_grad():
                total_loss, detailed = loss_fn(model, preds_t, targets_t, return_detailed=True)
            
            for comp_name in component_names:
                comp_total = detailed[comp_name]['total']
                if comp_total.ndim == 0:
                    comp_total = comp_total.unsqueeze(0).expand(batch_size)
                per_sample_timestep_losses[comp_name].append(comp_total.cpu().numpy())
                
                if 'per_channel' in detailed[comp_name]:
                    per_channel = detailed[comp_name]['per_channel']
                    if per_channel.ndim == 1:
                        per_channel = per_channel.unsqueeze(0).expand(batch_size, -1)
                    per_sample_timestep_channel_losses[comp_name].append(per_channel.cpu().numpy())
            
            if total_loss.ndim == 0:
                total_loss = total_loss.unsqueeze(0).expand(batch_size)
            per_sample_timestep_losses['total'].append(total_loss.cpu().numpy())
        
        # Stack into (B, T_flat)
        for key in per_sample_timestep_losses:
            per_sample_timestep_losses[key] = np.stack(per_sample_timestep_losses[key], axis=1)
        
        # Stack channel losses: (B, T_flat, C)
        has_timestep_channel_breakdown = {}
        for key in per_sample_timestep_channel_losses:
            if per_sample_timestep_channel_losses[key]:
                per_sample_timestep_channel_losses[key] = np.stack(
                    per_sample_timestep_channel_losses[key], axis=1
                )
                has_timestep_channel_breakdown[key] = True
            else:
                has_timestep_channel_breakdown[key] = False
        
        # Compute per-timestep statistics
        for comp_name in ['total'] + component_names:
            per_sample_overall_flat = per_sample_timestep_losses[comp_name]  # (B, T_flat)
            
            if has_timestep_channel_breakdown.get(comp_name, False):
                per_sample_channel_flat = per_sample_timestep_channel_losses[comp_name]  # (B, T_flat, C)
                
                per_timestep_channel_mean = per_sample_channel_flat.mean(axis=0)  # (T_flat, C)
                per_timestep_channel_std = per_sample_channel_flat.std(axis=0, ddof=0)
                per_timestep_overall_mean = per_sample_overall_flat.mean(axis=0)  # (T_flat,)
                per_timestep_overall_std = per_sample_overall_flat.std(axis=0, ddof=0)
                
                per_timestep_mean = np.concatenate(
                    [per_timestep_channel_mean, per_timestep_overall_mean[:, None]], axis=-1
                )
                per_timestep_std = np.concatenate(
                    [per_timestep_channel_std, per_timestep_overall_std[:, None]], axis=-1
                )
                
                # Cumulative
                cumulative_channel_flat = np.cumsum(per_sample_channel_flat, axis=1)
                cumulative_overall_flat = np.cumsum(per_sample_overall_flat, axis=1)
                
                cumulative_timestep_mean = np.concatenate([
                    cumulative_channel_flat.mean(axis=0),
                    cumulative_overall_flat.mean(axis=0)[:, None]
                ], axis=-1)
                cumulative_timestep_std = np.concatenate([
                    cumulative_channel_flat.std(axis=0, ddof=0),
                    cumulative_overall_flat.std(axis=0, ddof=0)[:, None]
                ], axis=-1)
            else:
                per_timestep_mean = per_sample_overall_flat.mean(axis=0)[:, None]
                per_timestep_std = per_sample_overall_flat.std(axis=0, ddof=0)[:, None]
                
                cumulative_overall_flat = np.cumsum(per_sample_overall_flat, axis=1)
                cumulative_timestep_mean = cumulative_overall_flat.mean(axis=0)[:, None]
                cumulative_timestep_std = cumulative_overall_flat.std(axis=0, ddof=0)[:, None]
            
            metric_results[comp_name].update({
                "per_timestep_mean": per_timestep_mean,
                "per_timestep_std": per_timestep_std,
                "cumulative_timestep_mean": cumulative_timestep_mean,
                "cumulative_timestep_std": cumulative_timestep_std,
            })
    
    return metric_results