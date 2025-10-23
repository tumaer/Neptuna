"""
Data Visualization and Progress Plotting for Model Training.

This module provides comprehensive visualization tools for monitoring model training progress
and evaluating prediction quality across different data dimensions and configurations.
Designed specifically for time-series and scientific computing applications with support
for multi-dimensional data visualization and error analysis.

Key Features:
- Multi-dimensional data plotting (1D, 2D, 3D spatial data)
- Autoregressive prediction visualization with rollout steps
- Side-by-side comparison of predictions, targets, and error metrics
- Support for conditioning inputs and multi-channel data
- Parallel processing for efficient plot generation
- Integration with Weights & Biases for experiment tracking
- Customizable time labeling and spatial aspect ratio handling

Supported Data Types:
- 1D: Time series, signal data with line plots and legends
- 2D: Spatial fields, images with colormaps and scientific notation
- 3D: Volumetric data (placeholder for future implementation)

Visualization Layout:
- Input data with temporal progression
- Optional conditioning inputs
- Model predictions vs ground truth targets
- Absolute and relative error analysis
- Configurable time step labeling and channel organization

Notes:
    This module uses a non-interactive matplotlib backend ('Agg') to support
    headless environments and server-side plot generation. All plots are saved
    to disk or logged to experiment tracking systems rather than displayed.
"""
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wandb
import io  # For in-memory PNG buffers
from PIL import Image  # To create a PIL image object for wandb
from utils.compute_stats import re_normalize_data
from typing import Tuple
import matplotlib.cm as cm
from matplotlib.patches import Patch
import math


def preprocess_for_plotting(
    inputs: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    data_config: dict,
    dataset,
    residual_config: dict | None,
    conditioning_inputs: np.ndarray | None = None,
):
    """Pre-process inputs, labels and predictions for plotting.

    This helper mirrors the renormalisation logic originally implemented in
    ``PlotOnEvalAndSaveCallback.on_plot`` so that it can be reused from other
    modules without code duplication.

    The function performs three main tasks:

    1. Renormalises *inputs*, *labels* and *predictions* back to their original
       physical scale using the statistics stored in *data_config*.
    2. Applies the same procedure to *conditioning_inputs* (if provided).
    3. Reconstructs raw values from residual predictions when residual learning
       is enabled via *residual_config*.

    Parameters
    ----------
    inputs, labels, predictions : np.ndarray
        Arrays with shapes ``(N, T, C, *spatial_dims)`` where the leading axes
        correspond to batch, time and channel respectively.
    data_config : dict
        Must contain the keys ``data_normalization_stats`` and
        ``data_normalization_strategy`` produced during dataset creation.
    dataset : Dataset
        Dataset instance that provides ``input_channels``, ``output_channels``
        and optionally ``conditioning_in_channels`` attributes.
    residual_config : dict | None
        Configuration dictionary controlling residual learning behaviour.  If
        *None*, no residual reconstruction is performed.
    conditioning_inputs : np.ndarray | None, optional
        Optional conditioning input array with the same leading dimensions as
        *inputs*.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str],
          Optional[np.ndarray], Optional[List[str]]]
        ``(inputs_renormed, labels_renormed, predictions_renormed,
        only_input_channel_names, output_channel_names,
        conditioning_inputs_renormed, conditioning_input_channel_names)``
    """

    # ------------------------------------------------------------------
    # Extract normalisation parameters and channel metadata
    # ------------------------------------------------------------------
    norm_stats = data_config["data_normalization_stats"]
    norm_strategy = data_config["data_normalization_strategy"]

    input_channel_names = getattr(dataset, "input_channels", None)
    output_channel_names = getattr(dataset, "output_channels", None)
    conditioning_input_channel_names = None

    # Make *independent* copies so that callers keep their originals intact.
    inputs_renormed = np.copy(inputs)
    labels_renormed = np.copy(labels)
    predictions_renormed = np.copy(predictions)
    conditioning_inputs_renormed = None

    # ------------------------------------------------------------------
    # Handle conditioning inputs (if present)
    # ------------------------------------------------------------------
    if conditioning_inputs is not None:
        conditioning_input_channel_names = [
            ch_name
            for ch_name in input_channel_names
            if ch_name in getattr(dataset, "conditioning_in_channels", [])
        ]
        conditioning_inputs_renormed = np.copy(conditioning_inputs)
        # Remove conditioning channels from *only* input channels
        only_input_channel_names = [
            ch_name for ch_name in input_channel_names if ch_name not in conditioning_input_channel_names
        ]
    else:
        only_input_channel_names = input_channel_names

    # ------------------------------------------------------------------
    # Renormalise input channels (inputs & optionally conditioning_inputs)
    # ------------------------------------------------------------------
    for c_idx, ch_name in enumerate(only_input_channel_names):
        if ch_name not in norm_stats:
            raise ValueError(f"Stats for input channel {ch_name} are unavailable.")
        stats = norm_stats[ch_name]
        if "mask" not in ch_name.lower():
            inputs_renormed[:, :, c_idx] = re_normalize_data(inputs[:, :, c_idx], stats, norm_strategy)

    if conditioning_inputs is not None:
        for c_idx, ch_name in enumerate(conditioning_input_channel_names):
            if ch_name not in norm_stats:
                raise ValueError(
                    f"Stats for conditioning_input channel {ch_name} are unavailable."
                )
            stats = norm_stats[ch_name]
            if "mask" not in ch_name.lower():
                conditioning_inputs_renormed[:, :, c_idx] = re_normalize_data(
                    conditioning_inputs[:, :, c_idx], stats, norm_strategy
                )

    # ------------------------------------------------------------------
    # Renormalise output channels (labels & predictions)
    # ------------------------------------------------------------------
    for c_idx, ch_name in enumerate(output_channel_names):
        if ch_name not in norm_stats:
            raise ValueError(f"Stats for output channel {ch_name} are unavailable.")
        norm_key = (
            ch_name
            if (
                (residual_config is None)
                or residual_config.get("add_base_value_with_raw_loss", False)
                or residual_config.get("add_predicted_value_with_raw_loss", False)
            )
            else f"{ch_name}_residual"
        )
        stats = norm_stats[norm_key]
        labels_renormed[:, :, c_idx] = re_normalize_data(labels[:, :, c_idx], stats, norm_strategy)
        predictions_renormed[:, :, c_idx] = re_normalize_data(predictions[:, :, c_idx], stats, norm_strategy)

    # ------------------------------------------------------------------
    # Reconstruct raw values for residual learning if requested
    # ------------------------------------------------------------------
    if residual_config is not None and residual_config.get("add_predicted_value_with_diff_loss", False):
        base_value = inputs_renormed[:, -1:, ]
        labels_renormed = labels_renormed.cumsum(axis=1) + base_value
        predictions_renormed = predictions_renormed.cumsum(axis=1) + base_value

    return (
        inputs_renormed,
        labels_renormed,
        predictions_renormed,
        only_input_channel_names,
        output_channel_names,
        conditioning_inputs_renormed,
        conditioning_input_channel_names,
    )


def _plot_data(ax, data, ndim, ch_names=None, vmin_arr=None, vmax_arr=None):
    """
    Plot data on given axes with dimension-specific formatting and styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on.
    data : numpy.ndarray
        Data array with shape (C, *spatial_dims) where C is number of channels.
    ndim : int
        Number of spatial dimensions (1, 2, or 3).
    ch_names : Optional[List[str]]
        Channel names for labeling. If None, uses default naming.
    vmin_arr, vmax_arr : Optional[Sequence[float]]
        Optional per-channel min/max to enforce consistent colorbar limits across
        related plots. Only used for 2D data.

    Returns
    -------
    Union[None, List[matplotlib.axes.Axes]]
        For multi-channel 2D data, returns list of sub-axes created.
        For other cases, returns None.

    Notes
    -----
    - 1D data: Creates line plots with legends and visible ticks
    - 2D data: Creates heatmaps with colorbars and scientific notation
    - 3D data: Currently shows placeholder text (not implemented)
    - Automatically handles aspect ratios for square vs rectangular domains
    - Uses 'coolwarm' colormap for 2D visualizations
    """
    C = data.shape[0]  # number of channels

    if ndim == 1:
        x = np.arange(data.shape[-1])
        for c in range(C):
            ax.plot(x, data[c], label=ch_names[c])
            ax.legend(fontsize=8, loc="upper right")
        # Show ticks for 1D plots
        ax.set_xticks(x[::len(x)//5])  # Show 5 ticks
        ax.tick_params(axis='both', labelsize=8)  # Set tick label size for both axes
        #ax.set_ylabel('Value', fontsize=10)

    elif ndim == 2:
        # Determine if the domain is square or rectangular
        h, w = data[0].shape
        is_rectangular = (h != w)
        # Use 'auto' for rectangular images to reduce whitespace; keep 'equal' for square
        aspect = 'auto' if is_rectangular else 'equal'
        
        # Guard against invalid vmin/vmax lengths
        use_vlims = vmin_arr is not None and vmax_arr is not None and len(vmin_arr) >= C and len(vmax_arr) >= C

        def _safe_vlims(c):
            if not use_vlims:
                return {}
            vmin = float(vmin_arr[c])
            vmax = float(vmax_arr[c])
            if vmin == vmax:
                eps = 1e-6 if vmin == 0.0 else abs(vmin) * 1e-6
                vmin -= eps
                vmax += eps
            return {"vmin": vmin, "vmax": vmax}
        
        if C == 1:
            im = ax.imshow(data[0], cmap="coolwarm", aspect=aspect, origin='lower', **_safe_vlims(0))
            # Keep adjustable box only for square plots where aspect is 'equal'
            if not is_rectangular:
                ax.set_adjustable('box')
            ax.set_title(ch_names[0], fontsize=8)
            # Hide x-axis tick labels to avoid overlap with horizontal colorbar
            ax.tick_params(axis='x', labelbottom=False)
            # Increase the padding for rectangular subplots to prevent overlap with axis tick labels.
            cbar_pad = 0.27 if is_rectangular else 0.12
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=cbar_pad, orientation='horizontal', location='bottom')
            cbar.ax.tick_params(labelsize=12)
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.formatter.set_useMathText(True)  # Use math text for consistent font
            offset_y = -1.2 if is_rectangular else -1.0
            cbar.ax.xaxis.offsetText.set_y(offset_y)  # Move the offset text further down
            cbar.ax.xaxis.offsetText.set_fontsize(14)  # Match the tick label font size
            cbar.update_ticks()
        else:
            ncols = C
            fig = ax.figure
            gs = ax.get_subplotspec().subgridspec(1, ncols, wspace=0.2)  # Tighter spacing between channel sub-axes
            ax.remove()
            sub_axes = []
            for c in range(C):
                sub_ax = fig.add_subplot(gs[0, c])
                im = sub_ax.imshow(data[c], cmap="coolwarm", aspect=aspect, origin='lower', **_safe_vlims(c))
                if not is_rectangular:
                    sub_ax.set_adjustable('box')
                sub_ax.set_title(ch_names[c], fontsize=8)
                # Hide x-axis tick labels to avoid overlap with horizontal colorbar
                sub_ax.tick_params(axis='x', labelbottom=False)
                # Increase the padding for rectangular subplots to prevent overlap with axis tick labels.
                cbar_pad = 0.29 if is_rectangular else 0.12
                cbar = fig.colorbar(im, ax=sub_ax, fraction=0.046, pad=cbar_pad, orientation='horizontal', location='bottom')
                cbar.ax.tick_params(labelsize=12)
                cbar.formatter.set_scientific(True)
                cbar.formatter.set_powerlimits((0, 0))
                cbar.formatter.set_useMathText(True)  # Use math text for consistent font
                offset_y = -1.2 if is_rectangular else -1.0
                cbar.ax.xaxis.offsetText.set_y(offset_y)  # Move the offset text further down
                cbar.ax.xaxis.offsetText.set_fontsize(14)  # Match the tick label font size
                cbar.update_ticks()
                sub_axes.append(sub_ax)
            return sub_axes

    elif ndim == 3:
        ax.text(0.5, 0.5, "3D multi-channel\nnot supported yet", ha="center", va="center", fontsize=10)
        ax.axis('off')


def plot_examples(
    input_array,
    prediction_array,
    target_array,
    only_input_channel_names,
    output_channel_names,
    conditioning_input_array=None,
    conditioning_input_channel_names=None,
    checkpoint_step=None,
    epoch=None,
    extra_info=None,
    ndim=1,
    num_examples=5,
    stride=1,
    save_dir="plots",
    log_to_wandb: bool = False,
    best_plot_at_train_end: bool = False,
    model_info: str | None = None,
    data_info: str | None = None,
    train_info: str | None = None,
    scheduler_info: str | None = None,
    example_indices: list[int] | None = None,
):
    """
    Generate comprehensive visualization plots comparing model predictions with targets.

    Creates side-by-side comparison plots showing input data, predictions, ground truth,
    and error metrics across multiple examples and time steps. Supports various data
    dimensions and handles conditioning inputs for complex model architectures.

    Parameters
    ----------
    input_array : numpy.ndarray
        Input data with shape (N, T_in, C, *spatial_shape) where:
        - N: batch size
        - T_in: input time steps
        - C: number of input channels
        - *spatial_shape: spatial dimensions (H, W for 2D)
    prediction_array : numpy.ndarray
        Model predictions with shape (N, T_pred, C_out, *spatial_shape).
    target_array : numpy.ndarray
        Ground truth targets with shape (N, T_pred, C_out, *spatial_shape).
    only_input_channel_names : List[str]
        Names of input channels for labeling and legends.
    output_channel_names : List[str]
        Names of output channels for labeling and legends.
    conditioning_input_array : Optional[numpy.ndarray]
        Optional conditioning inputs with shape (N, T_in, C_cond, *spatial_shape).
    conditioning_input_channel_names : Optional[List[str]]
        Names of conditioning input channels.
    checkpoint_step : Optional[int]
        Training checkpoint step number for plot titles.
    epoch : Optional[int]
        Training epoch number for plot titles.
    extra_info : Optional[str]
        Additional information string for plot titles (e.g., dataset name).
    ndim : int, default=1
        Number of spatial dimensions (1, 2, or 3).
    num_examples : int, default=5
        Number of examples to plot from the batch.
    stride : int, default=1
        Time step stride for temporal labeling.
    save_dir : str, default="plots"
        Directory to save plots when not logging to W&B.
    log_to_wandb : bool, default=False
        Whether to log plots to Weights & Biases instead of saving to disk.
    best_plot_at_train_end : bool, default=False
        When True, save with a "_best.png" suffix (used at train end).
    model_info : Optional[str], default=None
        Optional string to display the model info.
    data_info : Optional[str], default=None
        Optional string to display the data info.
    train_info : Optional[str], default=None
        Training configuration summary lines.
    scheduler_info : Optional[str], default=None
        Scheduler configuration summary lines.

    Returns
    -------
    Dict[str, wandb.Image]
        Dictionary of plot figures for W&B logging when log_to_wandb=True.
        Empty dictionary when saving to disk.

    Notes
    -----
    Plot Layout:
    - Column 1: Input data (temporal progression)
    - Column 2: Conditioning inputs (if provided)
    - Column 3: Model predictions
    - Column 4: Ground truth targets
    - Column 5: Absolute error |pred - target|
    - Column 6: Relative error |pred - target| / |target|

    Time Labeling:
    - Input columns: "t - N" to "t" (historical data)
    - Prediction columns: "t + 1" to "t + T_pred" (future predictions)

    Special Features:
    - Automatic aspect ratio handling for 2D spatial data
    - Scientific notation for colorbars
    - Parallel processing for efficient plot generation
    - Best metric checkpoint handling with file renaming
    - Configurable figure sizing based on data dimensions and channel counts
    """
    os.makedirs(save_dir, exist_ok=True)

    N, T_in, C, *spatial_shape = input_array.shape
    T_pred = prediction_array.shape[1]

    if example_indices is None:
        np.random.seed(42)
        num_pick = min(num_examples, N)
        example_indices = np.random.choice(N, size=num_pick, replace=False)
    else:
        example_indices = np.array(example_indices, dtype=int)

    returned_figs: dict[str, matplotlib.figure.Figure] = {}

    # Use ThreadPoolExecutor for parallelized file saving
    with ThreadPoolExecutor() as executor:
        for idx in example_indices:
            inp = input_array[idx]          # [T_in, C, ...]
            pred = prediction_array[idx]    # [T_pred, C, ...]
            tgt = target_array[idx]         # [T_pred, C, ...]

            abs_err = np.abs(pred - tgt)
            rel_err = np.abs((pred - tgt) / (np.abs(tgt) + 1e-8))

            # Determine spacing and widths based on channels
            has_conditioning = conditioning_input_array is not None
            nrows = max(T_in, T_pred)
            
            # Calculate relative widths for each column section
            input_channels = len(only_input_channel_names)
            output_channels = len(output_channel_names)
            
            # Compute per-channel vmin/vmax across Input, Prediction, and Target for consistent colorbars (2D only)
            vmins = vmaxs = None
            if ndim == 2:
                C_inout = output_channels  # assume same channels for pred/target
                vmins = np.zeros(C_inout, dtype=float)
                vmaxs = np.zeros(C_inout, dtype=float)
                for c_idx in range(C_inout):
                    # Stack input channel if present; input may include more channels (e.g. conditioning removed already)
                    # Find corresponding index for this channel name in inputs
                    ch_name = output_channel_names[c_idx]
                    if ch_name in only_input_channel_names:
                        in_c = only_input_channel_names.index(ch_name)
                        in_stack = inp[:, in_c]
                    else:
                        in_stack = None
                    pred_stack = pred[:, c_idx]
                    tgt_stack = tgt[:, c_idx]
                    stacks = [arr for arr in [in_stack, pred_stack, tgt_stack] if arr is not None]
                    all_vals = np.concatenate(stacks, axis=0)
                    vmin = float(np.nanmin(all_vals))
                    vmax = float(np.nanmax(all_vals))
                    if vmin == vmax:
                        eps = 1e-6 if vmin == 0.0 else abs(vmin) * 1e-6
                        vmin -= eps
                        vmax += eps
                    vmins[c_idx] = vmin
                    vmaxs[c_idx] = vmax
            
            if has_conditioning:
                conditioning_channels = len(conditioning_input_channel_names)
                # Allocate width proportional to the actual number of conditioning
                # channels.  This guarantees that every individual channel image
                # across all logical columns (Input / Conditioning / Prediction …)
                # has identical size, independent of how many channels each
                # column contains.
                column_widths = [input_channels, conditioning_channels, output_channels, output_channels, output_channels, output_channels]
                column_titles = ["Input", "Conditioning", "Prediction", "Target", "Abs Error = |Pred - Target|", "Rel Error = |Pred - Target|/|Target|"]
            else:
                if ndim == 2:
                    column_widths = [input_channels, output_channels, output_channels, output_channels, output_channels]
                else:
                    column_widths = [1, 1, 1, 1, 1]
                column_titles = ["Input", "Prediction", "Target", "Abs Error = |Pred - Target|", "Rel Error = |Pred - Target|/|Target|"]

            # Calculate total grid columns and positions
            total_grid_cols = sum(column_widths)
            col_positions = []
            current_pos = 0
            for width in column_widths:
                col_positions.append((current_pos, current_pos + width))
                current_pos += width

            # Layout tuning parameters
            dims_ratio = 0.8      # Reserved height for dims/info text row (new, to avoid overlaps)
            header_ratio = 0.15   # Height allocated for column titles
            time_label_ratio = 0.5 # Base height for time label row
            spacer_ratio = 0.2    # Spacer row ratio between time labels and footer (reduced)
            footer_ratio = 4.0    # Base footer height for more space under time labels

            # Determine rectangular 2D status upfront for spacing tweaks
            is_rectangular2d = False
            if ndim == 2 and len(spatial_shape) >= 2:
                H, W = spatial_shape[:2]
                is_rectangular2d = (H != W)

            # Padding between individual plot and its xlabel (time indicator). 
            # If this padding is too large, the xlabel from one subplot can overlap the
            # axes of the subplot in the row below.  Use a smaller value for 2-D plots
            # (which typically have less vertical space per row) and a moderate value
            # for 1-D plots.
            x_label_pad = (55 if is_rectangular2d else 40) if ndim == 2 else 30

            # Reduce horizontal spacing so each subplot takes up more space within its
            # column, effectively making the plots ~50 % larger without increasing the
            # overall figure size.
            main_wspace = 0.5
            # Increase vertical spacing between rows for rectangular plots to make room for xlabels
            main_hspace = 0.6 if is_rectangular2d else 0.4
            
            # Adjust figure width based on total content
            base_width_per_unit = 4.0 if ndim == 2 else 5.0
            fig_width = total_grid_cols * base_width_per_unit

            # For ndim=2, make plot cells match data aspect ratio by adjusting figure height.
            # The height of a plot row (where height_ratio=1) should equal the width of a grid cell, scaled by the data's aspect ratio.
            # IMPORTANT: Account for all grid rows (dims + header + plot rows + time label + footer) when computing total height.
            if ndim == 2:
                data_aspect_ratio = H / W
                # For rectangular domains, set height to half the current length (width): height/width = 0.5
                # For square domains, keep proportional to H/W (i.e., 1.0)
                target_height_scale = 0.5 if (H != W) else data_aspect_ratio
                # Height of one ratio unit (i.e., one plot row) in inches
                cell_height = base_width_per_unit * target_height_scale
                # Enforce a minimum cell height to prevent text overlaps
                min_cell_height = 1.2  # inches
                cell_height = max(cell_height, min_cell_height)
                total_ratio_units = dims_ratio + header_ratio + nrows + time_label_ratio + footer_ratio
                fig_height = total_ratio_units * cell_height
                # Ensure overall figure height grows proportionally with current length (width) for rectangular domains
                if is_rectangular2d:
                    fig_height = max(fig_height, fig_width * data_aspect_ratio)
            else:  # Original calculation for 1D plots which can be non-square
                fig_height = (dims_ratio + header_ratio + nrows + time_label_ratio + footer_ratio + 1.3 + 0.6) * 3.5  # extra offset for titles
            
            fig = plt.figure(figsize=(fig_width, fig_height))

            # Add main title
            # Place at a constant relative position; dedicated dims row avoids collisions
            suptitle_y_pos = 0.98
            # --------------------------------------------------------------
            # Title logic: include Checkpoint/Epoch only for validation plots
            # --------------------------------------------------------------
            include_ckpt = False
            try:
                # Determine if the parent directory name (split by "_") contains
                # the keyword "validation" in the second-to-last token.
                # Example: "plots/run_42_validation_2025" → parts[-2] == "validation".
                parts = os.path.normpath(save_dir).split('_')
                if len(parts) >= 2 and 'validation' in parts[-2].lower():
                    include_ckpt = True
            except Exception:
                include_ckpt = False

            if include_ckpt:
                title_str = (
                    f"{extra_info}\nCheckpoint Step: {checkpoint_step}, Epoch: {epoch} "
                    f"\nExample Index: {idx}"
                )
            else:
                title_str = f"{extra_info}\nExample Index: {idx}"

            fig.suptitle(
                title_str,
                fontsize=32,
                y=suptitle_y_pos,
                weight='bold'
            )
            
            # Build dimensions/info text to render inside a dedicated top row in the GridSpec
            dims_text = (
                f"Additional Info: Total number of examples={N}, Spatial_res={spatial_shape}, "
                f"# Input_frames={T_in}, # Input_channels={C}, # Prediction_frames={T_pred}, "
                f"# Prediction_channels={pred.shape[1]}"
            )

            # Keep dims_text focused on dataset statistics; append a one-line model summary.
            if model_info is not None:
                summary_line = model_info.split("\n")[0]  # e.g. "FNO | Params: 12.3M"
                dims_text += "\n" + summary_line

            # --------------------------------------------------------------
            # Footer: detailed model & data configuration (indented bullets)
            # --------------------------------------------------------------
            footer_lines: list[str] = []

            # Add remaining model configuration lines beneath a header
            if model_info is not None and "\n" in model_info:
                #detailed_cfg = "\n".join(model_info.split("\n")[1:])  # skip first line
                #indented_cfg = "\n".join(["    " + ln for ln in model_info.split("\n")])
                footer_lines.append("MODEL CONFIG:\n" + model_info + "\n")

            # Add data configuration lines similarly
            if data_info is not None:
                #indented_data = "\n".join(["    " + ln for ln in data_info.split("\n")])
                footer_lines.append("DATA CONFIG:\n" + data_info + "\n")

            if train_info is not None:
                #indented_train = "\n".join(["    " + ln for ln in train_info.split("\n")])
                footer_lines.append("TRAIN CONFIG:\n" + train_info + "\n")

            if scheduler_info is not None:
                #indented_sched = "\n".join(["    " + ln for ln in scheduler_info.split("\n")])
                footer_lines.append("SCHEDULER CONFIG:\n" + scheduler_info + "\n")

            # Create GridSpec with variable column widths and specific height ratios
            # New: include a dedicated dims row at the top to avoid overlapping with plots
            # For rectangular 2D plots, allocate extra space for time labels and footer
            time_label_ratio_eff = (time_label_ratio + 0.3) if is_rectangular2d else time_label_ratio
            spacer_ratio_eff = (spacer_ratio + 0.3) if is_rectangular2d else spacer_ratio
            footer_ratio_eff = (footer_ratio + 1.8) if is_rectangular2d else footer_ratio
            gs = gridspec.GridSpec(
                nrows + 5,
                total_grid_cols,
                figure=fig,
                top=0.94,
                height_ratios=[dims_ratio, header_ratio] + [1] * nrows + [time_label_ratio_eff, spacer_ratio_eff, footer_ratio_eff],
                hspace=main_hspace,
                wspace=main_wspace,
            )

            # Add dims/info text row at the very top
            dims_ax = fig.add_subplot(gs[0, :])
            dims_ax.axis('off')
            dims_ax.text(0.5, 0.8, dims_text, ha='center', va='center', fontsize=22)

            # Add column titles at the top
            for col_idx, (start_col, end_col) in enumerate(col_positions):
                title_ax = fig.add_subplot(gs[1, start_col:end_col])
                title_ax.axis('off')
                title_ax.text(0.5, 0.5, column_titles[col_idx], ha='center', va='center', fontsize=14, weight='bold')

            # Add time labels at the bottom  
            for col_idx, (start_col, end_col) in enumerate(col_positions):
                # Time labels now occupy the third-from-last row (because we added a spacer row)
                time_ax = fig.add_subplot(gs[-3, start_col:end_col])
                time_ax.axis('off')
                time_y = 0.62 if is_rectangular2d else 0.45
                if col_idx == 0:  # Input column
                    time_label = f"t - {stride * (T_in - 1)} to t"
                elif has_conditioning and col_idx == 1:  # Conditioning column
                    time_label = f"t - {stride * (T_in - 1)} to t"
                else:  # Other columns (prediction, target, errors)
                    time_label = f"t + {stride} to t + {stride * T_pred}"
                time_fs = 22 if is_rectangular2d else 25
                time_ax.text(0.5, time_y, time_label, ha='center', va='center', fontsize=time_fs)

            plot_time_fontsize = 20  # Font size for per-plot time labels

            for row in range(nrows):
                # +2 offset to account for [dims row, header row] at the top of the GridSpec
                row_offset = row + 2

                # Column 0: Input
                start_col, end_col = col_positions[0]
                if row < T_in:
                    input_ax = fig.add_subplot(gs[row_offset, start_col:end_col])
                    axes_to_label = _plot_data(input_ax, inp[row], ndim, only_input_channel_names, vmins, vmaxs)
                    if isinstance(axes_to_label, list):  # Multi-channel 2D case
                        # Ensure bottom ticks are shown without adding per-plot time labels
                        mid_channel = len(axes_to_label) // 2
                        axes_to_label[mid_channel].tick_params(labelbottom=True)
                    else:  # Single channel or 1D case
                        input_ax.tick_params(labelbottom=True)

                # Column 1: Conditioning (if present)
                if has_conditioning:
                    start_col, end_col = col_positions[1]
                    if row < T_in:
                        cond_inp = conditioning_input_array[idx]  # [T_in, C_cond, ...]
                        cond_ax = fig.add_subplot(gs[row_offset, start_col:end_col])
                        axes_to_label = _plot_data(cond_ax, cond_inp[row], ndim, conditioning_input_channel_names)
                        if isinstance(axes_to_label, list):  # Multi-channel 2D case
                            mid_channel = len(axes_to_label) // 2
                            axes_to_label[mid_channel].tick_params(labelbottom=True)
                        else:  # Single channel or 1D case
                            cond_ax.tick_params(labelbottom=True)

                # Determine column indices for remaining plots
                pred_col_idx = 2 if has_conditioning else 1
                target_col_idx = 3 if has_conditioning else 2
                abs_err_col_idx = 4 if has_conditioning else 3
                rel_err_col_idx = 5 if has_conditioning else 4

                # Prediction column
                start_col, end_col = col_positions[pred_col_idx]
                if row < T_pred:
                    pred_ax = fig.add_subplot(gs[row_offset, start_col:end_col])
                    axes_to_label = _plot_data(pred_ax, pred[row], ndim, output_channel_names, vmins, vmaxs)
                    if isinstance(axes_to_label, list):  # Multi-channel 2D case
                        mid_channel = len(axes_to_label) // 2
                        axes_to_label[mid_channel].tick_params(labelbottom=True)
                    else:  # Single channel or 1D case
                        pred_ax.tick_params(labelbottom=True)

                # Target column
                start_col, end_col = col_positions[target_col_idx]
                if row < T_pred:
                    target_ax = fig.add_subplot(gs[row_offset, start_col:end_col])
                    axes_to_label = _plot_data(target_ax, tgt[row], ndim, output_channel_names, vmins, vmaxs)
                    if isinstance(axes_to_label, list):  # Multi-channel 2D case
                        mid_channel = len(axes_to_label) // 2
                        axes_to_label[mid_channel].tick_params(labelbottom=True)
                    else:  # Single channel or 1D case
                        target_ax.tick_params(labelbottom=True)

                # Abs Error column
                start_col, end_col = col_positions[abs_err_col_idx]
                if row < T_pred:
                    abs_err_ax = fig.add_subplot(gs[row_offset, start_col:end_col])
                    axes_to_label = _plot_data(abs_err_ax, abs_err[row], ndim, output_channel_names)
                    if isinstance(axes_to_label, list):  # Multi-channel 2D case
                        mid_channel = len(axes_to_label) // 2
                        axes_to_label[mid_channel].tick_params(labelbottom=True)
                    else:  # Single channel or 1D case
                        abs_err_ax.tick_params(labelbottom=True)

                # Rel Error column
                start_col, end_col = col_positions[rel_err_col_idx]
                if row < T_pred:
                    rel_err_ax = fig.add_subplot(gs[row_offset, start_col:end_col])
                    axes_to_label = _plot_data(rel_err_ax, rel_err[row], ndim, output_channel_names)
                    if isinstance(axes_to_label, list):  # Multi-channel 2D case
                        mid_channel = len(axes_to_label) // 2
                        axes_to_label[mid_channel].tick_params(labelbottom=True)
                    else:  # Single channel or 1D case
                        rel_err_ax.tick_params(labelbottom=True)

            # --------------------------------------------------------------
            # Footer: detailed model & data configuration (indented bullets)
            # --------------------------------------------------------------
            if footer_lines:
                footer_text = "\n".join(footer_lines)
                footer_ax = fig.add_subplot(gs[-1, :])
                footer_ax.axis('off')
                # Place footer text at top-left within the footer axes to maximize clearance
                footer_ax.text(0.0, 0.98, footer_text, ha='left', va='top', fontsize=20, wrap=True)

            # --------------------------------------------------------------
            # Saving behaviour
            # --------------------------------------------------------------
            # Save best figures regardless of W&B logging. Otherwise, save only
            # when not logging to W&B (to avoid duplicating large artifacts).
            save_this_fig = (not log_to_wandb) or best_plot_at_train_end

            if save_this_fig:
                # Use a special suffix for best figures
                if best_plot_at_train_end:
                    filename = f"ckpt_{checkpoint_step}_epoch_{epoch}_example_{idx}_best.png"
                else:
                    filename = f"ckpt_{checkpoint_step}_epoch_{epoch}_example_{idx}.png"

                img_path = os.path.join(save_dir, filename)
                fig.savefig(img_path, dpi=150, bbox_inches="tight")

            # --------------------------------------------------------------
            # W&B logging (only create the image object here; actual logging
            # timing can be handled by the caller).
            # --------------------------------------------------------------

            if log_to_wandb and not best_plot_at_train_end:
                # Use an in-memory buffer with bbox_inches='tight' so nothing is cut off
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.1)
                buf.seek(0)
                pil_img = Image.open(buf)

                returned_figs[f"plot_progress/example_{idx}"] = wandb.Image(pil_img)

                # We purposely **do not** store the best plot in
                # ``returned_figs`` so that it lives **only on disk**.
                # A separate post-run routine (on_train_end) uploads the saved PNG to W&B.
            
            plt.close(fig)
    return returned_figs


def plot_rollout_metrics(step_metrics: dict, output_channel_names: list[str], save_dir: str, title: str | None = None, filename: str = "rollout_metrics.png", plot_type: str = "per_step", sequence_info: list[int] | tuple[int, int, int] | None = None) -> None:
    """Plot per-metric curves over time steps for IC-start evaluations.

    Parameters
    ----------
    step_metrics : dict
        Mapping from metric name to a dictionary with keys like
        "per_step_mean", "per_step_std", "cumulative_mean", "cumulative_std".
        Each value is an array of shape (T, C+1) where columns 0..C-1 are
        per-channel values and the last column is the overall value.
    output_channel_names : list[str]
        Names for the C output channels, used in the legend.
    save_dir : str
        Directory where the figure will be saved.
    title : Optional[str]
        Optional figure title.
    filename : str
        Output filename. Defaults to "rollout_metrics.png".
    plot_type : {"cumulative", "per_step"}
        Which statistics to plot. Defaults to "per_step".
    sequence_info : List[int] | Tuple[int, int, int] | None
        Sequence configuration [input_steps, output_steps, stride]. If provided,
        the x-axis is offset by input_steps and scaled by stride, and each
        rollout step advances by output_steps frames, i.e., time =
        stride * (input_steps + output_steps * rollout_step).
    """
    os.makedirs(save_dir, exist_ok=True)

    num_metrics = len(step_metrics)
    if num_metrics == 0:
        return

    # Determine keys for mean/std based on requested plot type
    if plot_type not in {"cumulative", "per_step"}:
        raise ValueError("plot_type must be 'cumulative' or 'per_step'")

    # Create a subplot grid: rows = num_metrics, cols = 2 (rollout | timestep)
    fig, axes = plt.subplots(num_metrics, 2, figsize=(14, max(3 * num_metrics, 4)), squeeze=False)
    # Column titles
    if num_metrics > 0:
        axes[0, 0].set_title("Rollout step metrics", fontsize=12)
        axes[0, 1].set_title("Timestep metrics", fontsize=12)

    # Prepare legends (overall label)
    overall_label = "overall"

    # Determine stride for x-axis scaling if provided
    input_steps = sequence_info[0]
    stride = sequence_info[2]

    # Key mapping for rollout vs. timestep metrics
    rollout_mean_key = "per_rollout_step_mean" if plot_type == "per_step" else "cumulative_rollout_step_mean"
    rollout_std_key = "per_rollout_step_std" if plot_type == "per_step" else "cumulative_rollout_step_std"
    timestep_mean_key = "per_timestep_mean" if plot_type == "per_step" else "cumulative_timestep_mean"
    timestep_std_key = "per_timestep_std" if plot_type == "per_step" else "cumulative_timestep_std"

    for row_idx, (metric_name, stats) in enumerate(step_metrics.items()):
        ax_rollout = axes[row_idx, 0]
        ax_timestep = axes[row_idx, 1]

        # ---------------------
        # Left column: Rollout
        # ---------------------
        if rollout_mean_key in stats and rollout_std_key in stats:
            means_r = stats[rollout_mean_key]  # (R, C+1)
            stds_r = stats[rollout_std_key]    # (R, C+1)
            R, total_cols_r = means_r.shape
            num_channels_r = total_cols_r - 1
            channel_legends_r = list(output_channel_names) if num_channels_r == len(output_channel_names) else [f"ch_{i}" for i in range(num_channels_r)]
            x_r = np.arange(1, R + 1)
            for c in range(num_channels_r):
                m = means_r[:, c]
                s = stds_r[:, c]
                line, = ax_rollout.plot(x_r, m, label=channel_legends_r[c], linewidth=1.5, alpha=0.95)
                ax_rollout.scatter(x_r, m, s=18, color=line.get_color(), edgecolors="none", zorder=3)
                ax_rollout.fill_between(x_r, m - s, m + s, color=line.get_color(), alpha=0.15)
            m_overall_r = means_r[:, -1]
            s_overall_r = stds_r[:, -1]
            ax_rollout.plot(x_r, m_overall_r, label=overall_label, linewidth=2.0, color="black")
            ax_rollout.scatter(x_r, m_overall_r, s=24, color="black", edgecolors="none", zorder=3)
            ax_rollout.fill_between(x_r, m_overall_r - s_overall_r, m_overall_r + s_overall_r, color="black", alpha=0.12)
            ax_rollout.set_xlabel("rollout step")
            ax_rollout.set_ylabel(metric_name)
            ax_rollout.grid(True, linestyle=":", alpha=0.6)
            ax_rollout.legend(fontsize=8, ncols=min(4, num_channels_r + 1))
        else:
            ax_rollout.text(0.5, 0.5, f"No rollout metrics for '{metric_name}'", ha="center", va="center")
            ax_rollout.axis("off")

        # ----------------------
        # Right column: Timestep
        # ----------------------
        if timestep_mean_key in stats and timestep_std_key in stats:
            means_t = stats[timestep_mean_key]  # (T_flat, C+1)
            stds_t = stats[timestep_std_key]    # (T_flat, C+1)
            Tflat, total_cols_t = means_t.shape
            num_channels_t = total_cols_t - 1
            channel_legends_t = list(output_channel_names) if num_channels_t == len(output_channel_names) else [f"ch_{i}" for i in range(num_channels_t)]
            #starts from index 0 so if input_steps=4: 0,1,2,3 then x_t starts from 4
            x_t = (input_steps - 1 + np.arange(1, Tflat + 1))*stride

            for c in range(num_channels_t):
                m = means_t[:, c]
                s = stds_t[:, c]
                line, = ax_timestep.plot(x_t, m, label=channel_legends_t[c], linewidth=1.5, alpha=0.95)
                ax_timestep.scatter(x_t, m, s=18, color=line.get_color(), edgecolors="none", zorder=3)
                ax_timestep.fill_between(x_t, m - s, m + s, color=line.get_color(), alpha=0.15)
            
            m_overall_t = means_t[:, -1]
            s_overall_t = stds_t[:, -1]
            ax_timestep.plot(x_t, m_overall_t, label=overall_label, linewidth=2.0, color="black")
            ax_timestep.scatter(x_t, m_overall_t, s=24, color="black", edgecolors="none", zorder=3)
            ax_timestep.fill_between(x_t, m_overall_t - s_overall_t, m_overall_t + s_overall_t, color="black", alpha=0.12)
            # Ensure uniform x-ticks at 'stride' between x_t[0] and x_t[-1], and also include 0
            try:
                xmin, xmax = ax_timestep.get_xlim()
                if xmin > 0:
                    ax_timestep.set_xlim(left=0)
                if len(x_t) > 0:
                    start_tick = float(x_t[0])
                    end_tick = float(x_t[-1])
                    step = float(stride) if float(stride) > 0 else max(1.0, end_tick - start_tick)
                    uniform_ticks = np.arange(start_tick, end_tick + 0.5 * step, step)
                    ticks_with_zero = np.unique(np.append(uniform_ticks, 0.0))
                    ax_timestep.set_xticks(ticks_with_zero)
            except Exception:
                pass
            ax_timestep.set_xlabel("time step")
            ax_timestep.set_ylabel(metric_name)
            ax_timestep.grid(True, linestyle=":", alpha=0.6)
            ax_timestep.legend(fontsize=8, ncols=min(4, num_channels_t + 1))
        else:
            ax_timestep.text(0.5, 0.5, f"No timestep metrics for '{metric_name}'", ha="center", va="center")
            ax_timestep.axis("off")

    if title:
        title_str = title
        if sequence_info is not None:
            title_str = f"{title}\nsequence_info={sequence_info}"
        fig.suptitle(title_str, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_info_strings(**kwargs) -> Tuple[str, str, str, str]:
    """Build summary strings for model, data, training and scheduler sections.

    This was moved from *bench/run.py* to centralise formatting utilities.
    """
    # ----------------- Model config string -----------------
    model_obj = kwargs["model_obj"]
    model_name = model_obj.__class__.__name__
    n_params = sum(p.numel() for p in model_obj.parameters())

    cfg_dict_raw = kwargs["model_config"]
    if cfg_dict_raw is not None:
        filtered_cfg = {
            k: v
            for k, v in cfg_dict_raw.items()
            if k != "model_name" and not isinstance(v, (dict, list, tuple)) and not k.startswith("_")
        }
        kv_pairs = [f"{k}={v}" for k, v in filtered_cfg.items()]
        lines = [", ".join(kv_pairs[i : i + 3]) for i in range(0, len(kv_pairs), 3)]
        config_block = "\n".join(lines) if lines else "-"
        model_info_str = f"{model_name} | Params: {n_params/1e6:.2f}M\nConfig: {config_block}"
    else:
        model_info_str = None

    # ----------------- Data configuration string -----------------
    flat_data = kwargs["data_config"]
    if flat_data is not None:
        filtered_data = {
            k: v
            for k, v in flat_data.items()
            if not isinstance(v, (dict, list, tuple)) and not k.startswith("_")
        }
        data_kv = [f"{k}={v}" for k, v in filtered_data.items()]
        data_lines = [", ".join(data_kv[i : i + 3]) for i in range(0, len(data_kv), 3)]
        data_info_str = "\n".join(data_lines) if data_lines else "-"
    else:
        data_info_str = None

    # ----------------- Train config strings -----------------
    train_cfg_raw = kwargs["train_config"]
    if train_cfg_raw is not None:
        filtered_train_cfg = {
            k: v
            for k, v in train_cfg_raw.items()
            if not isinstance(v, (dict, list, tuple)) and not k.startswith("_")
        }
        train_kv = [f"{k}={v}" for k, v in filtered_train_cfg.items()]
        train_lines = [", ".join(train_kv[i : i + 3]) for i in range(0, len(train_kv), 3)]
        train_info_str = "\n".join(train_lines) if train_lines else "-"
    else:
        train_info_str = None

    # ----------------- Scheduler config strings -----------------
    sched_cfg_raw = kwargs["scheduler_config"]
    if sched_cfg_raw is not None:
        filtered_sched_cfg = {
            k: v
            for k, v in sched_cfg_raw.items()
            if not isinstance(v, (dict, list, tuple)) and not k.startswith("_")
        }
        sched_kv = [f"{k}={v}" for k, v in filtered_sched_cfg.items()]
        sched_lines = [", ".join(sched_kv[i : i + 3]) for i in range(0, len(sched_kv), 3)]
        sched_info_str = "\n".join(sched_lines) if sched_lines else "-"
    else:
        sched_info_str = None

    return model_info_str, data_info_str, train_info_str, sched_info_str


def plot_multi_run_rollout_metrics(
    runs_step_metrics: dict,
    save_dir: str,
    title: str | None = None,
    filename: str = "all_runs_rollout_timestep_metrics.png",
    sequence_info: list[int] | tuple[int, int, int] | None = None,
    runs_sequence_info: dict | None = None,
) -> None:
    """Overlay per-metric rollout and timestep curves from multiple runs in one figure.

    Parameters
    ----------
    runs_step_metrics : dict[str, dict]
        Mapping from run label to the per-run step_metrics dict returned by
        `compute_metrics_for_n_rollouts(..., include_per_timestep=True)`.
    save_dir : str
        Directory to save the figure.
    title : Optional[str]
        Figure title.
    filename : str
        Output filename.
    sequence_info : List[int] | Tuple[int, int, int] | None
        Default sequence configuration [input_steps, output_steps, stride] used when a run
        does not supply its own configuration.
    runs_sequence_info : Optional[dict[str, list[int] | tuple[int, int, int]]]
        Optional mapping from run label to its sequence configuration; when provided,
        each run's timestep x-axis is computed using its own configuration.
    """
    os.makedirs(save_dir, exist_ok=True)

    if not runs_step_metrics:
        return

    # Collect union of metric names across runs
    metric_names: set[str] = set()
    for run_metrics in runs_step_metrics.values():
        metric_names.update(run_metrics.keys())
    metric_names = sorted(metric_names)

    # Prepare figure with two columns (rollout | timestep)
    num_metrics = len(metric_names)
    if num_metrics == 0:
        return

    fig, axes = plt.subplots(num_metrics, 2, figsize=(14, max(3 * num_metrics, 4)), squeeze=False)

    # Column titles
    axes[0, 0].set_title("Rollout step metrics", fontsize=12)
    axes[0, 1].set_title("Timestep metrics", fontsize=12)

    # Default x-axis scaling for timesteps (used if a run-specific value is absent)
    default_input_steps = sequence_info[0] if sequence_info is not None else 1
    default_stride = sequence_info[2] if sequence_info is not None else 1

    # Track one handle per run for a global legend
    run_label_to_handle: dict[str, any] = {}

    for row_idx, metric_name in enumerate(metric_names):
        ax_rollout = axes[row_idx, 0]
        ax_timestep = axes[row_idx, 1]

        # Keys expected in per-run stats (use per_step keys by default)
        rollout_mean_key = "per_rollout_step_mean"
        rollout_std_key = "per_rollout_step_std"
        timestep_mean_key = "per_timestep_mean"
        timestep_std_key = "per_timestep_std"

        # Plot each run's overall curve with its std band
        # Collect required x-ticks (each run's first/last x_t)
        timestep_ticks: set[float] = set()

        for run_label, run_metrics in runs_step_metrics.items():
            if metric_name not in run_metrics:
                continue
            stats = run_metrics[metric_name]

            # Left: rollout step overlay
            if rollout_mean_key in stats and rollout_std_key in stats:
                means_r = stats[rollout_mean_key]  # (R, C+1)
                stds_r = stats[rollout_std_key]    # (R, C+1)
                R = means_r.shape[0]
                x_r = np.arange(1, R + 1)
                m_overall_r = means_r[:, -1]
                s_overall_r = stds_r[:, -1]
                line, = ax_rollout.plot(x_r, m_overall_r, label=run_label, linewidth=2.0)
                ax_rollout.fill_between(x_r, m_overall_r - s_overall_r, m_overall_r + s_overall_r, color=line.get_color(), alpha=0.15)
                # Capture a handle for the global legend if not set yet
                if run_label not in run_label_to_handle:
                    run_label_to_handle[run_label] = line

            # Right: timestep overlay
            if timestep_mean_key in stats and timestep_std_key in stats:
                means_t = stats[timestep_mean_key]  # (T_flat, C+1)
                stds_t = stats[timestep_std_key]    # (T_flat, C+1)
                Tflat = means_t.shape[0]
                # Determine per-run sequence info for x-axis
                if runs_sequence_info is not None and run_label in runs_sequence_info and runs_sequence_info[run_label] is not None:
                    run_si = runs_sequence_info[run_label]
                    run_input_steps = int(run_si[0]) if len(run_si) > 0 else default_input_steps
                    run_stride = int(run_si[2]) if len(run_si) > 2 else default_stride
                else:
                    run_input_steps = default_input_steps
                    run_stride = default_stride
                # starts from index 0 so if input_steps=4: 0,1,2,3 then x_t starts from 4
                x_t = (run_input_steps - 1 + np.arange(1, Tflat + 1)) * run_stride
                m_overall_t = means_t[:, -1]
                s_overall_t = stds_t[:, -1]
                line, = ax_timestep.plot(x_t, m_overall_t, label=run_label, linewidth=2.0)
                ax_timestep.fill_between(x_t, m_overall_t - s_overall_t, m_overall_t + s_overall_t, color=line.get_color(), alpha=0.15)
                # Required ticks: include each run's first and last x_t
                if x_t.size > 0:
                    timestep_ticks.add(float(x_t[0]))
                    timestep_ticks.add(float(x_t[-1]))
                # Capture a handle for the global legend if not already captured from rollout panel
                if run_label not in run_label_to_handle:
                    run_label_to_handle[run_label] = line

        # Common styling per row
        ax_rollout.set_xlabel("rollout step")
        ax_rollout.set_ylabel(metric_name)
        ax_rollout.grid(True, linestyle=":", alpha=0.6)
        # No per-axes legend; use a single global legend instead

        ax_timestep.set_xlabel("time step")
        ax_timestep.set_ylabel(metric_name)
        ax_timestep.grid(True, linestyle=":", alpha=0.6)
        # Apply required timestep ticks (union of first/last for all runs in this row)
        if len(timestep_ticks) > 0:
            try:
                sorted_ticks = sorted(timestep_ticks)
                ax_timestep.set_xticks(sorted_ticks)
            except Exception:
                pass

    if title:
        title_str = title
        # if sequence_info is not None:
        #     title_str = f"{title}\nsequence_info={sequence_info}"
        fig.suptitle(title_str, fontsize=14)

    # Global legend at the bottom for all runs
    if len(run_label_to_handle) > 0:
        try:
            fig.legend(
                handles=list(run_label_to_handle.values()),
                labels=list(run_label_to_handle.keys()),
                loc="lower center",
                ncol=min(6, max(1, len(run_label_to_handle))),
                frameon=False,
            )
        except Exception:
            pass

    # Leave space at bottom for the global legend
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))

    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_rollout_metrics_bar_chart(
    runs_step_metrics: dict,
    save_dir: str,
    run_configs: dict,
    filename: str = "rollout_metrics_bar_chart.png",
) -> None:
    """Compute mean per-rollout step metrics and plot as grouped bar charts.

    Parameters
    ----------
    runs_step_metrics : dict[str, dict]
        Mapping from run label to the per-run step_metrics dict returned by
        `compute_metrics_for_n_rollouts(..., include_per_timestep=True)`.
    save_dir : str
        Directory to save the figure.
    run_configs : dict[str, dict]
        Mapping from run label to the run configuration dictionary.
    filename : str
        Output filename.

    Plotting logic:
     - Loads config for each run
     - By comparing configs, divides runs into two groupings
     - First grouping: runs that get plotted together in the same subplot
     - Second grouping: runs that get the same bar color/style within a subplot
     - The legend displays the parameters that differ between the runs in each grouping
     - By default: first grouping is by architecture, second grouping is by conditioning/input data
     - Can be customized by changing the conditioning_vars list in the code
    """
    # -------------------- Utility Functions --------------------
    def make_hashable(obj):
        """Convert a container to a hashable type recursively."""
        if isinstance(obj, (list, np.ndarray)):
            return tuple(make_hashable(item) for item in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((key, make_hashable(value)) for key, value in obj.items()))
        elif isinstance(obj, (set, frozenset)):
            return tuple(sorted(make_hashable(item) for item in obj))
        else:
            return obj

    # -------------------- Data Processing Functions --------------------

    def process_run_configs(run_configs, conditioning_vars):
        """Process run configurations into architectures, conditioning info, and other parameters."""
        run_architectures = {}
        run_conditioning_info = {}
        run_other_params = {}

        for run_name, config in run_configs.items():
            architecture = config.get('architectures', ['Unknown'])[0]
            run_architectures[run_name] = architecture

            conditioning_params = {var: config.get(var, None) for var in conditioning_vars}
            run_conditioning_info[run_name] = conditioning_params

            exclude_keys = ['architectures', 'conditioning', 'sequence_info', 'in_size', 'out_size']
            other_params = {k: v for k, v in config.items() if k not in exclude_keys}
            run_other_params[run_name] = other_params

        return run_architectures, run_conditioning_info, run_other_params

    def calculate_results(runs_step_metrics):
        """Calculate mean and pooled std for L1 and L2 errors."""
        results = {}
        for run_name, metrics in runs_step_metrics.items():
            l1_mean = metrics['l1_error']['per_rollout_step_mean'][:, -1]
            l1_std = metrics['l1_error']['per_rollout_step_std'][:, -1]
            l2_mean = metrics['l2_error']['per_rollout_step_mean'][:, -1]
            l2_std = metrics['l2_error']['per_rollout_step_std'][:, -1]

            l1_mean_avg = np.mean(l1_mean)
            l2_mean_avg = np.mean(l2_mean)
            l1_pooled_std = np.sqrt(np.sum(l1_std**2 + (l1_mean - l1_mean_avg)**2) / (len(l1_mean) + 1))
            l2_pooled_std = np.sqrt(np.sum(l2_std**2 + (l2_mean - l2_mean_avg)**2) / (len(l2_mean) + 1))

            results[run_name] = {
                'l1_mean_avg': l1_mean_avg,
                'l2_mean_avg': l2_mean_avg,
                'l1_pooled_std': l1_pooled_std,
                'l2_pooled_std': l2_pooled_std,
            }
        return results

    def group_runs_by_architecture(run_architectures):
        """Group runs by model architecture."""
        architecture_groups = {}
        for run_name, arch in run_architectures.items():
            if arch not in architecture_groups:
                architecture_groups[arch] = []
            architecture_groups[arch].append(run_name)
        return architecture_groups

    def group_runs_by_parameters(architecture_groups, run_other_params):
        """Group runs by differentiating parameters."""
        grouped_runs = {}
        for arch, run_names in architecture_groups.items():
            param_subgroups = {}
            for run_name in run_names:
                param_items = []
                for key, value in sorted(run_other_params[run_name].items()):
                    try:
                        hashable_value = make_hashable(value)
                        param_items.append((key, hashable_value))
                    except:
                        param_items.append((key, str(value)))
                param_tuple = tuple(param_items)
                if param_tuple not in param_subgroups:
                    param_subgroups[param_tuple] = []
                param_subgroups[param_tuple].append(run_name)
            grouped_runs[arch] = param_subgroups
        return grouped_runs

    def simplify_groups(grouped_runs):
        """Simplify groups to store only differentiating parameters."""
        simplified_groups = {}
        for arch, subgroups in grouped_runs.items():
            simplified_groups[arch] = []
            if len(subgroups) <= 1:
                for param_tuple, runs in subgroups.items():
                    simplified_groups[arch].append({'runs': runs, 'diff_params': {}})
                continue
            all_param_values = {}
            for param_tuple, _ in subgroups.items():
                params_dict = dict(param_tuple)
                for key, value in params_dict.items():
                    if key not in all_param_values:
                        all_param_values[key] = set()
                    all_param_values[key].add(value)
            diff_param_keys = [k for k, v in all_param_values.items() if len(v) > 1]
            for param_tuple, runs in subgroups.items():
                params_dict = dict(param_tuple)
                diff_params = {k: params_dict[k] for k in diff_param_keys if k in params_dict}
                simplified_groups[arch].append({'runs': runs, 'diff_params': diff_params})
        return simplified_groups

    def group_runs_by_conditioning(run_conditioning_info):
        """Group runs by conditioning and sequence info."""
        conditioning_groups = {}
        group_labels = {}
        group_counter = 1
        for run_name, cond_info in run_conditioning_info.items():
            hashable_cond = make_hashable(cond_info)
            if hashable_cond not in conditioning_groups:
                conditioning_groups[hashable_cond] = []
                group_labels[hashable_cond] = f"G{group_counter}"
                group_counter += 1
            conditioning_groups[hashable_cond].append(run_name)
        run_group_labels = {}
        for hashable_cond, runs in conditioning_groups.items():
            group_label = group_labels[hashable_cond]
            for run in runs:
                run_group_labels[run] = group_label
        return run_group_labels

    # -------------------- Plotting Functions --------------------

    def plot_grouped_l2_errors(simplified_groups, results, run_group_labels):
        """Plot grouped L2 errors."""
        total_subplots = sum(len(groups) for arch, groups in simplified_groups.items())
        n_cols = min(3, total_subplots)
        n_rows = math.ceil(total_subplots / n_cols)
        
        fig_width = 2.5 * n_cols
        fig_height = 3.5 * n_rows
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        if total_subplots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        unique_groups = sorted(set(run_group_labels.values()))  # Sort for consistent order
        group_colors = {}
        cmap = cm.get_cmap('tab10', len(unique_groups))
        
        for i, group in enumerate(unique_groups):
            group_colors[group] = cmap(i)
        
        # Get min/max for consistent y-axis
        all_l2_values = []
        for arch, groups in simplified_groups.items():
            for group in groups:
                runs = group['runs']
                for run in runs:
                    if run in results:
                        l2_error = results[run]['l2_mean_avg']
                        l2_std = results[run]['l2_pooled_std']
                        all_l2_values.append(l2_error + l2_std)
                        all_l2_values.append(max(l2_error - l2_std, 0.001))

        global_min = max(0.001, min(all_l2_values) / 1.5)
        global_max = max(all_l2_values) * 1.1
        
        # Plot each subgroup in its own subplot
        subplot_idx = 0
        for arch, groups in simplified_groups.items():
            for subgroup_idx, group in enumerate(groups):
                if subplot_idx >= len(axes):
                    break
                    
                ax = axes[subplot_idx]
                ax.set_yscale('log')
                ax.set_ylim(global_min, global_max)
                
                # Set title with architecture name and subgroup index
                ax.set_title(f"{arch} {subgroup_idx+1}", fontsize=10)
                if subplot_idx % n_cols == 0:
                    ax.set_ylabel("L2 Error", fontsize=10)
                else:
                    ax.tick_params(axis='y', which='both', labelleft=False)
                ax.set_yscale('log')

                ax.set_ylim(global_min, global_max)
                log_min = np.floor(np.log10(global_min))
                log_max = np.ceil(np.log10(global_max))
                num_ticks = int(log_max - log_min + 1)
                yticks = np.logspace(log_min, log_max, num=num_ticks)
                ax.set_yticks(yticks)
                ax.set_yticklabels([f"{tick:.1e}" for tick in yticks])
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                runs = group['runs']
                diff_params = group['diff_params']
                
                # Sort runs by their group labels for consistent order
                runs = sorted(runs, key=lambda run: run_group_labels.get(run, "Unknown"))
                
                all_x_positions = []
                all_labels = []
                all_colors = []
                
                bar_width = 0.7
                x_positions = [i for i in range(len(runs))]
                all_x_positions.extend(x_positions)
                
                for run in runs:
                    group_label = run_group_labels.get(run, "Unknown")
                    all_labels.append(group_label)
                    all_colors.append(group_colors[group_label])
                
                # Get L2 errors and standard deviations
                l2_errors = [results[run]['l2_mean_avg'] if run in results else 0 for run in runs]
                l2_stds = [results[run]['l2_pooled_std'] if run in results else 0 for run in runs]
                
                # Plot bars with colors based on conditioning group
                for i, (x, val, std, color) in enumerate(zip(x_positions, l2_errors, l2_stds, all_colors)):
                    bar = ax.bar(x, val, yerr=std, width=bar_width, alpha=0.7, 
                        color=color, capsize=3)
                
                # Set x-ticks and labels
                ax.set_xticks(all_x_positions)
                ax.set_xticklabels(all_labels, rotation=0, ha='center', fontsize=9)
                
                # Add parameter group legend if there are parameters
                if diff_params:
                    param_label = ", ".join([f"{k}={v}" for k, v in diff_params.items()])
                    ax.text(0.5, 0.95, param_label, 
                        transform=ax.transAxes, ha='center', 
                        va='top', fontsize=7, bbox=dict(facecolor='white', alpha=0.7))
                
                subplot_idx += 1
        
        # Hide unused subplots
        for i in range(subplot_idx, len(axes)):
            axes[i].set_visible(False)
        
        legend_elements = []
        conditioning_varies = len(set(cond_info['conditioning'] for cond_info in run_conditioning_info.values())) > 1
        
        for group_label in unique_groups:
            for run_name, label in run_group_labels.items():
                if label == group_label and run_name in run_conditioning_info:
                    cond_info = run_conditioning_info[run_name]
                    
                    legend_text = f"{group_label}: seq_info={cond_info['sequence_info']}"
                    if conditioning_varies:
                        legend_text += f", cond={cond_info['conditioning']}"
                        
                    legend_elements.append(Patch(facecolor=group_colors[group_label], 
                                                label=legend_text))
                    break

        param_legend_elements = []
        for arch, groups in simplified_groups.items():
            for subgroup_idx, group in enumerate(groups):
                diff_params = group['diff_params']
                if diff_params:
                    param_str = ", ".join([f"{k}={v}" for k, v in diff_params.items()])
                    label = f"{arch} {subgroup_idx+1}: {param_str}"
                else:
                    label = f"{arch} {subgroup_idx+1}: Default config"
                
                marker_style = 'o' if 'FNO' in arch else ('s' if 'CNO' in arch else '^')
                
                param_legend_elements.append(plt.Line2D([0], [0], marker=marker_style, 
                                            color='w', markerfacecolor=f'C{subgroup_idx%10}', 
                                            markersize=8, label=label))

        num_param_legend_rows = (len(param_legend_elements) // 2) + 1 
        extra_height = 0.6 + 0.2 * num_param_legend_rows

        fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1] + extra_height)

        fig.legend(handles=legend_elements, loc='lower center', fontsize=8, 
                bbox_to_anchor=(0.5, 0.2),
                ncol=1, frameon=True, title="Conditioning Groups")

        if param_legend_elements:
            fig.legend(handles=param_legend_elements, loc='lower center', 
                    fontsize=7, bbox_to_anchor=(0.5, 0.05),
                    ncol=min(2, len(param_legend_elements)), frameon=True, 
                    title="Parameter Configurations")

            bottom_margin = 0.4
            plt.subplots_adjust(bottom=bottom_margin)

        plt.tight_layout(rect=[0, bottom_margin, 1, 1])
        out_path = os.path.join(save_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

    # -------------------- Main Script --------------------
    conditioning_vars = ["conditioning", "sequence_info", "in_size", "out_size"]

    # Process data
    run_architectures, run_conditioning_info, run_other_params = process_run_configs(run_configs, conditioning_vars)
    results = calculate_results(runs_step_metrics)
    architecture_groups = group_runs_by_architecture(run_architectures)
    grouped_runs = group_runs_by_parameters(architecture_groups, run_other_params)
    simplified_groups = simplify_groups(grouped_runs)
    run_group_labels = group_runs_by_conditioning(run_conditioning_info)

    # Plot results
    plot_grouped_l2_errors(simplified_groups, results, run_group_labels)
