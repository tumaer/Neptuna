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
        aspect = 'equal' if h == w else 'auto'
        
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
            ax.set_title(ch_names[0], fontsize=8)
            # Increase the padding so the horizontal colorbar does not overlap the image grid.
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.12, orientation='horizontal', location='bottom')
            cbar.ax.tick_params(labelsize=12)
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.formatter.set_useMathText(True)  # Use math text for consistent font
            cbar.ax.xaxis.offsetText.set_y(-1.0)  # Move the offset text further down
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
                sub_ax.set_title(ch_names[c], fontsize=8)
                # Increase the padding so the horizontal colorbar does not overlap the image grid.
                cbar = fig.colorbar(im, ax=sub_ax, fraction=0.046, pad=0.12, orientation='horizontal', location='bottom')
                cbar.ax.tick_params(labelsize=12)
                cbar.formatter.set_scientific(True)
                cbar.formatter.set_powerlimits((0, 0))
                cbar.formatter.set_useMathText(True)  # Use math text for consistent font
                cbar.ax.xaxis.offsetText.set_y(-1.0)  # Move the offset text further down
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
    is_best_metric: bool = False,
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
    is_best_metric : bool, default=False
        Whether this represents the best metric checkpoint for special handling.
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
        # example_indices = example_indices[(example_indices >= 0) & (example_indices < N)]
        # if example_indices.size == 0:
        #     np.random.seed(42)
        #     num_pick = min(num_examples, N)
        #     example_indices = np.random.choice(N, size=num_pick, replace=False)

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
            header_ratio = 0.15  # Height allocated for column titles
            time_label_ratio = 0.5  # Height for time label row
            footer_ratio = 4.0   # Further increase footer height for more space under time labels

            # Padding between individual plot and its xlabel (time indicator). 
            # If this padding is too large, the xlabel from one subplot can overlap the
            # axes of the subplot in the row below.  Use a smaller value for 2-D plots
            # (which typically have less vertical space per row) and a moderate value
            # for 1-D plots.
            x_label_pad = 40 if ndim == 2 else 30

            # Reduce horizontal spacing so each subplot takes up more space within its
            # column, effectively making the plots ~50 % larger without increasing the
            # overall figure size.
            main_wspace = 0.3
            
            # Adjust figure width based on total content
            base_width_per_unit = 4.0 if ndim == 2 else 5.0
            fig_width = total_grid_cols * base_width_per_unit

            # For ndim=2, make plot cells match data aspect ratio by adjusting figure height.
            # The height of a plot row (where height_ratio=1) should equal the width of a grid cell, scaled by the data's aspect ratio.
            # The total height in ratio units is header_ratio + nrows*1 (plots) + footer_ratio.
            if ndim == 2:
                H, W = spatial_shape
                data_aspect_ratio = H / W
                # We want the height for a ratio of 1.0 to be `base_width_per_unit * data_aspect_ratio`.
                # So, total height = (total_ratio_units) * (height_of_one_ratio_unit).
                cell_height = base_width_per_unit * data_aspect_ratio
                fig_height = (nrows + header_ratio + footer_ratio) * cell_height
            else:  # Original calculation for 1D plots which can be non-square
                fig_height = (nrows + header_ratio + footer_ratio + 1.3) * 3.5  # extra offset for titles
            
            fig = plt.figure(figsize=(fig_width, fig_height))

            # Add main title
            # Adjust y-position to be relative to the new figure height calculation
            suptitle_y_pos = 1 - (0.10 / (nrows + header_ratio + footer_ratio)) if ndim == 2 else 0.965
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
            
            # Add dimensions info as subtitle, positioned farther below the suptitle to avoid overlap.
            dims_text = (
                f"Additional Info: Total number of examples={N}, Spatial_res={spatial_shape}, "
                f"# Input_frames={T_in}, # Input_channels={C}, # Prediction_frames={T_pred}, "
                f"# Prediction_channels={pred.shape[1]}"
            )

            # Keep dims_text focused on dataset statistics; append a one-line model summary.
            if model_info is not None:
                summary_line = model_info.split("\n")[0]  # e.g. "FNO | Params: 12.3M"
                dims_text += "\n" + summary_line

            # Increase spacing below the suptitle so dims_text never collides with it.
            text_y_pos = suptitle_y_pos - (0.06 if ndim == 2 else 0.06)
            fig.text(0.5, text_y_pos, dims_text, ha='center', va='center', fontsize=22)

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

            # Create gridspec with variable column widths and specific height ratios
            # The hspace and wspace from the old plt.subplots_adjust are used here.
            # Anchor the top of the gridspec to be just below the dims_text for consistent spacing.
            gs = gridspec.GridSpec(nrows + 3, total_grid_cols,
                                figure=fig,
                                top=text_y_pos - 0.02,
                                height_ratios=[header_ratio] + [1] * nrows + [time_label_ratio, footer_ratio],
                                hspace=0.4, wspace=main_wspace)

            # Add column titles at the top
            for col_idx, (start_col, end_col) in enumerate(col_positions):
                title_ax = fig.add_subplot(gs[0, start_col:end_col])
                title_ax.axis('off')
                title_ax.text(0.5, 0.5, column_titles[col_idx], ha='center', va='center', fontsize=14, weight='bold')

            # Add time labels at the bottom  
            for col_idx, (start_col, end_col) in enumerate(col_positions):
                time_ax = fig.add_subplot(gs[-2, start_col:end_col])
                time_ax.axis('off')
                if col_idx == 0:  # Input column
                    time_label = f"t - {stride * (T_in - 1)} to t"
                elif has_conditioning and col_idx == 1:  # Conditioning column
                    time_label = f"t - {stride * (T_in - 1)} to t"
                else:  # Other columns (prediction, target, errors)
                    time_label = f"t + {stride} to t + {stride * T_pred}"
                time_ax.text(0.5, 0.5, time_label, ha='center', va='center', fontsize=28)

            plot_time_fontsize = 20  # Font size for per-plot time labels

            for row in range(nrows):
                row_offset = row + 1

                # Column 0: Input
                start_col, end_col = col_positions[0]
                if row < T_in:
                    time_val = "t" if row == T_in - 1 else f"t - {stride * (T_in - 1 - row)}"
                    input_ax = fig.add_subplot(gs[row_offset, start_col:end_col])
                    axes_to_label = _plot_data(input_ax, inp[row], ndim, only_input_channel_names, vmins, vmaxs)
                    if isinstance(axes_to_label, list):  # Multi-channel 2D case
                        # Add time label only to the middle channel
                        mid_channel = len(axes_to_label) // 2
                        axes_to_label[mid_channel].set_xlabel(time_val, fontsize=plot_time_fontsize, labelpad=x_label_pad)
                        axes_to_label[mid_channel].tick_params(labelbottom=True)
                    else:  # Single channel or 1D case
                        input_ax.set_xlabel(time_val, fontsize=plot_time_fontsize, labelpad=x_label_pad)
                        input_ax.tick_params(labelbottom=True)

                # Column 1: Conditioning (if present)
                if has_conditioning:
                    start_col, end_col = col_positions[1]
                    if row < T_in:
                        time_val = "t" if row == T_in - 1 else f"t - {stride * (T_in - 1 - row)}"
                        cond_inp = conditioning_input_array[idx]  # [T_in, C_cond, ...]
                        cond_ax = fig.add_subplot(gs[row_offset, start_col:end_col])
                        axes_to_label = _plot_data(cond_ax, cond_inp[row], ndim, conditioning_input_channel_names)
                        if isinstance(axes_to_label, list):  # Multi-channel 2D case
                            mid_channel = len(axes_to_label) // 2
                            axes_to_label[mid_channel].set_xlabel(time_val, fontsize=plot_time_fontsize, labelpad=x_label_pad)
                            axes_to_label[mid_channel].tick_params(labelbottom=True)
                        else:  # Single channel or 1D case
                            cond_ax.set_xlabel(time_val, fontsize=plot_time_fontsize, labelpad=x_label_pad)
                            cond_ax.tick_params(labelbottom=True)

                # Determine column indices for remaining plots
                pred_col_idx = 2 if has_conditioning else 1
                target_col_idx = 3 if has_conditioning else 2
                abs_err_col_idx = 4 if has_conditioning else 3
                rel_err_col_idx = 5 if has_conditioning else 4

                # Prediction column
                start_col, end_col = col_positions[pred_col_idx]
                if row < T_pred:
                    time_val = f"t + {stride * (row + 1)}"
                    pred_ax = fig.add_subplot(gs[row_offset, start_col:end_col])
                    axes_to_label = _plot_data(pred_ax, pred[row], ndim, output_channel_names, vmins, vmaxs)
                    if isinstance(axes_to_label, list):  # Multi-channel 2D case
                        # Add time label only to the middle channel
                        mid_channel = len(axes_to_label) // 2
                        axes_to_label[mid_channel].set_xlabel(time_val, fontsize=plot_time_fontsize, labelpad=x_label_pad)
                        axes_to_label[mid_channel].tick_params(labelbottom=True)
                    else:  # Single channel or 1D case
                        pred_ax.set_xlabel(time_val, fontsize=plot_time_fontsize, labelpad=x_label_pad)
                        pred_ax.tick_params(labelbottom=True)

                # Target column
                start_col, end_col = col_positions[target_col_idx]
                if row < T_pred:
                    time_val = f"t + {stride * (row + 1)}"
                    target_ax = fig.add_subplot(gs[row_offset, start_col:end_col])
                    axes_to_label = _plot_data(target_ax, tgt[row], ndim, output_channel_names, vmins, vmaxs)
                    if isinstance(axes_to_label, list):  # Multi-channel 2D case
                        # Add time label only to the middle channel
                        mid_channel = len(axes_to_label) // 2
                        axes_to_label[mid_channel].set_xlabel(time_val, fontsize=plot_time_fontsize, labelpad=x_label_pad)
                        axes_to_label[mid_channel].tick_params(labelbottom=True)
                    else:  # Single channel or 1D case
                        target_ax.set_xlabel(time_val, fontsize=plot_time_fontsize, labelpad=x_label_pad)
                        target_ax.tick_params(labelbottom=True)

                # Abs Error column
                start_col, end_col = col_positions[abs_err_col_idx]
                if row < T_pred:
                    time_val = f"t + {stride * (row + 1)}"
                    abs_err_ax = fig.add_subplot(gs[row_offset, start_col:end_col])
                    axes_to_label = _plot_data(abs_err_ax, abs_err[row], ndim, output_channel_names)
                    if isinstance(axes_to_label, list):  # Multi-channel 2D case
                        # Add time label only to the middle channel
                        mid_channel = len(axes_to_label) // 2
                        axes_to_label[mid_channel].set_xlabel(time_val, fontsize=plot_time_fontsize, labelpad=x_label_pad)
                        axes_to_label[mid_channel].tick_params(labelbottom=True)
                    else:  # Single channel or 1D case
                        abs_err_ax.set_xlabel(time_val, fontsize=plot_time_fontsize, labelpad=x_label_pad)
                        abs_err_ax.tick_params(labelbottom=True)

                # Rel Error column
                start_col, end_col = col_positions[rel_err_col_idx]
                if row < T_pred:
                    time_val = f"t + {stride * (row + 1)}"
                    rel_err_ax = fig.add_subplot(gs[row_offset, start_col:end_col])
                    axes_to_label = _plot_data(rel_err_ax, rel_err[row], ndim, output_channel_names)
                    if isinstance(axes_to_label, list):  # Multi-channel 2D case
                        # Add time label only to the middle channel
                        mid_channel = len(axes_to_label) // 2
                        axes_to_label[mid_channel].set_xlabel(time_val, fontsize=plot_time_fontsize, labelpad=x_label_pad)
                        axes_to_label[mid_channel].tick_params(labelbottom=True)
                    else:  # Single channel or 1D case
                        rel_err_ax.set_xlabel(time_val, fontsize=plot_time_fontsize, labelpad=x_label_pad)
                        rel_err_ax.tick_params(labelbottom=True)

            # --------------------------------------------------------------
            # Footer: detailed model & data configuration (indented bullets)
            # --------------------------------------------------------------
            if footer_lines:
                footer_text = "\n".join(footer_lines)
                footer_ax = fig.add_subplot(gs[-1, :])
                footer_ax.axis('off')
                
                footer_ax.text(
                    -0.10,
                    -0.25,
                    footer_text,
                    ha="left",
                    va="bottom",
                    fontsize=20,
                    wrap=True
                )

            # --------------------------------------------------------------
            # Saving behaviour
            # --------------------------------------------------------------
            # We *always* save the *best* figure to disk so it can later be
            # uploaded to W&B after the training run has concluded.  If
            # ``log_to_wandb`` is *False* we additionally save all other
            # figures for offline inspection.

            save_this_fig = (not log_to_wandb) or is_best_metric

            if save_this_fig:
                # prepend the word "best" if this is the best metric figure
                if is_best_metric:
                    # Before writing the new "best" file, rename a previous
                    # best (if any) so it is no longer tagged as best.
                    for prev_file in os.listdir(save_dir):
                        if prev_file.endswith("_best.png"):
                            old_path = os.path.join(save_dir, prev_file)
                            new_file = prev_file.replace("_best.png", ".png")
                            new_path = os.path.join(save_dir, new_file)
                            try:
                                os.rename(old_path, new_path)
                            except OSError as e:
                                print(
                                    f"[plot_progress] Warning: could not rename previous best file '{prev_file}': {e}"
                                )
                    filename = f"ckpt_{checkpoint_step}_epoch_{epoch}_example_{idx}_best.png"
                else:
                    filename = f"ckpt_{checkpoint_step}_epoch_{epoch}_example_{idx}.png"

                img_path = os.path.join(save_dir, filename)
                fig.savefig(img_path, dpi=150, bbox_inches="tight")

                # When W&B logging is active we want *only* the best figure in
                # the directory.  Remove any other PNGs that do not correspond
                # to the freshly written best file.
                if log_to_wandb and is_best_metric:
                    for other_file in os.listdir(save_dir):
                        if other_file.endswith(".png") and other_file != filename:
                            try:
                                os.remove(os.path.join(save_dir, other_file))
                            except OSError as e:
                                print(f"[plot_progress] Warning: could not delete '{other_file}': {e}")

            # --------------------------------------------------------------
            # W&B logging (only create the image object here; actual logging
            # timing can be handled by the caller).
            # --------------------------------------------------------------

            if log_to_wandb:
                # Use an in-memory buffer with bbox_inches='tight' so nothing is cut off
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.1)
                buf.seek(0)
                pil_img = Image.open(buf)

                returned_figs[f"plot_progress/example_{idx}"] = wandb.Image(pil_img)

                # We purposely **do not** store the best plot in
                # ``returned_figs`` so that it lives **only on disk**.
                # A separate post-run routine uploads the saved PNG on_train_end inside WandbCallback.
            
            plt.close(fig)
    return returned_figs


def plot_rollout_metrics(step_metrics: dict, output_channel_names: list[str], save_dir: str, title: str | None = None, filename: str = "rollout_metrics.png", plot_type: str = "per_step") -> None:
    """Plot per-metric curves over rollout steps for IC-start evaluations.

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
        Which statistics to plot. Defaults to "cumulative". When set to
        "per_step", plots per_step_mean and per_step_std.
    """
    os.makedirs(save_dir, exist_ok=True)

    num_metrics = len(step_metrics)
    if num_metrics == 0:
        return

    # Determine keys for mean/std based on requested plot type
    if plot_type not in {"cumulative", "per_step"}:
        raise ValueError("plot_type must be 'cumulative' or 'per_step'")
    mean_key = f"{plot_type}_mean"
    std_key = f"{plot_type}_std"

    # Create one subplot per metric
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, max(3 * num_metrics, 4)), squeeze=False)
    axes = axes[:, 0]

    # Prepare legends (channels + overall)
    channel_legends = list(output_channel_names)
    overall_label = "overall"

    for ax, (metric_name, stats) in zip(axes, step_metrics.items()):
        if mean_key not in stats or std_key not in stats:
            # Skip metrics missing requested stats
            continue
        means = stats[mean_key]   # (T, C+1)
        stds = stats[std_key]     # (T, C+1)

        T, total_cols = means.shape
        num_channels = total_cols - 1
        if num_channels != len(output_channel_names):
            # Fallback if mismatch
            channel_legends = [f"ch_{i}" for i in range(num_channels)]

        x = np.arange(1, T + 1)
        # Plot per-channel mean with std bands, using scatter markers and connecting lines
        for c in range(num_channels):
            channel_mean = means[:, c]
            channel_std = stds[:, c]
            line, = ax.plot(x, channel_mean, label=channel_legends[c], linewidth=1.5, alpha=0.95)
            ax.scatter(x, channel_mean, s=18, color=line.get_color(), edgecolors="none", zorder=3)
            ax.fill_between(x, channel_mean - channel_std, channel_mean + channel_std, color=line.get_color(), alpha=0.15)
        # Plot overall mean with std band, with markers and connecting line
        overall_mean = means[:, -1]
        overall_std = stds[:, -1]
        ax.plot(x, overall_mean, label=overall_label, linewidth=2.0, color="black")
        ax.scatter(x, overall_mean, s=24, color="black", edgecolors="none", zorder=3)
        ax.fill_between(x, overall_mean - overall_std, overall_mean + overall_std, color="black", alpha=0.12)

        ax.set_xlabel("rollout step")
        ax.set_ylabel(metric_name)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8, ncols=min(4, num_channels + 1))

    if title:
        fig.suptitle(title, fontsize=14)
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