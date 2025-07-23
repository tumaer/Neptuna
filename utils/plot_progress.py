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

def _plot_data(ax, data, ndim, ch_names=None):
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
        
        if C == 1:
            im = ax.imshow(data[0], cmap="coolwarm", aspect=aspect, origin='lower')
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
            gs = ax.get_subplotspec().subgridspec(1, ncols, wspace=0.6)  # Add spacing between channels
            ax.remove()
            sub_axes = []
            for c in range(C):
                sub_ax = fig.add_subplot(gs[0, c])
                im = sub_ax.imshow(data[c], cmap="coolwarm", aspect=aspect, origin='lower')
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
    is_best_metric: bool = False
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

    extra_info = extra_info.split('/')[-1]

    np.random.seed(42)
    example_indices = np.random.choice(N, size=num_examples, replace=False)

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
            footer_ratio = 2.4   # Further increase footer height for more space under time labels

            # Padding between individual plot and its xlabel (time indicator). 
            # If this padding is too large, the xlabel from one subplot can overlap the
            # axes of the subplot in the row below.  Use a smaller value for 2-D plots
            # (which typically have less vertical space per row) and a moderate value
            # for 1-D plots.
            x_label_pad = 25 if ndim == 2 else 30

            # Use a uniform but larger wspace to create clearer separation between logical columns.
            main_wspace = 1.2  # Empirically chosen for good visual separation
            
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
            # Use a multi-line title so that all components are visible without being truncated.
            fig.suptitle(
                f"{extra_info}\nCheckpoint Step: {checkpoint_step}, Epoch: {epoch}\nExample Index: {idx}",
                fontsize=32,
                y=suptitle_y_pos,
                weight='bold'
            )
            
            # Add dimensions info as subtitle, positioned farther below the suptitle to avoid overlap.
            dims_text = (
                f"Additional Info: Total number of validation examples={N}, Spatial_res={spatial_shape}, "
                f"# Input_frames={T_in}, # Input_channels={C}, # Prediction_frames={T_pred}, "
                f"# Prediction_channels={pred.shape[1]}"
            )

            # Increase spacing below the suptitle further so that the dims text never
            # collides with the (multi-line) suptitle.  Empirically a 0.10 offset for
            # 2-D plots and 0.08 for 1-D plots provides sufficient clearance across a
            # wide range of figure heights.
            # Reduce the vertical offset so the dims_text sits closer to the suptitle.
            text_y_pos = suptitle_y_pos - (0.06 if ndim == 2 else 0.06)
            fig.text(0.5, text_y_pos, dims_text, ha='center', va='center', fontsize=22)

            # Create gridspec with variable column widths and specific height ratios
            # The hspace and wspace from the old plt.subplots_adjust are used here.
            # Anchor the top of the gridspec to be just below the dims_text for consistent spacing.
            gs = gridspec.GridSpec(nrows + 2, total_grid_cols,
                                figure=fig,
                                top=text_y_pos - 0.02,
                                height_ratios=[header_ratio] + [1] * nrows + [footer_ratio],
                                hspace=0.4, wspace=main_wspace)

            # Add column titles at the top
            for col_idx, (start_col, end_col) in enumerate(col_positions):
                title_ax = fig.add_subplot(gs[0, start_col:end_col])
                title_ax.axis('off')
                title_ax.text(0.5, 0.5, column_titles[col_idx], ha='center', va='center', fontsize=14, weight='bold')

            # Add time labels at the bottom  
            for col_idx, (start_col, end_col) in enumerate(col_positions):
                time_ax = fig.add_subplot(gs[-1, start_col:end_col])
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
                    axes_to_label = _plot_data(input_ax, inp[row], ndim, only_input_channel_names)
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
                    axes_to_label = _plot_data(pred_ax, pred[row], ndim, output_channel_names)
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
                    axes_to_label = _plot_data(target_ax, tgt[row], ndim, output_channel_names)
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
    # ------------------------------------------------------------------
    # Final cleanup: remove any duplicate plots that have both a `_best.png`
    # and the same name *without* the suffix.  This ensures only best plots
    # remain on disk when `wandb` logging is disabled. This is needed as inside _maybe_log_save_evaluate,
    # we perform one last evalutaion run even when the training ends.
    # ------------------------------------------------------------------

    # # Always remove non-best duplicates when a _best plot exists.
    # for fname in os.listdir(save_dir):
    #     if fname.endswith("_best.png"):
    #         base_name = fname.replace("_best.png", ".png")
    #         dup_path = os.path.join(save_dir, base_name)
    #         if os.path.isfile(dup_path):
    #             try:
    #                 os.remove(dup_path)
    #             except OSError as e:
    #                 print(f"[plot_progress] Warning: could not delete duplicate plot '{dup_path}': {e}")

    return returned_figs