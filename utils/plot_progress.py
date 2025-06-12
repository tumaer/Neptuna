import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt


def _plot_data(ax, data, ndim):
    C = data.shape[0]  # number of channels

    if ndim == 1:
        x = np.arange(data.shape[-1])
        for c in range(C):
            ax.plot(x, data[c], label=f"ch{c}")
        if C > 1:
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
            im = ax.imshow(data[0], cmap="coolwarm", aspect=aspect)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.1, orientation='horizontal', location='bottom')
            cbar.ax.tick_params(labelsize=4)  # Reduce font size by 50%
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.formatter.set_useMathText(True)  # Use math text for consistent font
            cbar.ax.xaxis.offsetText.set_y(-1.0)  # Move the offset text further down
            cbar.ax.xaxis.offsetText.set_fontsize(5)  # Match the tick label font size
            cbar.update_ticks()
        else:
            ncols = C
            fig = ax.figure
            gs = ax.get_subplotspec().subgridspec(1, ncols, wspace=0.6)  # Add spacing between channels
            ax.remove()
            sub_axes = []
            for c in range(C):
                sub_ax = fig.add_subplot(gs[0, c])
                im = sub_ax.imshow(data[c], cmap="coolwarm", aspect=aspect)
                sub_ax.set_title(f"ch{c}", fontsize=8)
                #sub_ax.set_xticks([])
                #sub_ax.set_yticks([])
                cbar = fig.colorbar(im, ax=sub_ax, fraction=0.046, pad=0.1, orientation='horizontal', location='bottom')
                cbar.ax.tick_params(labelsize=4)  # Reduce font size by 50%
                cbar.formatter.set_scientific(True)
                cbar.formatter.set_powerlimits((0, 0))
                cbar.formatter.set_useMathText(True)  # Use math text for consistent font
                cbar.ax.xaxis.offsetText.set_y(-1.0)  # Move the offset text further down
                cbar.ax.xaxis.offsetText.set_fontsize(5)  # Match the tick label font size
                cbar.update_ticks()
                sub_axes.append(sub_ax)
            return sub_axes

    elif ndim == 3:
        ax.text(0.5, 0.5, "3D multi-channel\nnot supported yet", ha="center", va="center", fontsize=10)
        ax.axis('off')


def plot_examples(input_array, prediction_array, target_array, checkpoint_step, epoch, ndim=1, num_examples=5, stride=1, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    N, T_in, C, *spatial_shape = input_array.shape
    T_pred = prediction_array.shape[1]

    np.random.seed(42)
    example_indices = np.random.choice(N, size=num_examples, replace=False)

    for idx in example_indices:
        inp = input_array[idx]          # [T_in, C, ...]
        pred = prediction_array[idx]    # [T_pred, C, ...]
        tgt = target_array[idx]         # [T_pred, C, ...]

        abs_err = np.abs(pred - tgt)
        rel_err = np.abs((pred - tgt) / (np.abs(tgt) + 1e-8))

        nrows = max(T_in, T_pred)
        ncols = 5

        # Increase figure size and adjust height ratio for better spacing
        fig = plt.figure(figsize=(ncols * 5, (nrows + 2) * 3.5))
        
        # Add main title
        fig.suptitle(f"Checkpoint Step: {checkpoint_step}, Epoch: {epoch}, Example Index: {idx}", fontsize=18, y=0.94, weight='bold')
        
        # Add dimensions info as subtitle
        dims_text = f"Additional Info: Total number of validation examples={N}, Problem dimension={ndim}, Spatial_res={spatial_shape}, # Input_frames={T_in}, # Input_channels={C}, # Prediction_frames={T_pred}, # Prediction_channels={pred.shape[1]}"
        fig.text(0.5, 0.93, dims_text, ha='center', va='center', fontsize=12)

        # Create gridspec with specific height ratios and spacing
        gs = fig.add_gridspec(nrows + 2, ncols, height_ratios=[0.3] + [1] * nrows + [0.5], hspace=0.5, wspace=0.4)
        axes = np.empty((nrows + 2, ncols), dtype=object)
        
        # Create all axes
        for i in range(nrows + 2):
            for j in range(ncols):
                axes[i, j] = fig.add_subplot(gs[i, j])

        # Add column titles at the top
        column_titles = ["Input", "Prediction", "Target", "Abs Error", "Rel Error"]
        for col in range(ncols):
            axes[0, col].axis('off')
            axes[0, col].text(0.5, 0.5, column_titles[col], ha='center', va='center', fontsize=14, weight='bold')

        # Add time labels at the bottom
        for col in range(ncols):
            axes[-1, col].axis('off')
            if col == 0:  # Input column
                time_label = f"t - {stride * (T_in - 1)} to t"
            else:  # Other columns
                time_label = f"t + 1 to t + {stride * T_pred}"
            axes[-1, col].text(0.5, 0.5, time_label, ha='center', va='center', fontsize=12)

        for row in range(nrows):
            row_offset = row + 1

            # Column 0: Input
            if row < T_in:
                time_val = "t" if row == T_in - 1 else f"t - {stride * (T_in - 1 - row)}"
                axes_to_label = _plot_data(axes[row_offset, 0], inp[row], ndim)
                if isinstance(axes_to_label, list):  # Multi-channel 2D case
                    # Add time label only to the middle channel
                    mid_channel = len(axes_to_label) // 2
                    axes_to_label[mid_channel].set_xlabel(time_val, fontsize=10, labelpad=30)
                    axes_to_label[mid_channel].tick_params(labelbottom=True)
                else:  # Single channel or 1D case
                    axes[row_offset, 0].set_xlabel(time_val, fontsize=10, labelpad=30)
                    axes[row_offset, 0].tick_params(labelbottom=True)
            else:
                axes[row_offset, 0].axis('off')

            # Column 1: Prediction
            if row < T_pred:
                time_val = f"t + {stride * (row + 1)}"
                axes_to_label = _plot_data(axes[row_offset, 1], pred[row], ndim)
                if isinstance(axes_to_label, list):  # Multi-channel 2D case
                    # Add time label only to the middle channel
                    mid_channel = len(axes_to_label) // 2
                    axes_to_label[mid_channel].set_xlabel(time_val, fontsize=10, labelpad=30)
                    axes_to_label[mid_channel].tick_params(labelbottom=True)
                else:  # Single channel or 1D case
                    axes[row_offset, 1].set_xlabel(time_val, fontsize=10, labelpad=30)
                    axes[row_offset, 1].tick_params(labelbottom=True)
            else:
                axes[row_offset, 1].axis('off')

            # Column 2: Target
            if row < T_pred:
                time_val = f"t + {stride * (row + 1)}"
                axes_to_label = _plot_data(axes[row_offset, 2], tgt[row], ndim)
                if isinstance(axes_to_label, list):  # Multi-channel 2D case
                    # Add time label only to the middle channel
                    mid_channel = len(axes_to_label) // 2
                    axes_to_label[mid_channel].set_xlabel(time_val, fontsize=10, labelpad=30)
                    axes_to_label[mid_channel].tick_params(labelbottom=True)
                else:  # Single channel or 1D case
                    axes[row_offset, 2].set_xlabel(time_val, fontsize=10, labelpad=30)
                    axes[row_offset, 2].tick_params(labelbottom=True)
            else:
                axes[row_offset, 2].axis('off')

            # Column 3: Abs Error
            if row < T_pred:
                time_val = f"t + {stride * (row + 1)}"
                axes_to_label = _plot_data(axes[row_offset, 3], abs_err[row], ndim)
                if isinstance(axes_to_label, list):  # Multi-channel 2D case
                    # Add time label only to the middle channel
                    mid_channel = len(axes_to_label) // 2
                    axes_to_label[mid_channel].set_xlabel(time_val, fontsize=10, labelpad=30)
                    axes_to_label[mid_channel].tick_params(labelbottom=True)
                else:  # Single channel or 1D case
                    axes[row_offset, 3].set_xlabel(time_val, fontsize=10, labelpad=30)
                    axes[row_offset, 3].tick_params(labelbottom=True)
            else:
                axes[row_offset, 3].axis('off')

            # Column 4: Rel Error
            if row < T_pred:
                time_val = f"t + {stride * (row + 1)}"
                axes_to_label = _plot_data(axes[row_offset, 4], rel_err[row], ndim)
                if isinstance(axes_to_label, list):  # Multi-channel 2D case
                    # Add time label only to the middle channel
                    mid_channel = len(axes_to_label) // 2
                    axes_to_label[mid_channel].set_xlabel(time_val, fontsize=10, labelpad=30)
                    axes_to_label[mid_channel].tick_params(labelbottom=True)
                else:  # Single channel or 1D case
                    axes[row_offset, 4].set_xlabel(time_val, fontsize=10, labelpad=30)
                    axes[row_offset, 4].tick_params(labelbottom=True)
            else:
                axes[row_offset, 4].axis('off')

        for col in range(ncols):
            for row in range(1, nrows + 1):
                axes[row, col].tick_params(labelbottom=True, labelsize=8)

        # Adjust layout to ensure labels are visible
        plt.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95, hspace=0.4, wspace=0.3)
        filename = f"ckpt_{checkpoint_step}_epoch_{epoch}_example_{idx}.png"       
        fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)