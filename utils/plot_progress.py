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
from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wandb
import io  # For in-memory PNG buffers
from PIL import Image  # To create a PIL image object for wandb
from utils.compute_stats import re_normalize_data
from typing import Tuple, List, Optional
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

# ------------------------------------------------------------------
# Rollout plotter
# ------------------------------------------------------------------

@dataclass
class LayoutConfig:
    """Immutable configuration for layout calculations."""
    # Visual dimensions
    base_visual_size: float = 3.5
    
    # Fixed margins (in inches)
    margin_between_plots_h: float = 0.65
    margin_between_plots_v: float = 0.65
    
    # Colorbar space (in inches, added to plot height/width)
    colorbar_thickness: float = 0.25
    colorbar_gap: float = 0.25
    
    # Non-content sections (in inches)
    dims_height: float = 2.0
    header_height: float = 2.0
    spacer_height: float = 1.0
    footer_height: float = 5.0
    
    # Figure margins (in inches)
    left_margin: float = 0.5
    right_margin: float = 0.5
    top_margin: float = 0.5
    bottom_margin: float = 0.5
    
    # Title column for horizontal layout
    title_column_width: float = 1.5

    # Timestamp row/column dimensions (in inches)
    timestamp_row_height: float = 0.2  # For horizontal layout
    timestamp_column_width: float = 0.3  # For vertical layout

@dataclass
class ColorbarConfig:
    """Configuration for colorbar appearance."""
    labelsize: int = 10
    pad_fraction: float = 0.16  # Fraction of axis for colorbar placement

@dataclass
class Slice3DConfig:
    """Configuration for 3D data slicing."""
    slice_axis: int = 0  # Which spatial axis to slice (0, 1, or 2)
    num_slices: int = 3  # Number of slices to show
    
    def get_slice_positions(self, axis_length: int) -> List[int]:
        """Get evenly spaced slice positions along the axis."""
        if self.num_slices == 1:
            return [axis_length // 2]
        return [int(i * (axis_length - 1) / (self.num_slices - 1)) 
                for i in range(self.num_slices)]


class LayoutCalculator(ABC):
    """Abstract base for layout calculations with fixed margins."""
    
    def __init__(self, config: LayoutConfig, cbar_config: ColorbarConfig,
                 slice_config: Optional[Slice3DConfig] = None):
        self.config = config
        self.cbar_config = cbar_config
        self.slice_config = slice_config or Slice3DConfig()
    
    @abstractmethod
    def calculate(self, spatial_shape: tuple, num_sections: int, 
                  num_time_steps: int, ndim: int) -> dict:
        """Calculate layout parameters.
        
        Returns
        -------
        dict with keys: fig_width, fig_height, visual_width, visual_height,
                       grid_cell_width, grid_cell_height, cbar_config
        """
        pass
    
    def _get_visual_dimensions(self, spatial_shape: tuple, ndim: int) -> Tuple[float, float]:
        """Calculate visual dimensions of plot content based on data aspect ratio.
        
        For 3D data, accounts for the complete grid of slices.
        
        Returns
        -------
        visual_width, visual_height : float, float
            Dimensions in inches (total visual area needed)
        """
        if ndim == 1:
            return self.config.base_visual_size * 1.5, self.config.base_visual_size * 0.8
        
        elif ndim == 2:
            if len(spatial_shape) >= 2:
                H, W = spatial_shape[:2]
                aspect_ratio = H / W - (1 - H/W)*0.3 # Slight adjustment for aesthetics
                
                visual_width = self.config.base_visual_size
                visual_height = self.config.base_visual_size * aspect_ratio
            else:
                visual_width = self.config.base_visual_size
                visual_height = self.config.base_visual_size
        
        elif ndim == 3:
            if len(spatial_shape) >= 3:
                slice_config = self.slice_config
                axis = slice_config.slice_axis
                
                # Get remaining dimensions after slicing
                dims = list(range(3))
                dims.remove(axis)
                H, W = spatial_shape[dims[0]], spatial_shape[dims[1]]
                
                # Single slice dimensions
                aspect_ratio = H / W
                single_slice_width = self.config.base_visual_size * 0.8
                single_slice_height = single_slice_width * aspect_ratio
                
                num_slices = slice_config.num_slices
                
                # Check if we're in a vertical or horizontal layout
                # This is determined by which calculator is being used
                is_vertical = isinstance(self, VerticalLayoutCalculator)
                
                if is_vertical:
                    # Vertical layout: slices stack vertically
                    # Width: single slice width (channels side-by-side handled by renderer)
                    # Height: all slices stacked + internal spacing
                    hspace_inches = single_slice_height * 0.25
                    visual_width = single_slice_width
                    visual_height = (num_slices * single_slice_height + 
                                (num_slices - 1) * hspace_inches)
                else:
                    # Horizontal layout: slices side-by-side
                    # Width: all slices in a row + internal spacing
                    # Height: single row height
                    wspace_inches = single_slice_width * 0.15
                    visual_width = (num_slices * single_slice_width + 
                                (num_slices - 1) * wspace_inches)
                    visual_height = single_slice_height
                
            else:
                visual_width = self.config.base_visual_size * self.slice_config.num_slices
                visual_height = self.config.base_visual_size
        
        else:
            raise ValueError(f"Unsupported dimensionality: {ndim}")
        
        return visual_width, visual_height


class VerticalLayoutCalculator(LayoutCalculator):
    """Layout calculator for vertical orientation (time as rows)."""
    
    def calculate(self, spatial_shape: tuple, num_columns: int, 
                  num_rows: int, ndim: int, is_timestamp: List[bool] = None) -> dict:
        
        # Get visual dimensions from data shape
        visual_width, visual_height = self._get_visual_dimensions(spatial_shape, ndim)
        
        # Colorbar is horizontal (below each plot)
        colorbar_space = self.config.colorbar_thickness + self.config.colorbar_gap
        
        # Grid cell dimensions = visual + colorbar
        grid_cell_width = visual_width
        grid_cell_height = visual_height + colorbar_space
        
        # Timestamp column width
        timestamp_width = self.config.timestamp_column_width
        
        # Calculate total figure dimensions
        # Width: margins + columns (mixing regular and timestamp widths) + spacing
        if is_timestamp is not None:
            total_width = sum(timestamp_width if is_ts else grid_cell_width 
                            for is_ts in is_timestamp)
            total_width += (num_columns - 1) * self.config.margin_between_plots_h
        else:
            total_width = num_columns * grid_cell_width + (num_columns - 1) * self.config.margin_between_plots_h
        
        fig_width = (self.config.left_margin + 
                    self.config.right_margin +
                    total_width)
        
        # Height: margins + headers + rows + spacing between rows + footer
        fig_height = (self.config.top_margin +
                     self.config.dims_height + 
                     self.config.header_height +
                     num_rows * grid_cell_height + 
                     (num_rows - 1) * self.config.margin_between_plots_v +
                     self.config.spacer_height + 
                     self.config.footer_height + 
                     self.config.bottom_margin)
        
        return {
            'fig_width': fig_width,
            'fig_height': fig_height,
            'visual_width': visual_width,
            'visual_height': visual_height,
            'grid_cell_width': grid_cell_width,
            'grid_cell_height': grid_cell_height,
            'timestamp_width': timestamp_width,
            'num_rows': num_rows,
            'num_columns': num_columns,
            'cbar_config': self.cbar_config,
        }


class HorizontalLayoutCalculator(LayoutCalculator):
    """Layout calculator for horizontal orientation (time as columns)."""
    
    def calculate(self, spatial_shape: tuple, num_rows: int, 
                  num_columns: int, ndim: int, is_timestamp: List[bool] = None) -> dict:
        
        # Get visual dimensions from data shape
        visual_width, visual_height = self._get_visual_dimensions(spatial_shape, ndim)
        
        # Colorbar is horizontal (below each plot)
        colorbar_space = self.config.colorbar_thickness + self.config.colorbar_gap
        
        # Grid cell dimensions = visual + colorbar
        grid_cell_width = visual_width
        grid_cell_height = visual_height + colorbar_space
        
        # Timestamp row height
        timestamp_height = self.config.timestamp_row_height
        
        # Calculate total figure dimensions
        # Width: margins + title column + columns + spacing
        fig_width = (self.config.left_margin + 
                    self.config.title_column_width +
                    self.config.margin_between_plots_h +
                    num_columns * grid_cell_width + 
                    (num_columns - 1) * self.config.margin_between_plots_h +
                    self.config.right_margin)
        
        # Height: margins + headers + rows (mixing regular and timestamp heights) + spacing + footer
        if is_timestamp is not None:
            total_height = sum(timestamp_height if is_ts else grid_cell_height 
                             for is_ts in is_timestamp)
            total_height += (num_rows - 1) * self.config.margin_between_plots_v
        else:
            total_height = num_rows * grid_cell_height + (num_rows - 1) * self.config.margin_between_plots_v
        
        fig_height = (self.config.top_margin +
                     self.config.dims_height + 
                     self.config.header_height +
                     total_height +
                     self.config.spacer_height + 
                     self.config.footer_height + 
                     self.config.bottom_margin)
        
        return {
            'fig_width': fig_width,
            'fig_height': fig_height,
            'visual_width': visual_width,
            'visual_height': visual_height,
            'grid_cell_width': grid_cell_width,
            'grid_cell_height': grid_cell_height,
            'timestamp_height': timestamp_height,
            'title_column_width': self.config.title_column_width,
            'num_rows': num_rows,
            'num_columns': num_columns,
            'cbar_config': self.cbar_config,
        }


class DataRenderer(ABC):
    """Abstract base for rendering data to axes."""
    
    @abstractmethod
    def render(self, ax: plt.Axes, data: np.ndarray, channel_names: List[str],
               vmin: Optional[np.ndarray] = None, vmax: Optional[np.ndarray] = None,
               **kwargs) -> Optional[List[plt.Axes]]:
        """Render data to axes."""
        pass


class Renderer1D(DataRenderer):
    """Renderer for 1D data."""
    
    def render(self, ax: plt.Axes, data: np.ndarray, channel_names: List[str],
               vmin: Optional[np.ndarray] = None, vmax: Optional[np.ndarray] = None,
               **kwargs) -> None:
        C = data.shape[0]
        x = np.arange(data.shape[-1])
        for c in range(C):
            ax.plot(x, data[c], label=channel_names[c])
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xticks(x[::len(x)//5])
        ax.tick_params(axis='both', labelsize=8)


class Renderer2D(DataRenderer):
    """Renderer for 2D data with automatic aspect ratio handling."""
    
    def render(self, ax: plt.Axes, data: np.ndarray, channel_names: List[str],
               vmin: Optional[np.ndarray] = None, vmax: Optional[np.ndarray] = None,
               cbar_config: Optional[ColorbarConfig] = None) -> Optional[List[plt.Axes]]:
        
        C = data.shape[0]
        h, w = data[0].shape
        
        # Use 'auto' aspect to let axes determine aspect from data
        aspect = 'auto'
        
        use_vlims = vmin is not None and vmax is not None
        
        def _safe_vlims(c):
            if not use_vlims:
                return {}
            vmin_val = float(vmin[c])
            vmax_val = float(vmax[c])
            if vmin_val == vmax_val:
                eps = 1e-6 if vmin_val == 0.0 else abs(vmin_val) * 1e-6
                vmin_val -= eps
                vmax_val += eps
            return {"vmin": vmin_val, "vmax": vmax_val}
        
        cbar_labelsize = cbar_config.labelsize if cbar_config else 10
        cbar_pad = cbar_config.pad_fraction if cbar_config else 0.02
        
        if C == 1:
            im = ax.imshow(data[0], cmap="coolwarm", aspect=aspect, 
                          origin='lower', **_safe_vlims(0))
            ax.set_title(channel_names[0], fontsize=8)
            ax.tick_params(axis='both', labelsize=7, labelbottom=False, labelleft=False, 
               bottom=False, left=False)
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=cbar_pad,
                               orientation='horizontal', location='bottom')
            self._format_colorbar(cbar, cbar_labelsize)
            return None
        else:
            fig = ax.figure
            gs = ax.get_subplotspec().subgridspec(1, C, wspace=0.2)
            ax.remove()
            sub_axes = []
            
            for c in range(C):
                sub_ax = fig.add_subplot(gs[0, c])
                im = sub_ax.imshow(data[c], cmap="coolwarm", aspect=aspect,
                                  origin='lower', **_safe_vlims(c))
                sub_ax.set_title(channel_names[c], fontsize=8)
                sub_ax.tick_params(axis='both', labelsize=7, labelbottom=False, labelleft=False,
                   bottom=False, left=False)
                
                cbar = fig.colorbar(im, ax=sub_ax, fraction=0.046, pad=cbar_pad,
                                   orientation='horizontal', location='bottom')
                self._format_colorbar(cbar, cbar_labelsize)
                sub_axes.append(sub_ax)
            
            return sub_axes
    
    def _format_colorbar(self, cbar, labelsize):
        """Apply consistent colorbar formatting."""
        cbar.ax.tick_params(labelsize=labelsize - 2)
        cbar.formatter.set_scientific(True)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.formatter.set_useMathText(True)
        cbar.ax.xaxis.offsetText.set_fontsize(labelsize - 2)
        cbar.update_ticks()


class Renderer3D(DataRenderer):
    """Renderer for 3D data using 2D slices."""
    
    def __init__(self, slice_config: Optional[Slice3DConfig] = None, 
                 orientation: str = 'vertical'):
        self.slice_config = slice_config or Slice3DConfig()
        self.orientation = orientation  # 'vertical' or 'horizontal'
    
    def render(self, ax: plt.Axes, data: np.ndarray, channel_names: List[str],
               vmin: Optional[np.ndarray] = None, vmax: Optional[np.ndarray] = None,
               cbar_config: Optional[ColorbarConfig] = None) -> Optional[List[plt.Axes]]:
        
        C = data.shape[0]
        spatial_shape = data[0].shape  # (D, H, W) or similar
        
        # Get slice positions
        slice_positions = self.slice_config.get_slice_positions(
            spatial_shape[self.slice_config.slice_axis]
        )
        
        use_vlims = vmin is not None and vmax is not None
        
        def _safe_vlims(c):
            if not use_vlims:
                return {}
            vmin_val = float(vmin[c])
            vmax_val = float(vmax[c])
            if vmin_val == vmax_val:
                eps = 1e-6 if vmin_val == 0.0 else abs(vmin_val) * 1e-6
                vmin_val -= eps
                vmax_val += eps
            return {"vmin": vmin_val, "vmax": vmax_val}
        
        cbar_labelsize = cbar_config.labelsize if cbar_config else 10
        cbar_pad = cbar_config.pad_fraction if cbar_config else 0.02
        
        # Create subgrid based on orientation
        fig = ax.figure
        
        if self.orientation == 'vertical':
            # Vertical layout: slices stack vertically
            # Grid: num_slices rows × C columns
            num_slices = self.slice_config.num_slices
            if C == 1:
                gs = ax.get_subplotspec().subgridspec(
                    num_slices, 1,
                    wspace=0, hspace=0.25
                )
            else:
                gs = ax.get_subplotspec().subgridspec(
                    num_slices, C,
                    wspace=0.15, hspace=0.25
                )
        else:
            # Horizontal layout: slices side-by-side
            # Grid: C rows × num_slices columns
            num_slices = self.slice_config.num_slices
            if C == 1:
                gs = ax.get_subplotspec().subgridspec(
                    1, num_slices,
                    wspace=0.15, hspace=0
                )
            else:
                gs = ax.get_subplotspec().subgridspec(
                    C, num_slices,
                    wspace=0.15, hspace=0.25
                )
        
        ax.remove()
        sub_axes = []
        
        if self.orientation == 'vertical':
            # Iterate: slices (rows) then channels (columns)
            for s_idx, slice_pos in enumerate(slice_positions):
                for c in range(C):
                    # Extract 2D slice
                    slice_data = self._extract_slice(data[c], slice_pos)
                    
                    if C == 1:
                        sub_ax = fig.add_subplot(gs[s_idx, 0])
                    else:
                        sub_ax = fig.add_subplot(gs[s_idx, c])
                    
                    im = sub_ax.imshow(slice_data, cmap="coolwarm", aspect='auto',
                                      origin='lower', **_safe_vlims(c))
                    
                    # Title: show channel and slice position
                    if C == 1:
                        title = f"{channel_names[c]}\nSlice {slice_pos}"
                    else:
                        # For multiple channels, show channel on first slice only
                        if s_idx == 0:
                            title = f"{channel_names[c]}\nSlice {slice_pos}"
                        else:
                            title = f"Slice {slice_pos}"
                    
                    sub_ax.set_title(title, fontsize=7)
                    sub_ax.tick_params(axis='both', labelsize=6, labelbottom=False, labelleft=False,
                   bottom=False, left=False)
                    
                    # Add colorbar
                    cbar = fig.colorbar(im, ax=sub_ax, fraction=0.046, pad=cbar_pad,
                                       orientation='horizontal', location='bottom')
                    self._format_colorbar(cbar, cbar_labelsize)
                    sub_axes.append(sub_ax)
        else:
            # Horizontal orientation: iterate channels (rows) then slices (columns)
            for c in range(C):
                for s_idx, slice_pos in enumerate(slice_positions):
                    # Extract 2D slice
                    slice_data = self._extract_slice(data[c], slice_pos)
                    
                    if C == 1:
                        sub_ax = fig.add_subplot(gs[0, s_idx])
                    else:
                        sub_ax = fig.add_subplot(gs[c, s_idx])
                    
                    im = sub_ax.imshow(slice_data, cmap="coolwarm", aspect='auto',
                                      origin='lower', **_safe_vlims(c))
                    
                    # Title: show channel name and slice position
                    if C == 1:
                        title = f"{channel_names[c]}\nSlice {slice_pos}"
                    else:
                        # For multiple channels, show channel on first slice only
                        if s_idx == 0:
                            title = f"{channel_names[c]}\nSlice {slice_pos}"
                        else:
                            title = f"Slice {slice_pos}"
                    
                    sub_ax.set_title(title, fontsize=7)
                    sub_ax.tick_params(axis='both', labelsize=6, labelbottom=False, labelleft=False,
                   bottom=False, left=False)
                    
                    # Add colorbar
                    cbar = fig.colorbar(im, ax=sub_ax, fraction=0.046, pad=cbar_pad,
                                       orientation='horizontal', location='bottom')
                    self._format_colorbar(cbar, cbar_labelsize)
                    sub_axes.append(sub_ax)
        
        return sub_axes
    
    def _extract_slice(self, data_3d: np.ndarray, slice_pos: int) -> np.ndarray:
        """Extract 2D slice from 3D data along configured axis."""
        axis = self.slice_config.slice_axis
        if axis == 0:
            return data_3d[slice_pos, :, :]
        elif axis == 1:
            return data_3d[:, slice_pos, :]
        elif axis == 2:
            return data_3d[:, :, slice_pos]
        else:
            raise ValueError(f"Invalid slice axis: {axis}")
    
    def _format_colorbar(self, cbar, labelsize):
        """Apply consistent colorbar formatting."""
        cbar.ax.tick_params(labelsize=labelsize - 2)
        cbar.formatter.set_scientific(True)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.formatter.set_useMathText(True)
        cbar.ax.xaxis.offsetText.set_fontsize(labelsize - 2)
        cbar.update_ticks()


class GridStructureBuilder:
    """Builds grid structure for different layouts."""
    
    def __init__(self, input_channels: List[str], output_channels: List[str],
                 conditioning_channels: Optional[List[str]] = None,
                 include_relative_error: bool = True):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conditioning_channels = conditioning_channels
        self.include_relative_error = include_relative_error
    
    def build_vertical(self) -> Tuple[List[int], List[str], List[bool]]:
        """Build structure for vertical layout (columns).
        
        Returns
        -------
        widths : List[int]
            Width multipliers for each column
        titles : List[str]
            Title for each column
        is_timestamp : List[bool]
            Whether each column is a timestamp column
        """
        has_conditioning = self.conditioning_channels is not None
        
        widths = []
        titles = []
        is_timestamp = []
        
        # Input section
        widths.append(len(self.input_channels))
        titles.append("Input")
        is_timestamp.append(False)
        
        # Conditioning section
        if has_conditioning:
            widths.append(len(self.conditioning_channels))
            titles.append("Conditioning")
            is_timestamp.append(False)
        
        # Output sections with timestamps after targets
        cols_per_field = 4 if self.include_relative_error else 3
        error_labels = (["Prediction", "Target", "Abs Error", "Rel Error"] 
                       if self.include_relative_error 
                       else ["Prediction", "Target", "Abs Error"])
        
        for ch_name in self.output_channels:
            for i, label in enumerate(error_labels):
                widths.append(1)
                titles.append(f"{ch_name}\n{label}")
                is_timestamp.append(False)
                
                # Add timestamp column after Target
                if label == "Target":
                    widths.append(1)
                    titles.append("Time")
                    is_timestamp.append(True)
        
        return widths, titles, is_timestamp
    
    def build_horizontal_2d(self) -> Tuple[List[int], List[str], List[bool]]:
        """Build structure for horizontal layout with 2D data (rows).
        
        Returns
        -------
        heights : List[int]
            Height multipliers for each row
        titles : List[str]
            Title for each row
        is_timestamp : List[bool]
            Whether each row is a timestamp row
        """
        has_conditioning = self.conditioning_channels is not None
        
        heights = []
        titles = []
        is_timestamp = []
        
        # Input rows (one per channel)
        for ch_name in self.input_channels:
            heights.append(1)
            titles.append(f"Input\n{ch_name}")
            is_timestamp.append(False)
        
        # Conditioning rows
        if has_conditioning:
            for ch_name in self.conditioning_channels:
                heights.append(1)
                titles.append(f"Conditioning\n{ch_name}")
                is_timestamp.append(False)
        
        # Output rows with timestamps after targets
        error_labels = (["Prediction", "Target", "Abs Error", "Rel Error"] 
                       if self.include_relative_error 
                       else ["Prediction", "Target", "Abs Error"])
        
        for ch_name in self.output_channels:
            for label in error_labels:
                heights.append(1)
                titles.append(f"{ch_name}\n{label}")
                is_timestamp.append(False)
                
                # Add timestamp row after Target
                if label == "Target":
                    heights.append(1)
                    titles.append("Time")
                    is_timestamp.append(True)
        
        return heights, titles, is_timestamp

class BasePlotter(ABC):
    """Abstract base plotter with template method pattern."""
    
    def __init__(
        self,
        input_array: np.ndarray,
        prediction_array: np.ndarray,
        target_array: np.ndarray,
        input_channel_names: List[str],
        output_channel_names: List[str],
        conditioning_input_array: Optional[np.ndarray] = None,
        conditioning_channel_names: Optional[List[str]] = None,
        ndim: int = 1,
        num_examples: int = 5,
        stride: int = 1,
        save_dir: str = "plots",
        layout_config: Optional[LayoutConfig] = None,
        cbar_config: Optional[ColorbarConfig] = None,
        slice_config: Optional[Slice3DConfig] = None,
        include_relative_error: bool = True,
        log_to_wandb: bool = False,
        best_plot_at_train_end: bool = False,
        example_indices: Optional[list[int]] = None,
        checkpoint_step: Optional[int] = None,
        epoch: Optional[int] = None,
        extra_info: Optional[str] = None,
        model_info: Optional[str] = None,
        data_info: Optional[str] = None,
        train_info: Optional[str] = None,
        scheduler_info: Optional[str] = None,
        **metadata
    ):  
        # Data
        self.input_array = input_array
        self.prediction_array = prediction_array
        self.target_array = target_array
        self.input_channel_names = input_channel_names
        self.output_channel_names = output_channel_names
        self.conditioning_input_array = conditioning_input_array
        self.conditioning_channel_names = conditioning_channel_names
        self.ndim = ndim
        self.num_examples = num_examples
        self.stride = stride
        self.save_dir = save_dir
        self.include_relative_error = include_relative_error

        # Save/log
        self.log_to_wandb = log_to_wandb
        self.best_plot_at_train_end = best_plot_at_train_end
        self.example_indices = example_indices
        
        # Metadata
        self.checkpoint_step = checkpoint_step
        self.epoch = epoch
        self.extra_info = extra_info
        self.model_info = model_info
        self.data_info = data_info
        self.train_info = train_info
        self.scheduler_info = scheduler_info
        self.metadata = metadata
        
        # Initialize components
        self.layout_config = layout_config or LayoutConfig()
        self.cbar_config = cbar_config or ColorbarConfig()
        self.slice_config = slice_config or Slice3DConfig()
        self.renderer = self._create_renderer()
        self.grid_builder = GridStructureBuilder(
            input_channel_names, output_channel_names,
            conditioning_channel_names, include_relative_error
        )
    
    def _create_renderer(self) -> DataRenderer:
        """Factory method for creating appropriate renderer."""
        if self.ndim == 1:
            return Renderer1D()
        elif self.ndim == 2:
            return Renderer2D()
        elif self.ndim == 3:
            orientation = 'vertical' if isinstance(self, VerticalPlotter) else 'horizontal'
            return Renderer3D(self.slice_config, orientation=orientation)
        else:
            raise ValueError(f"Unsupported dimensionality: {self.ndim}")
    
    @abstractmethod
    def _create_layout_calculator(self) -> LayoutCalculator:
        """Factory method for creating layout calculator."""
        pass
    
    @abstractmethod
    def plot(self) -> dict:
        """Main plotting method - template method."""
        pass
    
    def _compute_vminmax(self, inp: np.ndarray, pred: np.ndarray, 
                        tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-channel vmin/vmax for consistent colorbars."""
        if self.ndim not in [2, 3]:
            return None, None
        
        C = len(self.output_channel_names)
        vmins = np.zeros(C)
        vmaxs = np.zeros(C)
        
        for c_idx, ch_name in enumerate(self.output_channel_names):
            stacks = [pred[:, c_idx], tgt[:, c_idx]]
            
            if ch_name in self.input_channel_names:
                in_c = self.input_channel_names.index(ch_name)
                stacks.append(inp[:, in_c])
            
            all_vals = np.concatenate(stacks, axis=0)
            vmin = float(np.nanmin(all_vals))
            vmax = float(np.nanmax(all_vals))
            
            if vmin == vmax:
                eps = 1e-6 if vmin == 0.0 else abs(vmin) * 1e-6
                vmin -= eps
                vmax += eps
            
            vmins[c_idx] = vmin
            vmaxs[c_idx] = vmax
        
        return vmins, vmaxs
    
    def _get_time_label(self, step: int, T_in: int) -> str:
        """Generate time label for a given step."""
        if step < T_in:
            time_offset = self.stride * (T_in - 1 - step)
            return "t = 0" if time_offset == 0 else f"t - {time_offset}"
        else:
            pred_step = step - T_in + 1
            return f"t + {self.stride * pred_step}"

    def _add_title_section(self, fig, gs, idx: int, N: int, spatial_shape: tuple,
                           T_in: int, C: int, T_pred: int) -> None:
        """Add title section to figure."""
        include_ckpt = self.checkpoint_step is not None and self.epoch is not None
        
        if include_ckpt:
            title_str = (f"{self.extra_info or 'Prediction Visualization'}\n"
                        f"Checkpoint Step: {self.checkpoint_step}, Epoch: {self.epoch}\n"
                        f"Example Index: {idx}")
        else:
            title_str = f"{self.extra_info or 'Prediction Visualization'}\nExample Index: {idx}"
        
        fig.suptitle(title_str, fontsize=32, y=0.98, weight='bold')
    
    def _add_dims_section(self, fig, gs, idx: int, N: int, spatial_shape: tuple,
                          T_in: int, C: int, T_pred: int) -> None:
        """Add dimensions information section."""
        dims_text = (
            f"Additional Info: Total number of examples={N}, Spatial_res={spatial_shape}, "
            f"# Input_frames={T_in}, # Input_channels={C}, # Prediction_frames={T_pred}, "
            f"# Prediction_channels={self.prediction_array.shape[2]}"
        )
        
        # Add first line of model info if available
        if self.model_info is not None and "\n" in self.model_info:
            dims_text += "\n" + self.model_info.split("\n")[0]
        
        dims_ax = fig.add_subplot(gs[0, :])
        dims_ax.axis('off')
        dims_ax.text(0.0, 0.98, dims_text, ha='left', va='top', 
                          fontsize=20, wrap=True, family='monospace')
    
    def _add_footer_section(self, fig, gs, footer_row_idx: int) -> None:
        """Add configuration footer section."""
        footer_lines = []
        
        # Add model config (skip first line if it was in dims)
        if self.model_info is not None:
            if "\n" in self.model_info:
                footer_lines.append("MODEL CONFIG:\n" + self.model_info.split("\n", 1)[1] + "\n")
            else:
                footer_lines.append("MODEL CONFIG:\n" + self.model_info + "\n")
        
        # Add data config
        if self.data_info is not None:
            footer_lines.append("DATA CONFIG:\n" + self.data_info + "\n")
        
        # Add train config
        if self.train_info is not None:
            footer_lines.append("TRAIN CONFIG:\n" + self.train_info + "\n")
        
        # Add scheduler config
        if self.scheduler_info is not None:
            footer_lines.append("SCHEDULER CONFIG:\n" + self.scheduler_info + "\n")
        
        if footer_lines:
            footer_text = "\n".join(footer_lines)
            footer_ax = fig.add_subplot(gs[footer_row_idx, :])
            footer_ax.axis('off')
            footer_ax.text(0.0, 0.98, footer_text, ha='left', va='top', 
                          fontsize=20, wrap=True, family='monospace')

    def _save_or_log_figure(self, fig: plt.Figure, idx: int, 
                           returned_figs: dict, suffix: str = "") -> None:
        """Save figure to disk and/or log to wandb (legacy logic)."""
        save_this_fig = (not self.log_to_wandb) or self.best_plot_at_train_end
        
        if save_this_fig:
            best_suffix = "_best" if self.best_plot_at_train_end else ""
            filename = f"ckpt_{self.checkpoint_step}_epoch_{self.epoch}_example_{idx}{best_suffix}{suffix}.png"
            img_path = os.path.join(self.save_dir, filename)
            fig.savefig(img_path, dpi=150, bbox_inches="tight")
        
        if self.log_to_wandb and not self.best_plot_at_train_end:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.1)
            buf.seek(0)
            pil_img = Image.open(buf)
            key = f"plot_progress/example_{idx}{suffix}"
            returned_figs[key] = wandb.Image(pil_img)


class VerticalPlotter(BasePlotter):
    """Plotter for vertical layout with fixed margins."""
    
    def _create_layout_calculator(self) -> LayoutCalculator:
        return VerticalLayoutCalculator(self.layout_config, self.cbar_config, 
                                       self.slice_config)
    
    def plot(self) -> dict:
        """Plot data in vertical orientation."""
        os.makedirs(self.save_dir, exist_ok=True)
        
        N, T_in, C, *spatial_shape = self.input_array.shape
        T_pred = self.prediction_array.shape[1]
        
        # Select examples
        if self.example_indices is None:
            np.random.seed(42)
            num_pick = min(self.num_examples, N)
            example_indices = np.random.choice(N, size=num_pick, replace=False)
        else:
            example_indices = np.array(self.example_indices, dtype=int)
        
        returned_figs = {}
        
        for idx in example_indices:
            inp = self.input_array[idx]
            pred = self.prediction_array[idx]
            tgt = self.target_array[idx]
            
            # Compute errors
            abs_err = np.abs(pred - tgt)
            rel_err = np.abs((pred - tgt) / (np.abs(tgt) + 1e-8))
            
            # Compute value ranges
            vmins, vmaxs = self._compute_vminmax(inp, pred, tgt)
            
            # Build grid structure
            col_widths, col_titles, is_timestamp = self.grid_builder.build_vertical()
            
            # Calculate layout
            nrows = max(T_in, T_pred)
            total_cols = len(col_widths)
            layout_calculator = self._create_layout_calculator()
            layout_params = layout_calculator.calculate(
                spatial_shape, total_cols, nrows, self.ndim, is_timestamp
            )
            
            # Create figure with absolute positioning
            fig = self._create_figure_vertical(
                idx, N, spatial_shape, T_in, C, T_pred,
                col_widths, col_titles, is_timestamp, nrows, layout_params
            )
            
            # Plot data
            self._plot_data_vertical(
                fig, inp, pred, tgt, abs_err, rel_err,
                vmins, vmaxs, T_in, T_pred, layout_params
            )
            
            # Save/log figure
            self._save_or_log_figure(fig, idx, returned_figs, suffix="")
            plt.close(fig)
        
        return returned_figs
    
    def _create_figure_vertical(self, idx, N, spatial_shape, T_in, C, T_pred,
                                col_widths, col_titles, is_timestamp, nrows, layout_params):
        """Create figure with fixed margin grid."""
        fig = plt.figure(figsize=(layout_params['fig_width'], 
                                layout_params['fig_height']))
        
        # Calculate height ratios in absolute inches
        dims_h = self.layout_config.dims_height
        header_h = self.layout_config.header_height
        grid_h = layout_params['grid_cell_height']
        margin_v = self.layout_config.margin_between_plots_v
        spacer_h = self.layout_config.spacer_height
        footer_h = self.layout_config.footer_height
        
        # Convert to ratios (GridSpec needs ratios)
        heights_inches = [dims_h, header_h]
        for i in range(nrows):
            heights_inches.append(grid_h)
            if i < nrows - 1:
                heights_inches.append(margin_v)
        heights_inches.extend([spacer_h, footer_h])
        
        # Width ratios (accounting for timestamp columns)
        grid_w = layout_params['grid_cell_width']
        timestamp_w = layout_params['timestamp_width']
        margin_h = self.layout_config.margin_between_plots_h
        
        # Build widths
        widths_inches = []
        col_gs_indices = []
        current_idx = 0
        
        for i, (width, is_ts) in enumerate(zip(col_widths, is_timestamp)):
            col_gs_indices.append(current_idx)
            if is_ts:
                widths_inches.append(timestamp_w)
            else:
                widths_inches.append(grid_w * width)
            current_idx += 1
            
            if i < len(col_widths) - 1:
                widths_inches.append(margin_h)
                current_idx += 1
        
        gs = gridspec.GridSpec(
            len(heights_inches), len(widths_inches),
            figure=fig,
            height_ratios=heights_inches,
            width_ratios=widths_inches,
            hspace=0, wspace=0
        )
        
        # Add title section
        self._add_title_section(fig, gs, idx, N, spatial_shape, T_in, C, T_pred)
        
        # Add dimension info
        self._add_dims_section(fig, gs, idx, N, spatial_shape, T_in, C, T_pred)
        
        # Add column titles
        for col_idx, (gs_idx, title) in enumerate(zip(col_gs_indices, col_titles)):
            title_ax = fig.add_subplot(gs[1, gs_idx])
            title_ax.axis('off')
            title_ax.text(0.5, 0.5, title, 
                        ha='center', va='center', fontsize=14, weight='bold')
        
        # Add footer section
        footer_row_idx = len(heights_inches) - 1
        self._add_footer_section(fig, gs, footer_row_idx)
        
        fig.col_gs_indices = col_gs_indices
        fig.col_widths = col_widths
        fig.is_timestamp = is_timestamp
        fig.gs = gs
        fig.example_idx = idx
        
        return fig
    
    def _plot_data_vertical(self, fig, inp, pred, tgt, abs_err, rel_err,
                        vmins, vmaxs, T_in, T_pred, layout_params):
        """Plot all data rows for vertical layout."""
        gs = fig.gs
        col_gs_indices = fig.col_gs_indices
        col_widths = fig.col_widths
        nrows = max(T_in, T_pred)
        
        for row in range(nrows):
            row_gs_idx = 2 + row * 2
            col_idx = 0
            
            # Plot input
            if row < T_in:
                gs_col = col_gs_indices[col_idx]
                self._plot_cell(fig, gs, row_gs_idx, gs_col, 1,
                            inp[row], self.input_channel_names,
                            vmins, vmaxs, layout_params)
            col_idx += 1
            
            # Plot conditioning
            if self.conditioning_input_array is not None:
                if row < T_in:
                    cond_inp = self.conditioning_input_array[fig.example_idx]
                    gs_col = col_gs_indices[col_idx]
                    self._plot_cell(fig, gs, row_gs_idx, gs_col, 1,
                                cond_inp[row], self.conditioning_channel_names,
                                None, None, layout_params)
                col_idx += 1
            
            # Plot predictions/targets/errors
            if row < T_pred:
                col_idx = self._plot_predictions_vertical(
                    fig, gs, row_gs_idx, col_gs_indices, col_widths, col_idx,
                    pred[row], tgt[row], abs_err[row], rel_err[row],
                    vmins, vmaxs, layout_params, row, T_in
                )
    
    def _plot_cell(self, fig, gs, row_idx, col_idx, span, data, ch_names,
                   vmins, vmaxs, layout_params):
        """Plot a single data cell."""
        ax = fig.add_subplot(gs[row_idx, col_idx:col_idx + span])
        self.renderer.render(
            ax, data, ch_names, vmins, vmaxs,
            cbar_config=layout_params['cbar_config']
        )
    
    def _plot_predictions_vertical(self, fig, gs, row_idx, col_gs_indices, col_widths,
                                start_col_idx, pred, tgt, abs_err, rel_err, 
                                vmins, vmaxs, layout_params, current_time_step, T_in):
        """Plot prediction/target/error columns with timestamps after targets."""
        col_idx = start_col_idx
        is_timestamp = fig.is_timestamp
        
        if self.ndim == 2:
            for c_idx, ch_name in enumerate(self.output_channel_names):
                # Prediction
                gs_col = col_gs_indices[col_idx]
                self._plot_cell(fig, gs, row_idx, gs_col, 1,
                            pred[c_idx:c_idx+1], [ch_name],
                            vmins[c_idx:c_idx+1], vmaxs[c_idx:c_idx+1], layout_params)
                col_idx += 1
                
                # Target
                gs_col = col_gs_indices[col_idx]
                self._plot_cell(fig, gs, row_idx, gs_col, 1,
                            tgt[c_idx:c_idx+1], [ch_name],
                            vmins[c_idx:c_idx+1], vmaxs[c_idx:c_idx+1], layout_params)
                col_idx += 1
                
                # Timestamp after target
                if col_idx < len(is_timestamp) and is_timestamp[col_idx]:
                    gs_col = col_gs_indices[col_idx]
                    time_ax = fig.add_subplot(gs[row_idx, gs_col])
                    time_ax.axis('off')
                    time_ax.text(0.5, 0.5, self._get_time_label(current_time_step, T_in),
                                ha='center', va='center', fontsize=12, weight='bold')
                    col_idx += 1
                
                # Absolute error
                gs_col = col_gs_indices[col_idx]
                self._plot_cell(fig, gs, row_idx, gs_col, 1,
                            abs_err[c_idx:c_idx+1], [ch_name],
                            None, None, layout_params)
                col_idx += 1
                
                # Relative error
                if self.include_relative_error:
                    gs_col = col_gs_indices[col_idx]
                    self._plot_cell(fig, gs, row_idx, gs_col, 1,
                                rel_err[c_idx:c_idx+1], [ch_name],
                                None, None, layout_params)
                    col_idx += 1
        else:  # 1D
            for i, data in enumerate([pred, tgt, abs_err] + ([rel_err] if self.include_relative_error else [])):
                gs_col = col_gs_indices[col_idx]
                self._plot_cell(fig, gs, row_idx, gs_col, 1,
                            data, self.output_channel_names,
                            None, None, layout_params)
                col_idx += 1
                
                # Add timestamp after target (i == 1)
                if i == 1 and col_idx < len(is_timestamp) and is_timestamp[col_idx]:
                    gs_col = col_gs_indices[col_idx]
                    time_ax = fig.add_subplot(gs[row_idx, gs_col])
                    time_ax.axis('off')
                    time_ax.text(0.5, 0.5, self._get_time_label(current_time_step, T_in),
                                ha='center', va='center', fontsize=12, weight='bold')
                    col_idx += 1
        
        return col_idx


class HorizontalPlotter(BasePlotter):
    """Plotter for horizontal layout with fixed margins."""
    
    def _create_layout_calculator(self) -> LayoutCalculator:
        return HorizontalLayoutCalculator(self.layout_config, self.cbar_config,
                                         self.slice_config)
    
    def plot(self) -> dict:
        """Plot data in horizontal orientation."""
        os.makedirs(self.save_dir, exist_ok=True)
        
        N, T_in, C, *spatial_shape = self.input_array.shape
        T_pred = self.prediction_array.shape[1]
        
        # Select examples
        if self.example_indices is None:
            np.random.seed(42)
            num_pick = min(self.num_examples, N)
            example_indices = np.random.choice(N, size=num_pick, replace=False)
        else:
            example_indices = np.array(self.example_indices, dtype=int)
        
        returned_figs = {}
        max_time_steps = max(T_in, T_pred)
        
        for idx in example_indices:
            inp = self.input_array[idx]
            pred = self.prediction_array[idx]
            tgt = self.target_array[idx]
            
            # Compute errors
            abs_err = np.abs(pred - tgt)
            rel_err = np.abs((pred - tgt) / (np.abs(tgt) + 1e-8))
            
            # Compute value ranges
            vmins, vmaxs = self._compute_vminmax(inp, pred, tgt)
            
            # Build grid structure
            if self.ndim == 2:
                row_heights, row_titles, is_timestamp = self.grid_builder.build_horizontal_2d()
            else:
                row_heights, row_titles, is_timestamp = self._build_1d_horizontal_structure()
            
            # Calculate layout
            total_rows = len(row_heights)
            layout_calculator = self._create_layout_calculator()
            layout_params = layout_calculator.calculate(
                spatial_shape, total_rows, max_time_steps, self.ndim, is_timestamp
            )
            
            # Create figure
            fig = self._create_figure_horizontal(
                idx, N, spatial_shape, T_in, C, T_pred,
                row_heights, row_titles, is_timestamp, max_time_steps, layout_params
            )
            
            # Plot data
            self._plot_data_horizontal(
                fig, inp, pred, tgt, abs_err, rel_err,
                vmins, vmaxs, T_in, T_pred, layout_params
            )
            
            # Save/log figure
            self._save_or_log_figure(fig, idx, returned_figs, suffix="")
            plt.close(fig)
        
        return returned_figs
    
    def _build_1d_horizontal_structure(self) -> Tuple[List[int], List[str], List[bool]]:
        """Build structure for 1D data in horizontal layout."""
        has_conditioning = self.conditioning_input_array is not None
        
        heights = []
        titles = []
        is_timestamp = []
        
        # Input row
        heights.append(1)
        titles.append("Input")
        is_timestamp.append(False)
        
        # Conditioning row
        if has_conditioning:
            heights.append(1)
            titles.append("Conditioning")
            is_timestamp.append(False)
        
        # Output rows with timestamps after targets
        error_labels = (["Prediction", "Target", "Abs Error", "Rel Error"] 
                    if self.include_relative_error 
                    else ["Prediction", "Target", "Abs Error"])
        
        for label in error_labels:
            heights.append(1)
            titles.append(label)
            is_timestamp.append(False)
            
            # Add timestamp row after Target
            if label == "Target":
                heights.append(1)
                titles.append("Time")
                is_timestamp.append(True)
        
        return heights, titles, is_timestamp
    
    def _create_figure_horizontal(self, idx, N, spatial_shape, T_in, C, T_pred,
                                row_heights, row_titles, is_timestamp, max_time_steps, layout_params):
        """Create figure with fixed margin grid."""
        fig = plt.figure(figsize=(layout_params['fig_width'], 
                                layout_params['fig_height']))
        
        # Calculate dimensions in inches
        dims_h = self.layout_config.dims_height
        header_h = self.layout_config.header_height
        grid_h = layout_params['grid_cell_height']
        timestamp_h = layout_params['timestamp_height']
        margin_v = self.layout_config.margin_between_plots_v
        spacer_h = self.layout_config.spacer_height
        footer_h = self.layout_config.footer_height
        
        # Height ratios (accounting for timestamp rows)
        heights_inches = [dims_h, header_h]
        for i, (height, is_ts) in enumerate(zip(row_heights, is_timestamp)):
            if is_ts:
                heights_inches.append(timestamp_h)
            else:
                heights_inches.append(grid_h * height)
            if i < len(row_heights) - 1:
                heights_inches.append(margin_v)
        heights_inches.extend([spacer_h, footer_h])
        
        # Width ratios
        title_w = layout_params['title_column_width']
        grid_w = layout_params['grid_cell_width']
        margin_h = self.layout_config.margin_between_plots_h
        
        widths_inches = [title_w, margin_h]
        for i in range(max_time_steps):
            widths_inches.append(grid_w)
            if i < max_time_steps - 1:
                widths_inches.append(margin_h)
        
        gs = gridspec.GridSpec(
            len(heights_inches), len(widths_inches),
            figure=fig,
            height_ratios=heights_inches,
            width_ratios=widths_inches,
            hspace=0, wspace=0
        )
        
        # Add title section
        self._add_title_section(fig, gs, idx, N, spatial_shape, T_in, C, T_pred)
        
        # Add dimension info
        self._add_dims_section(fig, gs, idx, N, spatial_shape, T_in, C, T_pred)
        
        # Add time column headers
        for col in range(max_time_steps):
            col_gs_idx = 2 + col * 2
            header_ax = fig.add_subplot(gs[1, col_gs_idx])
            header_ax.axis('off')
            header_ax.text(0.5, 0.5, self._get_time_label(col, T_in), 
                        ha='center', va='center', fontsize=18, weight='bold')
        
        # Add row titles
        row_gs_indices = []
        gs_idx = 2
        for i, (height, is_ts) in enumerate(zip(row_heights, is_timestamp)):
            row_gs_indices.append(gs_idx)
            if i < len(row_heights) - 1:
                gs_idx += 2  # Include margin
            else:
                gs_idx += 1
        
        for row_idx, (gs_row, is_ts) in enumerate(zip(row_gs_indices, is_timestamp)):
            span = 1  # Each row is independent now
            title_ax = fig.add_subplot(gs[gs_row, 0])
            title_ax.axis('off')
            title_ax.text(0.5, 0.5, row_titles[row_idx], 
                        ha='center', va='center', fontsize=14, weight='bold')
        
        # Add footer section
        footer_row_idx = len(heights_inches) - 1
        self._add_footer_section(fig, gs, footer_row_idx)
        
        fig.row_gs_indices = row_gs_indices
        fig.row_heights = row_heights
        fig.is_timestamp = is_timestamp
        fig.gs = gs
        fig.example_idx = idx
        
        return fig
    
    def _plot_data_horizontal(self, fig, inp, pred, tgt, abs_err, rel_err,
                            vmins, vmaxs, T_in, T_pred, layout_params):
        """Plot all data columns for horizontal layout."""
        gs = fig.gs
        row_gs_indices = fig.row_gs_indices
        row_heights = fig.row_heights
        is_timestamp = fig.is_timestamp
        max_time_steps = max(T_in, T_pred)
        
        for col in range(max_time_steps):
            col_gs_idx = 2 + col * 2
            row_idx = 0
            
            # Plot input
            if col < T_in:
                if self.ndim == 2:
                    for in_c, ch_name in enumerate(self.input_channel_names):
                        gs_row = row_gs_indices[row_idx]
                        
                        if ch_name in self.output_channel_names:
                            out_idx = self.output_channel_names.index(ch_name)
                            input_vmin = vmins[out_idx:out_idx+1]
                            input_vmax = vmaxs[out_idx:out_idx+1]
                        else:
                            input_vmin = None
                            input_vmax = None
                        
                        self._plot_cell(fig, gs, gs_row, col_gs_idx, 1,
                                    inp[col, in_c:in_c+1], [ch_name],
                                    input_vmin, input_vmax, layout_params)
                        row_idx += 1
                else:  # 1D
                    gs_row = row_gs_indices[row_idx]
                    self._plot_cell(fig, gs, gs_row, col_gs_idx, 1,
                                inp[col], self.input_channel_names,
                                vmins, vmaxs, layout_params)
                    row_idx += 1
            else:
                row_idx += len(self.input_channel_names) if self.ndim == 2 else 1
            
            # Plot conditioning
            if self.conditioning_input_array is not None:
                if col < T_in:
                    cond_inp = self.conditioning_input_array[fig.example_idx]
                    if self.ndim == 2:
                        for cond_c, ch_name in enumerate(self.conditioning_channel_names):
                            gs_row = row_gs_indices[row_idx]
                            
                            if ch_name in self.output_channel_names:
                                out_idx = self.output_channel_names.index(ch_name)
                                cond_vmin = vmins[out_idx:out_idx+1]
                                cond_vmax = vmaxs[out_idx:out_idx+1]
                            else:
                                cond_vmin = None
                                cond_vmax = None
                            
                            self._plot_cell(fig, gs, gs_row, col_gs_idx, 1,
                                        cond_inp[col, cond_c:cond_c+1], [ch_name],
                                        cond_vmin, cond_vmax, layout_params)
                            row_idx += 1
                    else:
                        gs_row = row_gs_indices[row_idx]
                        self._plot_cell(fig, gs, gs_row, col_gs_idx, 1,
                                    cond_inp[col], self.conditioning_channel_names,
                                    None, None, layout_params)
                        row_idx += 1
                else:
                    row_idx += len(self.conditioning_channel_names) if self.ndim == 2 else 1
            
            # Plot predictions/targets/errors
            if col < T_pred:
                row_idx = self._plot_predictions_horizontal(
                    fig, gs, row_gs_indices, row_idx, col_gs_idx,
                    pred[col], tgt[col], abs_err[col], rel_err[col],
                    vmins, vmaxs, layout_params, col, T_in
                )
    
    def _plot_predictions_horizontal(self, fig, gs, row_gs_indices, start_row_idx,
                                    col_gs_idx, pred, tgt, abs_err, rel_err,
                                    vmins, vmaxs, layout_params, current_time_step, T_in):
        """Plot prediction/target/error rows with timestamps after targets."""
        row_idx = start_row_idx
        is_timestamp = fig.is_timestamp
        
        if self.ndim == 2:
            for c_idx, ch_name in enumerate(self.output_channel_names):
                # Prediction
                gs_row = row_gs_indices[row_idx]
                self._plot_cell(fig, gs, gs_row, col_gs_idx, 1,
                            pred[c_idx:c_idx+1], [ch_name],
                            vmins[c_idx:c_idx+1], vmaxs[c_idx:c_idx+1], layout_params)
                row_idx += 1
                
                # Target
                gs_row = row_gs_indices[row_idx]
                self._plot_cell(fig, gs, gs_row, col_gs_idx, 1,
                            tgt[c_idx:c_idx+1], [ch_name],
                            vmins[c_idx:c_idx+1], vmaxs[c_idx:c_idx+1], layout_params)
                row_idx += 1
                
                # Timestamp after target
                if row_idx < len(is_timestamp) and is_timestamp[row_idx]:
                    gs_row = row_gs_indices[row_idx]
                    time_ax = fig.add_subplot(gs[gs_row, col_gs_idx])
                    time_ax.axis('off')
                    time_ax.text(0.5, 0.5, self._get_time_label(current_time_step, T_in),
                                ha='center', va='center', fontsize=12, weight='bold')
                    row_idx += 1
                
                # Absolute error
                gs_row = row_gs_indices[row_idx]
                self._plot_cell(fig, gs, gs_row, col_gs_idx, 1,
                            abs_err[c_idx:c_idx+1], [ch_name],
                            None, None, layout_params)
                row_idx += 1
                
                # Relative error
                if self.include_relative_error:
                    gs_row = row_gs_indices[row_idx]
                    self._plot_cell(fig, gs, gs_row, col_gs_idx, 1,
                                rel_err[c_idx:c_idx+1], [ch_name],
                                None, None, layout_params)
                    row_idx += 1
        else:  # 1D
            data_list = [pred, tgt, abs_err]
            if self.include_relative_error:
                data_list.append(rel_err)
            
            for i, data in enumerate(data_list):
                gs_row = row_gs_indices[row_idx]
                self._plot_cell(fig, gs, gs_row, col_gs_idx, 1,
                            data, self.output_channel_names,
                            None, None, layout_params)
                row_idx += 1
                
                # Add timestamp after target (i == 1)
                if i == 1 and row_idx < len(is_timestamp) and is_timestamp[row_idx]:
                    gs_row = row_gs_indices[row_idx]
                    time_ax = fig.add_subplot(gs[gs_row, col_gs_idx])
                    time_ax.axis('off')
                    time_ax.text(0.5, 0.5, self._get_time_label(current_time_step, T_in),
                                ha='center', va='center', fontsize=12, weight='bold')
                    row_idx += 1
        
        return row_idx
    
    def _plot_cell(self, fig, gs, row_idx, col_idx, span, data, ch_names,
                   vmins, vmaxs, layout_params):
        """Plot a single data cell."""
        ax = fig.add_subplot(gs[row_idx, col_idx:col_idx + span])
        self.renderer.render(
            ax, data, ch_names, vmins, vmaxs,
            cbar_config=layout_params['cbar_config']
        )


def create_plotter(orientation: str = 'vertical', **kwargs) -> BasePlotter:
    """Factory function to create appropriate plotter."""
    if orientation == 'vertical':
        return VerticalPlotter(**kwargs)
    elif orientation == 'horizontal':
        return HorizontalPlotter(**kwargs)
    else:
        raise ValueError(f"Unknown orientation: {orientation}")



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
