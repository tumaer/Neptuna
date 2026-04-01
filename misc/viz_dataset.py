"""
HDF5 Dataset Visualization Tool.

This module provides a comprehensive visualization tool for HDF5 datasets commonly
used in computational fluid dynamics (CFD) and other scientific applications. It
can generate individual frame images and MP4 videos from time-series data stored
in HDF5 format, with support for both 2D and 3D datasets.

Key Features:
- Visualize time-series data from HDF5 files
- Support for 2D images and 3D volumes (with slice, max projection, and isometric views)
- Generate individual PNG frames and MP4 videos
- Flexible group selection (random, specific, or all groups)
- Customizable colormaps and frame rates
- Command-line interface for easy usage
- Optional dataset normalization using statistics loaded from YAML data-configs

Example Usage:
    # Visualize a random group with default settings
    python viz_dataset.py --h5-path data/train.h5
    
    # Visualize specific groups with custom settings
    python viz_dataset.py --h5-path data/train.h5 --groups "group_0" "group_1" \
        --fps 15 --cmap plasma --keep-frames --title <Figure_title> --out <output_dir> 

    python3 ./misc/viz_dataset.py --h5-path ./data/fluids/Laser_Droplet/2D/train.h5 --random --fps 20 --title "My Data" --out "./misc/ld"
    
    # Create video for all groups with frame range
    python viz_dataset.py --h5-path data/train.h5 --all --frame-range 50 100 \
        --step 5 --title "CFD Simulation - Group {g}, t={t}"
    
    # 3D volume visualization with isometric view
    python viz_dataset.py --h5-path data/3d_data.h5 --volume-proj iso \
        --groups "volume_data"

    # Normalized visualization using stats from a data-config YAML
    python viz_dataset.py --h5-path data/train.h5 \
        --data-config config/data_config/fluids/Laser_Droplet/2d_laser_droplet_data_default.yaml \
        --normalize

    # Normalized visualization with strategy override
    python viz_dataset.py --h5-path data/train.h5 \
        --data-config config/data_config/fluids/Laser_Droplet/2d_laser_droplet_data_default.yaml \
        --normalize --norm-strategy z_normalization

    # Plot only selected datasets (channels)
    python viz_dataset.py --h5-path data/train.h5 --channels "Velocity_X" "Velocity_Y" "Density" "Pressure"

    # Plot only specific channel indices (e.g. 0 and 2) for multi-channel datasets
    python viz_dataset.py --h5-path data/train.h5 --channel-indices 0 2

Dependencies:
    - h5py: HDF5 file handling
    - matplotlib: Plotting and visualization
    - numpy: Numerical operations
    - imageio: Video creation
    - scikit-image (optional): For marching cubes algorithm in 3D visualization
    - PyYAML: Loading normalization statistics from YAML configuration

Install dependencies:
    pip install h5py matplotlib numpy imageio imageio-ffmpeg scikit-image PyYAML
"""

import argparse
import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # for isometric rendering
from matplotlib.ticker import ScalarFormatter
import yaml

try:
    from skimage.measure import marching_cubes
except ImportError:  # skimage might be optional
    marching_cubes = None


def parse_args():
    """
    Parse command-line arguments for the visualization tool.
    
    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing all configuration options
        for the visualization process.
    
    Notes
    -----
    The parser supports the following arguments with examples:
    
    **Input/Output Options:**
    --h5-path : str
        Path to the HDF5 file to visualize.
        Example: --h5-path data/fluids/train.h5
    
    --out : str, default='trajectory_frames'
        Output directory for generated PNG frames and MP4 videos.
        Example: --out ./my_visualizations
    
    **Visualization Options:**
    --cmap : str, default='viridis'
        Colormap for scalar fields (velocity fields automatically use 'coolwarm').
        Example: --cmap plasma, --cmap inferno, --cmap jet
    
    --title : str, optional
        Template string for figure titles with format placeholders.
        Available placeholders: {t} (timestep), {g} (group), {proj} (projection), {slice} (slice index)
        Example: --title "CFD Simulation - Group {g}, t={t}"
        Example: --title "Pressure Field at t={t} - {proj} view"
    
    **Channel / dataset selection:**
    --channels : list of str, optional
        Only plot these dataset names (channel names). If omitted, all datasets
        in each group are plotted.
        Example: --channels velocity pressure
    
    --channel-indices : list of int, optional
        Only plot these channel indices (0-based) for multi-channel datasets.
        If omitted, all channels are plotted. Ignored for single-channel datasets.
        Example: --channel-indices 0 2
    
    **Group Selection (mutually exclusive):**
    --groups : list of str
        Specify one or more group names to process.
        Example: --groups "group_0" "group_1" "group_5"
    
    --all : flag
        Process all groups in the HDF5 file.
        Example: --all (no additional arguments needed)
    
    --random : flag
        Explicitly select a random group (default behavior).
        Example: --random (no additional arguments needed)
    
    **Frame Control:**
    --frame-range : int int
        Inclusive range of timestep indices to process.
        Example: --frame-range 10 50 (process timesteps 10 through 50)
        Example: --frame-range 0 99 (process first 100 timesteps)
    
    --step : int, default=1
        Save every N-th frame (must be ≥ 1).
        Example: --step 5 (save frames 0, 5, 10, 15, ...)
        Example: --step 10 (save frames 0, 10, 20, 30, ...)
    
    **Video Options:**
    --fps : float, default=10.0
        Frames per second for output MP4 video.
        Example: --fps 24.0 (cinematic frame rate)
        Example: --fps 5.0 (slower playback)
    
    --keep-frames : flag
        Keep individual PNG frames after MP4 creation.
        Example: --keep-frames (no additional arguments needed)
    
    **3D Volume Projection:**
    --volume-proj : str, default='slice', choices=['slice', 'max', 'iso']
        Method for projecting 3D volumes to 2D.
        Example: --volume-proj slice (show single slice)
        Example: --volume-proj max (maximum intensity projection)
        Example: --volume-proj iso (isometric 3D rendering)
    
    --slice-index : int, optional
        Index of slice to show for 'slice' projection (defaults to middle slice).
        Example: --slice-index 15 (show 15th slice)
        Example: --slice-index 0 (show first slice)
    
    **Reproducibility:**
    --seed : int, optional
        Random seed for deterministic group selection.
        Example: --seed 42 (always select same "random" group)
        Example: --seed 12345
    
    **Data Normalization Options:**
    --data-config : str, optional
        Path to data-config YAML under config/data_config/<case_name> containing
        'data_normalization_stats' (and optional 'data_normalization_strategy').
        Example: --data-config config/data_config/fluids/Laser_Droplet/2d_laser_droplet_data_default.yaml
    
    --normalize : flag
        Apply normalization using statistics from the provided data-config.
        Example: --normalize (requires --data-config)
    
    --norm-strategy : str, optional
        Override normalization strategy. If omitted, the strategy from YAML is
        used (defaulting to 'z_normalization' if not present in YAML).
        Choices: z_normalization, min_max_normalization, robust_normalization
    
    **Complete Example Commands:**
    
    # Basic usage - visualize random group
    python viz_dataset.py --h5-path data/train.h5
    
    # Advanced usage - specific groups with custom settings
    python viz_dataset.py --h5-path data/cfd_sim.h5 \
        --groups "simulation_1" "simulation_2" \
        --frame-range 50 150 --step 5 --fps 15 \
        --cmap plasma --title "CFD Results - {g} at t={t}" \
        --out ./results --keep-frames
    
    # 3D volume visualization
    python viz_dataset.py --h5-path data/3d_volume.h5 \
        --volume-proj iso --groups "volume_data" \
        --fps 8 --title "3D Volume - {proj} view"
    
    # Process all groups with reproducible random seed
    python viz_dataset.py --h5-path data/train.h5 --all \
        --seed 42 --step 10 --fps 20
    
    # Normalized visualization using YAML-provided statistics
    python viz_dataset.py --h5-path data/train.h5 \
        --data-config config/data_config/fluids/Aerobreakup/2d_aerobreakup_data_default.yaml \
        --normalize --fps 12 --cmap viridis
    """
    parser = argparse.ArgumentParser(
        description="Visualise all timesteps of a randomly chosen group inside a train.h5 file."
    )
    parser.add_argument(
        "--h5-path",
        dest="h5_path",
        required=True,
        help="Path to the .h5 file (required).",
    )
    parser.add_argument(
        "--out",
        default="trajectory_frames",
        help="Directory where the PNGs will be written",
    )
    parser.add_argument(
        "--cmap",
        default="coolwarm",
        help="Colormap to use for all fields (default: 'coolwarm')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic group selection",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Save every STEP-th frame (e.g. 5 means frames 0,5,10,…). Must be ≥1.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames-per-second for the output MP4 video.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title template for the figure. Available placeholders: {t} (timestep), {g} (group), {proj} (volume projection mode), {slice} (slice index, if applicable). Example: 'Group {g} – t={t} – {proj}{slice}'.",
    )
    parser.add_argument(
        "--frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Optional inclusive range of timestep indices to process (e.g. --frame-range 10 200).",
    )
    # Mutually exclusive options for choosing which group(s) to process.  If none
    # are given we fall back to selecting a single random group.
    group_sel = parser.add_mutually_exclusive_group()
    group_sel.add_argument(
        "--groups",
        "--group",
        nargs="+",
        metavar="GROUP",
        dest="groups",
        help="Name(s) of one or more groups in the HDF5 file to process (--group is an alias).",
    )
    group_sel.add_argument(
        "--all",
        action="store_true",
        help="Process *all* groups found in the file. An MP4 video will be created and frame PNGs are deleted afterwards.",
    )
    group_sel.add_argument(
        "--random",
        action="store_true",
        help="Explicitly request a random group to be processed (default behaviour if neither --groups nor --all is supplied).",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="If set, keep the individual PNG frame images instead of deleting them after MP4 creation.",
    )
    # 3-D visualisation options
    parser.add_argument(
        "--volume-proj",
        choices=["slice", "max", "iso"],
        default="slice",
        help="Method to project 3-D volumes onto 2-D for plotting. 'slice' shows a single slice (see --slice-index); 'max' shows a maximum-intensity projection along the first volume axis; 'iso' shows an isometric view of the volume.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Index of the slice to show when --volume-proj=slice. Defaults to the middle slice if not specified.",
    )
    # Normalization options
    parser.add_argument(
        "--data-config",
        dest="data_config",
        default=None,
        help="Path to the YAML under config/data_config/<case_name> to read normalization stats.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="If set, normalize datasets using statistics from the provided data-config before plotting.",
    )
    parser.add_argument(
        "--norm-strategy",
        dest="norm_strategy",
        choices=["z_normalization", "min_max_normalization", "robust_normalization"],
        default=None,
        help="Override normalization strategy. If omitted, uses strategy from the YAML.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        metavar="NAME",
        default=None,
        help="Only plot these dataset names (e.g. velocity, pressure). If omitted, plot all datasets.",
    )
    parser.add_argument(
        "--channel-indices",
        dest="channel_indices",
        nargs="+",
        type=int,
        metavar="IDX",
        default=None,
        help="Only plot these channel indices (0-based) for multi-channel datasets. If omitted, plot all channels.",
    )
    return parser.parse_args()


def choose_random_group(h5file: h5py.File, rng: random.Random) -> str:
    """
    Randomly select a group from the HDF5 file.
    
    Parameters
    ----------
    h5file : h5py.File
        Opened HDF5 file object.
    rng : random.Random
        Random number generator for reproducible selection.
        
    Returns
    -------
    str
        Name of the randomly selected group.
        
    Raises
    ------
    RuntimeError
        If the HDF5 file contains no groups.
    """
    groups = list(h5file.keys())
    if not groups:
        raise RuntimeError("HDF5 file does not contain any groups!")
    group_name = rng.choice(groups)
    print(f"Chosen group: {group_name}")
    return group_name


def load_group_datasets(group: h5py.Group):
    """
    Load all datasets from an HDF5 group into memory.
    
    Parameters
    ----------
    group : h5py.Group
        HDF5 group containing the datasets to load.
        
    Returns
    -------
    tuple[dict, int]
        A tuple containing:
        - dict: Dictionary mapping dataset names to numpy arrays
        - int: Number of timesteps (common to all datasets)
        
    Raises
    ------
    ValueError
        If datasets have inconsistent time dimensions or insufficient dimensions.
        
    Notes
    -----
    All datasets in the group must share the same leading time dimension.
    Each dataset must have at least 2 dimensions (time + spatial/feature dims).
    """
    data = {}
    time_dim = None
    for dset_name, dset in group.items():
        arr = dset[...]
        if arr.ndim < 2:
            raise ValueError(
                f"Dataset {dset_name} must have at least 2 dimensions (time + data): {arr.shape}"
            )
        if time_dim is None:
            time_dim = arr.shape[0]
        elif arr.shape[0] != time_dim:
            raise ValueError(
                f"Dataset {dset_name} has inconsistent time dimension {arr.shape[0]} (expected {time_dim})"
            )
        data[dset_name] = arr
    return data, time_dim


def _apply_normalization(arr: np.ndarray, stats: dict, strategy: str) -> np.ndarray:
    """
    Apply normalization to an array using channel-agnostic scalar stats.

    The same scalar stats are applied to every spatial/temporal element. If the
    array contains an explicit channel dimension (e.g., shape (T,C,H,W)), the
    same stats are applied to all channels.
    """
    eps = 1e-8
    if strategy == "z_normalization":
        mean = float(stats["mean"]) if "mean" in stats else 0.0
        std = float(stats.get("std", 1.0))
        std = std if abs(std) > eps else eps
        return (arr - mean) / std
    if strategy == "min_max_normalization":
        vmin = float(stats.get("min", 0.0))
        vmax = float(stats.get("max", 1.0))
        denom = (vmax - vmin) if (vmax - vmin) > eps else eps
        return (arr - vmin) / denom
    if strategy == "robust_normalization":
        median = float(stats["median"]) if "median" in stats else 0.0
        iqr = float(stats.get("iqr", 1.0))
        iqr = iqr if abs(iqr) > eps else eps
        return (arr - median) / iqr
    # Unknown strategy -> return unmodified
    return arr


def _lookup_stats(stats_map: dict, dset_name: str, channel_index: int | None) -> dict | None:
    """Find stats for dataset name, optionally trying a <name>_<idx> key for channels."""
    if dset_name in stats_map:
        return stats_map[dset_name]
    if channel_index is not None:
        key = f"{dset_name}_{channel_index}"
        return stats_map.get(key)
    return None


def normalize_datasets(datasets: dict, stats_map: dict, strategy: str) -> dict:
    """
    Normalize all datasets using the provided statistics map and strategy.

    - If stats are missing for a dataset, it is left unmodified (with a warning).
    - Supports arrays with shapes like (T,H,W), (T,C,H,W), (T,H,W,D), (T,C,H,W,D), etc.
    """
    normalized = {}
    for name, arr in datasets.items():
        # Determine channel dimension based on heuristics in plot_timestep
        if arr.ndim >= 5:
            # (T, C, ...)
            out = arr.copy()
            C = arr.shape[1]
            for c in range(C):
                stats = _lookup_stats(stats_map, name, c)
                if stats is None:
                    stats = _lookup_stats(stats_map, name, None)
                if stats is None:
                    print(f"[normalize] Missing stats for '{name}' (ch {c}); leaving unmodified.")
                    continue
                out[:, c] = _apply_normalization(out[:, c], stats, strategy)
        elif arr.ndim == 4 and arr.shape[1] <= 3:
            # (T, C, H, W) treated as channels
            out = arr.copy()
            C = arr.shape[1]
            for c in range(C):
                stats = _lookup_stats(stats_map, name, c)
                if stats is None:
                    stats = _lookup_stats(stats_map, name, None)
                if stats is None:
                    print(f"[normalize] Missing stats for '{name}' (ch {c}); leaving unmodified.")
                    continue
                out[:, c] = _apply_normalization(out[:, c], stats, strategy)
        else:
            # No explicit channel dim; apply one set of stats
            stats = _lookup_stats(stats_map, name, None)
            if stats is None:
                print(f"[normalize] Missing stats for '{name}'; leaving unmodified.")
                out = arr
            else:
                out = _apply_normalization(arr, stats, strategy)
        normalized[name] = out
    return normalized


def plot_timestep(
    datasets: dict,
    t: int,
    output_path: Path,
    cmap_scalar: str,
    title_tpl: str | None = None,
    group_name: str | None = None,
    volume_proj: str = "slice",
    slice_index: int | None = None,
    channel_indices: list[int] | None = None,
):
    """
    Create and save a visualization for a single timestep.
    
    This function generates a multi-panel figure showing all datasets and channels
    for a specific timestep. It handles both 2D and 3D data with various projection
    methods for 3D volumes.
    
    Parameters
    ----------
    datasets : dict
        Dictionary mapping dataset names to numpy arrays containing the data.
    t : int
        Timestep index to visualize.
    output_path : Path
        Path where the generated PNG image will be saved.
    cmap_scalar : str
        Colormap name for scalar fields (velocity fields use 'coolwarm').
    title_tpl : str, optional
        Template string for figure title with format placeholders.
    group_name : str, optional
        Name of the HDF5 group being visualized.
    volume_proj : str, default='slice'
        Method for 3D volume projection: 'slice', 'max', or 'iso'.
    slice_index : int, optional
        Index of slice to show for 'slice' projection (defaults to middle).
    channel_indices : list of int, optional
        If provided, only plot these channel indices (0-based) for multi-channel
        datasets. If None, plot all channels.
        
    Notes
    -----
    The function automatically determines the number of channels in each dataset
    using heuristics based on array dimensions:
    - ndim >= 5: assumes (T,C,...) layout
    - ndim == 4: treats second dim as channels if <= 3, otherwise as depth
    - ndim < 4: assumes single channel
    
    For 3D volumes, three projection methods are available:
    - 'slice': Shows a single 2D slice
    - 'max': Maximum intensity projection along first axis
    - 'iso': Isometric 3D rendering using marching cubes (requires scikit-image)
    
    Examples
    --------
    >>> datasets = {'velocity': vel_data, 'pressure': press_data}
    >>> plot_timestep(datasets, 42, Path('frame_042.png'), 'viridis',
    ...               title_tpl='Simulation t={t}', group_name='group_0')
    """
    # Determine how many subplots are needed, accounting for possible absence of channel dim
    def n_channels(arr: np.ndarray) -> int:
        """Return the number of visualisable channels for *arr*.

        Heuristic:
        • ndim ≥ 5 → assume layout (T,C,…) – return C.
        • ndim == 4:
            – If second dim ≤ 3 treat it as channel count (typical for vector
              fields such as velocity with 2 or 3 components).
            – Otherwise treat it as a depth dimension of a 3-D volume (no
              channel) and return 1.
        • everything else → 1 channel.
        """
        if arr.ndim >= 5:
            return arr.shape[1]
        if arr.ndim == 4:
            return arr.shape[1] if arr.shape[1] <= 3 else 1
        return 1

    def channels_to_plot(n_ch: int) -> list[int]:
        """Return channel indices to plot for a dataset with n_ch channels."""
        if channel_indices is None:
            return list(range(n_ch))
        return [c for c in channel_indices if 0 <= c < n_ch]

    n_plots = sum(
        len(channels_to_plot(n_channels(arr))) for arr in datasets.values()
    )
    ncols = int(np.ceil(np.sqrt(n_plots)))  # square-ish layout
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False
    )
    axes_iter = iter(axes.flat)
    plot_counter = 0  # keep track of subplot index for potential 3-D axes

    for dset_name, arr in datasets.items():
        n_ch = n_channels(arr)
        ch_list = channels_to_plot(n_ch)
        for ch in ch_list:
            ax = next(axes_iter)
            plot_counter += 1
            # Extract the frame/volume for plotting
            if arr.ndim >= 5:
                img = arr[t, ch]
            elif arr.ndim == 4:
                if arr.shape[1] <= 3:
                    img = arr[t, ch]
                else:
                    img = arr[t]
            elif arr.ndim >= 3:
                # (T, C?, H, W) or (T, H, W)
                if n_ch > 1:
                    img = arr[t, ch]
                else:
                    img = arr[t]
            else:
                img = arr[t]

            # Choose plotting method based on dimensionality
            if volume_proj == "iso" and img.ndim == 3:
                # Produce 3-D isometric plot of the volume
                # Convert current 2-D axis to 3-D
                ax.remove()
                ax = fig.add_subplot(nrows, ncols, plot_counter, projection="3d")
                # Use marching cubes if available, else fallback to voxel plot
                try:
                    cmap_obj = plt.get_cmap(cmap_scalar)
                    vmin, vmax = float(img.min()), float(img.max())

                    if marching_cubes is not None:
                        verts, faces, normals, verts_intensity = marching_cubes(
                            img, level=float(np.mean(img))
                        )
                        mesh = Poly3DCollection(verts[faces], alpha=0.7)
                        if verts_intensity is None or len(verts_intensity) == 0:
                            # derive vertex intensities by sampling original volume
                            idx = np.clip(
                                np.round(verts).astype(int), 0, np.array(img.shape) - 1
                            )
                            verts_intensity = img[idx[:, 0], idx[:, 1], idx[:, 2]]

                        face_vals = verts_intensity[faces].mean(axis=1)
                        colors = cmap_obj((face_vals - vmin) / (vmax - vmin + 1e-8))
                        mesh.set_facecolors(colors)
                        ax.add_collection3d(mesh)
                        ax.set_xlim(0, img.shape[0])
                        ax.set_ylim(0, img.shape[1])
                        ax.set_zlim(0, img.shape[2])

                        # Draw bounding box
                        dx, dy, dz = img.shape
                        corners = [
                            (0, 0, 0),
                            (dx, 0, 0),
                            (dx, dy, 0),
                            (0, dy, 0),
                            (0, 0, dz),
                            (dx, 0, dz),
                            (dx, dy, dz),
                            (0, dy, dz),
                        ]
                        edge_idx = [
                            (0, 1),
                            (1, 2),
                            (2, 3),
                            (3, 0),
                            (4, 5),
                            (5, 6),
                            (6, 7),
                            (7, 4),
                            (0, 4),
                            (1, 5),
                            (2, 6),
                            (3, 7),
                        ]
                        for i, j in edge_idx:
                            xs = [corners[i][0], corners[j][0]]
                            ys = [corners[i][1], corners[j][1]]
                            zs = [corners[i][2], corners[j][2]]
                            ax.plot(xs, ys, zs, color="black", linewidth=0.5)
                    else:
                        # Fallback rough voxel rendering
                        filled = img > np.mean(img)
                        img_norm = (img - vmin) / (vmax - vmin + 1e-8)
                        voxel_colors = cmap_obj(img_norm)
                        ax.voxels(filled, facecolors=voxel_colors, edgecolor="k")
                    ax.set_axis_off()

                    # Add colorbar for iso view
                    from matplotlib.cm import ScalarMappable

                    sm = ScalarMappable(
                        cmap=cmap_obj, norm=plt.Normalize(vmin=vmin, vmax=vmax)
                    )
                    sm.set_array([])
                    cb = fig.colorbar(sm, ax=ax, shrink=0.7)
                    cb.formatter = ScalarFormatter(useOffset=False, useMathText=True)
                    cb.update_ticks()
                except Exception as e:
                    ax.text(0.5, 0.5, 0.5, f"ISO err: {e}", ha="center")
                im_data = None  # nothing for 2-D processing further

            elif img.ndim == 3:  # 3-D volume -> project to 2-D
                if volume_proj == "max":
                    img2d = img.max(axis=0)
                else:  # 'slice'
                    idx = slice_index if slice_index is not None else img.shape[0] // 2
                    idx = max(0, min(idx, img.shape[0] - 1))
                    img2d = img[idx]
                im_data = img2d
            else:
                im_data = img

            if im_data is not None and im_data.ndim == 2:
                # 2D field -> image
                im = ax.imshow(im_data, cmap=cmap_scalar, origin="lower")

                # ------------------------------------------------------------------
                # Show resolution ticks (axes) – x/y in pixel coordinates
                # ------------------------------------------------------------------
                height, width = im_data.shape

                # Choose up to three tick positions (start, middle, end)
                def _ticks(n):
                    if n <= 1:
                        return [0]
                    if n == 2:
                        return [0, 1]
                    return [0, n // 2, n - 1]

                ax.set_xticks(_ticks(width))
                ax.set_yticks(_ticks(height))
                ax.set_xlim(0, width - 1)
                ax.set_ylim(0, height - 1)
                #ax.set_xlabel("X")
                #ax.set_ylabel("Y")

                cb = fig.colorbar(im, ax=ax, shrink=0.7)
                cb.formatter = ScalarFormatter(useOffset=False, useMathText=True)
                cb.update_ticks()
            elif im_data is not None and im_data.ndim == 1:
                ax.plot(im_data)
            else:
                if im_data is not None:
                    raise ValueError(
                        f"Unsupported data dimensionality for plotting: {im_data.shape}"
                    )

            # Title per subplot
            if n_ch > 1:
                ax.set_title(f"{dset_name} [ch {ch}] t={t}")
            else:
                ax.set_title(f"{dset_name} t={t}")

            # Keep axes visible for 2-D images (ticks added above).  For other
            # plots, we leave default axis behaviour.

    # Hide any remaining unused axes
    for ax in axes_iter:
        ax.axis("off")

    # Build formatting dictionary for titles
    title_vars = {
        "t": t,
        "proj": volume_proj,
        "slice": slice_index if slice_index is not None else "",
    }
    if group_name is not None:
        title_vars["g"] = group_name

    if title_tpl:
        fig.suptitle(title_tpl.format(**title_vars), fontsize=16)
    else:
        # Default title showing group & timestep only
        default_title = f"{group_name or ''}"
        fig.suptitle(default_title.strip(), fontsize=16)

    # Projection details placed just below the main title
    proj_descr = (
        f"{volume_proj}" if volume_proj != "slice" else f"slice {title_vars['slice']}"
    )
    fig.text(0.5, 0.90, proj_descr, ha="center", va="top", fontsize=10, style="italic")

    # Make room for the suptitle and projection description and add horizontal margins
    fig.subplots_adjust(left=0.06, right=0.94, top=0.86, bottom=0.06)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    """
    Main entry point for the visualization tool.
    
    This function orchestrates the entire visualization process:
    1. Parse command-line arguments
    2. Load and validate the HDF5 file
    3. Select groups to process based on user preferences
    4. Generate visualization frames for each selected group
    5. Create MP4 videos from the generated frames
    6. Clean up intermediate files if requested
    
    The function handles multiple groups, frame ranges, and various
    visualization options as specified through command-line arguments.
    
    Raises
    ------
    FileNotFoundError
        If the specified HDF5 file does not exist.
    ValueError
        If invalid arguments are provided (e.g., invalid frame range, step size).
    RuntimeError
        If the HDF5 file contains no groups.
    """
    args = parse_args()
    rng = random.Random(args.seed)

    h5_path = Path(args.h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"Cannot find file: {h5_path}")

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load normalization configuration if requested
    norm_stats = None
    norm_strategy = None
    if args.normalize:
        if args.data_config is None:
            raise ValueError("--normalize requires --data-config pointing to a YAML with data_normalization_stats")
        cfg_path = Path(args.data_config)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Cannot find data-config YAML: {cfg_path}")
        with open(cfg_path, "r") as fcfg:
            cfg_body = yaml.safe_load(fcfg) or {}
        norm_stats = cfg_body.get("data_normalization_stats")
        if not isinstance(norm_stats, dict) or not norm_stats:
            raise ValueError("No data_normalization_stats found in the provided data-config YAML")
        norm_strategy = args.norm_strategy or cfg_body.get("data_normalization_strategy", "z_normalization")

    with h5py.File(h5_path, "r") as f:
        # Decide which groups to process based on CLI flags
        if args.all:
            group_names = list(f.keys())
            if not group_names:
                raise RuntimeError("HDF5 file does not contain any groups!")
        elif args.groups:
            # Validate the requested group names exist in the file
            missing = [g for g in args.groups if g not in f]
            if missing:
                available = list(f.keys())
                raise ValueError(
                    "Requested group(s) not found in file: "
                    f"{', '.join(missing)}. "
                    f"\n Available groups:\n  " + "\n  ".join(available)
                )
            group_names = args.groups
        else:  # --random or default behaviour
            group_names = [choose_random_group(f, rng)]

        if args.step < 1:
            raise ValueError("--step must be at least 1")

        for group_name in group_names:
            datasets, n_timesteps = load_group_datasets(f[group_name])
            if args.channels is not None:
                missing = [c for c in args.channels if c not in datasets]
                if missing:
                    raise ValueError(
                        f"Requested channel(s) not found in group '{group_name}': {', '.join(missing)}. "
                        f"\n Available channels:\n  " + "\n  ".join(datasets.keys())
                    )
                datasets = {k: v for k, v in datasets.items() if k in args.channels}
            if args.normalize and norm_stats is not None:
                datasets_proc = normalize_datasets(datasets, norm_stats, norm_strategy)
                title_tpl = args.title or "{g} (norm: {proj})"
            else:
                datasets_proc = datasets
                title_tpl = args.title
            print(
                f"Processing group '{group_name}' containing {len(datasets)} channels, names: {list(datasets.keys())} and "
                f"{n_timesteps} timesteps. Saving every {args.step}-th frame."
            )

            # Determine valid timestep range
            if args.frame_range is not None:
                t_start, t_end = args.frame_range
                if t_start < 0 or t_end < 0 or t_start > t_end:
                    raise ValueError(
                        "Invalid --frame-range: START and END must be non-negative and START ≤ END"
                    )
                # Clip range to available timesteps
                t_start = max(t_start, 0)
                t_end = min(t_end, n_timesteps - 1)
            else:
                t_start, t_end = 0, n_timesteps - 1

            frame_paths = []
            for t in range(t_start, t_end + 1, args.step):
                out_file = output_dir / f"{group_name}_t{t:04d}.png"
                plot_timestep(
                    datasets_proc,
                    t,
                    out_file,
                    cmap_scalar=args.cmap,
                    title_tpl=title_tpl,
                    group_name=group_name,
                    volume_proj=args.volume_proj,
                    slice_index=args.slice_index,
                    channel_indices=args.channel_indices,
                )
                frame_paths.append(out_file)

            # Always create an MP4 video for the processed frames
            mp4_path = output_dir / f"{group_name}.mp4"
            print(f"Creating MP4 {mp4_path} (fps={args.fps}) …")
            try:
                with imageio.get_writer(
                    mp4_path, fps=args.fps, codec="libx264", format="ffmpeg", mode="I"
                ) as writer:
                    for frame_path in frame_paths:
                        frame = imageio.imread(frame_path)
                        frame_arr = np.asarray(frame)
                        # Ensure 3-channel RGB for all frames (grayscale→RGB, drop alpha)
                        if frame_arr.ndim == 2:  # grayscale
                            frame_arr = np.stack([frame_arr] * 3, axis=-1)
                        elif frame_arr.shape[2] == 4:  # RGBA
                            frame_arr = frame_arr[..., :3]
                        writer.append_data(frame_arr)
                print(f"MP4 saved to {mp4_path}")
            except Exception as e:
                print(f"Failed to create MP4: {e}")

            # Optionally delete intermediate PNG frames unless user asked to keep them
            if not args.keep_frames:
                for frame_path in frame_paths:
                    try:
                        frame_path.unlink()
                    except FileNotFoundError:
                        pass
                print(
                    f"Deleted {len(frame_paths)} frame image(s) for group '{group_name}'."
                )
            else:
                print(
                    f"Kept {len(frame_paths)} frame image(s) for group '{group_name}' as requested."
                )


if __name__ == "__main__":
    main()
