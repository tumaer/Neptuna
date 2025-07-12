"""
HDF5 Dataset Statistics Computation Utilities.

This module provides comprehensive statistical analysis capabilities for HDF5 datasets
commonly used in scientific computing and machine learning applications. It computes
essential statistics (mean, standard deviation, min, max, median, IQR) across multiple
files and groups, with support for temporal residual analysis and flexible data filtering.

Key Features:
- Multi-file statistical aggregation.
- Temporal residual statistics for time-series data
- Flexible filtering by groups, frames, and temporal stride
- Support for both on-the-fly and batch statistical computation
- Automatic channel expansion for multi-component fields
- Numerical precision(float64 internally)

Main Functions:
    compute_statistics: Primary API for computing dataset statistics
    
Internal Classes:
    _StatsAggregator: Utility class for accumulating statistics across data chunks

Statistical Measures:
    For each channel, the following statistics are computed:
    - mean: Arithmetic mean across all selected data points
    - std: Standard deviation (combined across groups using RMS)
    - min: Global minimum value
    - max: Global maximum value  
    - median: Median value (only when on_fly_stats=False)
    - iqr: Interquartile range (Q75 - Q25, only when on_fly_stats=False)

Residual Analysis:
    When residual_config is provided, additional statistics are computed for
    temporal residuals (frame-to-frame differences). These are stored with
    the suffix "_residual" and use the same statistical measures as the
    original channels.

Memory Considerations:
    The module offers two computation modes:
    - on_fly_stats=True: Memory-efficient but cannot compute median and IQR
    - on_fly_stats=False: Higher memory usage but complete statistics
"""

import h5py
import numpy as np
from typing import List, Dict, Tuple


__all__ = [
    "compute_statistics",
]


class _StatsAggregator:
    """
    Utility container to accumulate statistics for a single channel across groups/files.
    
    This class provides a memory-efficient way to compute combined statistics
    across multiple data chunks without loading all data into memory simultaneously.
    It maintains separate lists of statistics from each chunk and combines them
    appropriately at the end.
    
    Attributes
    ----------
    means : List[float]
        List of mean values from each processed data chunk.
    stds : List[float]
        List of standard deviations from each processed data chunk.
    mins : List[float]
        List of minimum values from each processed data chunk.
    maxs : List[float]
        List of maximum values from each processed data chunk.
    
    Examples
    --------
    >>> agg = _StatsAggregator()
    >>> agg.add(np.array([1, 2, 3, 4, 5]))
    >>> agg.add(np.array([6, 7, 8, 9, 10]))
    >>> combined_stats = agg.combined()
    >>> print(combined_stats['mean'])  # Combined mean of both arrays
    """

    def __init__(self):
        self.means: List[float] = []
        self.stds: List[float] = []
        self.mins: List[float] = []
        self.maxs: List[float] = []

    def add(self, arr: np.ndarray):
        """
        Accumulate statistics for a data array.
        
        This method computes statistics for the input array in float64 precision
        to avoid numerical precision issues, regardless of the input array's dtype.
        
        Parameters
        ----------
        arr : numpy.ndarray
            Input array of any shape and numeric dtype. Will be flattened
            internally for statistical computation.
            
        Notes
        -----
        All computations are performed in float64 precision.
        """
        # Promote to float64 for numerically stable statistics.
        flat = arr.reshape(-1).astype(np.float64)

        self.means.append(float(np.mean(flat, dtype=np.float64)))
        self.stds.append(float(np.std(flat, dtype=np.float64)))
        # Min/Max are exact so dtype promotion is not strictly required but we
        # keep everything in float64 for consistency.
        self.mins.append(float(np.min(flat)))
        self.maxs.append(float(np.max(flat)))

    def combined(self) -> Dict[str, float]:
        """
        Compute final combined statistics from all accumulated chunks.
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing combined statistics with keys:
            - 'mean': Average of all chunk means
            - 'std': Root-mean-square of all chunk standard deviations
            - 'min': Global minimum across all chunks
            - 'max': Global maximum across all chunks
            
        Notes
        -----
        The combined standard deviation is computed as the RMS of individual
        standard deviations, which provides a reasonable approximation when
        chunk sizes are similar.
        """
        return {
            "mean": float(np.mean(self.means)),
            "std": float(_combined_std(self.stds)),
            "min": float(np.min(self.mins)),
            "max": float(np.max(self.maxs)),
        }


def _combined_std(std_devs: List[float]) -> float:
    """
    Compute combined standard deviation from individual standard deviations.
    
    This function computes the root-mean-square (RMS) of individual standard
    deviations, which provides a reasonable approximation for the combined
    standard deviation when the underlying data chunks have similar sizes.
    
    Parameters
    ----------
    std_devs : List[float]
        List of standard deviations from individual data chunks.
        
    Returns
    -------
    float
        Combined standard deviation computed as RMS of input values.
        
    Notes
    -----
    The RMS formula used is: sqrt(mean(std_i^2)) where std_i are the individual
    standard deviations. This is an approximation that works well when chunk
    sizes are similar but may not be exact for highly variable chunk sizes.
    """
    variances = [s ** 2 for s in std_devs]
    return float(np.sqrt(np.mean(variances, dtype=np.float64)))


def _discover_metadata(first_h5_path: str, filter_groups: List[str] | None = None) -> Tuple[List[str], int]:
    """
    Extract channel names and spatial dimensionality from an HDF5 file.
    
    This function inspects the first available group in an HDF5 file to determine
    the expanded channel names and spatial dimensionality. It handles multi-component
    fields by expanding them into individual channels with appropriate naming.
    
    Parameters
    ----------
    first_h5_path : str
        Path to the HDF5 file to inspect for metadata.
    filter_groups : List[str], optional
        If provided, only groups whose names appear in this list will be
        considered. The first matching group will be used for metadata extraction.
        
    Returns
    -------
    Tuple[List[str], int]
        A tuple containing:
        - List[str]: Expanded channel names (e.g., ['velocity_0', 'velocity_1', 'pressure'])
        - int: Spatial dimensionality (2 for 2D, 3 for 3D, etc.)
        
    Raises
    ------
    ValueError
        If filter_groups is provided but none of the specified groups exist in the file.
        
    Notes
    -----
    Channel expansion rules:
    - Single-component fields (shape[1] == 1): Use field name as-is
    - Multi-component fields (shape[1] > 1): Expand to 'fieldname_0', 'fieldname_1', etc.
    
    The spatial dimensionality is inferred from the number of dimensions beyond
    the first two (time and channel dimensions).
    """
    with h5py.File(first_h5_path, "r") as f:
        # Determine which group to inspect for metadata
        if filter_groups is not None and len(filter_groups) > 0:
            # Pick the first group that both exists in the file and is listed in the filter
            first_group: str | None = None
            for candidate in filter_groups:
                if candidate in f:
                    first_group = candidate
                    break
            if first_group is None:
                raise ValueError(
                    "None of the groups specified in 'filter_groups' were found in the file "
                    f"{first_h5_path}."
                )
        else:
            # Default behaviour: simply take the first group present in the file
            first_group = list(f.keys())[0]

        grp = f[first_group]

        channel_names: List[str] = []
        problem_dimension: int | None = None

        for dset_name in grp:
            dset = grp[dset_name]
            channel_dim = dset.shape[1]

            # expanded channel names
            if channel_dim == 1:
                channel_names.append(dset_name)
            else:
                for ch in range(channel_dim):
                    channel_names.append(f"{dset_name}_{ch}")

            if problem_dimension is None:
                problem_dimension = len(dset.shape[2:])

        assert problem_dimension is not None, "Could not infer problem dimension."

    return channel_names, problem_dimension


def compute_statistics(
    h5_paths: List[str],
    residual_config: Dict[str, bool] | None = None,
    filter_groups: List[str] | None = None,
    filter_frames: List[int] | None = None,
    frame_stride: int = 1,
    on_fly_stats: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], List[str], int]:
    """
    Compute comprehensive statistics for HDF5 datasets with optional residual analysis.
    
    This function processes one or more HDF5 files to compute statistical measures
    (mean, std, min, max, median, IQR) for each channel. It supports temporal
    residual analysis, flexible data filtering, and memory-efficient computation modes.
    
    Parameters
    ----------
    h5_paths : List[str]
        List of paths to HDF5 files to process. All files should have compatible
        structure (same channel names and dimensions).
        
    residual_config : Dict[str, bool], optional
        Configuration for residual statistics computation. When provided, additional
        statistics are computed for temporal residuals (frame-to-frame differences).
        The dictionary should contain boolean flags for residual computation modes.
        If both "add_predicted_value" and "add_base_value" are True, raises ValueError.
        
    filter_groups : List[str], optional
        If provided, only HDF5 groups whose names appear in this list will be
        processed. Groups not in the list are ignored.
        
    filter_frames : List[int], optional
        If provided, must contain exactly two integers [start, end] defining an
        inclusive range of time-frame indices to process. 
        Use None for start/end to indicate first/last frame.
        
    frame_stride : int, default=1
        Step size when iterating over frames. Value of 1 processes every frame,
        2 processes every second frame, etc. Must be positive.
        
    on_fly_stats : bool, default=False
        Computation mode selector:
        - False: Accumulate all data in memory then compute statistics (enables median and IQR)
        - True: Compute statistics incrementally (memory efficient, no median and IQR)
        
    Returns
    -------
    Tuple[Dict[str, Dict[str, float]], List[str], int]
        A tuple containing:
        - Dict[str, Dict[str, float]]: Channel statistics mapping. Each channel maps to
          a dictionary with keys: 'mean', 'std', 'min', 'max', and optionally 'median', 'iqr'.
          When residual_config is provided, additional entries with '_residual' suffix.
        - List[str]: Expanded channel names discovered from the dataset.
        - int: Spatial dimensionality of the data (2D, 3D, etc.).
        
    Raises
    ------
    ValueError
        - If h5_paths is empty
        - If residual_config has conflicting boolean flags
        - If filter_frames is not a 2-element list
        - If frame_stride is not positive
        - If specified filter_groups don't exist in the files
        
    Examples
    --------
    >>> # Basic usage
    >>> stats, channels, dim = compute_statistics(["data.h5"])
    >>> print(f"Dataset has {len(channels)} channels in {dim}D")
    >>> 
    >>> # With residual analysis and filtering
    >>> residual_cfg = {"add_predicted_value": False, "add_base_value": True}
    >>> stats, channels, dim = compute_statistics(
    ...     ["train.h5", "val.h5"],
    ...     residual_config=residual_cfg,
    ...     filter_groups=["simulation_1", "simulation_2"],
    ...     filter_frames=[10, 100],
    ...     frame_stride=2,
    ...     on_fly_stats=False
    ... )
    >>> 
    >>> # Access statistics
    >>> velocity_stats = stats["velocity_0"]
    >>> print(f"Velocity mean: {velocity_stats['mean']:.3f}")
    >>> print(f"Velocity std: {velocity_stats['std']:.3f}")
    >>> 
    >>> # Access residual statistics (if computed)
    >>> if "velocity_0_residual" in stats:
    ...     residual_stats = stats["velocity_0_residual"]
    ...     print(f"Residual mean: {residual_stats['mean']:.3f}")
    
    Notes
    -----
    Statistical Computation:
    - All internal computations use float64 precision for numerical stability
    - Standard deviations are combined using root-mean-square across chunks
    - Median and IQR are only available when on_fly_stats=False
    
    Residual Analysis:
    - Residuals are computed as frame-to-frame differences: x[t] - x[t-1]
    - Residual statistics use the same measures as original channels
    - Residual keys are suffixed with "_residual"
    
    Memory Usage:
    - on_fly_stats=True: Memory usage scales with number of channels
    - on_fly_stats=False: Memory usage scales with total data size
    
    Channel Expansion:
    - Single-component fields: channel name = field name
    - Multi-component fields: channel names = "fieldname_0", "fieldname_1", etc.
    """

    if len(h5_paths) == 0:
        raise ValueError("h5_paths list is empty.")

    # Metadata from first file – honour group filter to guarantee consistency
    channel_names, problem_dim = _discover_metadata(h5_paths[0], filter_groups)

    # Aggregators for raw channels (created now but populated later)
    aggregators: Dict[str, _StatsAggregator] = {name: _StatsAggregator() for name in channel_names}

    # Optional aggregators for residuals, keyed by "<channel>_residual"
    residual_aggregators: Dict[str, _StatsAggregator] | None = None
    if residual_config is not None:
        add_pred = residual_config.get("add_predicted_value", False)
        add_base = residual_config.get("add_base_value", False)

        # Prevent ambiguous configuration where both modes are requested.
        if add_pred and add_base:
            raise ValueError("add_predicted_value and add_base_value cannot both be True.")

        # Regardless of which residual mode will later be used by the data
        # loader, we always compute frame-to-frame residual statistics so that
        # the same normalisation parameters apply in either case.
        residual_aggregators = {
            f"{name}_residual": _StatsAggregator() for name in channel_names
        }

    # When computing statistics after full accumulation, store arrays per channel.
    collected_data: Dict[str, List[np.ndarray]] | None = None
    collected_residual: Dict[str, List[np.ndarray]] | None = None
    if not on_fly_stats:
        collected_data = {name: [] for name in channel_names}
        if residual_aggregators is not None:
            collected_residual = {k: [] for k in residual_aggregators.keys()}

    # Iterate over all files and groups
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            for grp_name in f:
                # Skip groups not in the filter (if provided)
                if filter_groups is not None and grp_name not in filter_groups:
                    continue
                grp = f[grp_name]
                for field_name in grp:
                    field = grp[field_name]
                    ch_dim = field.shape[1]

                    for ch in range(ch_dim): #loop for each component of the field
                        channel_data = field[:, ch]

                        # Apply frame filter if requested
                        if filter_frames is not None:
                            if len(filter_frames) != 2:
                                raise ValueError(
                                    "filter_frames must be a 2-element list [start, end] denoting an inclusive range"
                                )

                            # Validate stride
                            if frame_stride <= 0:
                                raise ValueError("frame_stride must be a positive integer")

                            start_idx, end_idx = filter_frames  # inclusive range

                            # Convert None-like values to defaults (allowing [None, end] etc.)
                            if start_idx is None:
                                start_idx = 0
                            if end_idx is None:
                                end_idx = channel_data.shape[0] - 1

                            # Because NumPy slicing excludes the stop index, add 1 to make inclusive
                            channel_data = channel_data[start_idx : end_idx + 1 : frame_stride]
                        else:
                            if frame_stride != 1:
                                # Apply stride even when full range is included
                                channel_data = channel_data[::frame_stride]

                        key = field_name if ch_dim == 1 else f"{field_name}_{ch}"

                        # Safety: if new channel encountered that wasn't in metadata (unlikely but possible)
                        if key not in aggregators:
                            aggregators[key] = _StatsAggregator()

                        #NOTE: For both cases, residual_config, the statistics are computed by taking 
                        # the differences of the neighbors
                        
                        if on_fly_stats:
                            # Immediate accumulation of channel-wise statistics for that partcular group
                            # Dataset level stats computed at once in the end from the group-wise stats
                            aggregators[key].add(channel_data)

                            # Residuals (on the fly)
                            if residual_aggregators is not None and channel_data.shape[0] > 1:
                                # if residual_config["add_predicted_value"]:
                                #     residual_arr = np.diff(channel_data, axis=0)
                                # else:
                                #     residual_arr = channel_data[1:] - channel_data[0]
                                #     #residual_arr = np.diff(channel_data, axis=0)
                                residual_arr = np.diff(channel_data, axis=0)
                                res_key = f"{key}_residual"
                                residual_aggregators[res_key].add(residual_arr) 
                        else:
                            # Store data for later aggregation
                            collected_data[key].append(channel_data)
                            if residual_aggregators is not None and channel_data.shape[0] > 1:
                                # if residual_config["add_predicted_value"]:
                                #     residual_arr = np.diff(channel_data, axis=0)
                                # else:
                                #     residual_arr = channel_data[1:] - channel_data[0]
                                #     #residual_arr = np.diff(channel_data, axis=0)
                                residual_arr = np.diff(channel_data, axis=0)
                                res_key = f"{key}_residual"
                                collected_residual[res_key].append(residual_arr)

    # --------------------------------------------
    # Final combination / statistics computation
    # --------------------------------------------
    extra_stats: Dict[str, Dict[str, float]] = {}
    extra_residual_stats: Dict[str, Dict[str, float]] = {}
    if not on_fly_stats:
        # Perform a single add per channel with concatenated data
        assert collected_data is not None  # for type checker
        for k, array_list in collected_data.items():
            if len(array_list) == 0:
                continue  # no data for this channel (shouldn't happen)
            concatenated = np.concatenate(array_list, axis=0)
            aggregators[k].add(concatenated)
            # ----------------------------------------------------
            # Additional statistics: median and inter-quartile range 
            # (cannot be computed on the fly)
            # ----------------------------------------------------
            flat = concatenated.reshape(-1).astype(np.float64)
            med = float(np.median(flat))
            q25, q75 = np.percentile(flat, [25, 75])
            extra_stats[k] = {"median": med, "iqr": float(q75 - q25)}

        if residual_aggregators is not None and collected_residual is not None:
            for k, res_list in collected_residual.items():
                if len(res_list) == 0:
                    continue
                concatenated = np.concatenate(res_list, axis=0)
                residual_aggregators[k].add(concatenated)

                # Extra stats for residuals
                flat_r = concatenated.reshape(-1).astype(np.float64)
                med_r = float(np.median(flat_r))
                q25_r, q75_r = np.percentile(flat_r, [25, 75])
                extra_residual_stats[k] = {"median": med_r, "iqr": float(q75_r - q25_r)}

    # Combine stats from aggregators
    channel_stats = {k: agg.combined() for k, agg in aggregators.items()}

    # Attach median/IQR when available (i.e. offline aggregation)
    for k, extra in extra_stats.items():
        channel_stats.setdefault(k, {}).update(extra)

    # Merge residual stats if present
    if residual_aggregators is not None:
        channel_stats.update({k: agg.combined() for k, agg in residual_aggregators.items()})
        # Attach median/IQR when available (i.e. offline aggregation)
        for k, extra in extra_residual_stats.items():
            channel_stats.setdefault(k, {}).update(extra)

    return channel_stats, channel_names, problem_dim


# -----------------------------------------------------------------------------
# Re-normalization helper used during plotting
# -----------------------------------------------------------------------------

def re_normalize_data(arr: np.ndarray, stats: Dict[str, float], norm_strategy: str) -> np.ndarray:
    """Reverts normalization applied during loading for a single channel.
    Mainly used for plotting the normalized data.
    
    Parameters
    ----------
    arr : np.ndarray
        The normalized data to revert (any shape).
    stats : Dict[str, float]
        Dictionary containing either {"mean", "std"} or {"min", "max"} or {"median", "iqr"}. 
    norm_strategy : str
        Either "z_normalization" or "min_max_normalization" or "robust_normalization" or "no_normalization".
    
    Returns
    -------
    np.ndarray
        Renormalized array (new copy).
    """
    if norm_strategy == "z_normalization":
        return arr * stats["std"] + stats["mean"]
    elif norm_strategy == "min_max_normalization":
        return arr * (stats["max"] - stats["min"]) + stats["min"]
    elif norm_strategy == "robust_normalization":
        return arr * stats["iqr"] + stats["median"]
    elif norm_strategy == "no_normalization":
        return arr
    else:
        raise ValueError(f"Unknown normalization strategy: {norm_strategy}")

# -----------------------------------------------------------------------------
# Normalization helper used during data loading
# -----------------------------------------------------------------------------

def normalize_data(arr: np.ndarray, stats: Dict[str, float], strategy: str) -> np.ndarray:
    """Apply per-channel normalization to a NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        The data to normalize (any shape).
    stats : Dict[str, float]
        Dictionary containing either {"mean", "std"} or {"min", "max"} or {"median", "iqr"}.
    strategy : str
        Either "z_normalization" or "min_max_normalization" or "robust_normalization" or "no_normalization".

    Returns
    -------
    np.ndarray
        Normalized array (new copy).
    """
    eps = 1e-12  # small constant to prevent divide-by-zero

    if strategy == "z_normalization":
        return (arr - stats["mean"]) / (stats["std"]  + eps)
    elif strategy == "min_max_normalization":
        return (arr - stats["min"]) / ((stats["max"] - stats["min"]) + eps)
    elif strategy == "robust_normalization":
        return (arr - stats["median"]) / (stats["iqr"] + eps)
    elif strategy == "no_normalization":
        return arr
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")

# -----------------------------------------------------------------------------
# Test script
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    demo_paths = [
        "./KVS_original/train.h5",
        #"./KVS_original/test.h5",
    ]
    residual_config = {
        "add_predicted_value": False,
        "add_base_value": True,
    }
    #residual_config = None
    stats, names, dim = compute_statistics(demo_paths, residual_config)

    print("Channel names:", names)
    print(f"Problem dimension: {dim}D")
    print("\nStatistics:")
    for k, v in stats.items():
        print(f"  {k}: mean={v['mean']:.6e}, std={v['std']:.6e}, min={v['min']:.6e}, max={v['max']:.6e}")

