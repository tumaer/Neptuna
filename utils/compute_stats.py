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
import re  # Added for parameter parsing


__all__ = [
    "compute_statistics",
    "compute_parameter_statistics",
    "compute_statistics_parallel",
]


class _StatsAggregator:
    """
    Utility container to accumulate statistics for a single channel across
    multiple data chunks (groups / files) **without** storing the full data
    or even per-chunk statistics in memory.

    Implementation details
    ----------------------
    The class uses **Welford’s online algorithm** (and the parallel update
    from Chan et al., 1979) to maintain running estimates of mean and
    (population) variance while requiring only:

        n     – total sample count (int)
        mean  – running mean          (float64)
        M2    – aggregated sum of squared deviations (float64)
        min   – global minimum value (float64)
        max   – global maximum value (float64)

    All updates are performed in float64 precision to guarantee numerical
    stability for very large data sets.

    Attributes
    ----------
    n : int
        Total number of samples processed so far.
    mean : float
        Running arithmetic mean.
    M2 : float
        Accumulated sum of squared deviations from the running mean.  The
        population standard deviation is derived from this quantity.
    min : float
        Global minimum across all processed samples.
    max : float
        Global maximum across all processed samples.

    Examples
    --------
    >>> agg = _StatsAggregator()
    >>> agg.add(np.array([1, 2, 3, 4]))
    >>> agg.add(np.array([10, 20]))
    >>> agg.combined()
    {'mean': 6.666666666666667, 'std': 7.4535599249993, 'min': 1.0, 'max': 20.0}
    """

    def __init__(self):
        # running sample count
        self.n: int = 0
        # running mean
        self.mean: float = 0.0
        # aggregated sum of squares of differences from the current mean
        self.M2: float = 0.0
        # running extrema
        self.min: float = np.inf
        self.max: float = -np.inf

    def add(self, arr: np.ndarray):
        """Accumulate statistics for *arr* (any shape, any numeric dtype)."""
        flat = arr.reshape(-1).astype(np.float64)
        cnt = flat.size
        if cnt == 0:
            return  # nothing to do

        # update extrema (exact)
        chunk_min = float(flat.min())
        chunk_max = float(flat.max())
        if self.n == 0:
            # first chunk → initialise directly
            self.min = chunk_min
            self.max = chunk_max
        else:
            if chunk_min < self.min:
                self.min = chunk_min
            if chunk_max > self.max:
                self.max = chunk_max

        # online update of mean and variance (via M2)
        arr_mean = float(np.mean(flat, dtype=np.float64))
        arr_var = float(np.var(flat, dtype=np.float64))  # population var

        delta = arr_mean - self.mean
        total_n = self.n + cnt

        # update mean
        self.mean += delta * cnt / total_n

        # combine the two M2 values (see Chan et al., 1979)
        self.M2 += arr_var * cnt + (delta ** 2) * self.n * cnt / total_n

        self.n = total_n

    def combined(self) -> Dict[str, float]:
        """Return a dictionary with combined statistics."""
        std = float(np.sqrt(self.M2 / max(self.n, 1)))  # population std
        return {
            "mean": float(self.mean),
            "std": std,
            "min": float(self.min),
            "max": float(self.max),
        }

    # ------------------------------------------------------------------
    # Parallel reduction helper
    # ------------------------------------------------------------------
    def add_from_aggregator(self, other: "_StatsAggregator") -> None:
        """Merge another aggregator *other* into *self* (parallel Welford)."""
        if other.n == 0:
            return
        if self.n == 0:
            # copy other's state (avoid code duplication)
            self.n = other.n
            self.mean = other.mean
            self.M2 = other.M2
            self.min = other.min
            self.max = other.max
            return

        delta = other.mean - self.mean
        total_n = self.n + other.n

        # update mean
        self.mean += delta * other.n / total_n

        # update M2 (Chan et al., 1979)
        self.M2 += other.M2 + delta * delta * self.n * other.n / total_n

        # update extrema & count
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)
        self.n = total_n


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


def _parse_numeric_token(token: str) -> float | None:
    """Extract a floating point number from an arbitrary string token.

    The token often contains a parameter name followed by its value, e.g.
    "Re1000" → 1000.0 or "M0.6" → 0.6.  This helper searches the first
    numeric substring (including optional sign and exponent) and converts it
    to *float*. Tokens that end with ``NaN`` are treated as placeholders and
    ignored (returns ``None``). If no numeric substring is found a
    *ValueError* is raised.
    """
    if token.endswith("NaN"):
        return np.nan

    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", token)
    if match:
        return float(match.group(0))
    raise ValueError(f"Cannot parse numeric value from token '{token}'.")


def compute_parameter_statistics(
    h5_paths: List[str],
    delimiter: str = "_",
    eps: float = 1e-12,
) -> Dict[int, Dict[str, float]]:
    """Compute min-max normalised simulation parameters encoded in group names.

    The parameters are encoded directly in the HDF5 group names using a delimiter (default: ``_``).
    For example::

        "Re100_M0.6"          -> [100.0, 0.6]
        "Re200_M0.3"          -> [200.0, 0.3]

    This utility scans *all* supplied files, extracts the numeric values of
    every parameter for every group, aggregates the full data set and finally
    applies min-max normalisation **independently for each parameter**::

        x_norm = (x - x_min) / (x_max - x_min + eps)

    Parameters
    ----------
    h5_paths : List[str]
        One or more HDF5 file paths to inspect.
    delimiter : str, default "_"
        Character used to split group names into parameter tokens.
    eps : float, default 1e-12
        Small constant preventing division by zero when *x_max == x_min*.

    Returns
    -------
    Dict[int, Dict[str, float]]
        ``param_ranges`` – A dictionary mapping the *parameter index* (int) to
        a ``{"min": float, "max": float}`` dictionary describing the range
        computed from all groups.

    Notes
    -----
    * If a token cannot be parsed into a float, a ``ValueError`` is raised.
    * All numeric conversions are executed in *float64* precision.
    * The function does **not** attempt to deduplicate groups across multiple
      files.  If the same group name occurs in two files, its parameter values
      will appear twice in the returned array.
    """
    if len(h5_paths) == 0:
        raise ValueError("h5_paths list is empty.")

    raw_params: List[List[float]] = []  # per-group list of parameter values

    # ------------------------------------------------------------------
    # Collect parameters from every group in every provided file
    # ------------------------------------------------------------------
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            for grp_name in f:
                tokens = grp_name.split(delimiter)
                values: List[float] = []
                for tok in tokens:
                    # Keep only parameter-like tokens that contain both
                    # alphabetic characters and numeric characters, plus NaN tokens.
                    has_letters_and_digits = bool(
                        re.search(r"[A-Za-z]", tok) and re.search(r"\d", tok)
                    )
                    if not (has_letters_and_digits or "NaN" in tok):
                        continue
                    parsed_value = _parse_numeric_token(tok)
                    values.append(parsed_value)
                raw_params.append(values)

    if len(raw_params) == 0:
        raise RuntimeError("No groups found in the supplied HDF5 files.")

    # Ensure parameter dimensionality is consistent
    n_params = len(raw_params[0])
    # if not all(len(p) == n_params for p in raw_params):
    #     raise ValueError(
    #         "Inconsistent number of parameters detected across group names."
    #     )

    raw_arr = np.asarray(raw_params, dtype=np.float64)  # (N, P)

    # ------------------------------------------------------------------
    # Min-max statistics per parameter dimension (ignoring missing entries)
    # ------------------------------------------------------------------
    mins = np.nanmin(raw_arr, axis=0)
    maxs = np.nanmax(raw_arr, axis=0)

    # denom = maxs - mins
    # denom[denom == 0.0] = eps  # avoid divide-by-zero for constant parameters

    # norm_arr = (raw_arr - mins) / denom

    # Assemble ranges into a dictionary for convenience
    param_ranges: Dict[int, Dict[str, float]] = {
        idx: {"min": float(mins[idx]), "max": float(maxs[idx])} for idx in range(n_params)
    }
    param_ranges["max_number_of_parameters"] = raw_arr.shape[-1]

    return param_ranges

# Deprecated function (will be removed in future)
def compute_statistics(
    h5_paths: List[str],
    residual_config: Dict[str, bool] | None = None,
    filter_groups: List[str] | None = None,
    filter_frames: List[int] | None = None,
    frame_stride: int = 1,
    on_fly_stats: bool = False,
    log_transform_channels: List[str] | None = None,
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

    log_channels_set = set(log_transform_channels or [])

    # --------------------------------------------------------------
    # Discover metadata (channel names and spatial dimensionality)
    # --------------------------------------------------------------
    metadata_found = False
    for _meta_path in h5_paths:
        try:
            channel_names, problem_dim = _discover_metadata(_meta_path, filter_groups)
            metadata_found = True
            break  # success – stop searching
        except ValueError:
            # _discover_metadata raises ValueError only when *filter_groups* is
            # provided but none of the requested groups are present in the file.
            # In that scenario we simply continue searching the remaining files.
            continue

    if not metadata_found:
        raise ValueError(
            "None of the provided HDF5 files contained any of the groups "
            f"specified in filter_groups={filter_groups}."
        )

    # Aggregators for raw channels (created now but populated later)
    aggregators: Dict[str, _StatsAggregator] = {name: _StatsAggregator() for name in channel_names}
    # Pre-create log-channel aggregators for channels marked for log transform
    for name in channel_names:
        if name in log_channels_set:
            aggregators[f"log_{name}"] = _StatsAggregator()

    # Optional aggregators for residuals, keyed by "<channel>_residual"
    residual_aggregators: Dict[str, _StatsAggregator] | None = None
    if residual_config is not None:
        if log_channels_set:
            raise ValueError("Residual computation for log-transformed channels is not implemented yet.")
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
        for name in channel_names:
            if name in log_channels_set:
                collected_data[f"log_{name}"] = []
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

                    # ------------------------------------------------------
                    # Perform a single HDF5 read for the entire field and
                    # slice frames/stride only once.  Subsequent per-channel
                    # operations work on in-memory views, massively reducing
                    # I/O overhead.
                    # ------------------------------------------------------
                    if filter_frames is not None:
                        if len(filter_frames) != 2:
                            raise ValueError(
                                "filter_frames must be a 2-element list [start, end] denoting an inclusive range"
                            )
                        if frame_stride <= 0:
                            raise ValueError("frame_stride must be a positive integer")

                        start_idx, end_idx = filter_frames
                        if start_idx is None:
                            start_idx = 0
                        if end_idx is None:
                            end_idx = field.shape[0] - 1
                        frame_slice = slice(start_idx, end_idx + 1, frame_stride)
                    else:
                        frame_slice = slice(None, None, frame_stride if frame_stride != 1 else None)

                    # Single read → numpy array in RAM (shape: (T, C, ...))
                    field_data = field[frame_slice]

                    # Pre-compute residuals if needed (diff along time axis)
                    if residual_aggregators is not None and field_data.shape[0] > 1:
                        residual_data = np.diff(field_data, axis=0)
                    else:
                        residual_data = None

                    # Iterate over every channel component but now slice in memory
                    for ch in range(ch_dim):
                        channel_data = field_data[:, ch]
                        key = field_name if ch_dim == 1 else f"{field_name}_{ch}"

                        if key not in aggregators:
                            aggregators[key] = _StatsAggregator()
                            if key in log_channels_set:
                                aggregators[f"log_{key}"] = _StatsAggregator()

                        if on_fly_stats:
                            aggregators[key].add(channel_data)
                            if key in log_channels_set:
                                if np.any(channel_data <= 0):
                                    raise ValueError(f"Channel {key} contains negative or zero values. Cannot apply log transform.")
                                log_key = f"log_{key}"
                                log_channel_data = np.log(channel_data)
                                aggregators[log_key].add(log_channel_data)
                            if residual_data is not None:
                                res_key = f"{key}_residual"
                                residual_aggregators[res_key].add(residual_data[:, ch])
                        else:
                            collected_data[key].append(channel_data)
                            if key in log_channels_set:
                                if np.any(channel_data <= 0):
                                    raise ValueError(f"Channel {key} contains negative or zero values. Cannot apply log transform.")
                                log_key = f"log_{key}"
                                log_channel_data = np.log(channel_data)
                                collected_data[log_key].append(log_channel_data)
                            if residual_data is not None:
                                res_key = f"{key}_residual"
                                collected_residual[res_key].append(residual_data[:, ch])

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


# =====================================================================
# Parallel CPU version – high-level convenience wrapper
# =====================================================================

from concurrent.futures import ProcessPoolExecutor
from functools import reduce


def _process_single_file(
    path: str,
    residual_config: Dict[str, bool] | None,
    filter_groups: List[str] | None,
    filter_frames: List[int] | None,
    frame_stride: int,
    on_fly_stats: bool,
    log_transform_channels: List[str] | None = None,
) -> tuple[Dict[str, _StatsAggregator], int]:
    """Compute per-channel aggregators for *one* HDF5 file (worker)."""

    log_channels_set = set(log_transform_channels or [])

    # Discover metadata for this file (may raise if filter_groups missing)
    try:
        channel_names, _ = _discover_metadata(path, filter_groups)
    except ValueError:
        # no matching groups → return empty dict
        return {}

    aggs: Dict[str, _StatsAggregator] = {n: _StatsAggregator() for n in channel_names}
    # Pre-create log-channel aggregators for any channel marked for log transform
    for n in channel_names:
        if n in log_channels_set:
            aggs[f"log_{n}"] = _StatsAggregator()
    residual_aggs: Dict[str, _StatsAggregator] | None = None
    if residual_config is not None:
        residual_aggs = {f"{n}_residual": _StatsAggregator() for n in channel_names}

    epsilon = 1e-10

    with h5py.File(path, "r") as f:
        groups_processed = 0
        for grp_name in f:
            if filter_groups is not None and grp_name not in filter_groups:
                continue
            grp = f[grp_name]
            groups_processed += 1
            for field_name in grp:
                field = grp[field_name]
                ch_dim = field.shape[1]

                # Build slice for frames / stride
                if filter_frames is not None:
                    if len(filter_frames) != 2:
                        raise ValueError("filter_frames must be length-2 [start, end]")
                    if frame_stride <= 0:
                        raise ValueError("frame_stride must be positive")
                    start_idx, end_idx = filter_frames
                    if start_idx is None:
                        start_idx = 0
                    if end_idx is None:
                        end_idx = field.shape[0] - 1
                    frame_slice = slice(start_idx, end_idx + 1, frame_stride)
                else:
                    #by default, process all frames of a trajectory with the variable 'frame_stride' 
                    frame_slice = slice(None, None, frame_stride if frame_stride != 1 else None)

                data = field[frame_slice]
                
                for ch in range(ch_dim):
                    key = field_name if ch_dim == 1 else f"{field_name}_{ch}"
                    
                    # Extract channel data
                    channel_data = data[:, ch]
                    
                    # ============================================================
                    # Apply log transform BEFORE computing statistics
                    # ============================================================
                    if key in log_channels_set:
                        # Raise an error if the channel data contains negative values
                        if np.any(channel_data <= 0):
                            raise ValueError(f"Channel {key} contains negative or zero values. Cannot apply log transform.")
                        # add a new key to the aggregator with the log-transformed data
                        log_key = f"log_{key}"
                        log_channel_data = np.log(channel_data)
                        #add is the function to add the statistics to the aggregator
                        aggs[log_key].add(log_channel_data)
                        #channel_data = np.log(np.maximum(channel_data, epsilon))
                    
                    # Add to aggregator
                    aggs[key].add(channel_data)
                    
                    # ============================================================
                    # Handle residuals (TODO: Implement residual stats computation for log-transformed channels)
                    # ============================================================
                    if residual_aggs is not None and data.shape[0] > 1:
                        # if the log_channel_set is not empty, raise an error
                        if log_channels_set:
                            raise ValueError("Residual computation for log-transformed channels is not implemented yet.")
                        res_key = f"{key}_residual"
                        
                        if key in log_channels_set:
                            # For log-transformed channels:
                            # Compute residuals in log space as log(x[t]) - log(x[t-1])
                            # This is NOT the same as log(x[t] - x[t-1])
                            log_original = np.log(np.maximum(data[:, ch], epsilon))
                            res_data = np.diff(log_original, axis=0)
                        else:
                            # For normal channels: standard temporal difference
                            res_data = np.diff(data[:, ch], axis=0)
                        
                        residual_aggs[res_key].add(res_data)

    # merge residuals into main dict (if any)
    if residual_aggs is not None:
        aggs.update(residual_aggs)

    return aggs, groups_processed


def _merge_aggregator_dicts(a: Dict[str, _StatsAggregator], b: Dict[str, _StatsAggregator]) -> Dict[str, _StatsAggregator]:
    """In-place merge of two aggregator dictionaries, returns *a* for chaining."""
    for k, agg_b in b.items():
        if k in a:
            a[k].add_from_aggregator(agg_b)
        else:
            a[k] = agg_b
    return a


def compute_statistics_parallel(
    h5_paths: List[str],
    residual_config: Dict[str, bool] | None = None,
    filter_groups: List[str] | None = None,
    filter_frames: List[int] | None = None,
    frame_stride: int = 1,
    on_fly_stats: bool = False,
    num_workers: int | None = None,
    log_transform_channels: List[str] | None = None,
) -> Tuple[Dict[str, Dict[str, float]], List[str], int]:
    """Parallel CPU implementation of :func:`compute_statistics`.

    Each HDF5 file is processed in its own **process** and the per-channel
    aggregators are merged in the parent.  The numerical results are
    identical to the serial version; only the run-time can change.
    """

    if len(h5_paths) == 0:
        raise ValueError("h5_paths list is empty.")

    #log_channels_set = set(log_transform_channels or [])

    # ------------------------------------------------------------------
    # Discover metadata once (first file that fits the filters)
    # ------------------------------------------------------------------
    metadata_found = False
    channel_names: List[str] = []
    problem_dim: int = 0
    for p in h5_paths:
        try:
            channel_names, problem_dim = _discover_metadata(p, filter_groups)
            #remove the channels that start with 'log_', we just need the statistics for the log trnsformed channel.
            channel_names = [ch for ch in channel_names if not ch.startswith("log_")]
            metadata_found = True
            break
        except ValueError:
            continue

    if not metadata_found:
        raise ValueError(
            "None of the provided HDF5 files contained any of the requested groups "
            f"filter_groups={filter_groups}."
        )

    # ------------------------------------------------------------------
    # Launch workers
    # ------------------------------------------------------------------
    import itertools as _it
    from concurrent.futures import as_completed
    try:
        from tqdm.auto import tqdm as _tqdm  # type: ignore
    except Exception:
        _tqdm = None

    # If we're effectively running one file, do a true per-group tqdm in-process
    # (multiprocessing can't provide fine-grained progress without IPC and can
    # be unstable with many concurrent HDF5 opens).
    if len(h5_paths) == 1 or (num_workers is not None and num_workers <= 1):
        path = h5_paths[0]
        with h5py.File(path, "r") as f:
            grp_names = [g for g in f.keys() if filter_groups is None or g in filter_groups]
        if not grp_names:
            raise ValueError(
                "None of the provided HDF5 files contained any of the requested groups "
                f"filter_groups={filter_groups}."
            )

        done = grp_names
        if _tqdm is not None:
            done = _tqdm(done, total=len(grp_names), desc="Computing statistics", unit="group")

        agg_dicts: list[Dict[str, _StatsAggregator]] = []
        for g in done:
            # reuse existing worker logic by restricting filter_groups to [g]
            d, _n = _process_single_file(
                path,
                residual_config,
                [g],
                filter_frames,
                frame_stride,
                on_fly_stats,
                log_transform_channels,
            )
            if d:
                agg_dicts.append(d)
    else:
        # Multi-file parallelism (stable) with progress measured in groups.
        # tqdm advances by the number of groups completed in each finished file.
        group_counts: dict[str, int] = {}
        total_groups = 0
        for p in h5_paths:
            try:
                with h5py.File(p, "r") as f:
                    if filter_groups is None:
                        c = len(list(f.keys()))
                    else:
                        c = sum(1 for g in filter_groups if g in f)
                group_counts[p] = c
                total_groups += c
            except OSError:
                group_counts[p] = 0

        if total_groups == 0:
            raise ValueError(
                "None of the provided HDF5 files contained any of the requested groups "
                f"filter_groups={filter_groups}."
            )

        pbar = None
        if _tqdm is not None:
            pbar = _tqdm(total=total_groups, desc="Computing statistics", unit="group")

        agg_dicts = []
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [
                pool.submit(
                    _process_single_file,
                    p,
                    residual_config,
                    filter_groups,
                    filter_frames,
                    frame_stride,
                    on_fly_stats,
                    log_transform_channels,
                )
                for p in h5_paths
            ]

            for fut in as_completed(futures):
                d, n_groups = fut.result()
                if d:
                    agg_dicts.append(d)
                if pbar is not None:
                    pbar.update(n_groups)

        if pbar is not None:
            pbar.close()

    # ------------------------------------------------------------------
    # Merge aggregator dictionaries
    # ------------------------------------------------------------------
    global_aggs: Dict[str, _StatsAggregator] = {}
    for d in agg_dicts:
        _merge_aggregator_dicts(global_aggs, d)

    # ------------------------------------------------------------------
    # Build final statistics dictionary
    # ------------------------------------------------------------------
    channel_stats = {k: agg.combined() for k, agg in global_aggs.items()}

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
    eps = 1e-12 # small constant to prevent divide-by-zero

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

