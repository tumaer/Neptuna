# Library to compute dataset statistics (mean, std, min, max) and metadata across
# one or multiple HDF5 files. Designed to be imported and called from other
# modules (e.g. `main.py`) instead of running as a standalone script.

import h5py
import numpy as np
import os
from typing import List, Dict, Tuple


__all__ = [
    "compute_statistics",  # main API
]


class _StatsAggregator:
    """Utility container to accumulate stats for a single channel across groups/files."""

    def __init__(self):
        self.means: List[float] = []
        self.stds: List[float] = []
        self.mins: List[float] = []
        self.maxs: List[float] = []

    def add(self, arr: np.ndarray):
        """Accumulate statistics for *arr* in float-64 precision irrespective of
        the input array dtype. This avoids precision loss when datasets are
        stored in single-precision formats (e.g. FP32)."""

        # Promote to float64 for numerically stable statistics.
        flat = arr.reshape(-1).astype(np.float64)

        self.means.append(float(np.mean(flat, dtype=np.float64)))
        self.stds.append(float(np.std(flat, dtype=np.float64)))
        # Min/Max are exact so dtype promotion is not strictly required but we
        # keep everything in float64 for consistency.
        self.mins.append(float(np.min(flat)))
        self.maxs.append(float(np.max(flat)))

    def combined(self) -> Dict[str, float]:
        return {
            "mean": float(np.mean(self.means)),
            "std": float(_combined_std(self.stds)),
            "min": float(np.min(self.mins)),
            "max": float(np.max(self.maxs)),
        }


def _combined_std(std_devs: List[float]) -> float:
    """Root-mean-square of individual standard deviations."""
    variances = [s ** 2 for s in std_devs]
    return float(np.sqrt(np.mean(variances, dtype=np.float64)))


def _discover_metadata(first_h5_path: str, filter_groups: List[str] | None = None) -> Tuple[List[str], int]:
    """Extract expanded channel names and spatial dimension from the first group
    of the first H5 file provided. If *filter_groups* is supplied, the first
    group whose name matches an entry in *filter_groups* is used instead of the
    very first group in the file. This makes the routine compatible with the
    group-filtering logic used inside :func:`compute_statistics`."""

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
    """Compute combined mean/std/min/max for each channel-expanded list across one
    or many HDF5 files. Optionally, when *residual_config* is supplied, additional
    statistics for the temporal residuals (frame-to-frame differences) of every
    channel are computed. These residual statistics are stored under the key
    f"{channel_name}_residual".

    Parameters
    ----------
    h5_paths : List[str]
        List of HDF5 file paths.
        
    residual_config : Dict[str, bool] | None, optional
        When *residual_config* is not ``None`` the function additionally
        returns statistics for residual fields, where residuals are always
        defined as frame-to-frame differences (x_t - x_{t-1}). These are
        stored under keys "<channel>_residual".  The contents of
        *residual_config* are used elsewhere in the codebase, but here they
        serve only for validation; if both "add_predicted_value" and
        "add_base_value" are set to True a ValueError is raised to
        prevent ambiguous configuration.

    filter_groups : List[str] | None, optional
        If provided, statistics are computed only for the HDF5 groups whose
        names are included in *filter_groups*. Any groups not listed here are
        ignored.

    filter_frames : List[int] | None, optional
        If provided, *filter_frames* **must** contain exactly two integers
        ``[start, end]`` which define an inclusive range of time-frame indices
        to consider (i.e. all frames with indices ``start <= t <= end`` will be
        used).  Negative indices follow normal NumPy semantics.  Use ``None``
        for *start* or *end* to indicate the first / last frame respectively.
        If *filter_frames* is ``None`` every frame is processed.

    frame_stride : int, default=1
        Step size when iterating over frames inside the selected range. A value
        of ``1`` means every frame is used, ``2`` every second frame, and so
        on.  Must be a positive integer.

    on_fly_stats : bool, default=False
        If ``False`` (default) all selected frames for each channel are first
        gathered in memory; statistics are then computed once per channel at
        the very end.  When ``True`` statistics are accumulated *on the fly*
        for every slice of data as it is encountered (previous behaviour). The
        latter uses less memory but can be slightly slower.

    Returns
    -------
    channel_stats : Dict[str, Dict[str, float]]
        Mapping: ``channel_name`` (and ``channel_name_residual`` when
        *residual_config* is provided) -> ``{mean, std, min, max, median, iqr}``.
    channel_names : List[str]
        Expanded channel names discovered from the first (selected) group.
    problem_dim : int
        Number of spatial dimensions (2 for 2D, 3 for 3D, ...).
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


    