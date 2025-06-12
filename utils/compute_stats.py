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
    """Utility container to accumulate stats for a single field across groups/files."""

    def __init__(self):
        self.means: List[float] = []
        self.stds: List[float] = []
        self.mins: List[float] = []
        self.maxs: List[float] = []

    def add(self, arr: np.ndarray):
        flat = arr.reshape(-1)  # keep original dtype
        self.means.append(float(np.mean(flat)))
        self.stds.append(float(np.std(flat)))
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
    return float(np.sqrt(np.mean(variances)))


def _discover_metadata(first_h5_path: str) -> Tuple[List[str], int]:
    """Extract expanded field names and spatial dimension from the first group
    of the first H5 file provided."""

    with h5py.File(first_h5_path, "r") as f:
        first_group = list(f.keys())[0]
        grp = f[first_group]

        field_names: List[str] = []
        problem_dimension: int | None = None

        for dset_name in grp:
            dset = grp[dset_name]
            channel_dim = dset.shape[1]

            # expanded channel names
            if channel_dim == 1:
                field_names.append(dset_name)
            else:
                for ch in range(channel_dim):
                    field_names.append(f"{dset_name}_{ch}")

            if problem_dimension is None:
                problem_dimension = len(dset.shape[2:])

        assert problem_dimension is not None, "Could not infer problem dimension."

    return field_names, problem_dimension


def compute_statistics(h5_paths: List[str]) -> Tuple[Dict[str, Dict[str, float]], List[str], int]:
    """Compute combined mean/std/min/max for each channel-expanded field across one
    or many HDF5 files.

    Parameters
    ----------
    h5_paths : List[str]
        List of HDF5 file paths.

    Returns
    -------
    field_stats : Dict[str, Dict[str, float]]
        Mapping: field_name -> {mean, std, min, max}
    field_names : List[str]
        Expanded field names discovered from the first file's first group.
    problem_dim : int
        Number of spatial dimensions (2 for 2D, 3 for 3D, ...).
    """

    if len(h5_paths) == 0:
        raise ValueError("h5_paths list is empty.")

    # Metadata from first file
    field_names, problem_dim = _discover_metadata(h5_paths[0])

    # Aggregators keyed by expanded field name
    aggregators: Dict[str, _StatsAggregator] = {name: _StatsAggregator() for name in field_names}

    # Iterate over all files and groups
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            for grp_name in f:
                grp = f[grp_name]
                for dset_name in grp:
                    dset = grp[dset_name]
                    ch_dim = dset.shape[1]

                    for ch in range(ch_dim):
                        channel_data = dset[:, ch]
                        key = dset_name if ch_dim == 1 else f"{dset_name}_{ch}"

                        # Safety: if new field encountered that wasn't in metadata (unlikely but possible)
                        if key not in aggregators:
                            aggregators[key] = _StatsAggregator()

                        aggregators[key].add(channel_data)

    # Combine
    field_stats = {k: agg.combined() for k, agg in aggregators.items()}

    return field_stats, field_names, problem_dim


# -----------------------------------------------------------------------------
# If executed as a script, run a quick demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    demo_paths = [
        "./KVS_original/train.h5",
        "./KVS_original/test.h5",
    ]

    stats, names, dim = compute_statistics(demo_paths)

    print("Field names:", names)
    print(f"Problem dimension: {dim}D")
    print("\nStatistics:")
    for k, v in stats.items():
        print(f"  {k}: mean={v['mean']:.6e}, std={v['std']:.6e}, min={v['min']:.6e}, max={v['max']:.6e}")


    