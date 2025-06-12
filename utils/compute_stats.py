import h5py
import numpy as np
import os
#Please refer to the following structure of .h5 file for statistics calculation
"""
    - Group 0: 
        - Dataset:
            - Field_name: density
                - Shape: (Timesteps, Channels, *Spatial_Dims)
                - Dtype: (float32, float64, etc.)
            - Field_name: velocity
                - Shape: (Timesteps, Channels, *Spatial_Dims)
                - Dtype: (float32, float64, etc.)
            - ...
    - Group 1:
        - Dataset:
            - Field_name: density
                - Shape: (Timesteps, Channels, *Spatial_Dims)
                - Dtype: (float32, float64, etc.)
            - Field_name: velocity
                - Shape: (Timesteps, Channels, *Spatial_Dims)
                - Dtype: (float32, float64, etc.)
            - ...
    
    - ...
    - ...

    - Group N:
        - Dataset:
            - Field_name: density
                - Shape: (Timesteps, Channels, *Spatial_Dims)
                - Dtype: (float32, float64, etc.)
            - Field_name: velocity
                - Shape: (Timesteps, Channels, *Spatial_Dims)
                - Dtype: (float32, float64, etc.)
            - ...
                
"""

# -----------------------------------------------------------------------------
# 1.  Provide a list of input .h5 files you want to merge before calculating statistics
# -----------------------------------------------------------------------------
data_paths = [
    "./KVS_original/train.h5",
    "./KVS_original/test.h5",
    # Add more .h5 files here if needed
]

# Path of the temporary combined file that will be created. This will be deleted after the script is run.
combined_path = "combined_dataset.h5"

# -----------------------------------------------------------------------------
# 2.  Merge the input files into a single combined .h5 file
# -----------------------------------------------------------------------------

def merge_h5_files(input_paths: list[str], output_path: str):
    """Merge multiple HDF5 files into a single file.

    If groups with the same name appear in multiple files, the later file's group
    is renamed with a suffix to avoid clashes (e.g. `Group -> Group_file1`).
    """

    # Remove any existing combined file to avoid appending to old data.
    if os.path.exists(output_path):
        os.remove(output_path)

    with h5py.File(output_path, "w") as fout:
        for idx, in_path in enumerate(input_paths):
            print(f"Merging: {in_path} -> {output_path}")
            with h5py.File(in_path, "r") as fin:
                for grp_name in fin:
                    dest_name = grp_name
                    if dest_name in fout:
                        dest_name = f"{grp_name}_file{idx}"
                    fin.copy(grp_name, fout, name=dest_name)


# Perform the merge once at script start
merge_h5_files(data_paths, combined_path)

# Use the combined file for all subsequent processing
data_path = combined_path

# -----------------------------------------------------------------------------
# 3.  Statistics computation
# -----------------------------------------------------------------------------

with h5py.File(data_path, "r") as f:
    class ComputeStatistics:
        """Class to gather dataset metadata (field names, spatial dimension) and
        compute per-field mean/std statistics over an HDF5 dataset."""

        def __init__(self, data_path: str):
            self.data_path = data_path
            self.group_name = ""
            self.first_group_name = None  # Track the very first group encountered
            self.field_names: list[str] = []
            self.first_group_processed = False
            self.problem_dimension: int | None = None

        # ---------- helper (static) ----------
        @staticmethod
        def _combined_std(std_devs: list[float]) -> float:
            variances = [s ** 2 for s in std_devs]
            mean_variance = sum(variances) / len(variances)
            return float(np.sqrt(mean_variance))

        # ---------- visitor interface ----------
        def __call__(self, name, obj):
            """Callable for h5py.Group.visititems to collect metadata from first group."""
            if isinstance(obj, h5py.Group):
                if self.first_group_name is None:
                    self.first_group_name = name
                elif name != self.first_group_name:
                    self.first_group_processed = True  # we entered a second group

                self.group_name = name
                print(f"Group: {name}")

            elif isinstance(obj, h5py.Dataset):
                dataset_name = name
                field_name = dataset_name.replace(self.group_name, '').strip('/')
                print(f"  Field_name: {field_name} - Shape: {obj.shape}, Dtype: {obj.dtype}")

                if not self.first_group_processed:
                    channel_dim = obj.shape[1]
                    if channel_dim == 1:
                        self.field_names.append(field_name)
                    else:
                        for ch in range(channel_dim):
                            self.field_names.append(f"{field_name}_{ch}")

                    if self.problem_dimension is None:
                        self.problem_dimension = len(obj.shape[2:])  # spatial dims

        # ---------- statistics over entire file ----------
        def compute_field_statistics(self):
            """Compute mean, std, min and max for each (possibly channel-expanded) field across the file."""
            per_field_group_stats: dict[str, dict[str, list[float]]] = {}

            with h5py.File(self.data_path, 'r') as f:
                for grp_name in f:
                    grp = f[grp_name]
                    for dset_name in grp:
                        dset = grp[dset_name]
                        data = dset[:]  # (T, C, ...)
                        ch_dim = data.shape[1]

                        for ch in range(ch_dim):
                            channel_data = data[:, ch]
                            key = dset_name if ch_dim == 1 else f"{dset_name}_{ch}"

                            flat = channel_data.reshape(-1)  # keep original dtype
                            grp_mean = float(np.mean(flat))
                            grp_std = float(np.std(flat))
                            grp_min = float(np.min(flat))
                            grp_max = float(np.max(flat))

                            if key not in per_field_group_stats:
                                per_field_group_stats[key] = {
                                    'means': [],
                                    'stds': [],
                                    'mins': [],
                                    'maxs': []
                                }

                            per_field_group_stats[key]['means'].append(grp_mean)
                            per_field_group_stats[key]['stds'].append(grp_std)
                            per_field_group_stats[key]['mins'].append(grp_min)
                            per_field_group_stats[key]['maxs'].append(grp_max)

            # aggregate across groups
            stats: dict[str, dict[str, float]] = {}
            for key, agg in per_field_group_stats.items():
                combined_mean = float(np.mean(agg['means']))
                combined_std_val = self._combined_std(agg['stds'])
                combined_min = float(np.min(agg['mins']))
                combined_max = float(np.max(agg['maxs']))
                stats[key] = {
                    'mean': combined_mean,
                    'std': combined_std_val,
                    'min': combined_min,
                    'max': combined_max
                }

            return stats

    # Instantiate statistics computer and gather metadata
    stats_computer = ComputeStatistics(data_path)
    f.visititems(stats_computer)

    print(f"\nField names from first group (expanded by channel): {stats_computer.field_names}")
    print(f"\nProblem dimension: {stats_computer.problem_dimension}D")

    # Compute mean and standard deviation for each field
    field_stats = stats_computer.compute_field_statistics()
    print("\nComputed field statistics (mean, std, min, max):")
    for field, stat in field_stats.items():
        print(
            f"  {field}: "
            f"mean={stat['mean']:.6e}, "
            f"std={stat['std']:.6e}, "
            f"min={stat['min']:.6e}, "
            f"max={stat['max']:.6e}"
        )

# -----------------------------------------------------------------------------
# 4.  Cleanup temporary combined file
# -----------------------------------------------------------------------------

if os.path.exists(combined_path):
    os.remove(combined_path)
    print(f"\nTemporary file '{combined_path}' has been removed.")


    