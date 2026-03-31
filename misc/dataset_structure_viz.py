"""
Dataset Structure Visualization Utility.

This module provides a simple utility to explore and visualize the hierarchical
structure of HDF5 datasets. It recursively traverses
the HDF5 file structure and prints information about groups and datasets,
including their shapes and data types.
"""

import h5py
import numpy as np
import os
import argparse

def visualize_dataset_structure(data_path):
    """
    Visualize the hierarchical structure of an HDF5 dataset file.
    
    This function opens an HDF5 file and recursively prints information about
    all groups and datasets contained within it, including their shapes and
    data types. This is useful for understanding the organization of CFD
    datasets before processing.
    
    Parameters
    ----------
    data_path : str
        Path to the HDF5 file to be analyzed. Can be relative or absolute.
        
    Notes
    -----
    The function prints the structure to stdout in a hierarchical format:
    - Groups are printed with "Group: <name>"
    - Datasets are printed with "  Field: <name> - Shape: <shape>, Dtype: <dtype>"
    
    Examples
    --------
    >>> visualize_dataset_structure("./data/fluids/KVS_trimmed/2D/train.h5")
    Group: /
      Field: velocity_x - Shape: (1000, 1, 64, 64), Dtype: float32
      Field: velocity_y - Shape: (1000, 1, 64, 64), Dtype: float32
      Field: pressure - Shape: (1000, 1, 64, 64), Dtype: float32
    """
    data_path = os.path.abspath(data_path)
    print("data_path:", data_path)
    
    with h5py.File(data_path, "r") as f:
        # ------------------------------------------------------------------
        # Count top-level groups (same criterion as scripts/count_h5_groups.py)
        # ------------------------------------------------------------------
        num_top_groups = sum(1 for _ in f.keys())
        print(f"Number of top-level groups: {num_top_groups}\n")

        # ------------------------------------------------------------------
        # Recursively print the structure
        # ------------------------------------------------------------------
        # def print_structure(name, obj):
        #     if isinstance(obj, h5py.Group):
        #         print(f"Group: {name}")
        #     elif isinstance(obj, h5py.Dataset):
        #         print(f"  Field: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")

        # f.visititems(print_structure)

        def read_h5(group, prefix=""):
            for key in group:
                item = group[key]
                path = f"{prefix}/{key}" if prefix else key

                if isinstance(item, h5py.Dataset):
                    print(f"  Field: {key} - Shape: {item.shape}, Dtype: {item.dtype}, Compression: {item.compression}")
                elif isinstance(item, h5py.Group):
                    print(f"Group: {key}")
                    read_h5(item, path)
        read_h5(f)

# -----------------------------------------------------------------------------
# Usage (terminal):
#     python misc/dataset_structure_viz.py /path/to/dataset.h5
# If no path is supplied, the default path defined below will be used.
# -----------------------------------------------------------------------------

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the hierarchical structure of an HDF5 dataset file."
    )
    parser.add_argument(
        "data_path",
        type=str,
        nargs="?",  # optional positional argument
        default="./data/fluids/Laser_Droplet/2D/test.h5",
        help="Path to the HDF5 file to analyze (default: %(default)s)",
    )

    args = parser.parse_args()

    visualize_dataset_structure(args.data_path)
