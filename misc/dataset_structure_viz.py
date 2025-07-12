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
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"  Field: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")
        
        f.visititems(print_structure)

# Main execution
if __name__ == "__main__":
    data_path = "./data/fluids/KVS_trimmed/2D/train.h5"
    visualize_dataset_structure(data_path)
