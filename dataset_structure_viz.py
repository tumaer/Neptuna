import h5py
import numpy as np
import os

data_path="./data/fluids/KS/1D/train.h5"
data_path=os.path.abspath(data_path)
print("data_path:",data_path)

with h5py.File(data_path, "r") as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")
    f.visititems(print_structure)