import h5py
import numpy as np
import os

files = [
    "/home/sjd1998/repos2/cfd_bench/data/fluids/3dKVS/train.h5",
    "/home/sjd1998/repos2/cfd_bench/data/fluids/3dKVS/test.h5"
]

for path in files:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
    with h5py.File(path, 'r+') as f:
        for group_name in list(f.keys()):
            group = f[group_name]
            for dataset_name in list(group.keys()):
                data = group[dataset_name][...]
                new_shape = data.shape + (6,)
                new_data = np.zeros(new_shape, dtype=data.dtype)
                new_data[..., 0] = data
                del group[dataset_name]
                group.create_dataset(dataset_name, data=new_data)