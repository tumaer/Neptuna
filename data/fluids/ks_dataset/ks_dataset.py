import numpy as np
import h5py 
# Load the .npy file
tensor = np.load('3d_ks_sample.npy')

# Print the shape
print(tensor.shape)
data_path = "3d_ks_train.h5"
with h5py.File(data_path, 'w') as f:
    for i in range(tensor.shape[0]):  # Loop over the seeds (trajectories)
        group = f.create_group(f'seed {i}')
        group.create_dataset('velocity', data=tensor[i])

with h5py.File(data_path, "r") as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")

    f.visititems(print_structure)