import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class MultiGroupHDF5Dataset(Dataset):
    def __init__(self, h5_path, groups=None, channels=None,
                 input_steps=3, label_steps=5,
                 input_stride=1, label_stride=1,
                 t_min=None, t_max=None,
                 transform=None):
        super().__init__()
        self.h5_path = h5_path
        self.transform = transform
        self.input_steps = input_steps
        self.label_steps = label_steps
        self.input_stride = input_stride
        self.label_stride = label_stride
        self.index_map = []

        # Compute total number of frames required per sample
        last_input_idx = (input_steps - 1) * input_stride
        last_label_idx = (label_steps - 1) * label_stride
        self.total_span = last_input_idx + 1 + last_label_idx

        with h5py.File(self.h5_path, 'r') as f:
            all_groups = list(f.keys())
            self.groups = groups if groups is not None else all_groups

            # Infer channels from first group
            first_group = f[self.groups[0]]
            available_channels = list(first_group.keys())
            self.channels = channels if channels is not None else available_channels

            for group_name in self.groups:
                group = f[group_name]
                num_samples = group[self.channels[0]].shape[0]

                # Apply t_min and t_max bounds
                tmin = 0 if t_min is None else t_min
                tmax = num_samples if t_max is None else min(t_max, num_samples)

                # Last valid starting index to keep the sample within range
                max_start_idx = tmax - self.total_span + 1

                for i in range(tmin, max_start_idx):
                    self.index_map.append((group_name, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        group_name, start_idx = self.index_map[idx]
        with h5py.File(self.h5_path, 'r') as f:
            group = f[group_name]
            input_chunks = []
            label_chunks = []

            for channel in self.channels:
                # Time indices for inputs and labels
                input_indices = [start_idx + i * self.input_stride for i in range(self.input_steps)]
                label_start = input_indices[-1] + 1
                label_indices = [label_start + i * self.label_stride for i in range(self.label_steps)]

                input_seq = np.stack([group[channel][i] for i in input_indices], axis=0)
                label_seq = np.stack([group[channel][i] for i in label_indices], axis=0)

                input_chunks.append(input_seq)
                label_chunks.append(label_seq)

            inputs = np.concatenate(input_chunks, axis=1)
            labels = np.concatenate(label_chunks, axis=1)

        sample = {
            "group": group_name, #can be used to condition the model
            "inputs": torch.from_numpy(inputs).float(),
            "labels": torch.from_numpy(labels).float()
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__=="__main__":


    from torch.utils.data import DataLoader

    dataset = MultiGroupHDF5Dataset(
    "./data/KVS/train.h5",
    input_steps=3,
    label_steps=5,
    input_stride=1,
    label_stride=1,
    t_min=100,     # only use timesteps 100 to 499
    t_max=500
)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in dataloader:
        print(batch["inputs"].shape)  # (4, 3, C_total, H, W)
        print(batch["labels"].shape)  # (4, 5, C_total, H, W)





#"./data/KVS/train.h5"