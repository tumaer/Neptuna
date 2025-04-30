###this file is used to load the data from the h5 file
##train_strategies:
# all2all : makes sense only when the number of timesteps are less
# many2many (includes many2one and one2many)
# many2all ?
# autoregressive??
# downsampling the data automatically to a specific resolution
#####
"""
Each dataset inside the .h5 file should have the following format:
1. Group name: Ex: ['Re_100', 'Re_200', 'Re_300', 'Re_400', 'Re_500']
2. Inside each group: Fields: Ex: ['velocity', 'pressure', 'density', 'temperature']
"""
import h5py
from torch.utils.data import Dataset
from typing import Optional
import os
import numpy as np
import torch

def fetch_dataset(dataset_name: str, 
                  **kwargs):
    """
    Factory function to create a dataset instance based on the dataset name.
    """
    if dataset_name == "KarmanVortexStreet":
        from data.fluids.incompressible import KarmanVortexStreetDataset as LoadedDataset
        return LoadedDataset(dataset_name=dataset_name, **kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented yet.")



class BaseDataset(Dataset):
    def __init__(self, 
                 dataset_directory_path: str,
                 mode: str, #train, test, val
                 strategy: str = 'many2many', #TODO: all2all
                 dataset_name: Optional[str] = None,
                 groups: Optional[list] = None, #specific groups inside dataset to be used for training
                 fields: Optional[list] = None, #specific fields inside dataset to be used for training
                 sequence_info: Optional[list] = [[1, 1, 1, 1]], #sequence_info[0][0]: input_seq_len, sequence_info[0][1]: label_seq_len, 
                                                                 #sequence_info[0][2]: input_sequence_stride, sequence_info[0][3]: label_sequence_stride
                 filter_frame: Optional[list] = None, #filter_frame[0][0]: min_frame, filter_frame[0][1]: max_frame
                 transform= None, #transform: any transform to be applied to the data
                 ):
        super().__init__()
        
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.dataset_name = dataset_name
        self.transform = transform
        self.index_map = []

        if dataset_name is None: #infer dataset name from the directory path
            dataset_name = os.path.basename(os.path.normpath(dataset_directory_path))
            print("dataset_name:", dataset_name)
        if self.mode == "train" or self.mode == "val":
            self.h5file_path = os.path.abspath(dataset_directory_path+"/train.h5")
            print("h5_file_path:", self.h5file_path)

        else:
            self.h5file_path = os.path.abspath(dataset_directory_path+"/test.h5")
            print("h5_file_path=", self.h5file_path)
        
        self.input_seq_len = sequence_info[0][0] #number of historic steps to be considered in the input
        self.label_seq_len = sequence_info[0][1] #number of future steps to be predicted
        self.input_seq_stride = sequence_info[0][2] #stride for the input sequence
        self.label_seq_stride = sequence_info[0][3] #stride for the output sequence
        
        self.strategy = strategy    

        # Compute total number of frames required per sample
        last_input_idx = (self.input_seq_len - 1) * self.input_seq_stride
        last_label_idx = (self.label_seq_len - 1) * self.label_seq_stride 
        self.total_span = last_input_idx + 1 + last_label_idx #this is the window to be extracted
        
        with h5py.File(self.h5file_path, 'r') as f:
            all_groups = list(f.keys())
            self.groups = groups if groups is not None else all_groups

            # Infer fields from first group
            first_group = f[self.groups[0]]
            available_fields = list(first_group.keys())
            self.fields = fields if fields is not None else available_fields

            for group_name in self.groups:
                group = f[group_name]
                # Apply t_min and t_max bounds
                self.min_frame = 0 if filter_frame is None else filter_frame[0][0] #selecting specific timesteps from the whole trajectory
                num_samples = group[self.fields[0]].shape[0]
                self.max_frame = num_samples-1 if filter_frame is None else min(filter_frame[0][1], num_samples)

                # Last valid starting index to keep the sample within range
                #max_start_idx = self.max_frame - self.total_span + 1
                # for i in range(self.min_frame, max_start_idx):
                #     self.index_map.append((group_name, i))

                half_input = (self.input_seq_len - 1) * self.input_seq_stride # Compute how far back we need to go from the center index to collect the full input sequence
                half_label = (self.label_seq_len - 1) * self.label_seq_stride # Compute how far forward we need to go from the center index to collect the full label sequence

                # The earliest valid center index must be far enough from the start (min_frame) so that we can extract all input frames to the left
                min_valid_idx = self.min_frame + half_input
                # The latest valid center index must be far enough from the end (max_frame)
                # so that we can extract all label frames to the right
                # We subtract 1 to ensure we don’t index beyond the available range
                max_valid_idx = self.max_frame - half_label - 1  

                for i in range(min_valid_idx, max_valid_idx + 1):
                    self.index_map.append((group_name, i))
            print("Index map built")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.strategy == "many2many":
            group_name, start_idx = self.index_map[idx]
            with h5py.File(self.h5file_path, 'r') as f:
                group = f[group_name]
                input_chunks = []
                label_chunks = []

                for field in self.fields:
                    # Time indices for inputs and labels
                    # input_indices = [start_idx + i * self.input_seq_stride for i in range(self.input_seq_len)]
                    # label_start = input_indices[-1] + 1
                    # label_indices = [label_start + i * self.label_seq_stride for i in range(self.label_seq_len)]

                    center_idx = start_idx
                    input_indices = [center_idx - (self.input_seq_len - 1 - i) * self.input_seq_stride for i in range(self.input_seq_len)]
                    label_indices = [center_idx + (i + 1) * self.label_seq_stride for i in range(self.label_seq_len)]

                    input_seq = np.stack([group[field][i] for i in input_indices], axis=0)
                    label_seq = np.stack([group[field][i] for i in label_indices], axis=0)

                    input_chunks.append(input_seq)
                    label_chunks.append(label_seq)

                inputs = np.concatenate(input_chunks, axis=1)
                labels = np.concatenate(label_chunks, axis=1)

            sample = {
                "group": group_name, #can be used to condition the model
                "input_data": torch.from_numpy(inputs).float(),
                "labels": torch.from_numpy(labels).float()  #NOTE: the key should be named "labels" 
            }
            if self.transform:
                sample = self.transform(sample) #TODO: add transform

            return  sample
        else:
            raise NotImplementedError("The specified strategy is not implemented.")

    #maybe in this child class one can write the code for conditioning the data
        
####testing
if __name__ == "__main__":
    from ..data.fluids.incompressible import KarmanVortexStreetDataset
    
    kvs_ds = KarmanVortexStreetDataset(
        dataset_directory_path="./data/KVS/",
        mode="train",
        strategy="many2many",
        dataset_name="KVS",
        groups=["Re_100", "Re_300", "Re_400"],
        fields=["velocity", "density"],
        sequence_info=[[8, 4, 1, 1]], #input_seq_len, label_seq_len, input_sequence_stride, label_sequence_stride
        filter_frame=[[100, 500]], #min_frame, max_frame
        transform=None 

    )

    dataloader = torch.utils.data.DataLoader(kvs_ds, batch_size=32, shuffle=True)

    for batch in dataloader:
        print(batch["inputs"].shape)  # (B, ip_seq_len, C_total, H, W)
        print(batch["labels"].shape)  # (B, label_seq_len, C_total, H, W)