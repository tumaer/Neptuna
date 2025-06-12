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
2. Inside each group: Channels: Ex: ['velocity', 'pressure', 'density', 'temperature']
"""
import h5py
from torch.utils.data import Dataset
from typing import Optional, List, Tuple
import os
import numpy as np
import torch
import math
import random


def build_index_map(h5py_file, group_list, field_list, filter_frame, window_size):
    index_map = []
    for group_name in group_list:
        group = h5py_file[group_name]
        num_samples = group[field_list[0]].shape[0]

        min_frame = (filter_frame[0] - 1 if filter_frame and filter_frame[0] is not None else 0)
        max_frame = (filter_frame[1] - 1 if filter_frame and filter_frame[1] is not None else num_samples - 1)
        max_frame = min(max_frame, num_samples - 1)

        max_start_idx = max_frame - window_size + 1
        for start_idx in range(min_frame, max_start_idx + 1, 1):
            end_idx = start_idx + window_size - 1
            index_map.append((group_name, start_idx, end_idx))
    return index_map

def create_train_eval_index_map(h5file_path: str,
                    groups: Optional[list],
                    channels: Optional[list],
                    input_seq_len: int,
                    label_seq_len: int,
                    stride: int, 
                    filter_frame: Optional[list] = None, #filter_frame[0]: min_frame, filter_frame[1]: max_frame
                    n_max_pf_train_rollouts: Optional[int] = 0,
                    n_eval_rollouts: Optional[int] = 1,
                    eval_split_ratio: Optional[float] = 0.2 #TODO: Handle the case where evel-split ratio is 0.0
                    ) -> list:
     
    random.seed(42) # Set random seed for reproducibility
    
    print("train_or_eval_h5_file_path:", h5file_path)

    with h5py.File(h5file_path, 'r') as f:
        all_groups = sorted(list(f.keys()))
        groups = groups if groups is not None else all_groups 

        # Infer channels from first group
        first_group = f[groups[0]]
        available_channels = list(first_group.keys())
        channels = channels if channels is not None else available_channels

        total_groups = len(groups)
        n_eval_groups = int(round(total_groups * eval_split_ratio))

        # --- Step 1 & 2: Split eval groups 50-50 between extremes and middle ---
        n_extreme = (n_eval_groups // 2) + 1 #+1 is added for the case where n_extreme is zero
        n_middle = n_eval_groups - n_extreme  # remaining from middle

        eval_groups = []

        # Get extremes: alternate picking from start and end
        extreme_indices = []
        start_idx = 0
        end_idx = total_groups - 1
        
        while len(extreme_indices) < n_extreme and start_idx <= end_idx:
            if len(extreme_indices) < n_extreme:
                extreme_indices.append(end_idx)
                end_idx -= 1
            if len(extreme_indices) < n_extreme and start_idx <= end_idx:
                extreme_indices.append(start_idx)
                start_idx += 1

        eval_groups = [groups[i] for i in extreme_indices]

        # Remaining groups to choose middle candidates from
        remaining_groups = [g for i, g in enumerate(groups) if i not in extreme_indices]

        # Randomly sample from middle
        middle_groups = remaining_groups[1:-1] if len(remaining_groups) > 2 else remaining_groups
        middle_sample = random.sample(middle_groups, min(n_middle, len(middle_groups)))

        eval_groups.extend(middle_sample)
        eval_groups = list(dict.fromkeys(eval_groups))  # ensure uniqueness
        train_groups = [g for g in groups if g not in eval_groups]

        print(f"Train groups ({len(train_groups)}): {train_groups}")
        print(f"Eval groups ({len(eval_groups)}): {eval_groups}")

        # --- Window sizes ---
        train_window_size = (input_seq_len + label_seq_len - 1 + n_max_pf_train_rollouts * label_seq_len) * stride + 1
        eval_window_size = (input_seq_len + label_seq_len - 1 + n_eval_rollouts * label_seq_len) * stride + 1

        #For example, if the input_seq_len=4, label_seq_len=3, stride=2, n_max_pf_train_rollouts=2, then the train_window_size = 25
        # as an example, if start_idx = 21, then the end_idx = 21 + 25 - 1 = 45
        # the window size is 25, so the input sequence is [21, 23, 25, 27] 
        # and the label sequence is [29, 31, 33], first pf-label indices: [35, 37, 39] and second pf-label indices: [41, 43, 45]
        # pushforward only kicks in according to the current epoch and the relative probabilities at that epoch, but we have to slice and select the labels for the max number of pf-rollouts

        # --- Build both train and eval maps ---
        train_index_map = build_index_map(f, train_groups, channels, filter_frame, train_window_size)
        eval_index_map = build_index_map(f, eval_groups, channels, filter_frame, eval_window_size)
        print(f"Length of train index map: {len(train_index_map)}")
        print(f"Length of eval index map: {len(eval_index_map)}")
        
        return train_index_map, eval_index_map, groups, channels

def create_test_index_map(h5file_path: str,
                    groups: Optional[list],
                    channels: Optional[list],
                    input_seq_len: int,
                    label_seq_len: int,
                    stride: int, 
                    filter_frame: Optional[list] = None,  # filter_frame[0]: min_frame, filter_frame[1]: max_frame
                    n_test_rollouts: Optional[int] = 1
                    ) -> list:
    
    print("test_h5_file_path:", h5file_path)

    with h5py.File(h5file_path, 'r') as f:
        all_groups = sorted(list(f.keys()))
        groups = groups if groups is not None else all_groups 

        # Infer channels from first group
        first_group = f[groups[0]]
        available_channels = list(first_group.keys())
        channels = channels if channels is not None else available_channels

        # --- Test window size ---
        test_window_size = (input_seq_len + label_seq_len - 1 + n_test_rollouts * label_seq_len) * stride + 1

        test_index_map = build_index_map(f, groups, channels, filter_frame, test_window_size)

        print(f"Length of test index map: {len(test_index_map)}")
        return test_index_map, groups, channels
     
def fetch_dataset(dataset_name: str, 
                  mode: str = "train",  # train, eval, or test
                  **kwargs):
    """
    Factory function to create a dataset instance based on the dataset name.
    
    Args:
        dataset_name: Name of the dataset to load
        mode: One of "train", "eval", or "test". For train/eval, uses train.h5. For test, uses test.h5
        **kwargs: Additional arguments passed to the dataset constructor
    """
    if dataset_name == "KarmanVortexStreet":
        from data.fluids.incompressible import KarmanVortexStreetDataset as LoadedDataset
    elif dataset_name == "KuramotoSivashinsky":
        from data.fluids.incompressible import KuramotoSivashinskyDataset as LoadedDataset
    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented yet.")

    # Determine h5 file path based on mode
    h5file_name = "test.h5" if mode == "test" else "train.h5" #use the same train.h5 for both train and eval data
    h5file_path = os.path.abspath(kwargs["dataset_directory_path"]+"/"+h5file_name)
    sequence_info = kwargs.get("sequence_info", [1, 1, 1])
    
    if mode == "train":
        #for pushforward training trick, we need to create a train index map with the extra sequence length
        n_max_pf_train_rollouts = kwargs.get("max_pf_train_rollouts", 0) #only used for the pf trick
        n_eval_rollouts = kwargs.get("n_eval_rollouts", 0)

        train_index_map, eval_index_map, groups, channels = create_train_eval_index_map(
            h5file_path=h5file_path,
            groups=kwargs.get("groups"),
            channels=kwargs.get("channels"),
            input_seq_len=sequence_info[0],
            label_seq_len=sequence_info[1],
            stride=sequence_info[2],
            filter_frame=kwargs.get("filter_frame", None),
            n_max_pf_train_rollouts=n_max_pf_train_rollouts,
            n_eval_rollouts=n_eval_rollouts,
            eval_split_ratio=kwargs.get("eval_split_ratio")
        )
        
        kwargs["groups"] = groups
        kwargs["channels"] = channels

        # Create datasets with their respective indices
        train_dataset = LoadedDataset(
            dataset_name=dataset_name,
            h5file_path=h5file_path,
            mode="train",
            indices=train_index_map,
            **kwargs
        )
        
        eval_dataset = LoadedDataset(
            dataset_name=dataset_name,
            h5file_path=h5file_path,
            mode="eval",
            indices=eval_index_map,
            **kwargs
        )
        
        return train_dataset, eval_dataset
    #########################################################
    else: 
        n_test_rollouts = kwargs.get("n_test_rollouts", 0)
        
        test_index_map, groups, channels = create_test_index_map(
            h5file_path=h5file_path,
            groups=kwargs.get("groups"),
            channels=kwargs.get("channels"),
            input_seq_len=sequence_info[0],
            label_seq_len=sequence_info[1],
            stride=sequence_info[2],
            n_test_rollouts=n_test_rollouts
        )
        #update the kwargs with the groups and channels
        kwargs["groups"] = groups
        kwargs["channels"] = channels

        test_dataset = LoadedDataset(
            dataset_name=dataset_name,
            h5file_path=h5file_path,
            mode="test",
            indices=test_index_map,
            **kwargs
        )
        return test_dataset

class BaseDataset(Dataset):
    def __init__(self, 
                 dataset_name: str,
                 h5file_path: str,
                 mode: str, #train, test, eval
                 indices: List[Tuple[str, int]],  # List of (group_name, start_idx) tuples
                 groups: List,
                 channels: List, #specific channels inside dataset to be used for training
                 strategy: str = 'many2many', #TODO: all2all
                 sequence_info: Optional[list] = [1, 1, 1], #sequence_info[0]: input_seq_len, sequence_info[1]: label_seq_len, 
                                                           #sequence_info[2]: stride
                 transform = None, #transform: any transform to be applied to the data
                 ):
        super().__init__()
        
        assert mode in ["train", "eval", "test"]
        assert indices is not None, "indices must be provided"
        assert channels is not None and len(channels) > 0, "channels must be provided and non-empty"
        
        self.mode = mode
        self.dataset_name = dataset_name
        self.h5file_path = h5file_path
        self.transform = transform
        self.index_map = indices #NOTE: this is a list of (group_name, start_idx) tuples and depends on the train/eval/test mode
        self.groups = groups
        self.channels = channels
        
        self.input_seq_len = sequence_info[0] 
        self.label_seq_len = sequence_info[1] 
        self.stride = sequence_info[2]
        
        self.strategy = strategy    #many2many

    def __len__(self):
        return len(self.index_map) #idx in __getitem__ is generated between 0 and len(self.index_map)-1
    #train_dataset, eval_dataset, test_dataset are all of type BaseDataset and each have their own len(self.index_map).

    def __getitem__(self, idx):
        
        if self.strategy == "many2many":
            group_name, start_idx, end_idx = self.index_map[idx]
            with h5py.File(self.h5file_path, 'r') as f:
                group = f[group_name]
                input_chunks = []
                label_chunks = []

                for channel in self.channels:
                    # input_indices is a list of indices for the input sequence starting from the left
                    # for example if start_idx is 21 and input_seq_length=6 and stride=2, then input_indices = [21, 23, 25, 27, 29, 31]
                    # label_indices is a list of indices for the label sequence starting with stride
                    # for example if start_idx is 21 and label_seq_length=7 and stride=2, then label_indices = [33, 35, 37, 39, 41, 43, 45]
                    input_indices = [i for i in range(start_idx, start_idx + (self.input_seq_len * self.stride), self.stride)]
                    #label_indices = [start_idx + (self.input_seq_len + i) * self.stride for i in range(self.label_seq_len)]
                    label_indices = [i for i in range(start_idx + (self.input_seq_len * self.stride), end_idx+1, self.stride)]

                    #input_seq_per_channel has shape [input_seq_len, C_channel, X_res, Y_res, Z_res]: C_channel=1 for scalar channels like density and pressure, C_channel=3 for 3D vector-channels like velocity
                    input_seq_per_channel = np.stack([group[channel][i] for i in input_indices], axis=0)
                    #label_seq_per_channel has shape [label_seq_len, C_channel, X_res, Y_res, Z_res]: C_channel=1 for scalar channels like density and pressure, C_channel=3 for 3D vector-channels like velocity
                    label_seq_per_channel = np.stack([group[channel][i] for i in label_indices], axis=0)

                    #append the input and label sequences to the respective lists
                    input_chunks.append(input_seq_per_channel)
                    label_chunks.append(label_seq_per_channel)

                # inputs are the input sequences for all channels, shape = [input_seq_len, C_total, X_res, Y_res, Z_res] where: C_total = sum(C_channel) for all channels
                # labels are the label sequences for all channels, shape = [label_seq_len, C_total, X_res, Y_res, Z_res]
                inputs = np.concatenate(input_chunks, axis=1)
                labels = np.concatenate(label_chunks, axis=1)

            sample = {
                "group": group_name, #can be used to condition the model
                "input_data": torch.from_numpy(inputs).float(),
                "label_including_rollouts": torch.from_numpy(labels).float()
            }
        
        else:
            raise NotImplementedError("The specified strategy/mode is not implemented.")       
        
        if self.transform:
            sample = self.transform(sample) #TODO: add transform

        return  sample
    #maybe in the child class of BaseDataset one can write the code for conditioning the data
        
####testing the dataloader
if __name__ == "__main__":
    import os
    import sys
    import torch
    
    # Add the project root directory to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_root)
    
    dataset_directory_path = "./data/fluids/KVS/2D"
    
    # Test fetch_dataset function
    print("\nTesting fetch_dataset function...")
    train_dataset, eval_dataset = fetch_dataset(
        dataset_name="KarmanVortexStreet",
        mode="train",
        dataset_directory_path=dataset_directory_path,
        groups=["Re_100", "Re_200", "Re_300", "Re_400", "Re_500"],
        channels=["velocity"],
        sequence_info=[8, 4, 1],
        filter_frame=[100, 500],
        eval_split_ratio=0.2
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Test dataloader
    print("\nTesting dataloader...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=False,
        num_workers=0
    )
    
    # Test a single batch
    for batch in train_loader:
        print("\nBatch shapes:")
        print(f"Input shape: {batch['input_data'].shape}")  # (B, input_seq, C_total, H, W)
        print(f"Label shape: {batch['labels'].shape}")      # (B, label_seq, C_total, H, W)
        # print(f"Group: {batch['group']}")                   # Group name for conditioning
        break  # Only test first batch