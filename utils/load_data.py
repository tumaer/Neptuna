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
from typing import Optional, List, Tuple
import os
import numpy as np
import torch

def create_index_map(h5file_path: str,
                    groups: Optional[list],
                    fields: Optional[list],
                    input_seq_len: int,
                    label_seq_len: int,
                    stride: int, 
                    filter_frame: Optional[list] = None #filter_frame[0]: min_frame, filter_frame[1]: max_frame
                    ) -> list:
    """
    Create index map for the dataset.
    Returns a list of (group_name, start_idx) tuples.
    stride is the same for both input and label sequences because we also want to perform autoregressive rollout
    In AR, we need to  take the last input_seq_len frames from the prediction and pass it to the model as input, so the prediction cannot have a different stride.
    """
    
    print("train_or_eval_h5_file_path:", h5file_path)
    index_map = []
    with h5py.File(h5file_path, 'r') as f:
        all_groups = list(f.keys())
        groups = groups if groups is not None else all_groups

        # Infer fields from first group
        first_group = f[groups[0]]
        available_fields = list(first_group.keys())
        fields = fields if fields is not None else available_fields

        for group_name in groups:
            group = f[group_name]
            num_samples = group[fields[0]].shape[0]  # number of timesteps

            min_frame = (filter_frame[0] - 1 if filter_frame and len(filter_frame) > 0 and filter_frame[0] is not None else 0)
            max_frame = (filter_frame[1] - 1 if filter_frame and len(filter_frame) > 1 and filter_frame[1] is not None else num_samples - 1)
            max_frame = min(max_frame, num_samples - 1)

            # # Example: input_seq_len=6, stride=2, label_seq_len=7, max_frame=50
            # # For index 21:
            # #   Input sequence: [21, 23, 25, 27, 29, 31] (6 frames with stride 2)
            # #   Label sequence: [33, 35, 37, 39, 41, 43, 45] (7 frames with stride 2)
            # #   This ensures we use all frames up to max_frame=50

            min_valid_idx = min_frame
            max_valid_idx = max_frame - (input_seq_len + label_seq_len - 1) * stride

            assert max_valid_idx >= min_valid_idx, (
                f"Group '{group_name}' cannot produce any valid window with input_seq_len={input_seq_len}, "
                f"label_seq_len={label_seq_len}, stride={stride}, max_frame={max_frame}, min_frame={min_frame}."
                f"Consider reducing either input_seq_len or label_seq_len or stride."
            )

            for i in range(min_valid_idx, max_valid_idx + 1):
                index_map.append((group_name, i))
        
        print(f"Using groups: {groups}")
        print(f"Using fields: {fields}")
    return index_map, groups, fields

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
        extra_train_sequence_info = kwargs.get("extra_train_sequence_info", None)

    index_map, groups, fields = create_index_map(
        h5file_path=h5file_path,
        groups=kwargs.get("groups"), #all groups to be used for training by default
        fields=kwargs.get("fields"), #all fields to be used for training by default
        input_seq_len=sequence_info[0], #input sequence length
        label_seq_len=sequence_info[1], #label sequence length
        stride=sequence_info[2], #stride for both input and label sequences
        filter_frame=kwargs.get("filter_frame") #filter frame
    )
    #update the kwargs with the groups and fields
    kwargs["groups"] = groups
    kwargs["fields"] = fields
    
    if mode == "test":
        # For test data, use all indices without splitting
        test_dataset = LoadedDataset(
            dataset_name=dataset_name,
            h5file_path=h5file_path,
            mode="test",
            indices=index_map,
            **kwargs
        )
        return test_dataset
    
    # For train/eval, split the indices
    np.random.seed(0) 
    # Shuffle the list directly
    np.random.shuffle(index_map)
    
    # Split indices for train and eval
    train_size = int(len(index_map) * (1-kwargs.get("eval_split_ratio")))
    train_indices = index_map[:train_size]
    eval_indices = index_map[train_size:]
    
    # Create datasets with their respective indices
    train_dataset = LoadedDataset(
        dataset_name=dataset_name,
        h5file_path=h5file_path,
        mode="train",
        indices=train_indices,
        **kwargs
    )
    
    eval_dataset = LoadedDataset(
        dataset_name=dataset_name,
        h5file_path=h5file_path,
        mode="eval",
        indices=eval_indices,
        **kwargs
    )
    
    return train_dataset, eval_dataset

class BaseDataset(Dataset):
    def __init__(self, 
                 dataset_name: str,
                 h5file_path: str,
                 mode: str, #train, test, eval
                 indices: List[Tuple[str, int]],  # List of (group_name, start_idx) tuples
                 groups: List,
                 fields: List, #specific fields inside dataset to be used for training
                 strategy: str = 'many2many', #TODO: all2all
                 sequence_info: Optional[list] = [1, 1, 1], #sequence_info[0]: input_seq_len, sequence_info[1]: label_seq_len, 
                                                           #sequence_info[2]: stride
                 transform = None, #transform: any transform to be applied to the data
                 ):
        super().__init__()
        
        assert mode in ["train", "eval", "test"]
        assert indices is not None, "indices must be provided"
        assert fields is not None and len(fields) > 0, "fields must be provided and non-empty"
        
        self.mode = mode
        self.dataset_name = dataset_name
        self.h5file_path = h5file_path
        self.transform = transform
        self.index_map = indices
        self.groups = groups
        self.fields = fields
        
        print(f"{self.mode} index map size: {len(self.index_map)}")
        
        self.input_seq_len = sequence_info[0] 
        self.label_seq_len = sequence_info[1] 
        self.stride = sequence_info[2]
        
        self.strategy = strategy    #all2all, many2many, autoregressive
        
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
                    # input_indices is a list of indices for the input sequence starting from the left
                    # for example if start_idx is 21 and input_seq_length=6 and stride=2, then input_indices = [21, 23, 25, 27, 29, 31]
                    # label_indices is a list of indices for the label sequence starting with stride
                    # for example if start_idx is 21 and label_seq_length=7 and stride=2, then label_indices = [33, 35, 37, 39, 41, 43, 45]
                    input_indices = [start_idx + i * self.stride for i in range(self.input_seq_len)]
                    label_indices = [start_idx + (self.input_seq_len + i) * self.stride for i in range(self.label_seq_len)]
                    #input_seq has shape [input_seq_len, C_field, X_res, Y_res, Z_res]: C_field=1 for scalar fields like density and pressure, C_field=3 for 3D vector-fields like velocity
                    input_seq = np.stack([group[field][i] for i in input_indices], axis=0)
                    #label_seq has shape [label_seq_len, C_field, X_res, Y_res, Z_res]: C_field=1 for scalar fields like density and pressure, C_field=3 for 3D vector-fields like velocity
                    label_seq = np.stack([group[field][i] for i in label_indices], axis=0)

                    #append the input and label sequences to the respective lists
                    input_chunks.append(input_seq)
                    label_chunks.append(label_seq)

                # inputs are the input sequences for all fields, shape = [input_seq_len, C_total, X_res, Y_res, Z_res] where: C_total = sum(C_field) for all fields
                # labels are the label sequences for all fields, shape = [label_seq_len, C_total, X_res, Y_res, Z_res]
                inputs = np.concatenate(input_chunks, axis=1)
                labels = np.concatenate(label_chunks, axis=1)

            sample = {
                "group": group_name, #can be used to condition the model
                "input_data": torch.from_numpy(inputs).float(),
                "labels": torch.from_numpy(labels).float()  #NOTE: the key should be named "labels" 
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
        groups=["Re_100", "Re_300", "Re_400"],
        fields=["velocity"],
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
        batch_size=32, 
        shuffle=True,
        num_workers=0
    )
    
    # Test a single batch
    for batch in train_loader:
        print("\nBatch shapes:")
        print(f"Input shape: {batch['input_data'].shape}")  # (B, input_seq, C_total, H, W)
        print(f"Label shape: {batch['labels'].shape}")      # (B, label_seq, C_total, H, W)
        # print(f"Group: {batch['group']}")                   # Group name for conditioning
        break  # Only test first batch