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
import random
import warnings
from utils.feature_utils import normalize_data

def build_eval_groups(filtered_groups, eval_split_ratio):
    
    # Set random seed for reproducibility of sampling groups from the middle for eval
    random.seed(42) 
    
    total_groups = len(filtered_groups)
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

    eval_groups = [filtered_groups[i] for i in extreme_indices]

    # Remaining groups to choose middle candidates from
    remaining_groups = [g for i, g in enumerate(filtered_groups) if i not in extreme_indices]

    # Randomly sample from middle
    middle_groups = remaining_groups[1:-1] if len(remaining_groups) > 2 else remaining_groups
    middle_sample = random.sample(middle_groups, min(n_middle, len(middle_groups)))

    eval_groups.extend(middle_sample)
    eval_groups = list(dict.fromkeys(eval_groups))  # ensure uniqueness
    return eval_groups

#---------------------------------------
# Helper functions for steady-state data
#---------------------------------------

def build_steady_state_index_map(group_list):
    """Return a simple list of group names for steady-state datasets."""
    return list(group_list)

def create_train_eval_steady_state_index_map(h5file_path: str,
                    filter_groups: Optional[list],
                    eval_split_ratio: Optional[float] = 0.2, #TODO: Handle the case where evel-split ratio is 0.0
                    eval_groups: Optional[list] = None,
                    ) -> list:

    print("train_or_eval_h5_file_path:", h5file_path)

    with h5py.File(h5file_path, 'r') as f:
        all_groups = sorted(list(f.keys()))
        filtered_groups = filter_groups if filter_groups is not None else all_groups
        
        if eval_groups is None:
            eval_groups = build_eval_groups(filtered_groups, eval_split_ratio)  
        
        else:
            warnings.warn("eval_split_ratio not obeyed as the evaluation groups are user-specified; using provided eval_groups instead.")
            eval_groups = eval_groups
        
        train_groups = [g for g in filtered_groups if g not in eval_groups]
        
        # --- Build both train and eval maps ---
        train_index_map = build_steady_state_index_map(train_groups)
        eval_index_map = build_steady_state_index_map(eval_groups)
        
        #Check if the train_index_map and eval_index_map are not empty
        assert len(train_index_map) > 0, "train_index_map is empty"
        assert len(eval_index_map) > 0, "eval_index_map is empty"
        
        print(f"Length of train index map: {len(train_index_map)}")
        print(f"Length of eval index map: {len(eval_index_map)}")
        
        return train_index_map, eval_index_map, filtered_groups

def create_test_steady_state_index_map(h5file_path: str,
                    filter_groups: Optional[list],
                    ) -> list:
    
    print("test_h5_file_path:", h5file_path)

    with h5py.File(h5file_path, 'r') as f:
        all_groups = sorted(list(f.keys()))
        filtered_groups = filter_groups if filter_groups is not None else all_groups

        test_index_map = build_steady_state_index_map(filtered_groups)

        print(f"Length of test index map: {len(test_index_map)}")
        return test_index_map, filtered_groups

#---------------------------------------
# Helper functions for transient data
#---------------------------------------

def build_transient_index_map(h5py_file, group_list, filter_frame, window_size):
    index_map = []
    channel_names_in_h5_file = list(h5py_file[group_list[0]].keys())
    for group_name in group_list:
        group = h5py_file[group_name]
        num_samples = group[channel_names_in_h5_file[0]].shape[0]

        min_frame = (filter_frame[0] - 1 if filter_frame and filter_frame[0] is not None else 0)
        max_frame = (filter_frame[1] - 1 if filter_frame and filter_frame[1] is not None else num_samples - 1)
        max_frame = min(max_frame, num_samples - 1)

        max_start_idx = max_frame - window_size + 1
        for start_idx in range(min_frame, max_start_idx + 1, 1):
            end_idx = start_idx + window_size - 1
            index_map.append((group_name, start_idx, end_idx))
    return index_map

def create_train_eval_transient_index_map(h5file_path: str,
                    filter_groups: Optional[list],
                    input_seq_len: int,
                    label_seq_len: int,
                    stride: int, 
                    filter_frames: Optional[list] = None, #filter_frames[0]: min_frame, filter_frame[1]: max_frame
                    n_max_pf_train_rollouts: Optional[int] = 0,
                    n_eval_rollouts: Optional[int] = 1,
                    eval_split_ratio: Optional[float] = 0.2, #TODO: Handle the case where evel-split ratio is 0.0
                    eval_groups: Optional[list] = None,
                    ) -> list:
    print("train_or_eval_h5_file_path:", h5file_path)

    with h5py.File(h5file_path, 'r') as f:
        all_groups = sorted(list(f.keys()))
        filtered_groups = filter_groups if filter_groups is not None else all_groups
        
        if eval_groups is None:
            eval_groups = build_eval_groups(filtered_groups, eval_split_ratio)  
        
        else:
            warnings.warn("eval_split_ratio not obeyed as the evaluation groups are user-specified; using provided eval_groups instead.")
            eval_groups = eval_groups
        
        train_groups = [g for g in filtered_groups if g not in eval_groups]

        #print(f"Train groups ({len(train_groups)}): {train_groups}")
        #print(f"Eval groups ({len(eval_groups)}): {eval_groups}")
        # --- Window sizes ---
        train_window_size = (input_seq_len + label_seq_len - 1 + n_max_pf_train_rollouts * label_seq_len) * stride + 1
        eval_window_size = (input_seq_len + label_seq_len - 1 + n_eval_rollouts * label_seq_len) * stride + 1

        #For example, if the input_seq_len=4, label_seq_len=3, stride=2, n_max_pf_train_rollouts=2, then the train_window_size = 25
        # as an example, if start_idx = 21, then the end_idx = 21 + 25 - 1 = 45
        # the window size is 25, so the input sequence is [21, 23, 25, 27] 
        # and the label sequence is [29, 31, 33], first pf-label indices: [35, 37, 39] and second pf-label indices: [41, 43, 45]
        # pushforward only kicks in according to the current epoch and the relative probabilities at that epoch, but we have to slice and select the labels for the max number of pf-rollouts
        
        # --- Build both train and eval maps ---
        train_index_map = build_transient_index_map(f, train_groups, filter_frames, train_window_size)
        eval_index_map = build_transient_index_map(f, eval_groups, filter_frames, eval_window_size)
        
        #Check if the train_index_map and eval_index_map are not empty
        assert len(train_index_map) > 0, "train_index_map is empty"
        assert len(eval_index_map) > 0, "eval_index_map is empty"
        
        print(f"Length of train index map: {len(train_index_map)}")
        print(f"Length of eval index map: {len(eval_index_map)}")
        
        return train_index_map, eval_index_map, filtered_groups

def create_test_transient_index_map(h5file_path: str,
                    filter_groups: Optional[list],
                    input_seq_len: int,
                    label_seq_len: int,
                    stride: int, 
                    filter_frames: Optional[list] = None,  # filter_frame[0]: min_frame, filter_frame[1]: max_frame
                    n_test_rollouts: Optional[int] = 1
                    ) -> list:
    
    print("test_h5_file_path:", h5file_path)

    with h5py.File(h5file_path, 'r') as f:
        all_groups = sorted(list(f.keys()))
        groups = filter_groups if filter_groups is not None else all_groups

        # --- Test window size ---
        test_window_size = (input_seq_len + label_seq_len - 1 + n_test_rollouts * label_seq_len) * stride + 1

        test_index_map = build_transient_index_map(f, groups, filter_frames, test_window_size)

        print(f"Length of test index map: {len(test_index_map)}")
        return test_index_map, groups
     
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
    # Determine h5 file path based on mode
    h5file_name = "test.h5" if mode == "test" else "train.h5" #use the same train.h5 for both train and eval data
    h5file_path = os.path.abspath(kwargs["dataset_directory_path"]+"/"+h5file_name)
    # Accept an optional ``sequence_info`` kwarg.  When the caller passes
    # ``None`` (or omits the key altogether) we fall back to a sensible
    # default of [1, 1, 1]. 
    sequence_info = kwargs.get("sequence_info") or [1, 1, 1]
    
    # ------------------------------------------------------------------
    # Select code path: transient (default) vs steady-state prediction
    # ------------------------------------------------------------------
    is_steady_state = bool(kwargs.get("is_steady_state_prediction"))

    if not is_steady_state:
        if mode == "train":
            #for pushforward training trick, we need to create a train index map with the extra sequence length
            if kwargs.get("pushforward_config") is not None:
                n_max_pf_train_rollouts = kwargs["pushforward_config"]["max_allowed_unroll_steps"][-1] 
            else:
                n_max_pf_train_rollouts = 0
            n_eval_rollouts = kwargs.get("n_eval_rollouts") or 0

            train_index_map, eval_index_map, all_groups = create_train_eval_transient_index_map(
                h5file_path=h5file_path,
                input_seq_len=sequence_info[0],
                label_seq_len=sequence_info[1],
                stride=sequence_info[2],
                filter_groups=kwargs["filter_groups"],
                filter_frames=kwargs["filter_frames"],
                n_max_pf_train_rollouts=n_max_pf_train_rollouts,
                n_eval_rollouts=n_eval_rollouts,
                eval_split_ratio=kwargs["eval_split_ratio"],
                eval_groups = kwargs["eval_groups"],
            )
        
            kwargs["groups"] = all_groups

            # Create datasets with their respective indices
            train_dataset = TransientDataset(
                dataset_name=dataset_name,
                h5file_path=h5file_path,
                mode="train",
                index_map=train_index_map,
                **kwargs
            )
            
            eval_dataset = TransientDataset(
                dataset_name=dataset_name,
                h5file_path=h5file_path,
                mode="eval",
                index_map=eval_index_map,
                **kwargs
            )
            
            return train_dataset, eval_dataset
        else: 
            n_test_rollouts = kwargs.get("n_test_rollouts") or 0
                
            test_index_map, all_groups = create_test_transient_index_map(
                    h5file_path=h5file_path,
                    filter_groups=kwargs["filter_groups"],
                    filter_frames=kwargs["filter_frames"],
                    input_seq_len=sequence_info[0],
                    label_seq_len=sequence_info[1],
                    stride=sequence_info[2],
                    n_test_rollouts=n_test_rollouts
                )
            #update the kwargs with the groups and channels
            kwargs["groups"] = all_groups
            
            test_dataset = TransientDataset(
                dataset_name=dataset_name,
                h5file_path=h5file_path,
                mode="test",
                index_map=test_index_map,
                **kwargs
            )
            return test_dataset
    else:
        if mode == "train":
            train_index_map, eval_index_map, all_groups = create_train_eval_steady_state_index_map(
                h5file_path=h5file_path,
                filter_groups=kwargs["filter_groups"],
                eval_split_ratio=kwargs["eval_split_ratio"],
                eval_groups = kwargs["eval_groups"],
            )
        
            kwargs["groups"] = all_groups

            # Create datasets with their respective indices
            train_dataset = SteadyStateDataset(
                dataset_name=dataset_name,
                h5file_path=h5file_path,
                mode="train",
                index_map=train_index_map,
                **kwargs
            )
            
            eval_dataset = SteadyStateDataset(
                dataset_name=dataset_name,
                h5file_path=h5file_path,
                mode="eval",
                index_map=eval_index_map,
                **kwargs
            )
            
            return train_dataset, eval_dataset
        
        else:
            test_index_map, all_groups = create_test_steady_state_index_map(
                    h5file_path=h5file_path,
                    filter_groups=kwargs["filter_groups"]
                )
            #update the kwargs with the groups and channels
            kwargs["groups"] = all_groups
            
            test_dataset = SteadyStateDataset(
                dataset_name=dataset_name,
                h5file_path=h5file_path,
                mode="test",
                index_map=test_index_map,
                **kwargs
            )
            return test_dataset

def _parse_group_name_to_params(group_name: str) -> List[object]:
    """Extract numeric/string parameter values from an HDF5 *group name*.

    Example
    -------
    "Re_100_Ma_0.05" → [100, 0.05]

    The function assumes the name is made of alternating *key* and *value*
    tokens separated by underscores (``_``).  Any number of key–value pairs is
    supported.  Each *value* token is converted to ``int`` when possible,
    otherwise ``float``, and finally left as ``str`` when neither cast
    succeeds.
    """

    tokens = group_name.split("_")
    values: List[object] = []

    # Values are every second token starting from index 1
    for i in range(1, len(tokens), 2):
        value_str = tokens[i]

        # Attempt to cast to int, then float, otherwise keep as str
        try:
            value: object = int(value_str)
        except ValueError:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str

        values.append(value)

    return values

class TransientDataset(Dataset):
    def __init__(self, 
                 dataset_name: str,
                 h5file_path: str,
                 mode: str, #train, test, eval
                 index_map: List[Tuple[str, int]],  # List of (group_name, start_idx) tuples
                 filter_in_channels: List, #specific input channels
                 conditioning_in_channels: List, #specific conditioning channels
                 filter_out_channels: List, #specific output channels
                 sequence_info: Optional[list] = [1, 1, 1], #sequence_info[0]: input_seq_len, sequence_info[1]: label_seq_len, 
                                                           #sequence_info[2]: stride
                 data_normalization_stats: dict = None,
                 data_normalization_strategy: str = None,
                 **kwargs
                 ):
        super().__init__()
        
        assert mode in ["train", "eval", "test"]
        assert index_map is not None, "index_map must be provided"
        assert filter_in_channels is not None and len(filter_in_channels) > 0, "filter_in_channels must be provided and non-empty"
        assert filter_out_channels is not None and len(filter_out_channels) > 0, "filter_out_channels must be provided and non-empty"
        
        self.mode = mode
        self.dataset_name = dataset_name
        self.h5file_path = h5file_path
        self.data_normalization_stats = data_normalization_stats
        self.data_normalization_strategy = data_normalization_strategy
        self.index_map = index_map #NOTE: this is a list of (group_name, start_idx, end_idx) tuples and depends on the train/eval/test mode
        self.input_channels = filter_in_channels
        self.conditioning_in_channels = conditioning_in_channels
        self.output_channels = filter_out_channels
        self.include_conditioning_parameters = kwargs["include_conditioning_parameters"] or False

        self.residual_config = kwargs["residual_config"]
        if self.residual_config is not None:
            assert not (self.residual_config["add_base_value"] and self.residual_config["add_predicted_value"]), "Both add_base_value and add_predicted_value cannot be true"

        if len(self.input_channels) != len(self.output_channels):
            warnings.warn("Number of input and label channels are different")
        
        #NOTE: Following assert statements ensure smoother AR rollout
        if self.conditioning_in_channels is not None:
            assert set(conditioning_in_channels).issubset(set(self.input_channels)), "conditioning_in_channels must be a subset of input_channels"
            assert not set(conditioning_in_channels).intersection(set(self.output_channels)), "conditioning_in_channels must not overlap with output_channels"
            assert set(self.output_channels + self.conditioning_in_channels) == set(self.input_channels), "For AR rollout,output_channels + conditioning_in_channels must be the same as input_channels"

        self.input_seq_len = sequence_info[0] 
        self.label_seq_len = sequence_info[1] 
        self.stride = sequence_info[2]
        
    def __len__(self):
        return len(self.index_map) #idx in __getitem__ is generated between 0 and len(self.index_map)-1

    def __getitem__(self, idx):
        #many2many train-test strategy:
        group_name, start_idx, end_idx = self.index_map[idx]
        # Build input index sequences
        input_indices = [i for i in range(start_idx, start_idx + (self.input_seq_len * self.stride), self.stride)]
        # Build label index sequences
        label_indices = [i for i in range(start_idx + (self.input_seq_len * self.stride), end_idx + 1, self.stride)]
        
        if self.residual_config is not None:
            base_input_index = input_indices[-1]
            label_indices = [base_input_index] + label_indices
        
        with h5py.File(self.h5file_path, 'r') as f:
            group = f[group_name]
            # Separate containers for normal input/label channels and conditioning channels
            input_chunks = []               # data for non-conditioning input channels
            label_chunks = []               # data for non-conditioning output channels
            conditioning_input_chunks = []  # data for conditioning input channels

            # Process input channels
            for channel in self.input_channels:
                is_conditioning = channel in self.conditioning_in_channels if self.conditioning_in_channels is not None else False
                # Resolve the dataset name and, if needed, the component index (e.g. "velocity_0" -> dataset "velocity", comp_idx 0)
                if channel in group:
                    channel_name = channel
                    component_idx = None  # use full vector/scalar stored in dataset
                else:
                    # Attempt to parse names like "velocity_0", "vorticity_1", ...
                    if "_" in channel:
                        base_name, suffix = channel.rsplit("_", 1)
                        if base_name in group and suffix.isdigit():
                            channel_name = base_name
                            component_idx = int(suffix)
                            base_name = None
                        else:
                            raise KeyError(f"Channel '{channel}' could not be resolved in the HDF5 group '{group_name}'.")
                    else:
                        raise KeyError(f"Channel '{channel}' not found in the HDF5 group '{group_name}'.")

                # Fetch data for input sequences
                if component_idx is None:
                    # Use the entire dataset slice (shape retains original channel dimension)
                    input_seq_per_channel = np.stack([group[channel_name][i] for i in input_indices], axis=0)
                else:   
                    # Extract the specific component and keep a singleton channel dim for consistency
                    input_seq_per_channel = np.stack([group[channel_name][i][component_idx:component_idx + 1] for i in input_indices], axis=0)

                # Skip normalization for mask channels
                if "mask" not in channel.lower():
                    input_seq_per_channel = normalize_data(
                        input_seq_per_channel,
                        self.data_normalization_stats[channel],
                        self.data_normalization_strategy,
                    )

                # Append to the appropriate container
                if is_conditioning:
                    conditioning_input_chunks.append(input_seq_per_channel)
                else:
                    input_chunks.append(input_seq_per_channel)
            
            # Process output channels
            for channel in self.output_channels:
                # Resolve the dataset name and, if needed, the component index (e.g. "velocity_0" -> dataset "velocity", comp_idx 0)
                if channel in group:
                    channel_name = channel
                    component_idx = None  # use full vector/scalar stored in dataset
                else:
                    # Attempt to parse names like "velocity_0", "vorticity_1", ...
                    if "_" in channel:
                        base_name, suffix = channel.rsplit("_", 1)
                        if base_name in group and suffix.isdigit():
                            channel_name = base_name
                            component_idx = int(suffix)
                            base_name = None
                        else:
                            raise KeyError(f"Channel '{channel}' could not be resolved in the HDF5 group '{group_name}'.")
                    else:
                        raise KeyError(f"Channel '{channel}' not found in the HDF5 group '{group_name}'.")

                # Fetch data for label sequences
                if component_idx is None:
                    # Use the entire dataset slice (shape retains original channel dimension)
                    if self.residual_config is None:
                        label_seq_per_channel = np.stack([group[channel_name][i] for i in label_indices], axis=0)
                    else: # here the label_indices include the final index of the input sequence at its first index
                        if self.residual_config["add_predicted_value"]: # add the previous value
                            label_seq_per_channel = np.stack([group[channel_name][label_indices[i]] - group[channel_name][label_indices[i-1]] for i in range(1, len(label_indices))], axis=0)
                        else: #add the base_value
                            label_seq_per_channel = np.stack([group[channel_name][label_indices[i]] - group[channel_name][label_indices[0]] for i in range(1, len(label_indices))], axis=0)
                else:   
                    # Extract the specific component and keep a singleton channel dim for consistency
                    if self.residual_config is None:
                        label_seq_per_channel = np.stack([group[channel_name][i][component_idx:component_idx + 1] for i in label_indices], axis=0)
                    else:
                        if self.residual_config["add_predicted_value"]:
                            label_seq_per_channel = np.stack([group[channel_name][label_indices[i]][component_idx:component_idx + 1] - group[channel_name][label_indices[i-1]][component_idx:component_idx + 1] for i in range(1, len(label_indices))], axis=0)
                        else: #add the base_value
                            label_seq_per_channel = np.stack([group[channel_name][label_indices[i]][component_idx:component_idx + 1] - group[channel_name][label_indices[0]][component_idx:component_idx + 1] for i in range(1, len(label_indices))], axis=0)

                # Select appropriate normalisation stats depending on residual mode
                norm_key = channel if self.residual_config is None else f"{channel}_residual"
                label_seq_per_channel = normalize_data(
                    label_seq_per_channel, #label_seq_per_channel is the residual of that particular channel if residual_config is not None
                    self.data_normalization_stats[norm_key],
                    self.data_normalization_strategy,
                )

                # append to non-conditioning label list
                label_chunks.append(label_seq_per_channel)

            # inputs are the input sequences for all channels, shape = [input_seq_len, C_total, X_res, Y_res, Z_res] where: C_total = sum(C_channel) for all channels
            # labels are the label sequences for all channels, shape = [label_seq_len, C_total, X_res, Y_res, Z_res]
            inputs = np.concatenate(input_chunks, axis=1)
            labels = np.concatenate(label_chunks, axis=1)
            # Build conditioning input tensor only if such channels exist
            conditioning_inputs = None
            if self.conditioning_in_channels is not None and len(conditioning_input_chunks) > 0:
                conditioning_inputs = np.concatenate(conditioning_input_chunks, axis=1)            
            
        # ------------------------------------------------------------------
        # Assemble the sample dictionary and add conditioning tensors only if
        # they are available.
        # ------------------------------------------------------------------

        sample = {
            "group": group_name,
            "input_data": torch.from_numpy(inputs).float(),
            "label_including_rollouts": torch.from_numpy(labels).float(),
        }

        if self.include_conditioning_parameters:
            sample["conditioning_parameters"] = _parse_group_name_to_params(group_name)

        if conditioning_inputs is not None:
            sample["conditioning_input_data"] = torch.from_numpy(conditioning_inputs).float()

        return  sample

class SteadyStateDataset(Dataset):
    def __init__(self, 
                 dataset_name: str,
                 h5file_path: str,
                 mode: str, #train, test, eval
                 index_map: List[str],  # List of group_names
                 filter_in_channels: List, #specific input channels
                 conditioning_in_channels: List, #specific conditioning channels
                 filter_out_channels: List, #specific output channels
                 data_normalization_stats: dict = None,
                 data_normalization_strategy: str = None,
                 **kwargs
                 ):
        super().__init__()
        #NOTE: It is possible that the self.input_channels and self.output_channels are completely different
        # For example, in the case of steady state prediction, the input_channels could be the binary mask and the output could be the steady state density and velocity 
        assert mode in ["train", "eval", "test"]
        assert index_map is not None, "index_map must be provided"
        assert filter_in_channels is not None and len(filter_in_channels) > 0, "filter_in_channels must be provided and non-empty"
        assert filter_out_channels is not None and len(filter_out_channels) > 0, "filter_out_channels must be provided and non-empty"
        
        self.mode = mode
        self.dataset_name = dataset_name
        self.h5file_path = h5file_path
        self.data_normalization_stats = data_normalization_stats
        self.data_normalization_strategy = data_normalization_strategy
        self.index_map = index_map #NOTE: this is a list of group_names
        self.input_channels = filter_in_channels
        self.conditioning_in_channels = conditioning_in_channels
        self.output_channels = filter_out_channels
        self.include_conditioning_parameters = kwargs["include_conditioning_parameters"] or False

        if len(self.input_channels) != len(self.output_channels):
            warnings.warn("Number of input and label channels are different")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        group_name = self.index_map[idx]
        with h5py.File(self.h5file_path, 'r') as f:
            group = f[group_name]
            # Separate containers for normal input/label channels and conditioning channels
            input_chunks = []               # data for non-conditioning input channels
            label_chunks = []               # data for non-conditioning output channels
            conditioning_input_chunks = []  # data for conditioning input channels

            for channel in self.input_channels:
                is_conditioning = channel in self.conditioning_in_channels if self.conditioning_in_channels is not None else False
                # Resolve the dataset name and, if needed, the component index (e.g. "velocity_0" -> dataset "velocity", comp_idx 0)
                if channel in group:
                    channel_name = channel
                    component_idx = None  # use full vector/scalar stored in dataset
                else:
                    # Attempt to parse names like "velocity_0", "vorticity_1", ...
                    if "_" in channel:
                        base_name, suffix = channel.rsplit("_", 1)
                        if base_name in group and suffix.isdigit():
                            channel_name = base_name
                            component_idx = int(suffix)
                        else:
                            raise KeyError(f"Channel '{channel}' could not be resolved in the HDF5 group '{group_name}'.")
                    else:
                        raise KeyError(f"Channel '{channel}' not found in the HDF5 group '{group_name}'.")
                     
                # For steady-state we always use the first (and only) frame
                input_data = group[channel_name][0]

                # Slice out the requested component (if any) while keeping a singleton channel dimension
                if component_idx is not None:
                    input_data = input_data[component_idx:component_idx + 1]

                # Add a leading time dimension so the final shape is [T, C, H, W, ...]
                input_seq_per_channel = input_data[np.newaxis, ...]

                # Skip normalization for mask channels
                if "mask" not in channel.lower():
                    input_seq_per_channel = normalize_data(
                        input_seq_per_channel,
                        self.data_normalization_stats[channel],
                        self.data_normalization_strategy,
                    )

                # Append to the appropriate container
                if is_conditioning:
                    conditioning_input_chunks.append(input_seq_per_channel)
                else:
                    input_chunks.append(input_seq_per_channel)

            # Process output channels
            for channel in self.output_channels:
                # Resolve the dataset name and, if needed, the component index (e.g. "velocity_0" -> dataset "velocity", comp_idx 0)
                if channel in group:
                    channel_name = channel
                    component_idx = None  # use full vector/scalar stored in dataset
                else:
                    # Attempt to parse names like "velocity_0", "vorticity_1", ...
                    if "_" in channel:
                        base_name, suffix = channel.rsplit("_", 1)
                        if base_name in group and suffix.isdigit():
                            channel_name = base_name
                            component_idx = int(suffix)
                        else:
                            raise KeyError(f"Channel '{channel}' could not be resolved in the HDF5 group '{group_name}'.")
                    else:
                        raise KeyError(f"Channel '{channel}' not found in the HDF5 group '{group_name}'.")

                # For steady-state we always fetch the first (and only) frame
                label_data = group[channel_name][0]

                # Slice out the requested component (if any) while keeping a singleton channel dimension
                if component_idx is not None:
                    label_data = label_data[component_idx:component_idx + 1]

                # Add a leading time dimension so the final shape is [T, C, H, W, ...]
                label_seq_per_channel = label_data[np.newaxis, ...]

                label_seq_per_channel = normalize_data(
                    label_seq_per_channel,
                    self.data_normalization_stats[channel],
                    self.data_normalization_strategy,
                )

                # append to non-conditioning label list
                label_chunks.append(label_seq_per_channel)
            
            inputs = np.concatenate(input_chunks, axis=1)
            labels = np.concatenate(label_chunks, axis=1)

            # Build conditioning input tensor only if such channels exist
            conditioning_inputs = None
            if self.conditioning_in_channels is not None and len(conditioning_input_chunks) > 0:
                conditioning_inputs = np.concatenate(conditioning_input_chunks, axis=1)

        sample = {
            "group": group_name,
            "input_data": torch.from_numpy(inputs).float(),
            "label_including_rollouts": torch.from_numpy(labels).float(),
        }

        if self.include_conditioning_parameters:
            sample["conditioning_parameters"] = _parse_group_name_to_params(group_name)

        if conditioning_inputs is not None:
            sample["conditioning_input_data"] = torch.from_numpy(conditioning_inputs).float()

        return  sample

####testing the dataloader
if __name__ == "__main__":
    import os
    import sys
    import torch
    import yaml
    # Add the project root directory to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_root)
    
    dataset_directory_path = "./data/synthetic/SF/2D"
    # Load the YAML configuration and extract data normalization stats
    with open("./config/data_config/synthetic/SF/2d_synthetic_data.yaml", 'r') as f:
        config = yaml.safe_load(f)
    data_normalization_stats = config['data_normalization_stats']
    # Test fetch_dataset function
    print("\nTesting fetch_dataset function...")
    train_dataset, eval_dataset = fetch_dataset(
        dataset_name="SyntheticFlow",
        mode="train",
        dataset_directory_path=dataset_directory_path,
        filter_groups=None,
        input_channels=['density', 'schlieren', 'velocity_0', 'velocity_1'],
        output_channels=['density', 'velocity_0', 'velocity_1'],
        sequence_info=[4, 3, 2],
        filter_frames=None,
        eval_split_ratio=0.2,
        eval_groups=None,
        data_normalization_stats=data_normalization_stats,
        data_normalization_strategy="z_normalization",
        max_pf_train_rollouts=2,
    )
    
    #print(f"Train dataset size: {len(train_dataset)}")
    #print(f"Eval dataset size: {len(eval_dataset)}")
    
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
        print(f"Label shape: {batch['label_including_rollouts'].shape}")      # (B, label_seq, C_total, H, W)
        # print(f"Group: {batch['group']}")                   # Group name for conditioning
        break  # Only test first batch