"""
HDF5 Dataset Loading and Management.

This module provides comprehensive dataset loading capabilities for HDF5-based
datasets, with specialized support for both transient (time-series) and steady-state
data. It includes flexible data splitting, normalization, channel filtering, and
support for various training strategies including pushforward learning and residual
modeling.

Key Features:
- Dual dataset types: TransientDataset for time-series data, SteadyStateDataset for steady-state
- Flexible train/eval/test splitting with configurable ratios and group selection
- Flexible channel filtering with support for multi-component fields
- Comprehensive data normalization with multiple strategies
- Residual learning support with configurable modes
- Conditioning parameter extraction from group names
- Data loading with configurable sequence lengths and strides
- Pushforward training support for improved temporal modeling

Dataset Types:
    TransientDataset: Handles time-series data with configurable sequence lengths,
                     strides, and rollout capabilities for autoregressive modeling
    SteadyStateDataset: Manages single-frame steady-state data for direct input-output
                       mapping problems

Data Organization:
    HDF5 Structure:
    - Groups: Named containers (e.g., "Re_100_Ma_0.05") with parameter information
    - Datasets: Named arrays within groups (e.g., "velocity", "pressure", "density")
    - Components: Multi-component fields expanded to individual channels ("velocity_0", "velocity_1")

Splitting Strategies:
    - Random sampling: Uniform random selection of groups for evaluation
    - Extreme sampling: Biased selection toward parameter space extremes (not used in the current implementation)
    - User-defined: Explicit specification of evaluation groups

Main Functions:
    fetch_dataset: Primary factory function for dataset creation
    build_eval_groups: Random evaluation group selection
    build_eval_groups_extremes: Extreme-biased evaluation group selection
    create_train_eval_transient_index_map: Index mapping for transient data splitting
    create_train_eval_steady_state_index_map: Index mapping for steady-state data splitting

Configuration Options:
    Data Filtering:
    - filter_groups: Specify which HDF5 groups to include
    - filter_frames: Temporal range filtering for transient data
    - filter_in_channels/filter_out_channels: Channel selection
    - conditioning_in_channels: Channels that remain constant during rollouts
    
    Normalization:
    - data_normalization_stats: Pre-computed statistics for normalization
    - data_normalization_strategy: "z_normalization", "min_max_normalization", robust_normalization.
    
    Training Strategies:
    - pushforward_config: Configuration for pushforward training
    - residual_config: Settings for residual learning modes
    - n_eval_rollouts/n_test_rollouts: Rollout lengths for evaluation/testing

Notes:
    Channel Naming Convention:
    - Single-component fields: Use dataset name directly (e.g., "pressure")
    - Multi-component fields: Append component index (e.g., "velocity_0", "velocity_1")
    
    Residual Learning Modes:
    - add_predicted_value_with_diff_loss: Predict frame-to-frame differences (residual) and compute loss on the difference
    - add_predicted_value_with_raw_loss: Add previous predicted value to the difference (residual) and compute loss on the raw data
    - add_base_value_with_raw_loss: Add base value to predicted difference (residuals) and compute loss on the raw data
    
"""
import h5py
from torch.utils.data import Dataset
from typing import Optional, List, Tuple
import os
import numpy as np
import torch
import random
import warnings
from utils.compute_stats import normalize_data

#NOTE: This function is not used in the current implementation.
def build_eval_groups_extremes(filtered_groups, eval_split_ratio):
    """
    Select evaluation groups biased toward extremes and middle samples.

    The function first determines the target number of evaluation groups as
    ``round(len(filtered_groups) * eval_split_ratio)``.  It then splits this
    quota approximately 50-50 between *extreme* and *middle* groups:

    1. *Extreme* groups are taken alternately from the end (largest index) and
       the beginning (smallest index) of ``filtered_groups`` until the extreme
       quota is reached.
    2. The remaining quota is filled by uniform random sampling (without
       replacement) from the groups that lie between the extremes.

    The random seed is fixed to 42 so that repeated runs with the same inputs
    yield identical splits.

    Parameters
    ----------
    filtered_groups : list
        Sequence of group names remaining after any filtering and already
        sorted in the desired order (e.g. by Reynolds number).
    eval_split_ratio : float
        Fraction of groups to include in the evaluation split (0 ≤ ratio ≤ 1).

    Returns
    -------
    list
        The combined list of extreme and middle group names to be used for
        evaluation.
        
    Notes
    -----
    This function is designed for certain datasets where parameter space
    extremes are particularly important for model evaluation. The extreme
    selection helps ensure the model is tested on boundary conditions and
    edge cases that may be critical for certain applications.
    
    The function is currently unused in the main implementation but provided
    for specialized evaluation strategies.
    """
    
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

# ---------------------------------------------------------------------------
# random sampling of evaluation groups
# ---------------------------------------------------------------------------
def build_eval_groups(filtered_groups, eval_split_ratio):
    """
    Randomly pick evaluation groups based on *eval_split_ratio*.

    This simple strategy samples groups uniformly at random (without
    replacement). The sampling is seeded for reproducibility so that repeated
    runs with the same inputs yield the same split.

    Parameters
    ----------
    filtered_groups : list
        Sequence of group names remaining after any filtering.
    eval_split_ratio : float
        Fraction of groups to include in the evaluation split (0 ≤ ratio ≤ 1).

    Returns
    -------
    list
        Randomly-selected evaluation group names.
        
    Notes
    -----
    Uses a fixed random seed (42) to ensure reproducible splits across
    multiple runs. This is the default evaluation group selection strategy
    used throughout the codebase.
    
    If the computed number of evaluation groups rounds to 0, raise an error.
    """

    # Reuse the global seed for deterministic behaviour.
    random.seed(42)

    total_groups = len(filtered_groups)
    n_eval_groups = int(round(total_groups * eval_split_ratio))

    # Guard against corner cases where the ratio rounds to 0.
    if n_eval_groups <= 0:
        raise ValueError(f"Number of evaluation groups is 0. Please increase the eval_split_ratio: {eval_split_ratio}")

    return random.sample(filtered_groups, n_eval_groups)

#---------------------------------------
# Helper functions for steady-state data
#---------------------------------------

def build_steady_state_index_map(group_list):
    """
    Return a simple list of group names for steady-state datasets.
    
    This function creates an index map for steady-state datasets where each
    group contains a single data point. The index map is simply a list of
    group names that can be used to access data during training.
    
    Parameters
    ----------
    group_list : list
        List of group names to include in the index map.
        
    Returns
    -------
    list
        List of group names (identical to input for steady-state case).
        
    Notes
    -----
    This function exists for consistency with the transient dataset interface
    where index maps are more complex, containing (group, start_idx, end_idx)
    tuples for temporal sequences.
    """
    return list(group_list)

def create_train_eval_steady_state_index_map(h5file_path: str,
                    filter_groups: Optional[list],
                    eval_split_ratio: Optional[float] = 0.2, #TODO: Handle the case where evel-split ratio is 0.0
                    eval_groups: Optional[list] = None,
                    ) -> list:
    """
    Create train and evaluation index maps for steady-state datasets.
    
    This function splits the available groups in an HDF5 file into training
    and evaluation sets for steady-state prediction tasks. Each group
    represents a single data point with different parameter configurations.
    
    Parameters
    ----------
    h5file_path : str
        Path to the HDF5 file containing the dataset.
    filter_groups : list, optional
        List of group names to include. If None, all groups are used.
    eval_split_ratio : float, default=0.2
        Fraction of groups to use for evaluation (0 ≤ ratio ≤ 1).
    eval_groups : list, optional
        Explicit list of groups to use for evaluation. If provided,
        eval_split_ratio is ignored.
        
    Returns
    -------
    tuple
        A tuple containing:
        - train_index_map (list): Group names for training
        - eval_index_map (list): Group names for validation  
        - filtered_groups (list): All groups considered after filtering
        
    Raises
    ------
    AssertionError
        If either train or evaluation index maps are empty.
        
    Notes
    -----
    When eval_groups is provided, a warning is issued that eval_split_ratio
    is being ignored. This allows for reproducible evaluation sets across
    different experiments.
    
    The function prints diagnostic information about the dataset size and
    split ratios for monitoring purposes.
    """

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
        
        #print(f"Length of train index map: {len(train_index_map)}")
        #print(f"Length of eval index map: {len(eval_index_map)}")
        
        return train_index_map, eval_index_map, filtered_groups

def create_test_steady_state_index_map(h5file_path: str,
                    filter_groups: Optional[list],
                    ) -> list:
    """
    Create test index map for steady-state datasets.
    
    This function creates an index map for test datasets in steady-state
    prediction tasks. Unlike train/eval splitting, test data uses all
    available groups (after filtering) for comprehensive evaluation.
    
    Parameters
    ----------
    h5file_path : str
        Path to the HDF5 file containing the test dataset.
    filter_groups : list, optional
        List of group names to include. If None, all groups are used.
        
    Returns
    -------
    tuple
        A tuple containing:
        - test_index_map (list): Group names for testing
        - filtered_groups (list): All groups considered after filtering
        
    """
    
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
    """
    Build index map for transient datasets with temporal windowing.
    
    This function creates an index map for transient (time-series) datasets
    by generating all possible temporal windows within the specified frame
    range for each group. 
    
    Parameters
    ----------
    h5py_file : h5py.File
        Open HDF5 file object containing the dataset.
    group_list : list
        List of group names to process.
    filter_frame : list or None
        Two-element list [min_frame, max_frame] specifying the temporal
        range to consider. If None, uses the full temporal range.
    window_size : int
        Size of the temporal window needed for input and label sequences.
        
    Returns
    -------
    list
        List of tuples (group_name, start_idx, end_idx) representing
        all valid temporal windows across all groups.
        
    Notes
    -----
    The function automatically handles:
    - Frame range validation to prevent out-of-bounds access
    - Window size constraints to ensure sufficient data
    - Sliding window generation with unit stride
    
    Window size calculation should account for:
    - Input sequence length
    - Label sequence length  
    - Stride between temporal samples
    - Number of rollout steps for pushforward training
    """
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
                    filter_frames: Optional[list] = None, 
                    n_max_pf_train_rollouts: Optional[int] = 0,
                    n_eval_rollouts: Optional[int] = 1,
                    eval_split_ratio: Optional[float] = 0.2, #TODO: Handle the case where evel-split ratio is 0.0
                    eval_groups: Optional[list] = None,
                    ) -> list:
    """
    Create train and validation index maps for transient datasets.
    
    This function splits transient (time-series) data into training and
    validation sets, creating index maps that define all valid temporal
    sequences for each split. It handles different window sizes for
    training and validation to accommodate pushforward trick while training and a different rollout length during validation.
    
    Parameters
    ----------
    h5file_path : str
        Path to the HDF5 file containing the dataset.
    filter_groups : list, optional
        List of group names to include. If None, all groups are used.
    input_seq_len : int
        Length of input sequences for the model.
    label_seq_len : int
        Length of label sequences for the model.
    stride : int
        Stride between consecutive temporal samples in sequences.
    filter_frames : list, optional
        Two-element list [min_frame, max_frame] for temporal filtering.
        If None, uses the full temporal range available.
    n_max_pf_train_rollouts : int, default=0
        Maximum number of pushforward rollouts during training.
        Increases training window size to accommodate longer sequences.
    n_eval_rollouts : int, default=1
        Number of rollouts to perform during validation.
    eval_split_ratio : float, default=0.2
        Fraction of groups to use for validation (0 ≤ ratio ≤ 1).
    eval_groups : list, optional
        Explicit list of groups for validation. If provided,
        eval_split_ratio is ignored.
        
    Returns
    -------
    tuple
        A tuple containing:
        - train_index_map (list): Training sequence indices as (group, start, end) tuples
        - eval_index_map (list): Evaluation sequence indices as (group, start, end) tuples
        - filtered_groups (list): All groups considered after filtering
        
    Raises
    ------
    AssertionError
        If either train or validation index maps are empty.
    """
    print("train_or_eval_h5_file_path:", h5file_path)

    with h5py.File(h5file_path, 'r') as f:
        all_groups = sorted(list(f.keys()))
        filtered_groups = filter_groups if filter_groups is not None else all_groups
        
        if eval_groups is None:
            eval_groups = build_eval_groups(filtered_groups, eval_split_ratio)  
        
        else:
            warnings.warn("eval_split_ratio not obeyed as the evaluation groups are user-specified; using provided eval_groups instead.")
            assert set(eval_groups).issubset(set(filtered_groups)), "eval_groups are not a subset of the filtered_groups"
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
        
        #print(f"Length of train index map: {len(train_index_map)}")
        #print(f"Length of eval index map: {len(eval_index_map)}")
        
        return train_index_map, eval_index_map, filtered_groups

def create_test_transient_index_map(h5file_path: str,
                    filter_groups: Optional[list],
                    input_seq_len: int,
                    label_seq_len: int,
                    stride: int, 
                    filter_frames: Optional[list] = None,  # filter_frame[0]: min_frame, filter_frame[1]: max_frame
                    n_test_rollouts: Optional[int] = 1
                    ) -> list:
    """
    Create test index map for transient datasets.
    
    This function creates an index map for test datasets in transient
    prediction tasks. It generates all valid temporal sequences within
    the specified constraints for comprehensive model evaluation.
    
    Parameters
    ----------
    h5file_path : str
        Path to the HDF5 file containing the test dataset.
    filter_groups : list, optional
        List of group names to include. If None, all groups are used.
    input_seq_len : int
        Length of input sequences for the model.
    label_seq_len : int
        Length of label sequences for the model.
    stride : int
        Stride between consecutive temporal samples in sequences.
    filter_frames : list, optional
        Two-element list [min_frame, max_frame] for temporal filtering.
    n_test_rollouts : int, default=1
        Number of rollouts to perform during testing.
        
    Returns
    -------
    tuple
        A tuple containing:
        - test_index_map (list): Test sequence indices as (group, start, end) tuples
        - filtered_groups (list): All groups considered after filtering
    """
    
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
    Factory function to create dataset instances based on configuration.
    
    This is the primary entry point for dataset creation, handling both
    transient and steady-state datasets with comprehensive configuration
    options.
    
    Parameters
    ----------
    dataset_name : str
        Name identifier for the dataset (used for logging and identification).
    mode : str, default="train"
        Dataset mode: "train" returns (train_ds, eval_ds), "test" returns test_ds.
    **kwargs : dict
        Comprehensive configuration options including:
        
        Required:
        - dataset_directory_path : str
            Path to directory containing HDF5 files
        - filter_in_channels : list
            Input channel names to load
        - filter_out_channels : list  
            Output channel names to load
        - data_normalization_stats : dict
            Pre-computed normalization statistics
        - data_normalization_strategy : str
            Normalization method ("z_normalization", "min_max_normalization", etc.)
            
        Optional:
        - is_steady_state_prediction : bool, default=False
            Whether to use steady-state (True) or transient (False) dataset
        - sequence_info : list, default=[1, 1, 1]
            [input_seq_len, label_seq_len, stride] for transient data
        - filter_groups : list, optional
            Specific groups to include
        - filter_frames : list, optional
            Temporal range [min_frame, max_frame] for transient data
        - eval_split_ratio : float, default=0.2
            Fraction of data for validation
        - eval_groups : list, optional
            Explicit validation groups
        - conditioning_in_channels : list, optional
            Channels that do not get predicted (used for conditioning the model)
        - pushforward_config : dict, optional
            Configuration for pushforward basedtraining
        - residual_config : dict, optional
            Configuration for residual learning
        - n_eval_rollouts : int, optional
            Number of rollouts to be performed during validation
        - n_test_rollouts : int, optional
            Number of rollouts to be performed during testing
        - include_conditioning_parameters : bool, default=False
            Whether to extract conditioning parameters from group names
    
    Returns
    -------
    Dataset or tuple
        - For mode="train": tuple of (train_dataset, eval_dataset)
        - For mode="test": single test_dataset
        
    Raises
    ------
    AssertionError
        If required parameters are missing or invalid.
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
            # For pushforward training we enlarge the training window only when the
            # feature is actually enabled. The extended windows are created with the max_allowed_unroll_stepsbefore training. The toggle lives inside
            # pushforward_config["enabled"].
            pf_cfg = kwargs.get("pushforward_config")
            if pf_cfg is not None and bool(pf_cfg.get("enabled", True)):
                n_max_pf_train_rollouts = pf_cfg["max_allowed_unroll_steps"][-1]
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
    """
    Extract numeric/string parameter values from an HDF5 *group name*.

    This function parses group names that follow a key-value naming convention
    to extract parameter values for conditioning or analysis. It handles
    automatic type conversion for numeric parameters.

    Parameters
    ----------
    group_name : str
        Group name following the pattern "key1_value1_key2_value2_..."
        
    Returns
    -------
    List[object]
        List of parameter values with automatic type conversion:
        - int if the value can be parsed as integer
        - float if the value can be parsed as float
        - str if neither numeric conversion succeeds
        
    Examples
    --------
    >>> _parse_group_name_to_params("Re_100_Ma_0.05")
    [100, 0.05]
    >>> 
    
    Notes
    -----
    The function assumes group names follow the pattern:
    "key1_value1_key2_value2_..." where keys and values alternate.
    
    Type conversion priority:
    1. int() - for integer values
    2. float() - for floating-point values  
    3. str - for non-numeric values (fallback)
    
    This is commonly used for extracting physical parameters like Reynolds
    numbers, Mach numbers, geometric parameters, etc., from simulation
    group names for conditioning or analysis purposes.
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
    """
    PyTorch Dataset for transient (time-series) scientific data.
    
    This dataset handles time-series data stored in HDF5 format, providing
    flexible temporal sequence extraction, multi-channel support, and
    advanced features for scientific machine learning including residual
    learning, pushforward training, and conditioning parameter extraction.
    
    Parameters
    ----------
    dataset_name : str
        Identifier for the dataset (used for logging and debugging).
    h5file_path : str
        Path to the HDF5 file containing the data.
    mode : str
        Dataset mode: "train", "eval", or "test".
    index_map : List[Tuple[str, int, int]]
        List of (group_name, start_idx, end_idx) tuples defining valid
        temporal sequences for this dataset split.
    filter_in_channels : List[str]
        Input channel names to load from the HDF5 file.
    conditioning_in_channels : List[str]
        Channels that do not get predicted (used for conditioning the model)
    filter_out_channels : List[str]
        Output channel names to load from the HDF5 file.
    sequence_info : List[int], default=[1, 1, 1]
        [input_seq_len, label_seq_len, stride] configuration.
    data_normalization_stats : dict
        Pre-computed statistics for data normalization.
    data_normalization_strategy : str
        Normalization method ("z_normalization", "min_max_normalization", "robust_normalization", "none").
    **kwargs : dict
        Additional configuration options including:
        - residual_config: Configuration for residual learning modes
        - include_conditioning_parameters: Whether to extract parameters from group names
        
    Attributes
    ----------
    mode : str
        Dataset mode (train/eval/test).
    dataset_name : str
        Dataset identifier.
    h5file_path : str
        Path to HDF5 data file.
    index_map : List[Tuple[str, int, int]]
        Temporal sequence definitions.
    input_channels : List[str]
        Input channel names.
    output_channels : List[str]
        Output channel names.
    conditioning_in_channels : List[str]
        Conditioning channel names.
    input_seq_len : int
        Length of input sequences.
    label_seq_len : int
        Length of output sequences.
    stride : int
        Temporal stride for sequence sampling.
    residual_config : dict
        Residual learning configuration.
    include_conditioning_parameters : bool
        Whether to extract conditioning parameters.
        
    Methods
    -------
    __len__()
        Return the number of temporal sequences in the dataset.
    __getitem__(idx)
        Load and return a single temporal sequence sample.
        
    Notes
    -----
    Channel Naming:
    - Single-component fields: Use dataset name directly (e.g., "pressure")
    - Multi-component fields: Use "fieldname_component" (e.g., "velocity_0")
    
    Sample Structure:
    Each sample returned by __getitem__ contains:
    - "group": Group name for identification
    - "input_data": Input sequences [T_in, C_in, H, W, ...]
    - "label_including_rollouts": Output sequences [T_out, C_out, H, W, ...]
    - "conditioning_input_data": Conditioning sequences (if applicable)
    - "conditioning_parameters": Extracted parameters (if enabled)
    """
    
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
        """
        Initialize the TransientDataset.
        
        Sets up the dataset with comprehensive validation of parameters and
        configuration of temporal sequence handling, channel management,
        and normalization strategies.
        
        Raises
        ------
        AssertionError
            If required parameters are missing or invalid configurations are detected.
        UserWarning
            If input and output channel counts differ (may be intentional).
        """
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
            assert not (self.residual_config["add_base_value_with_raw_loss"] and self.residual_config["add_predicted_value_with_diff_loss"] and self.residual_config["add_predicted_value_with_raw_loss"]), "Only one can be true at a time, currently more than one is true"
            #all 3 of them should not be false at the same time.
            assert (self.residual_config["add_base_value_with_raw_loss"] or self.residual_config["add_predicted_value_with_diff_loss"] or self.residual_config["add_predicted_value_with_raw_loss"]), "Only one can be true at a time, currently all are false, set the residual_config to null in the data_config"
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
        """
        Return the number of temporal sequences in the dataset.
        
        Returns
        -------
        int
            Number of valid temporal sequences available for training/validation/testing.
        """
        return len(self.index_map) #idx in __getitem__ is generated between 0 and len(self.index_map)-1

    def __getitem__(self, idx):
        """
        Load and return a single temporal sequence sample.
        
        This method handles the complete data loading pipeline including:
        1. Temporal sequence extraction based on index map
        2. Multi-component field handling and channel expansion
        3. Residual computation (if configured)
        4. Data normalization
        5. Conditioning parameter extraction
        
        Parameters
        ----------
        idx : int
            Index of the temporal sequence to load (0 <= idx < len(dataset)).
            
        Returns
        -------
        dict
            Sample dictionary containing:
            - "group" : str
                Group name for identification and conditioning
            - "input_data" : torch.Tensor
                Input sequences with shape [T_in, C_in, H, W, ...]
            - "label_including_rollouts" : torch.Tensor
                Output sequences with shape [T_out, C_out, H, W, ...]
            - "conditioning_input_data" : torch.Tensor, optional
                Conditioning sequences with shape [T_in, C_cond, H, W, ...]
            - "conditioning_parameters" : List[object], optional
                Extracted parameters from group name
                
        Notes
        -----
        Temporal Sequence Construction:
        - Input indices: [start_idx, start_idx + stride, ..., start_idx + (input_seq_len-1)*stride]
        - Label indices: [start_idx + input_seq_len*stride, ..., end_idx] with stride
        
        Residual Handling:
        - For residual modes, label indices include the final input frame as base
        - Temporal differences computed as: label[t] - label[t-1]
        - Normalization uses either raw or residual statistics based on mode
        
        Channel Processing:
        - Handles both single-component like density and multi-component channels like velocity
        - Automatic component extraction for multi-component channels (e.g., "velocity_0")
        - Separate handling of conditioning vs. regular input channels
        - Mask channels skip normalization to preserve binary values
        
        The method ensures all data is properly normalized and formatted for
        PyTorch training pipelines.
        """
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
                        if self.residual_config["add_predicted_value_with_diff_loss"]: # add the previous value
                            label_seq_per_channel = np.stack([group[channel_name][label_indices[i]] - group[channel_name][label_indices[i-1]] for i in range(1, len(label_indices))], axis=0)
                        elif self.residual_config["add_predicted_value_with_raw_loss"]:
                            label_seq_per_channel = np.stack([group[channel_name][label_indices[i]] for i in range(1, len(label_indices))], axis=0)
                        else: #add_base_value_with_raw_loss
                            label_seq_per_channel = np.stack([group[channel_name][label_indices[i]] for i in range(1, len(label_indices))], axis=0)
                else:   
                    # Extract the specific component and keep a singleton channel dim for consistency
                    if self.residual_config is None:
                        label_seq_per_channel = np.stack([group[channel_name][i][component_idx:component_idx + 1] for i in label_indices], axis=0)
                    else:
                        if self.residual_config["add_predicted_value_with_diff_loss"]:
                            label_seq_per_channel = np.stack([group[channel_name][label_indices[i]][component_idx:component_idx + 1] - group[channel_name][label_indices[i-1]][component_idx:component_idx + 1] for i in range(1, len(label_indices))], axis=0)
                        elif self.residual_config["add_predicted_value_with_raw_loss"]:
                            label_seq_per_channel = np.stack([group[channel_name][label_indices[i]][component_idx:component_idx + 1] for i in range(1, len(label_indices))], axis=0)
                        else: #add_base_value_with_raw_loss
                            label_seq_per_channel = np.stack([group[channel_name][label_indices[i]][component_idx:component_idx + 1] for i in range(1, len(label_indices))], axis=0)

                # Select appropriate normalisation stats depending on residual mode
                norm_key = channel if ((self.residual_config is None) or (self.residual_config["add_base_value_with_raw_loss"]) or (self.residual_config["add_predicted_value_with_raw_loss"])) else f"{channel}_residual"
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
    """
    PyTorch Dataset for steady-state scientific data.
    
    This dataset handles steady-state (single time-step) data stored in HDF5
    format, designed for direct input-output mapping problems such as
    geometry-to-flow prediction, parameter-to-solution mapping, and other
    non-temporal scientific computing applications.
    
    The dataset provides:
    - Single-frame data loading with multi-channel support
    - Multi-component field handling with automatic channel expansion
    - Comprehensive data normalization
    - Conditioning parameter extraction from group names
    - Flexible input-output channel mapping
    
    Parameters
    ----------
    dataset_name : str
        Identifier for the dataset (used for logging and debugging).
    h5file_path : str
        Path to the HDF5 file containing the data.
    mode : str
        Dataset mode: "train", "eval", or "test".
    index_map : List[str]
        List of group names defining the dataset split.
    filter_in_channels : List[str]
        Input channel names to load from the HDF5 file.
    conditioning_in_channels : List[str]
        Subset of input channels used for conditioning. NOTE: conditioning channels are not predicted
    filter_out_channels : List[str]
        Output channel names to load from the HDF5 file.
    data_normalization_stats : dict
        Pre-computed statistics for data normalization.
    data_normalization_strategy : str
        Normalization method ("z_normalization", "min_max_normalization", etc.).
    **kwargs : dict
        Additional configuration options including:
        - include_conditioning_parameters: Whether to extract parameters from group names
        
    Attributes
    ----------
    mode : str
        Dataset mode (train/eval/test).
    dataset_name : str
        Dataset identifier.
    h5file_path : str
        Path to HDF5 data file.
    index_map : List[str]
        Group names for this dataset split.
    input_channels : List[str]
        Input channel names.
    output_channels : List[str]
        Output channel names.
    conditioning_in_channels : List[str]
        Conditioning channel names.
    include_conditioning_parameters : bool
        Whether to extract conditioning parameters.
        
    Methods
    -------
    __len__()
        Return the number of groups in the dataset.
    __getitem__(idx)
        Load and return a single steady-state sample.
        
    Notes
    -----
    Data Structure:
    - Each group contains steady-state data (single time frame)
    - Input and output channels can be completely different
    - Common use case: geometry/boundary conditions → flow solution
    
    Sample Structure:
    Each sample returned by __getitem__ contains:
    - "group": Group name for identification
    - "input_data": Input data [1, C_in, H, W, ...]
    - "label_including_rollouts": Output data [1, C_out, H, W, ...]
    - "conditioning_input_data": Conditioning data (if applicable)
    - "conditioning_parameters": Extracted parameters (if enabled)
    
    The leading dimension of 1 maintains consistency with the transient
    dataset interface where the first dimension represents time steps.
    """
    
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
        """
        Initialize the SteadyStateDataset.
        
        Sets up the dataset with validation of parameters and configuration
        of channel management and normalization strategies for steady-state
        data processing.
        
        Raises
        ------
        AssertionError
            If required parameters are missing or invalid.
        UserWarning
            If input and output channel counts differ (often expected for steady-state).
        """
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
        """
        Return the number of groups in the dataset.
        
        Returns
        -------
        int
            Number of groups available for training/evaluation.
        """
        return len(self.index_map)

    def __getitem__(self, idx):
        """
        Load and return a single steady-state sample.
        
        This method handles the complete data loading pipeline for steady-state
        data including:
        1. Group selection based on index map
        2. Single-frame data extraction (first time step)
        3. Multi-component field handling and channel expansion
        4. Data normalization
        5. Conditioning parameter extraction
        
        Parameters
        ----------
        idx : int
            Index of the group to load (0 <= idx < len(dataset)).
            
        Returns
        -------
        dict
            Sample dictionary containing:
            - "group" : str
                Group name for identification and conditioning
            - "input_data" : torch.Tensor
                Input data with shape [1, C_in, H, W, ...]
            - "label_including_rollouts" : torch.Tensor
                Output data with shape [1, C_out, H, W, ...]
            - "conditioning_input_data" : torch.Tensor, optional
                Conditioning data with shape [1, C_cond, H, W, ...]
            - "conditioning_parameters" : List[object], optional
                Extracted parameters from group name
                
        Notes
        -----
        Data Processing:
        - Always uses the first (index 0) time frame from each dataset
        - Adds leading time dimension for consistency with transient interface
        - Handles both single-component and multi-component fields
        - Automatic component extraction for multi-component channels (e.g., "velocity_0")
        - Separate handling of conditioning vs. regular input channels
        - Mask channels skip normalization to preserve binary values
        """
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