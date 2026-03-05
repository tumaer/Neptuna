
import torch
from torch import nn
from typing import List, Optional, Dict, Tuple, Union, Any
from transformers.trainer import *
from transformers import Trainer as Trainer_
import numpy as np
from utils.compute_stats import re_normalize_data, normalize_data
from utils.load_data import fetch_dataset
from utils.custom_callbacks import (
    WandbCallback,
    CallbackHandler,
    #DefaultFlowCallback,
    LossStatisticsCallback,
    AdaptiveWeightCallback,
)  # custom callbacks
from transformers.trainer_callback import DefaultFlowCallback as DefaultFlowCallback_
from transformers.integrations.integration_utils import WandbCallback as WandbCallback_
from utils.trainer_utils import EvalPrediction
from utils.loss_utils import (
    fetch_loss_metric,
    get_loss_weighting_strategy_entry,
    create_loss_weighting_strategy,
)
from omegaconf import OmegaConf, ListConfig
from collections.abc import Mapping  # locally import to avoid top-of-file change
import json
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"           
import h5py
import time



def compute_curriculum_start_epochs(train_strategy_config) -> list:
    """
    Compute and validate the list of curriculum start epochs from the strategy config.
    """
    if train_strategy_config is None:
        raise ValueError("train_strategy_config is required to compute start epochs.")

    curriculum_cfg = getattr(train_strategy_config, "curriculum", None)
    if curriculum_cfg is None:
        raise ValueError("train_strategy_config.curriculum is required to compute start epochs.")

    curriculum_change_epochs = []

    for idx, block in enumerate(curriculum_cfg):
        start_epoch = block.get("start_epoch", None)
        if start_epoch is None:
            raise ValueError(
                f"start_epoch missing in train_strategy_config.curriculum[{idx}]"
            )
        curriculum_change_epochs.append(start_epoch)

    if not curriculum_change_epochs:
        raise ValueError("train_strategy_config.curriculum must contain at least one block.")

    #num_train_epochs = None
    num_train_epochs = train_strategy_config.get("num_train_epochs")
    if num_train_epochs is None:
        raise ValueError("train_strategy_config.num_train_epochs is required to compute start epochs.")

    if curriculum_change_epochs[-1] > num_train_epochs:
        raise ValueError(
            "last element of curriculum_change_epochs is greater than num_train_epochs"
        )
    #Append the num_train_epochs to the list of curriculum change epochs, to avoid updating the loss weights at the last epoch.
    curriculum_change_epochs.append(num_train_epochs)
    return curriculum_change_epochs


class Trainer(Trainer_):
    """    
    This class extends the HuggingFace Trainer with specialized features.
    
    Parameters
    ----------
    model_config : Dict
        Model configuration dictionary containing architecture parameters.
    data_config : Dict
        Data configuration dictionary with dataset and preprocessing parameters.
    train_config : Dict
        Training configuration dictionary with optimization and strategy parameters.
    scheduler_config : Dict
        Scheduler configuration dictionary with learning rate scheduling parameters.
    infer_config : Dict
        Inference configuration dictionary with inference parameters.
    output_log_config : Dict
        Output and logging configuration dictionary.
    **kwargs
        Additional keyword arguments passed to the base Trainer class.
        
    Attributes
    ----------
    eval_or_test_rollout_steps : Optional[int]
        Number of rollout steps for validation/testing.
    output_all_steps : bool
        Whether to output predictions during validation/testing.
    original_label_seq_len : int
        Original sequence length for labels without rollout.
    get_prediction_loss_for_eval_windows : bool
        Whether to compute prediction loss for validation/testing windows.
    residual_config : Dict
        Configuration for residual learning strategies.
    """
    def __init__(self, **kwargs):
        
        self.eval_or_test_rollout_steps = None
        self.output_all_steps = False
        self.data_config = kwargs.pop("data_config", None)
        self.model_config = kwargs.pop("model_config", None)
        self.train_config = kwargs.pop("train_config", None)
        self.scheduler_config = kwargs.pop("scheduler_config", None)
        self.infer_config = kwargs.pop("infer_config", None)
        self.output_log_config = kwargs.pop("output_log_config", None)
        self.train_strategy_config = kwargs.pop("train_strategy_config", None) #each entry of the list has a range of epoch for which the train strategy and loss metric is active.

        # Pre-extract curriculum start epochs for convenience
        self.curriculum_start_epochs = compute_curriculum_start_epochs(self.train_strategy_config)

        super().__init__(**kwargs)

        #self.original_label_seq_len = self.data_config.sequence_info[1] #number of predicted timesteps from the model (#no rollout timesteps considered)
        
        self.get_prediction_loss_for_eval_windows = False #TODO: Find a way to not hardcode this.
        self.num_epochs_between_eval = max(
            1,
            int(
                self.train_strategy_config.get(
                    "num_epochs_between_eval",
                    self.train_config.get("num_epochs_between_eval", 1),
                )
            ),
        )

        self.residual_config = self.data_config["residual_config"]

        # ------------------------------------------------------------------
        # Precompute conditioning flags and a fast model-forward function to
        # avoid repeated if/else branches during every forward call.
        # ------------------------------------------------------------------
        conditioning_features = self.data_config["conditioning_features"]
        self._use_cond_input_data = conditioning_features["conditioning_in_channels"] is not None
        self._use_cond_parameters = False if conditioning_features["conditioning_method"] is None else True

        def _build_model_forward_fn(use_cond_input, use_cond_params):
            if use_cond_input and use_cond_params:
                return lambda m, i: m(
                    input_data=i["input_data"],
                    conditioning_input_data=i["conditioning_input_data"],
                    conditioning_parameters=i["conditioning_parameters"],
                )
            elif use_cond_params:
                return lambda m, i: m(
                    input_data=i["input_data"],
                    conditioning_parameters=i["conditioning_parameters"],
                )
            elif use_cond_input:
                return lambda m, i: m(
                    input_data=i["input_data"],
                    conditioning_input_data=i["conditioning_input_data"],
                )
            else:
                return lambda m, i: m(input_data=i["input_data"])

        #pre-build a partial fuction that builds the model forward function based on the conditioning flags.
        self._model_forward_fn = _build_model_forward_fn(
            self._use_cond_input_data, self._use_cond_parameters
        )

        # Capture any callbacks created by the base Trainer, replace the HF default flow callback
        # with our custom DefaultFlowCallback, then re-instantiate using our CallbackHandler.
        existing_callbacks = list(getattr(self.callback_handler, "callbacks", []) or [])
        filtered_callbacks = [
            cb for cb in existing_callbacks if not isinstance(cb, DefaultFlowCallback_)
        ]
        filtered_callbacks.append(DefaultFlowCallback())

        self.callback_handler = CallbackHandler(
            filtered_callbacks,
            self.model,
            getattr(self, "tokenizer", None),
            self.optimizer,
            self.lr_scheduler,
        )
        # Initialize loss function from the first train strategy block.
        self.loss_fn = self.get_loss_fn(self.train_strategy_config.curriculum[0].train_loss)

        # Flag to indicate if detailed losses should be collected for adaptive weighting
        self._collect_detailed_losses = getattr(self, '_collect_detailed_losses', False)

        # Flag to indicate if component-wise gradient norms should be collected for adaptive weighting
        self._collect_gradients = getattr(self, '_collect_gradients', False)
        self._grad_stat_names = getattr(self, '_grad_stat_names', [])

        self._weight_per_channel = self.train_strategy_config.curriculum[0].train_loss.train_loss_weighting_strategy.get("weight_per_channel", False)
        self._weight_sub_components = self.train_strategy_config.curriculum[0].train_loss.train_loss_weighting_strategy.get("weight_sub_components", False)
        self._loss_history_interval = self.train_strategy_config.curriculum[0].train_loss.train_loss_weighting_strategy.get("loss_history_interval", 1)
        self._grad_history_interval = self.train_strategy_config.curriculum[0].train_loss.train_loss_weighting_strategy.get("grad_history_interval", 1)
        
        # Configuration for gradient statistics computation
        self._grad_stats_last_layer_only = self.train_strategy_config.curriculum[0].train_loss.train_loss_weighting_strategy.get("grad_stats_last_layer_only", False)
        self._grad_stats_layer_pattern = self.train_strategy_config.curriculum[0].train_loss.train_loss_weighting_strategy.get("grad_stats_layer_pattern", None)
        self._grad_stats_num_last_params = self.train_strategy_config.curriculum[0].train_loss.train_loss_weighting_strategy.get("grad_stats_num_last_params", 2)

        # Inject a reference to this Trainer into all registered callbacks so they can
        # access training context (datasets, model, args, etc.).
        try:
            for cb in getattr(self.callback_handler, "callbacks", []) or []:
                setattr(cb, "trainer", self)
        except Exception:
            pass

        if self.output_log_config is not None and self.output_log_config["logging"]["wandb"]:
            self.callback_handler.remove_callback(WandbCallback_)
            self.add_callback(WandbCallback())
            # Ensure the newly added callback also gets a trainer reference
            try:
                for cb in getattr(self.callback_handler, "callbacks", []) or []:
                    setattr(cb, "trainer", self)
            except Exception:
                pass
    
    ##custom function, not inside transformers library
    def get_loss_fn(self, loss_config):
        """
        Build a training loss function from a train_loss config block and place
        it on the appropriate device.
        """
        #self.initial_loss_config = train_loss_config
        if loss_config is not None:
            loss_fn = fetch_loss_metric(self.data_config, loss_config)
            logger.info("Initialized composite train loss function from config")

            device = (
                self.args.device
                if hasattr(self.args, "device")
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            return loss_fn.to(device)

        logger.warning("No loss_config provided, using default MSE loss") # ? Is this true during only_inference??
        return None

    #custom function, not inside transformers library
    def _get_train_strategy_block_for_epoch(
        self,
        epoch: Optional[int] = None,
    ):
        """
        Return the curriculum block active for the given epoch.

        Parameters
        ----------
        epoch : Optional[int]
            Epoch index to resolve. Defaults to the current trainer state epoch.
        train_strategy_config : Optional[dict]
            Strategy configuration; defaults to ``self.train_strategy_config``.

        Returns
        -------
        Any
            The selected curriculum block.
        """
        cfg = self.train_strategy_config
        if cfg is None:
            raise ValueError("train_strategy_config is required to resolve the strategy block.")

        # Extract curriculum in a way that works for both OmegaConf objects and plain dicts.
        curriculum = getattr(cfg, "curriculum", None)
        if curriculum is None and isinstance(cfg, Mapping):
            curriculum = cfg.get("curriculum")

        if not curriculum:
            raise ValueError("train_strategy_config.curriculum is missing or empty.")

        # Use current epoch if not provided.
        current_epoch = int(epoch if epoch is not None else getattr(self.state, "epoch", 0) or 0)

        def _start_epoch(block) -> int:
            # Support both Mapping (dict) and OmegaConf/DotMap-like objects.
            if isinstance(block, Mapping):
                return int(block.get("start_epoch", 0))
            return int(getattr(block, "start_epoch", 0))

        # Sort blocks by start_epoch to pick the latest block that has started.
        sorted_blocks = sorted(curriculum, key=_start_epoch)
        selected = sorted_blocks[0]
        for block in sorted_blocks:
            if current_epoch >= _start_epoch(block):
                selected = block
            else:
                break

        return selected

    #custom function, not inside transformers library
    def _compute_raw_prediction(self, prediction: torch.Tensor, base_value: torch.Tensor):
        """
        Compute raw predictions from residual predictions using vectorized operations.

        Parameters
        ----------
        prediction : torch.Tensor
            Shape (B, T, C, *spatial_dims). Prediction is the residual of the channel values.
        base_value : torch.Tensor
            Shape (B, 1, C, *spatial_dims). Last known physical state used as an additive baseline.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            raw_prediction : The residual-augmented prediction with the same shape as *prediction*.
            new_base_value : The latest timestep of *raw_prediction* (when add_predicted_value=True) or the
                              unchanged *base_value* (when add_base_value=True). This is fed into subsequent
                              autoregressive steps.
        """
        #prediction.shape = torch.Size([B, label_seq_length, C_labels, x_resolution, y_resolution, ...])
        #base_value.shape = torch.Size([B, 1, C_labels, x_resolution, y_resolution, ...]) which is the last time step of the input data
        if self.residual_config["add_predicted_value_with_diff_loss"] or self.residual_config["add_predicted_value_with_raw_loss"]:
            # Cumulative sum along temporal axis replicates the iterative loop:
            # for i in range(self.data_config.sequence_info[1]):
            #     raw_prediction.append(prediction[:,i:i+1,:,:,:] + base_value)
            #     base_value = raw_prediction[-1]
            # raw_prediction = torch.cat(raw_prediction, dim=1)
            # Assume the base_value is x_0 and the prediction is dx1 (x1-x0), dx2 (x2-x1), dx3 (x3-x2)
            #step 1: cumsum along the temporal axis: dx1 (x1-x0), dx1+dx2 (x2-x0), dx1+dx2+dx3 (x3-x0)
            #step 2: add the base_value (x0) to the cumulative sum to obtain: x1, x2, x3. 
            # The + operator, broadcasts the base_value (B, 1) to the shape of the cumulative sum (B, label_seq_length).
            raw_prediction = prediction.cumsum(dim=1) + base_value
            new_base_value = raw_prediction[:, -1:]
        else:  # add_base_value == True (add_predicted_value is False)
            # for i in range(self.data_config.sequence_info[1]):
            #     raw_prediction.append(prediction[:,i:i+1,:,:,:] + base_value)
            # raw_prediction = torch.cat(raw_prediction, dim=1)
            # Assume the base_value is x_0 (B, 1 , ...)and the prediction is dx1 (x1-x0), dx2 (x2-x0), dx3 (x3-x0) of shape (B, label_seq_length, ...).
            # step 1: add the base_value, x_0 to the prediction: dx1+x0, dx2+x0, dx3+x0 resulting in x1, x2, x3 of shape (B, label_seq_length, ...).
            # The + operator, broadcasts the base_value (B, 1) to the shape of the prediction (B, label_seq_length).
            raw_prediction = prediction + base_value
            new_base_value = base_value
        return raw_prediction, new_base_value
    
    #custom function, not inside transformers library
    def _rebuild_datasets(self):
        """
        Recreate training and evaluation datasets with updated hyperparameters.
        
        This method is called during hyperparameter search to rebuild datasets
        with the new configuration parameters after a trial update.
        """
        self.train_dataset, self.eval_dataset = fetch_dataset(
            dataset_name=self.data_config["dataset_name"],
            dataset_directory_path=self.data_config["dataset_directory_path"],
            sequence_info=self.data_config["sequence_info"],
            train_filter_frames=self.data_config["filter_features"]["train_filter_frames"],
            train_filter_groups=self.data_config["filter_features"]["train_filter_groups"],
            infer_filter_frames=self.data_config["filter_features"]["infer_filter_frames"],
            infer_filter_groups=self.data_config["filter_features"]["infer_filter_groups"],
            filter_in_channels=self.data_config["filter_features"]["filter_in_channels"],
            conditioning_in_channels=self.data_config["conditioning_features"]["conditioning_in_channels"],
            include_conditioning_parameters=self.data_config["conditioning_features"]["include_conditioning_parameters"],
            parameter_min_max_stats=self.data_config["conditioning_features"]["parameter_min_max_stats"],
            filter_out_channels=self.data_config["filter_features"]["filter_out_channels"],
            data_normalization_stats=self.data_config["data_normalization_stats"],
            data_normalization_strategy=self.data_config["data_normalization_strategy"],
            eval_split_ratio=self.train_config["eval_split_ratio"] if self.train_config is not None else None,
            eval_groups=self.data_config["eval_groups"],
            is_steady_state_prediction=self.data_config["is_steady_state_prediction"],
            residual_config=self.data_config["residual_config"],
            n_eval_rollouts=self.train_config["n_eval_rollouts"] if self.train_config is not None else None,
            n_infer_rollouts=self.infer_config["n_infer_rollouts"] if self.infer_config is not None else None,
        )
        
    ##overrides the one in the  base class from transformers library
    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Optional[Callable[[Dataset], torch.utils.data.Sampler]] = None,
        is_training: bool = False,
        dataloader_key: Optional[str] = None,
    ) -> DataLoader:
        """Create a [`~torch.utils.data.DataLoader`] from the given dataset."""

        data_collator = self.data_collator
        ## NOTE:commented out code from the base class
        # if is_datasets_available() and isinstance(dataset, datasets.Dataset):
        #     dataset = self._remove_unused_columns(dataset, description=description)
        # else:
        #     data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )

        dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

        # Store the prepared dataloader for subsequent evaluations if using persistent workers.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return dataloader

    ##custom function, not inside transformers library
    def _forward_model_train(self, model, inputs):
        """
        Forward pass for training.

        Parameters
        ----------
        model : torch.nn.Module
            The model to forward.
        inputs : Dict[str, torch.Tensor]
            Input tensors including data and optional conditioning inputs.

        Returns
        -------
        Tuple[torch.Tensor, int]
            prediction : Model predictions with shape (B, T, C, *spatial_dims).
        """
        batch_size, _, _, *spatial_dims = inputs["input_data"].shape
        base_value = inputs["input_data"][:,-1:,]
            
        prediction = self._model_forward_fn(model, inputs)
        
        prediction = prediction.reshape(batch_size, self.data_config["sequence_info"][1], len(self.data_config["filter_features"]["filter_out_channels"]), *spatial_dims)
        
        if self.residual_config is not None and (self.residual_config["add_predicted_value_with_raw_loss"] or self.residual_config["add_base_value_with_raw_loss"]):
            #NOTE: For the cases: add_predicted_value_with_raw_loss or add_base_value_with_raw_loss, we need the raw values before loss is computed inside compute_loss().
            raw_prediction, base_value = self._compute_raw_prediction(prediction, base_value)
            prediction = raw_prediction

        return prediction

    ##overrides the one in the base class from transformers library
    # ! heavily modified compared to the base class
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): 
        """
        Compute training loss.

        Parameters
        ----------
        model : torch.nn.Module
            The model to compute loss for.
        inputs : Dict[str, torch.Tensor]
            Input tensors and labels.
        return_outputs : bool, default=False
            Whether to return model outputs along with loss.
        num_items_in_batch : Optional[int]
            Number of items in the batch.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Loss tensor, or tuple of (loss, outputs) if return_outputs=True.
        """
        #return_outputs is true only when doing eval or test. By default it is false for training.
        ########################################################
        prediction = self._forward_model_train(model, inputs)
        
        # Get labels for the current rollout step
        labels = inputs["label_including_rollouts"][:,0:self.data_config.sequence_info[1]]

        input_frames = inputs["input_data"]
        
        # Check if we should collect detailed losses at this step
        should_collect_losses = (
            self._collect_detailed_losses and 
            self.state.global_step % self._loss_history_interval == 0
        )
        
        
        if not should_collect_losses:
            loss = self.loss_fn(
                model=model,
                predictions=prediction,
                labels=labels,
                input_frames=input_frames,
                return_detailed=False
            )
        else: #for adaptive weighting of train_loss components
            # Check if we should collect gradients at this step
            should_collect_grads = (
                self._collect_gradients and 
                self.state.global_step % self._grad_history_interval == 0
            )

            loss, detailed = self.loss_fn(
                model=model,
                predictions=prediction,
                labels=labels,
                input_frames=input_frames,
                return_detailed=True,
                preserve_component_grads=should_collect_grads
            )

            if not hasattr(self, '_detailed_loss_accumulator'):
                self._detailed_loss_accumulator = {}

            if should_collect_grads and not hasattr(self, '_gradient_accumulator'):
                self._gradient_accumulator = {}
            
            for component_name, component_detailed in detailed.items():
                # Extract loss value
                loss_value = component_detailed.get('total', component_detailed)
                
                # Register gradient hooks if needed (only when collecting gradients)
                if should_collect_grads and loss_value.requires_grad:
                    loss_value.retain_grad()
                    self._register_gradient_hook(component_name, loss_value)

                # Convert to detached GPU tensor
                loss_scalar = loss_value.detach()
                
                # Accumulate training loss for total
                if component_name not in self._detailed_loss_accumulator:
                    self._detailed_loss_accumulator[component_name] = []
                self._detailed_loss_accumulator[component_name].append(loss_scalar)
                
                # Optionally accumulate per-channel losses
                if self._weight_per_channel and 'per_channel' in component_detailed:
                    per_channel_value = component_detailed['per_channel']
                    
                    # Iterate over each channel index
                    for ch_idx in range(per_channel_value.numel()):
                        ch_scalar = per_channel_value[ch_idx]
                        per_channel_key = f"{component_name}/channel_{ch_idx}"
                        
                        if per_channel_key not in self._detailed_loss_accumulator:
                            self._detailed_loss_accumulator[per_channel_key] = []
                        
                        # Accumulate the detached scalar
                        per_channel_detached = ch_scalar.detach()
                        self._detailed_loss_accumulator[per_channel_key].append(per_channel_detached)
                        
                        # Register gradient hooks for each channel scalar if needed
                        if should_collect_grads and ch_scalar.requires_grad:
                            ch_scalar.retain_grad()
                            self._register_gradient_hook(per_channel_key, ch_scalar)
                    
                    # Optionally accumulate per-component losses
                    if self._weight_sub_components and 'per_component' in component_detailed:
                        per_component_dict = component_detailed['per_component']
                        for sub_component_name, sub_component_value in per_component_dict.items():
                            sub_component_key = f"{component_name}/{sub_component_name}"
                            if sub_component_key not in self._detailed_loss_accumulator:
                                self._detailed_loss_accumulator[sub_component_key] = []
                            
                            sub_component_detached = sub_component_value.detach() if torch.is_tensor(sub_component_value) else torch.tensor(float(sub_component_value), device=loss.device)
                            self._detailed_loss_accumulator[sub_component_key].append(sub_component_detached)
                            
                            # Register gradient hooks for per-component if needed
                            if should_collect_grads and sub_component_value.requires_grad:
                                sub_component_value.retain_grad()
                                self._register_gradient_hook(sub_component_key, sub_component_value)
            
        return (loss, prediction) if return_outputs else loss

    ##custom function, not inside transformers library
    def _register_gradient_hook(self, component_name: str, component_loss: torch.Tensor):
        """
        Store component loss for later gradient computation w.r.t. model parameters.
        """
        # Store the component loss (don't hook it yet)
        if not hasattr(self, '_component_losses_for_grad'):
            self._component_losses_for_grad = {}
        self._component_losses_for_grad[component_name] = component_loss
    
    ##custom function, not inside transformers library
    def _get_params_for_grad_stats(self, model):
        """
        Get the parameters to use for gradient statistics computation.
        
        Returns an iterable of (name, param) tuples based on configuration:
        - If grad_stats_last_layer_only is False: returns all parameters
        - If grad_stats_layer_pattern is specified: returns parameters matching the pattern
        - Otherwise: returns the last N parameters (default N=2 for weight and bias of final layer)
        
        Parameters
        ----------
        model : torch.nn.Module
            The model whose parameters to filter.
            
        Returns
        -------
        List[Tuple[str, torch.nn.Parameter]]
            List of (name, parameter) tuples to use for gradient statistics.
        """
        all_params = list(model.named_parameters())
        
        if not self._grad_stats_last_layer_only:
            # Use all parameters
            return all_params
        
        if self._grad_stats_layer_pattern is not None:
            # Filter by pattern (e.g., "output", "head", "final")
            pattern = self._grad_stats_layer_pattern.lower()
            filtered = [(name, param) for name, param in all_params if pattern in name.lower()]
            if filtered:
                return filtered
            else:
                logger.warning(
                    f"No parameters matched pattern '{self._grad_stats_layer_pattern}'. "
                    f"Falling back to last {self._grad_stats_num_last_params} parameters."
                )
        
        # Use last N parameters
        n = min(self._grad_stats_num_last_params, len(all_params))
        return all_params[-n:]

    def _compute_component_gradient_stats(self):
        """
        Compute gradient statistics of each component w.r.t. model parameters.
        Stats are controlled by self._grad_stat_names (e.g., ["norm", "var", "max"]).
        """
        if not hasattr(self, '_component_losses_for_grad'):
            return

        if not getattr(self, "_grad_stat_names", None):
            return

        if not hasattr(self, '_gradient_accumulator'):
            self._gradient_accumulator = {}

        stat_names = set(self._grad_stat_names)
        model = self.model

        # Get filtered parameters for gradient statistics
        params_for_stats = self._get_params_for_grad_stats(model)
        
        # Store current gradients if any exist
        saved_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                saved_grads[name] = param.grad.clone()

        for component_name, component_loss in self._component_losses_for_grad.items():
            # Zero gradients
            model.zero_grad(set_to_none=True)

            # Compute gradient of this component w.r.t. parameters
            component_loss.backward(retain_graph=True)

            # Accumulators
            total_norm_sq = 0.0
            max_abs = 0.0
            sum_vals = 0.0
            sum_sq = 0.0
            count = 0

            # Only iterate over filtered parameters for statistics
            for name, param in params_for_stats:
                if param.grad is None:
                    continue
                g = param.grad.detach()

                if "norm" in stat_names:
                    # Use sum of squares to avoid extra sqrt each param
                    total_norm_sq += float(g.pow(2).sum().item())

                if "max" in stat_names:
                    max_abs = max(max_abs, float(g.abs().max().item()))

                if "var" in stat_names:
                    sum_vals += float(g.sum().item())
                    sum_sq += float(g.pow(2).sum().item())
                    count += g.numel()

            # Prepare component entry
            if component_name not in self._gradient_accumulator:
                self._gradient_accumulator[component_name] = {}

            device = component_loss.device

            if "norm" in stat_names:
                grad_norm = total_norm_sq ** 0.5
                self._gradient_accumulator[component_name].setdefault("norm", []).append(
                    torch.tensor(grad_norm, device=device)
                )

            if "max" in stat_names:
                self._gradient_accumulator[component_name].setdefault("max", []).append(
                    torch.tensor(max_abs, device=device)
                )

            if "var" in stat_names:
                if count > 0:
                    mean = sum_vals / count
                    var = max((sum_sq / count) - (mean ** 2), 0.0)
                else:
                    var = 0.0
                self._gradient_accumulator[component_name].setdefault("var", []).append(
                    torch.tensor(var, device=device)
                )

        # Restore previous gradients
        model.zero_grad(set_to_none=True)
        for name, param in model.named_parameters():
            if name in saved_grads:
                param.grad = saved_grads[name]

        # Clear stored component losses
        self._component_losses_for_grad = {}


    # Overriden from the base class in transformers library
    def training_step(
        self, 
        model: nn.Module, 
        inputs: dict[str, Union[torch.Tensor, Any]], 
        num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Prepare buffers for context parallelism

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            #loss here is the CompositeLoss scalar, which initiates the backward pass.
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            should_collect_grads = (
                self._collect_gradients and 
                self.state.global_step % self._grad_history_interval == 0
            )

            # Added: (Not in base class) compute component gradient stats if needed
            if should_collect_grads:
                self._compute_component_gradient_stats()

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                if is_torch_xpu_available():
                    torch.xpu.empty_cache()
                elif is_torch_mlu_available():
                    torch.mlu.empty_cache()
                elif is_torch_musa_available():
                    torch.musa.empty_cache()
                elif is_torch_npu_available():
                    torch.npu.empty_cache()
                elif is_torch_mps_available():
                    torch.mps.empty_cache()
                elif is_torch_hpu_available():
                    logger.warning(
                        "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                    )
                else:
                    torch.cuda.empty_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learning rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                from apex import amp

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
                if (
                    not self.model_accepts_loss_kwargs or num_items_in_batch is None
                ) and self.compute_loss_func is None:
                    # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                    loss = loss / self.current_gradient_accumulation_steps

                # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
                # https://github.com/huggingface/transformers/pull/35808
                if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                    kwargs["scale_wrt_gas"] = False

                self.accelerator.backward(loss, **kwargs)

            return loss.detach()


    ##custom function, not inside transformers library
    def _forward_model_eval_or_test(self, model, inputs):
        """
        Forward pass for evaluation or testing.

        Parameters
        ----------
        model : torch.nn.Module
            The model to forward.
        inputs : Dict[str, torch.Tensor]
            Input tensors including data and optional conditioning inputs.

        Returns
        -------
        torch.Tensor
            Model predictions with shape (B, T, C, *spatial_dims).
        """
        batch_size, _, _, *spatial_dims = inputs["input_data"].shape
        
        prediction = self._model_forward_fn(model, inputs)
        
        prediction = prediction.reshape(batch_size, self.data_config["sequence_info"][1], len(self.data_config["filter_features"]["filter_out_channels"]), *spatial_dims)
        return prediction
    
    #NOTE: There are two functions for eval_loss (both not inside transformers library): 
    # 1) compute_eval_loss, one which computes the mse loss of each window of the batch, takes the mean across the batch and assigns the same scalar value to all entries of the batch. 
    # 2) compute_eval_without_loss, no window loss is computed, the predictions are accumulated as the batches get processed and loss is computed inside compute_metrics inside run.py
    def compute_eval_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute evaluation loss with autoregressive rollout and window-based loss computation.

        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate.
        inputs : Dict[str, torch.Tensor]
            Input tensors and labels.
        return_outputs : bool, default=False
            Whether to return model outputs along with loss.
        num_items_in_batch : Optional[int]
            Number of items in the batch.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Loss tensor, or tuple of (loss, outputs) if return_outputs=True.
        """
        #########################################################
        #Autoregressive prediction (for eval and test)
        #########################################################
        # loss is computed per window for a batch of windows.
        if self.output_all_steps: #this is set to true when self.rollout_steps is set in main.py
            losses_ = []
            predictions_ = []
        else:
            total_loss = 0
        
        prediction = self._forward_model_eval_or_test(model,inputs) 

        # loss_fn = nn.functional.mse_loss   #this is the eval_loss, which is NOT used for saving the best model.
        if self.residual_config is None:
            loss = self.loss_fn(
                model=model,
                predictions=prediction,
                labels=inputs["label_including_rollouts"][:,0:self.data_config.sequence_info[1]],
                return_detailed=False
                ) 
        
        else:
            if self.residual_config["add_predicted_value_with_diff_loss"]:
                #here the labels are the residuals.
                base_value = inputs["input_data"][:,-1:,]
                raw_prediction, base_value = self._compute_raw_prediction(prediction, base_value)
                #raw predictions are needed for continuing the autoregressive rollout.
                loss = self.loss_fn(
                    model=model,
                    predictions=prediction, 
                    labels=inputs["label_including_rollouts"][:,0:self.data_config.sequence_info[1]],
                    return_detailed=False
                    ) 

            if (self.residual_config["add_predicted_value_with_raw_loss"] or self.residual_config["add_base_value_with_raw_loss"]):
                base_value = inputs["input_data"][:,-1:,]
                raw_prediction, base_value = self._compute_raw_prediction(prediction, base_value)
                #here the labels are the raw values. raw_predictions are needed for both loss computation and for continuing the autoregressive rollout.
                loss = self.loss_fn(
                    model=model,
                    predictions=raw_prediction, 
                    labels=inputs["label_including_rollouts"][:,0:self.data_config.sequence_info[1]],
                    return_detailed=False
                    )

        if self.output_all_steps:            
            if self.residual_config is None:
                # raw values are added  
                predictions_.append(prediction.detach())
            elif self.residual_config is not None and self.residual_config["add_predicted_value_with_diff_loss"]:
                # predictions correspond to the difference and they are accumulated for the metrics.
                predictions_.append(prediction.detach())
            elif self.residual_config is not None and self.residual_config["add_predicted_value_with_raw_loss"]:
                # raw_predictions are accumulated for the metrics as the loss is computed with the raw values.
                predictions_.append(raw_prediction.detach())
            else: #add_base_value_with_raw_loss
                # raw_predictions are accumulated for the metrics as the loss is computed with the raw values.
                predictions_.append(raw_prediction.detach())

            #predictions_.append(raw_prediction.detach() if self.residual_config is not None else prediction.detach()) 
            losses_.append(loss)
        else:
            total_loss += loss #loss is added up across all rollout_steps and we obtain a scalar. This is divided by rollout_steps at the end of the "if" statement

        if not self.data_config.is_steady_state_prediction:
        
            for i in range(1,self.rollout_steps+1): 
                #logger.debug(f"Eval/Test rollout step {i+1} of {self.rollout_steps+1}")
                #prediction.shape = torch.Size([B, label_seq_length,C_labels, x_resolution, y_resolution, ...]) 
                #recreate the inputs to be fed to the model for the next step
                if (self.data_config.sequence_info[1] >= self.data_config.sequence_info[0]): #label_sequence length > input_sequence length
                    inputs = {
                        **inputs,
                        **{ #this part replaces the "input_data" of input with the output of the model. 
                            #So the new input is the output from the previous step.
                            "input_data": (
                                prediction[:,(self.data_config.sequence_info[1] - self.data_config.sequence_info[0]):,].detach() #slice the predictions so as to extract the input_sequence.
                            ) if self.residual_config is None else (
                                raw_prediction[:,(self.data_config.sequence_info[1] - self.data_config.sequence_info[0]):,].detach() #slice the predictions so as to extract the input_sequence.
                            )
                        },
                }
                else: #input_sequence length > label_sequence length (the more usual case)
                    inputs = {
                        **inputs,
                        **{ #this part replaces the "input_data" of input with the output of the model. 
                            #So the new input is the output from the previous step.
                            "input_data": ( 
                                torch.cat([inputs["input_data"][:,self.data_config.sequence_info[1]:,], prediction.detach()], dim=1) #slice a part of the input_data so as to extract the input_sequence.
                            ) if self.residual_config is None else (
                                torch.cat([inputs["input_data"][:,self.data_config.sequence_info[1]:,], raw_prediction.detach()], dim=1) #slice a part of the input_data so as to extract the input_sequence.
                            )
                        },
                }
                
                prediction = self._forward_model_eval_or_test(model,inputs) 

                #loss = loss_fn(prediction, inputs["label_including_rollouts"][:,i*self.data_config.sequence_info[1]:(i+1)*self.data_config.sequence_info[1]]) 
                #if self.residual_config is not None:
                #    raw_prediction, base_value = self._compute_raw_prediction(prediction, base_value)

                if self.residual_config is None:
                    #here the predictions are the raw values and also the corresponding labels. 
                    loss = self.loss_fn(
                        model=model,
                        predictions=prediction, 
                        labels=inputs["label_including_rollouts"][:,i*self.data_config.sequence_info[1]:(i+1)*self.data_config.sequence_info[1]],
                        return_detailed=False
                        ) 
                
                else:
                    if self.residual_config["add_predicted_value_with_diff_loss"]:
                        raw_prediction, base_value = self._compute_raw_prediction(prediction, base_value)
                        #here the labels are the residuals.
                        loss = self.loss_fn(
                            model=model,
                            predictions=prediction,
                            labels=inputs["label_including_rollouts"][:,i*self.data_config.sequence_info[1]:(i+1)*self.data_config.sequence_info[1]],
                            return_detailed=False) 

                    if (self.residual_config["add_predicted_value_with_raw_loss"] or self.residual_config["add_base_value_with_raw_loss"]):
                        #here the base_value is not reinitalized, it is continued from the previous step.
                        raw_prediction, base_value = self._compute_raw_prediction(prediction, base_value)
                        #here the labels are the raw values.
                        loss = self.loss_fn(
                            model=model,
                            predictions=raw_prediction,
                            labels=inputs["label_including_rollouts"][:,i*self.data_config.sequence_info[1]:(i+1)*self.data_config.sequence_info[1]],
                            return_detailed=False)                

                if self.output_all_steps:
                    if self.residual_config is None:
                        predictions_.append(prediction.detach())
                    elif self.residual_config is not None and self.residual_config["add_predicted_value_with_diff_loss"]:
                        predictions_.append(prediction.detach())
                    elif self.residual_config is not None and self.residual_config["add_predicted_value_with_raw_loss"]:
                        predictions_.append(raw_prediction.detach())
                    else: #add_base_value_with_raw_loss
                        predictions_.append(raw_prediction.detach())
                    losses_.append(loss) 
                else:
                    total_loss += loss 

        if self.output_all_steps:
            predictions= torch.stack(predictions_, dim=1) #predictions.shape = torch.Size([B, rollout_steps+1, label_seq_length, C_output, *spatial_resolution])
            loss = torch.stack(losses_, dim=0) #shape: (rollout_steps+1)

        else:
            loss = total_loss / (self.rollout_steps+1) #take the mean of the loss across all rollout_steps

        return (loss, predictions) if return_outputs else loss 
        #loss is a scalar which is the same for all the windows of the batch.
        # inside evalutaion_loop(), the loss is repeated batch times before appending to the list of all_losses. 

    def compute_eval_without_loss(self, model, inputs):
        """
        Compute evaluation predictions without loss computation for memory efficiency.

        This re-implementation significantly speeds up the method by:
        1. Pre-allocating the full predictions tensor and filling it in-place (avoids Python list
           growth and the final `torch.stack`).
        2. Reducing Python-level overhead by caching frequently accessed attributes locally.
        3. Updating `input_data` in-place instead of recreating a new `dict` every rollout step.

        The numerical behaviour is identical to the original implementation.
        """
        # ------------------------------------------------------------------
        # Fast path preparation
        # ------------------------------------------------------------------
        batch_size, _, _, *spatial_dims = inputs["input_data"].shape
        rollout_steps = 0 if self.data_config.is_steady_state_prediction else self.rollout_steps
        total_steps = rollout_steps + 1  # first step + autoregressive rollouts

        # First forward pass (t = 0)
        prediction = self._forward_model_eval_or_test(model, inputs)

        # -------------------------------------------------------------
        # Handle residual learning variants once per step
        # -------------------------------------------------------------
        use_residuals = self.residual_config is not None
        if use_residuals:
            base_value = inputs["input_data"][:, -1:]  # last known physical state
            raw_prediction, base_value = self._compute_raw_prediction(prediction, base_value)

        # -------------------------------------------------------------
        # Pre-allocate storage when the caller wants *all* steps
        # -------------------------------------------------------------
        if self.output_all_steps:
            seq_len_out = self.data_config["sequence_info"][1]
            n_channels_out = prediction.shape[2]
            predictions = torch.empty(
                (batch_size, total_steps, seq_len_out, n_channels_out, *spatial_dims),
                dtype=prediction.dtype,
                device=prediction.device,
            )
            if not use_residuals or self.residual_config["add_predicted_value_with_diff_loss"]:
                predictions[:, 0].copy_(prediction.detach())
            else:
                predictions[:, 0].copy_(raw_prediction.detach())
        else:
            predictions = None  # we will only return the last prediction

        # Early exit when no autoregressive rollout is required
        if rollout_steps == 0:
            return predictions if self.output_all_steps else prediction.detach()

        # -------------------------------------------------------------
        # Local aliases to avoid repeated attribute look-ups inside loop
        # -------------------------------------------------------------
        seq_inp, seq_out, _ = self.data_config.sequence_info
        forward_fn = self._forward_model_eval_or_test
        curr_inputs = inputs  # will be updated in-place

        # -------------------------------------------------------------
        # Autoregressive rollout loop
        # -------------------------------------------------------------
        for step in range(1, total_steps):
            # Prepare `input_data` for the next timestep -------------------
            if seq_out >= seq_inp:
                # label sequence ≥ input sequence (slice needed)
                slice_from = seq_out - seq_inp
                next_input = (
                    prediction[:, slice_from:].detach()
                    if not use_residuals
                    else raw_prediction[:, slice_from:].detach()
                )
            else:
                # input sequence > label sequence (concatenate needed)
                next_input = torch.cat(
                    [
                        curr_inputs["input_data"][:, seq_out:],
                        prediction.detach() if not use_residuals else raw_prediction.detach(),
                    ],
                    dim=1,
                )
            # Update the inputs dict (shallow copy keeps other keys intact)
            curr_inputs = {**curr_inputs, "input_data": next_input}

            # Forward pass for this rollout step -------------------------
            prediction = forward_fn(model, curr_inputs)

            if use_residuals:
                raw_prediction, base_value = self._compute_raw_prediction(prediction, base_value)

            # Store the prediction if requested --------------------------
            if self.output_all_steps:
                if not use_residuals or self.residual_config["add_predicted_value_with_diff_loss"]:
                    #raw values are added OR predictions correspond to the difference and they are accumulated for the metrics.
                    predictions[:, step].copy_(prediction.detach())
                else:
                    #raw_predictions are accumulated for the metrics as the loss is computed with the raw values.
                    predictions[:, step].copy_(raw_prediction.detach())

        # Return either the full tensor (B, S, T, C, *spatial*) or the last prediction
        return predictions if self.output_all_steps else prediction.detach()

    ## overrides the one in the base class from transformers library
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self.get_total_train_batch_size(args)

        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = model is self.model

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    # We should avoid accelerate preparing the model in TP case since we dont need it as it is handled by transformers from_pretrained and also it goes into DDP based preparation.
                    if self.is_tp_enabled:
                        self.optimizer = self.accelerator.prepare(self.optimizer)
                    else:
                        model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        
        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)
        
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )
            def trace_handler(p):
                print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
                p.export_chrome_trace("./trace.json")

            profile_training_cfg = self.train_config.get("profile_training", {})
            profile_training_enabled = bool(profile_training_cfg.get("enabled", False))
            profile_wait = int(profile_training_cfg.get("wait", 0))
            profile_warmup = int(profile_training_cfg.get("warmup", 150))
            profile_active = int(profile_training_cfg.get("active_steps", 30))
            profile_repeat = int(profile_training_cfg.get("repeat", 1))
            profiler_context = (
                torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, 
                                torch.profiler.ProfilerActivity.CUDA, 
                                torch.profiler.ProfilerActivity.XPU], 
                    record_shapes=False, 
                    schedule=torch.profiler.schedule(
                        wait=profile_wait,
                        warmup=profile_warmup,
                        active=profile_active,
                        repeat=profile_repeat,
                    ),
                    on_trace_ready=trace_handler,
                    profile_memory=True,
                    with_stack=True
                )
                if profile_training_enabled
                else contextlib.nullcontext()
            )

            with profiler_context as prof:
                for _ in range(total_updates):
                    update_step += 1
                    num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                    batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                    # Store the number of batches for current gradient accumulation
                    # This is used to correctly scale the loss when the last accumulation step has fewer batches
                    self.current_gradient_accumulation_steps = len(batch_samples)
                    for i, inputs in enumerate(batch_samples):
                        step += 1
                        do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                        # Since we perform prefetching, we need to manually set sync_gradients
                        self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                        if self.args.include_num_input_tokens_seen:
                            main_input_name = getattr(self.model, "main_input_name", "input_ids")
                            if main_input_name not in inputs:
                                logger.warning(
                                    "Tried to track the number of tokens seen, however the current model is "
                                    "not configured properly to know what item is the input. To fix this, add "
                                    "a `main_input_name` attribute to the model class you are using."
                                )
                            else:
                                input_tokens = inputs[main_input_name].numel()
                                input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                                self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()
                        if rng_to_sync:
                            self._load_rng_state(resume_from_checkpoint)
                            rng_to_sync = False
                        
                        # Skip past any already trained steps if resuming training
                        if steps_trained_in_current_epoch > 0:
                            steps_trained_in_current_epoch -= 1
                            if steps_trained_progress_bar is not None:
                                steps_trained_progress_bar.update(1)
                            if steps_trained_in_current_epoch == 0:
                                self._load_rng_state(resume_from_checkpoint)
                            continue
                        elif steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.close()
                            steps_trained_progress_bar = None
                        
                        if step % args.gradient_accumulation_steps == 0:
                            self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                        # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                        context = (
                            functools.partial(self.accelerator.no_sync, model=model)
                            if i != len(batch_samples) - 1
                            and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                            else contextlib.nullcontext
                        )
                        with context():
                            tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                        if (
                            args.logging_nan_inf_filter
                            and not is_torch_xla_available()
                            and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                        ):
                            # if loss is nan or inf simply add the average of previous logged losses
                            tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                        else:
                            if tr_loss.device != tr_loss_step.device:
                                raise ValueError(
                                    f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                                )
                            tr_loss = tr_loss + tr_loss_step

                        self.current_flos += float(self.floating_point_ops(inputs))

                        if do_sync_step:
                            # Since we perform prefetching, we need to manually set sync_gradients to True
                            self.accelerator.gradient_state._set_sync_gradients(True)

                            # Gradient clipping
                            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                                if is_sagemaker_mp_enabled() and args.fp16:
                                    _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                                elif self.use_apex:
                                    from apex import amp

                                    # Revert to normal clipping otherwise, handling Apex or full precision
                                    _grad_norm = nn.utils.clip_grad_norm_(
                                        amp.master_params(self.optimizer),
                                        args.max_grad_norm,
                                    )
                                else:
                                    grad_norm_context = contextlib.nullcontext
                                    if self.is_tp_enabled:
                                        from torch.distributed._tensor.experimental import implicit_replication

                                        grad_norm_context = implicit_replication
                                    with grad_norm_context():
                                        _grad_norm = self.accelerator.clip_grad_norm_(
                                            model.parameters(),
                                            args.max_grad_norm,
                                        )

                                if (
                                    is_accelerate_available()
                                    and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                                ):
                                    grad_norm = model.get_global_grad_norm()
                                    # In some cases the grad norm may not return a float
                                    if hasattr(grad_norm, "item"):
                                        grad_norm = grad_norm.item()
                                else:
                                    grad_norm = _grad_norm

                            self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                            context = contextlib.nullcontext
                            if self.is_tp_enabled:
                                from torch.distributed._tensor.experimental import implicit_replication

                                context = implicit_replication

                            with context():
                                self.optimizer.step()

                            self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                            # get leaning rate before update
                            learning_rate = self._get_learning_rate()

                            if not self.accelerator.optimizer_step_was_skipped:
                                # Delay optimizer scheduling until metrics are generated
                                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    self.lr_scheduler.step()

                            model.zero_grad()
                            self.state.global_step += 1
                            self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                            self._maybe_log_save_evaluate(
                                tr_loss,
                                grad_norm,
                                model,
                                trial,
                                epoch,
                                ignore_keys_for_eval,
                                start_time,
                                learning_rate=learning_rate,
                            )
                        else:
                            self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                        # PyTorch/XLA relies on the data loader to insert the mark_step for
                        # each step. Since we are breaking the loop early, we need to manually
                        # insert the mark_step here.
                        if self.control.should_epoch_stop or self.control.should_training_stop:
                            if is_torch_xla_available():
                                xm.mark_step()
                            break
                    if profile_training_enabled:
                        prof.step() #to save the profile
                    # We also need to break out of the nested loop
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            
            if (epoch+1)%self.num_epochs_between_eval==0:
                self._maybe_log_save_evaluate(
                    tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
                )

            #! Modify the callbacks if the curriculum block changed based on the epoch (not in the base class).
            # using epoch instead of self.state.epoch because there is a possibility that self.state.epoch is fractional.
            if ((epoch + 1) in self.curriculum_start_epochs) and ((epoch + 1) < num_train_epochs):
                next_block = self._get_train_strategy_block_for_epoch(epoch + 1)
                train_loss_dict_next_block = next_block.train_loss
                validation_loss_dict_next_block = next_block.validation_loss
                loss_weighting_strategy_next_block = create_loss_weighting_strategy(train_loss_dict_next_block)

                if loss_weighting_strategy_next_block is not None:
                    grad_stats = get_loss_weighting_strategy_entry(
                        train_loss_dict_next_block.train_loss_weighting_strategy.type
                    ).get("grad_stats", [])
                    use_gradients = bool(grad_stats)

                    self._collect_detailed_losses = True
                    self._collect_gradients = use_gradients

                    self._grad_stat_names = grad_stats

                    self._weight_per_channel = train_loss_dict_next_block.train_loss_weighting_strategy.get("weight_per_channel", False)
                    self._weight_sub_components = train_loss_dict_next_block.train_loss_weighting_strategy.get("weight_sub_components", False)
                    self._loss_history_interval = train_loss_dict_next_block.train_loss_weighting_strategy.get("loss_history_interval", 1)
                    self._grad_history_interval = train_loss_dict_next_block.train_loss_weighting_strategy.get("grad_history_interval", 1)

                    loss_stats_callback_next_block = LossStatisticsCallback(
                        collect_train_losses=True, grad_stats=grad_stats, collect_gradients=use_gradients, trainer=self
                    )
                    loss_source = train_loss_dict_next_block.train_loss_weighting_strategy.get("loss_source", "train")
                    adaptive_weight_callback_next_block = AdaptiveWeightCallback(
                        loss_weighting_strategy=loss_weighting_strategy_next_block,
                        stats_callback=loss_stats_callback_next_block,
                        trainer=self,
                        loss_source=loss_source,
                        use_gradients=use_gradients,
                        grad_stats=grad_stats,
                        curriculum_start_epochs=self.curriculum_start_epochs
                    )

                    callbacks = list(getattr(self.callback_handler, "callbacks", []) or [])
                    replaced_ls = False
                    replaced_aw = False

                    # Replace existing callbacks if present; otherwise append new ones.
                    for idx, cb in enumerate(callbacks):
                        if isinstance(cb, LossStatisticsCallback):
                            callbacks[idx] = loss_stats_callback_next_block
                            replaced_ls = True
                        if isinstance(cb, AdaptiveWeightCallback):
                            callbacks[idx] = adaptive_weight_callback_next_block
                            replaced_aw = True

                    if not replaced_ls:
                        callbacks.append(loss_stats_callback_next_block)
                    if not replaced_aw:
                        callbacks.append(adaptive_weight_callback_next_block)

                    self.callback_handler.callbacks = callbacks
                else:
                    # Remove LossStatisticsCallback and AdaptiveWeightCallback from the list of callbacks if the next block has no train_loss weighting strategy.
                    callbacks = list(getattr(self.callback_handler, "callbacks", []) or [])
                    callbacks = [
                        cb for cb in callbacks
                        if not isinstance(cb, (LossStatisticsCallback, AdaptiveWeightCallback))
                    ]
                    self.callback_handler.callbacks = callbacks

                    self._collect_detailed_losses = False
                    self._collect_gradients = False

                    self._grad_stat_names = []

                    self._weight_per_channel = False
                    self._weight_sub_components = False
                    self._loss_history_interval = 1
                    self._grad_history_interval = 1

                self.loss_fn = self.get_loss_fn(train_loss_dict_next_block)
                self.eval_loss_fn = self.get_loss_fn(validation_loss_dict_next_block)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")
        
        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                pass
                #*dist.barrier() (!commented to avoidproblems with intel XPUs)
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)
        
        #! Only rank 0 runs end-of-train callbacks; others wait at barriers (! this isnot in the base class).
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        self.accelerator.wait_for_everyone()
        
        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    #custom function, not inside transformers library
    def last_evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        "This function is just for plotting, and is done solely by rank 0"

        RANK = int(os.environ.get("RANK", -1))
        IS_MAIN_PROCESS = RANK in [-1, 0]

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self.model

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        if IS_MAIN_PROCESS:
            logger.info(f"\n***** Running the LAST {description} for plotting from the best checkpoint *****")
            if has_length(dataloader):
                logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            else:
                logger.info("  Num examples: Unknown")
            logger.info(f"  Batch size = {batch_size}")

        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        # Initialize containers
        collect_full_eval_tensors = True if description == "Prediction" else False
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = (
            EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
            if collect_full_eval_tensors
            else None
        )
        all_labels = (
            EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
            if collect_full_eval_tensors
            else None
        )
        all_inputs = (
            EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
            if collect_full_eval_tensors
            else None
        )
        all_conditioning_inputs = (
            EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
            if collect_full_eval_tensors
            else None
        )
        example_logits = None
        example_labels = None
        example_inputs = None
        example_conditioning_inputs = None

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        log_channels = self.data_config.get("log_transform_channels") or []
        dim = self.data_config.get("dimension")
        channel_axis = -2 if dim == 1 else (-3 if dim == 2 else -4)
        channel_names = getattr(eval_dataset, "output_channels")
        norm_stats = self.data_config.get("data_normalization_stats")
        norm_strategy = self.data_config.get("data_normalization_strategy")

        def _apply_log_inverse(arr, log_channels, channel_names, norm_stats, norm_strategy, channel_axis):
            for ch_name in log_channels:
                if (
                    ch_name not in channel_names
                    or norm_stats is None
                    or norm_strategy is None
                ):
                    continue
                stats_key = f"log_{ch_name}"
                if stats_key not in norm_stats or ch_name not in norm_stats:
                    continue
                ch_idx = channel_names.index(ch_name)
                slicer = [slice(None)] * arr.ndim
                slicer[channel_axis] = ch_idx
                log_space = re_normalize_data(arr[tuple(slicer)], norm_stats[stats_key], norm_strategy)
                physical_space = torch.exp(log_space)
                physical_space_normalized = normalize_data(
                    physical_space, norm_stats[ch_name], norm_strategy
                )
                arr[tuple(slicer)] = physical_space_normalized
            return arr

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )
            #NOTE: The following is added on top of the base class
            conditioning_input_name = getattr(self.model, "conditioning_input_name", "conditioning_input_data")
            conditioning_input_decode = (  ##To include the inputs in the metrics computation
                self._prepare_input(inputs[conditioning_input_name]) if "conditioning_inputs" in args.include_for_metrics else None
            )

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function(losses.repeat(batch_size))
                all_losses.add(losses)
            if collect_full_eval_tensors and inputs_decode is not None:
                inputs_decode = self.gather_function(inputs_decode)
                #if not self.args.batch_eval_metrics or description == "Prediction":
                all_inputs.add(inputs_decode)
            if collect_full_eval_tensors and conditioning_input_decode is not None:
                conditioning_input_decode = self.gather_function(conditioning_input_decode)
                #if not self.args.batch_eval_metrics or description == "Prediction":
                all_conditioning_inputs.add(conditioning_input_decode)
            if logits is not None:
                logits = _apply_log_inverse(logits, log_channels, channel_names, norm_stats, norm_strategy, channel_axis)
                if collect_full_eval_tensors:
                    gathered_logits = self.gather_function(logits)
                #if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(gathered_logits)
            if labels is not None:
                labels = _apply_log_inverse(labels, log_channels, channel_names, norm_stats, norm_strategy, channel_axis)
                if collect_full_eval_tensors:
                    gathered_labels = self.gather_function(labels)
                #if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(gathered_labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # if self.args.batch_eval_metrics:
            #     #not implemented
            #     pass
            is_last_step = self.accelerator.gradient_state.end_of_dataloader
            if not collect_full_eval_tensors and is_last_step and IS_MAIN_PROCESS:
                #just choosing the first example from the last batch to plot
                example_logits = logits[-1:].detach().cpu().numpy()
                example_labels = labels[-1:].detach().cpu().numpy()
                example_inputs = inputs_decode[-1:].detach().cpu().numpy()
                example_conditioning_inputs = conditioning_input_decode[-1:].detach().cpu().numpy() if conditioning_input_decode is not None else None
            
            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if collect_full_eval_tensors and args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()
                all_conditioning_inputs.to_cpu_and_numpy()
                if not self.args.batch_eval_metrics:
                    del losses, logits, labels, inputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if torch.xpu.is_available():
                        torch.xpu.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        all_losses = all_losses.get_arrays() if collect_full_eval_tensors else None
        all_preds = all_preds.get_arrays() if collect_full_eval_tensors else example_logits
        all_labels = all_labels.get_arrays() if collect_full_eval_tensors else example_labels
        all_inputs = all_inputs.get_arrays() if collect_full_eval_tensors else example_inputs
        all_conditioning_inputs = (
            all_conditioning_inputs.get_arrays() if collect_full_eval_tensors else example_conditioning_inputs
        )
        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples
        
        #The purpose of this method is to plot the best checkpoint for which we only need the prediction, labels and input tensors. 
        #So we don't need to compute the metrics.
        metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples), all_inputs, all_conditioning_inputs

    #custom function, not inside transformers library, for plotting the best checkpoint.
    def last_evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval"):

        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        eval_loop = self.last_evaluation_loop
        
        output, input, conditioning_input = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )        

        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            output.metrics,
            # NOTE: kwargs added to be used in PlotOnEvalAndSaveCallback()
            predictions=output.predictions,
            labels=output.label_ids,
            inputs=input,
            conditioning_inputs=conditioning_input,
            eval_dataset=eval_dataset,
            data_config=self.data_config,
            train_config=self.train_config,
            train_strategy_config=self.train_strategy_config,
            output_log_config=self.output_log_config,
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
        )

        #NOTE: stop training if NaN is encountered in the loss and set the control flags to False.
        if self.control.should_training_stop_due_to_nan and self.state.epoch<self.train_config['num_train_epochs']:
            self.control.should_evaluate = False
            self.control.should_save = False
            self.control.should_plot = False
        
        logger.info(f"\n***** Finished the LAST Evaluation for plotting from the best checkpoint *****")

    ##custom function, not inside transformers library
    def set_eval_or_test_rollout_steps(self, rollout_steps=None, output_all_steps=False):
        """
        Configure rollout steps for evaluation or testing.

        Parameters
        ----------
        rollout_steps : Optional[int]
            Number of autoregressive rollout steps. If None, no rollout is performed.
        output_all_steps : bool, default=False
            Whether to output predictions for all rollout steps.
        """
        self.rollout_steps = rollout_steps 
        if self.rollout_steps is not None and output_all_steps:
            self.output_all_steps = True
    
    ##overrides the one in the base class from transformers library
    #* introduced compute_eval_loss and compute_eval_without_loss in this function which are not in the base class.
    def prediction_step( 
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool, #True if compute_metrics is not provided
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform a single prediction step with custom loss computation and output handling.

        Parameters
        ----------
        model : nn.Module
            The model to evaluate.
        inputs : Dict[str, Union[torch.Tensor, Any]]
            Input tensors and labels.
        prediction_loss_only : bool
            Whether to return only the loss (True if compute_metrics is not provided).
        ignore_keys : Optional[List[str]]
            Keys to ignore when gathering predictions.

        Returns
        -------
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
            Tuple containing (loss, logits, labels).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels: #has_labels is true for validation and testing
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled(): #doesn't go here by default, this is for distributed inference
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else: #not sure why this 'else' is needed.
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if (has_labels or loss_without_labels) and self.get_prediction_loss_for_eval_windows: #NOTE: self.get_prediction_loss_for_eval_windows is added on top of the base class
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_eval_loss( 
                            model, inputs, return_outputs=True
                        ) #return_output is true only when doing eval or inference.. By default it is false
                    loss = loss.detach().mean() #mean() is used when: self.output_all_steps = True which results in loss being a tensor of shape (num_rollout_steps+1,) and we take the mean

                    if isinstance(outputs, dict): 
                        logits = tuple( 
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]# ignores the keys
                        )
                    else: # Enters here as outputs is a tensor. logits is the outputs tensor (#NOTE: in the base class it is outputs[1:] as the 0th index is the loss)
                        logits = outputs
                
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = self.compute_eval_without_loss(model, inputs)
                        ##in the base class, the above line is outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs

        if prediction_loss_only: #prediction_loss_only is True if compute_metrics is not provided
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1 and isinstance(logits, tuple): 
            logits = logits[0] 
        
        return (loss, logits, labels)

    ### overrides the one in the base class from transformers library
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Extended evaluation loop with conditioning input support and custom metric computation.

        Parameters
        ----------
        dataloader : DataLoader
            Evaluation dataloader.
        description : str
            Description of the evaluation phase.
        prediction_loss_only : Optional[bool]
            Whether to compute only prediction loss.
        ignore_keys : Optional[List[str]]
            Keys to ignore during evaluation.
        metric_key_prefix : str, default="eval"
            Prefix for metric keys.

        Returns
        -------
        EvalLoopOutput
            Evaluation results including predictions, labels, and metrics.
        """
        RANK = int(os.environ.get("RANK", -1))
        IS_MAIN_PROCESS = RANK in [-1, 0]

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
            #print("XPU cache emptied before evaluation, rank: ", RANK, flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            #print("CUDA cache emptied before evaluation, rank: ", RANK, flush=True)
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8" and not self.args.torch_compile)
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        if IS_MAIN_PROCESS:
            logger.info(f"\n***** Running {description} *****")
            if has_length(dataloader):
                logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            else:
                logger.info("  Num examples: Unknown")
            logger.info(f"  Batch size = {batch_size}")

        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        collect_full_eval_tensors = True if (description == "Prediction" or not self.args.batch_eval_metrics) else False
        #* Description is "Prediction" only when doing inference.
        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100) if collect_full_eval_tensors else None
        all_preds = (
            EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
            if collect_full_eval_tensors
            else None
        )
        all_labels = (
            EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
            if collect_full_eval_tensors
            else None
        )
        all_inputs = (
            EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
            if collect_full_eval_tensors
            else None
        )
        all_conditioning_inputs = (
            EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
            if collect_full_eval_tensors
            else None
        )

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        #! not in the base class
        # Renormalize log-transformed channels in predictions and labels using log statistics to obtain log-transformed channels in physical units.
        # Then take the exponential of the log-transformed channels in physical units to obtain the channels in physical units.
        # Finally, normalize the channels in physical units using the original statistics to get the channels in normalized (physical) units.
        log_channels = self.data_config.get("log_transform_channels") or []
        dim = self.data_config.get("dimension")
        channel_axis = -2 if dim == 1 else (-3 if dim == 2 else -4)
        channel_names = getattr(eval_dataset, "output_channels")
        norm_stats = self.data_config.get("data_normalization_stats")
        norm_strategy = self.data_config.get("data_normalization_strategy")
        
        def _apply_log_inverse(arr, log_channels, channel_names, norm_stats, norm_strategy, channel_axis):
            for ch_name in log_channels:
                if (
                    ch_name not in channel_names
                    or norm_stats is None
                    or norm_strategy is None
                ):
                    continue
                stats_key = f"log_{ch_name}"
                if stats_key not in norm_stats or ch_name not in norm_stats:
                    continue
                ch_idx = channel_names.index(ch_name)
                slicer = [slice(None)] * arr.ndim
                slicer[channel_axis] = ch_idx
                log_space = re_normalize_data(  #Ex: Here we obtain log(Density) in physical units 
                    arr[tuple(slicer)], norm_stats[stats_key], norm_strategy
                )
                physical_space = torch.exp(log_space) #Ex: Here we obtain Density in physical units
                physical_space_normalized = normalize_data(
                    physical_space, norm_stats[ch_name], norm_strategy #Ex: Here we normalize Density 
                )
                arr[tuple(slicer)] = physical_space_normalized #Ex: Here we store the normalized Density
            return arr
        #########################################################
        # Main evaluation loop
        #########################################################
        #print length of dataloader
        for step, inputs in enumerate(dataloader):
            #print(f"Step {step} of {len(dataloader)}")
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step (the losses are for a batch of inputs given by the dataloader)
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (  ##To include the inputs in the metrics computation
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )
            #NOTE: The following is added on top of the base class
            conditioning_input_name = getattr(self.model, "conditioning_input_name", "conditioning_input_data")
            conditioning_input_decode = (  ##To include the inputs in the metrics computation
                self._prepare_input(inputs[conditioning_input_name]) if "conditioning_inputs" in args.include_for_metrics else None
            )

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function(losses.repeat(batch_size)) 
                #NOTE: repeat is used to ensure that each window of the batch owns the same loss value.
                all_losses.add(losses)
            if collect_full_eval_tensors and inputs_decode is not None:
                #inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function(inputs_decode)
                #if not self.args.batch_eval_metrics or description == "Prediction":
                all_inputs.add(inputs_decode)
            #NOTE: The following is added on top of the base class
            #########################################################
            if collect_full_eval_tensors and conditioning_input_decode is not None:
                #conditioning_input_decode = self.accelerator.pad_across_processes(conditioning_input_decode, dim=1, pad_index=-100)
                conditioning_input_decode = self.gather_function(conditioning_input_decode)
                #if not self.args.batch_eval_metrics or description == "Prediction":
                all_conditioning_inputs.add(conditioning_input_decode)
            #########################################################
            if logits is not None:
                logits = _apply_log_inverse(logits, log_channels, channel_names, norm_stats, norm_strategy, channel_axis)
                if collect_full_eval_tensors:
                    gathered_logits = self.gather_function(logits)
                    #if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(gathered_logits)
            if labels is not None:
                labels = _apply_log_inverse(labels, log_channels, channel_names, norm_stats, norm_strategy, channel_axis)
                if collect_full_eval_tensors:
                    gathered_labels = self.gather_function(labels)
                    #if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(gathered_labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    # batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    # batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    # batch_kwargs["conditioning_inputs"] = conditioning_input_decode if "conditioning_inputs" in args.include_for_metrics else None
                    #NOTE: inputs is a dict which has the input_data and conditioning_input_data
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )
                    #just one example to plot
                    if not collect_full_eval_tensors and is_last_step and IS_MAIN_PROCESS:
                        #just choosing the first example from the last batch to plot
                        example_logits = logits[-1:].detach().cpu().numpy()
                        example_labels = labels[-1:].detach().cpu().numpy()
                        example_inputs = inputs_decode[-1:].detach().cpu().numpy()
                        example_conditioning_inputs = conditioning_input_decode[-1:].detach().cpu().numpy() if conditioning_input_decode is not None else None

                del losses, logits, labels, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.xpu.is_available():
                    torch.xpu.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if collect_full_eval_tensors and args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()
                all_conditioning_inputs.to_cpu_and_numpy()
        
                if not self.args.batch_eval_metrics:
                    del losses, logits, labels, inputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if torch.xpu.is_available():
                        torch.xpu.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays() if all_losses is not None else None #all_losses.shape = torch.Size([B*(steps+1) , ]) 
        all_preds = (
            all_preds.get_arrays() if collect_full_eval_tensors else example_logits
        ) #all_preds.shape = torch.Size([B*(steps+1), n_eval_rollouts+1, label_seq_length, C_output, x_resolution, y_resolution, ...]) 
        all_labels = (
            all_labels.get_arrays() if collect_full_eval_tensors else example_labels
        ) #all_labels.shape = torch.Size([B*(steps+1), (n_eval_rollouts+1)*label_seq_length, C_output, x_resolution, y_resolution, ...]) 
        all_inputs = (
            all_inputs.get_arrays() if collect_full_eval_tensors else example_inputs
        ) #all_inputs.shape = torch.Size([B*(steps+1), input_seq_length, C_input, x_resolution, y_resolution, ...]) 
        all_conditioning_inputs = (
            all_conditioning_inputs.get_arrays() if collect_full_eval_tensors else example_conditioning_inputs
        ) #all_conditioning_inputs.shape = torch.Size([B*(steps+1), conditioning_seq_length, C_conditioning, x_resolution, y_resolution, ...]) 

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics! (to be removed in future versions)
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            eval_set_kwargs["conditioning_inputs"] = all_conditioning_inputs if "conditioning_inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs)
            )
        
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if key.startswith(metric_key_prefix):
                continue
            metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples), all_inputs, all_conditioning_inputs
        #* all_inputs and all_conditioning_inputs are the additional return arguments compared to the evaluation_loop() function in the base class.

    ### overrides the one in the base class from transformers library
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Run evaluation with custom callback support and extended metric computation.

        Parameters
        ----------
        eval_dataset : Optional[Union[Dataset, Dict[str, Dataset]]]
            Evaluation dataset(s). If None, uses self.eval_dataset.
        ignore_keys : Optional[List[str]]
            Keys to ignore during evaluation.
        metric_key_prefix : str, default="eval"
            Prefix for metric keys.

        Returns
        -------
        Dict[str, float]
            Dictionary of evaluation metrics.
        """
        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()

        #########################################################
        #NOTE: Main evaluation loop
        eval_loop_start_time = time.time()
        output, input, conditioning_input = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        # Record the wall-clock duration of the evaluation loop (in seconds)
        output.metrics[f"{metric_key_prefix}_eval_loop_time"] = round(time.time() - eval_loop_start_time, 4)
        #print(f"eval_loop_time: {output.metrics[f'{metric_key_prefix}_eval_loop_time']}")
        #########################################################
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        # Optionally log with the epoch corresponding to the best checkpoint
        # (used for the final evaluation pass at the end of training to avoid
        # logging the last epoch instead of the best one).
        _logged_with_best_epoch = False
        try:
            if getattr(self, "_force_best_epoch_for_logging", False):
                ckpt_path = getattr(self.state, "best_model_checkpoint", None)
                if ckpt_path is not None:
                    state_path = os.path.join(ckpt_path, "trainer_state.json")
                    if os.path.isfile(state_path):
                        try:
                            with open(state_path, "r") as _fp:
                                _state_json = json.load(_fp)
                            _best_epoch = _state_json.get("epoch", None)
                            if _best_epoch is not None:
                                _prev_epoch = self.state.epoch
                                self.state.epoch = _best_epoch
                                self.log(output.metrics)  # logs using the best epoch value
                                self.state.epoch = _prev_epoch
                                _logged_with_best_epoch = True
                        except Exception:
                            pass
        finally:
            pass

        if not _logged_with_best_epoch:
            self.log(output.metrics) #NOTE: logs into wandb during evaluation

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            output.metrics,
            #! kwargs added to be used in PlotOnEvalAndSaveCallback() (not in the base class).
            predictions=output.predictions,
            labels=output.label_ids,
            inputs=input,
            conditioning_inputs=conditioning_input,
            eval_dataset=eval_dataset,
            data_config=self.data_config,
            train_config=self.train_config,
            train_strategy_config=self.train_strategy_config,
            output_log_config=self.output_log_config,
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)
        #! stop training if NaN is encountered in the loss and set the control flags to False (not in the base class).
        if self.control.should_training_stop_due_to_nan and self.state.epoch<self.train_config['num_train_epochs']:
            self.control.should_evaluate = False
            self.control.should_save = False
            self.control.should_plot = False

        return output.metrics

    ### overrides the one in the base class from transformers library
    # * no change compared to the base class
    def _evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
        """
        Internal evaluation method with learning rate scheduler support.

        Parameters
        ----------
        trial : Any
            Hyperparameter search trial object.
        ignore_keys_for_eval : List[str]
            Keys to ignore during evaluation.
        skip_scheduler : bool, default=False
            Whether to skip learning rate scheduler step.

        Returns
        -------
        Dict[str, float]
            Evaluation metrics.
        """
        metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
        self._report_to_hp_search(trial, self.state.global_step, metrics)

        # Run delayed LR scheduler now that metrics are populated
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and not skip_scheduler:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            try:
                self.lr_scheduler.step(metrics[metric_to_check])
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', "
                    f"which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. "
                    f"Please ensure that the `compute_metrics` function returns a dictionary that includes '{metric_to_check}' or "
                    f"consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc
        return metrics

    ### overrides the one in the base class from transformers library
    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[list[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        callbacks_backup = list(getattr(self.callback_handler, "callbacks", []))
        # Remove both HF's WandB callback and the custom one if present
        try:
            self.callback_handler.remove_callback(WandbCallback_)
        except Exception:
            pass
        try:
            self.callback_handler.remove_callback(WandbCallback)
        except Exception:
            pass

        try:
            # memory metrics - must set up as early as possible
            self._memory_tracker.start()
            test_dataloader = self.get_test_dataloader(test_dataset)
            start_time = time.time()

            output, input, conditioning_input = self.evaluation_loop(
                test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
            )
            total_batch_size = self.args.eval_batch_size * self.args.world_size
            if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
                start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
            if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
                start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
            output.metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )

            self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
            self._memory_tracker.stop_and_update_metrics(output.metrics)

            return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics) , input, conditioning_input
        finally:
            # Restore callbacks if we temporarily removed WandB for predict
            if callbacks_backup is not None:
                self.callback_handler.callbacks = callbacks_backup

    ### overrides the one in the base class from transformers library
    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
        ):
        """
        Handle logging, saving, and evaluation during training with custom plotting support.

        Parameters
        ----------
        tr_loss : torch.Tensor
            Training loss tensor.
        grad_norm : Union[torch.Tensor, float]
            Gradient norm value.
        model : torch.nn.Module
            The model being trained.
        trial : Any
            Hyperparameter search trial object.
        epoch : int
            Current training epoch.
        ignore_keys_for_eval : List[str]
            Keys to ignore during evaluation.
        start_time : float
            Training start time.
        """
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time) #NOTE: logs into wandb for training
        
        metrics = None
        RANK = int(os.environ.get("RANK", -1))
        IS_MAIN_PROCESS = RANK in [-1, 0]

        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval) 
            if IS_MAIN_PROCESS:
                logger.info(f"Model checkpointing is done based on: eval_{self.args.metric_for_best_model}")
            #* added predictions, labels, inputs as additional return arguments compared to the base class.
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save: 
            # Remove any transient keys related to plotting progress from the trainer state
            # before saving checkpoints to avoid checkpoint serialization errors.
            try:
                state_slot = self.__dict__.get("state", None)
                if state_slot is not None:
                    state_dict = getattr(state_slot, "__dict__", None)
                    if isinstance(state_dict, dict):
                        # NOTE: filter out any log history entries that contain keys starting with "plot_progress" 
                        # as wandb images cannot be saved in .json files.
                        log_history = state_dict.get("log_history")
                        if isinstance(log_history, list):
                            filtered_history = []
                            for _entry in log_history:
                                if isinstance(_entry, dict) and any(
                                    isinstance(_kk, str) and _kk.startswith("plot_progress") for _kk in _entry.keys()
                                ):
                                    continue
                                filtered_history.append(_entry)
                            # mutate in place to preserve references held elsewhere
                            log_history[:] = filtered_history
            except Exception:
                # Never fail checkpointing due to cleanup issues
                pass
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        #RANK = int(os.environ.get("RANK", -1))
        if self.control.should_plot and (IS_MAIN_PROCESS):  
            self.control = self.callback_handler.on_plot(self.args, self.state, self.control, is_new_best_metric=is_new_best_metric)
    
    # Override the _save_checkpoint method to include loss configuration saving
    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save checkpoint with comprehensive loss configuration including current weights and metric parameters.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to save
        trial : Any
            Hyperparameter search trial object
        metrics : Optional[Dict], default=None
            Metrics dictionary (not used by parent class but kept for compatibility)
        """
        super()._save_checkpoint(model, trial)

        # Helper to convert tensors to lists
        def _tensorize_for_json(obj):
            """Recursively convert tensors to lists for JSON serialization."""
            if isinstance(obj, torch.Tensor):
                return obj.cpu().tolist()
            elif isinstance(obj, dict):
                return {k: _tensorize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_tensorize_for_json(item) for item in obj]
            else:
                return obj
        
        if self.train_strategy_config is not None:
            try:
                # Get the output directory
                output_dir = self.args.output_dir
                
                # Construct checkpoint folder name based on HF's logic
                if self.args.save_strategy == "no":
                    # No checkpoints saved
                    return
                else:  # "steps" (default)
                    checkpoint_folder = os.path.join(
                        output_dir, 
                        f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                    )
                
                loss_config_path = os.path.join(checkpoint_folder, "loss_config.json")
                
                current_block = self._get_train_strategy_block_for_epoch(self.state.epoch-1)

                current_train_strategy_dict = OmegaConf.to_container(current_block, resolve=True)
                
                if self.loss_fn is not None:
                    # Get loss weights at the time of checkpointing
                    weight_dict = self.loss_fn.get_loss_weight_dict()

                    current_train_loss_dict = current_train_strategy_dict["train_loss"]
                    
                    for component in current_train_loss_dict['components']:
                        comp_name = component.get('name', component['type'])
                        if comp_name in weight_dict:
                            component['current_weights'] = _tensorize_for_json(weight_dict[comp_name])
            
                # Save
                with open(loss_config_path, 'w') as f:
                    json.dump(current_train_strategy_dict , f, indent=2)
                
                logger.info(f"Saved loss_config to {loss_config_path}")
                
            except Exception as e:
                logger.warning(f"Failed to save loss_config: {e}")
    
    ### overrides the one in the base class from transformers library (NOTE: Not updated yet to 4.56.0 version of the base class)
    def _hp_search_setup(self, trial: Union["optuna.Trial", dict[str, Any]]):
        """
        Set up hyperparameter search with nested configuration support.

        Parameters
        ----------
        trial : Union[optuna.Trial, Dict[str, Any]]
            Hyperparameter search trial object or parameter dictionary.
        """
        self._trial = trial

        if self.hp_search_backend is None or trial is None:
            return
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            params = self.hp_space(trial)
        elif self.hp_search_backend == HPSearchBackend.RAY:
            params = trial
            params.pop("wandb", None)
        elif self.hp_search_backend == HPSearchBackend.SIGOPT:
            params = {k: int(v) if isinstance(v, str) else v for k, v in trial.assignments.items()}
        elif self.hp_search_backend == HPSearchBackend.WANDB:
            params = trial
        #####HERE THE HYPERPARAMETERS ARE REPLACED #################
        # Accept hyperparameter keys in dot-path format (e.g. ``train_config.max_steps``) and apply them
        # to the corresponding (potentially nested) attribute inside ``self``.
        for key, value in params.items():
            attr_path = key.split(".")  # Traverse nested attributes via dot notation
            target_obj = self

            # Walk down the hierarchy until the parent of the final attribute
            for part in attr_path[:-1]:
                if hasattr(target_obj, part):
                    target_obj = getattr(target_obj, part)
                else:
                    logger.warning(
                        f"Trying to set {key} in the hyperparameter search but `{part}` attribute was not found."
                    )
                    target_obj = None
                    break

            if target_obj is None:
                continue  # Skip keys that cannot be resolved

            final_part = attr_path[-1]

            # Retrieve the existing value (if any) to perform type casting later on
            old_attr = getattr(target_obj, final_part, None)

            # Attempt to cast the new value to the type of the existing value, when available
            if old_attr is not None:
                try:
                    value = type(old_attr)(value)
                except Exception:
                    # Fallback to the supplied type if casting fails (e.g. incompatible types)
                    pass

            # Finally, set/overwrite the attribute
            setattr(target_obj, final_part, value)

            # Mirror the change in ``self.args`` if the attribute exists there as well.
            if hasattr(self.args, final_part):
                old_attr_args = getattr(self.args, final_part, None)
                # Perform type-casting similar to above when possible
                value_for_args = value
                if old_attr_args is not None:
                    try:
                        value_for_args = type(old_attr_args)(value)
                    except Exception:
                        pass
                setattr(self.args, final_part, value_for_args)
        #NOTE:add trial number to self.args, which will be passed to WandbCallback 
        RANK = int(os.environ.get("RANK", -1))
        IS_MAIN_PROCESS = RANK in [-1, 0]

        if hasattr(trial, "number"):
            setattr(self.args, "trial_number", trial.number)
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            #set 2 blank lines
            if IS_MAIN_PROCESS:
                logger.info("-----------------------------------------------------------------------")
                logger.info("-----------------------------------------------------------------------")
                logger.info("")
                logger.info(f"Trial: {trial.params}")
                logger.info("")
                logger.info("-----------------------------------------------------------------------")
                logger.info("-----------------------------------------------------------------------")
            
            # ------------------------------------------------------------------
            # Retrieve all channel names present in the underlying HDF5 dataset.
            # We open the ``train.h5`` file, inspect the first group and collect
            # all dataset keys (each key corresponds to a channel).  This gives us
            # a robust, order-preserving list of *all* channels that exist in the
            # raw data, independent of any prior filtering decisions.
            # ------------------------------------------------------------------

            h5file_path = os.path.abspath(os.path.join(self.data_config["dataset_directory_path"], "train.h5"))

            # Lazily load the keys to keep the file access lightweight; we only
            # need the name list, not the actual data.
            with h5py.File(h5file_path, "r") as _h5f:
                first_group = next(iter(_h5f.keys()))  # assume at least one group exists

                grp = _h5f[first_group]

                channel_names = []  # expanded list

                for dset_name in grp:
                    dset = grp[dset_name]
                    channel_dim = dset.shape[1]

                    # Single-component field → keep original name
                    if channel_dim == 1:
                        channel_names.append(dset_name)
                    # Multi-component field → expand to ``name_0``, ``name_1``, ...
                    else:
                        for ch in range(channel_dim):
                            channel_names.append(f"{dset_name}_{ch}")

            filter_in_keywords = self.data_config["filter_features"]["filter_in_channels"]
            filtered_in_channels = (
                [n for n in channel_names if any(n.startswith(k) for k in filter_in_keywords)]
                if filter_in_keywords
                else channel_names
            )

            filter_cond_in_keywords = self.data_config["conditioning_features"]["conditioning_in_channels"]
            filtered_cond_in_channels = (
                [
                    n
                    for n in channel_names
                    if any(n.startswith(k) for k in filter_cond_in_keywords)
                ]
                if filter_cond_in_keywords
                else None
            )

            filter_out_keywords = self.data_config["filter_features"]["filter_out_channels"]
            filtered_out_channels = (
                [n for n in channel_names if any(n.startswith(k) for k in filter_out_keywords)]
                if filter_out_keywords
                else channel_names
            )

            self.data_config["filter_features"]["filter_in_channels"] = filtered_in_channels
            self.data_config["filter_features"]["filter_out_channels"] = filtered_out_channels
            self.data_config["conditioning_features"]["conditioning_in_channels"] = filtered_cond_in_channels            
            
            # -------------------------------
            # Also log constant parameters
            # -------------------------------
            def _flatten_dict(d, parent_key=""):
                """Recursively flattens a nested dict using dot notation keys."""
                flat = {}
                for k, v in d.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    if isinstance(v, dict):
                        flat.update(_flatten_dict(v, new_key))
                    else:
                        flat[new_key] = v
                return flat

            # Gather all major config sections that influence a run
            flat_cfg = {}
            for section_name in ["model_config", "data_config", "train_config"]:
                cfg_section = getattr(self, section_name, None)
                
                if isinstance(cfg_section, Mapping):
                    flat_cfg.update(_flatten_dict(cfg_section, section_name))

            # Complete parameter list (including both constant and trial-sampled values)
            def _sanitize(value):
                """Prepare values for JSON serialization while preserving numeric precision."""
                # Handle mappings and sequences recursively
                if isinstance(value, Mapping):
                    return {k: _sanitize(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [_sanitize(v) for v in value]

                # Convert numpy scalars to native Python types if numpy is available
                try:
                    import numpy as np  # local import to avoid hard dep if not installed
                    if isinstance(value, (np.integer, np.floating)):
                        value = value.item()
                except ModuleNotFoundError:
                    pass

                return value

            complete_params = _sanitize(flat_cfg)  # include everything; no filtering
            formatted = json.dumps(complete_params, indent=2, sort_keys=True, default=str)
            if IS_MAIN_PROCESS:
                logger.info("All Config Params (%d):\n%s", len(complete_params), formatted)

        if self.hp_search_backend == HPSearchBackend.SIGOPT and IS_MAIN_PROCESS:
            logger.info(f"SigOpt Assignments: {trial.assignments}")
        if self.hp_search_backend == HPSearchBackend.WANDB and IS_MAIN_PROCESS:
            logger.info(f"W&B Sweep parameters: {trial}")
        if self.is_deepspeed_enabled:
            if self.args.deepspeed is None:
                raise ValueError("For sweeps with deepspeed, `args.deepspeed` must be set")

            self.accelerator.free_memory()

            # Rebuild the deepspeed config to reflect the updated training parameters
            from accelerate.utils import DeepSpeedPlugin

            from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

            self.args.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.args.deepspeed)
            self.args.hf_deepspeed_config.trainer_config_process(self.args)
            self.args.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.args.hf_deepspeed_config)

            # From 1.0 on, we need to fully wipe the DS plugin when doing sweeps.
            # Simply calling `_reset_state` is enough and doesn't need a version pin.
            AcceleratorState()._reset_state()

        self.create_accelerator_and_postprocess()

        # Recreate datasets so they reflect the updated hyper-parameters
        self._rebuild_datasets()

        try:
            # --------------------------------------------------------------
            # Determine the *trial-specific* output directory.  We build the
            # directory name from one of the following (in order of priority):
            #   1. The ``hp_name`` callable provided to ``hyperparameter_search``.
            #   2. The Optuna trial number (``trial.number``).
            #   3. Fallback: reuse ``self.args.output_dir`` directly.
            # --------------------------------------------------------------
            trial_name = None

            # (1) User-supplied naming function via ``hp_name`` ----------------
            hp_name_fn = getattr(self, "hp_name", None)
            if callable(hp_name_fn):
                try:
                    trial_name = hp_name_fn(trial)  # may raise / return None
                except Exception:
                    trial_name = None

            # (2) Optuna / backend default -------------------------------------
            if trial_name is None:
                if hasattr(trial, "number"):
                    trial_name = f"trial{trial.number}"
                elif isinstance(trial, dict) and "number" in trial:
                    trial_name = f"trial{trial['number']}"

            # Build final path --------------------------------------------------
            trial_output_dir = (
                os.path.join(self.args.output_dir, trial_name)
                if trial_name is not None
                else self.args.output_dir
            )

            if IS_MAIN_PROCESS:
                os.makedirs(trial_output_dir, exist_ok=True)

            # --------------------------------------------------------------
            # Convert (possibly OmegaConf) DictConfig → regular Python container
            # before serialising to JSON.  We resolve all interpolations so
            # the stored config is fully explicit.
            # --------------------------------------------------------------
            cfg_serialisable = None
            try:
                from omegaconf import DictConfig as _DC, OmegaConf as _OC  # type: ignore

                if isinstance(self.data_config, _DC):
                    cfg_serialisable = _OC.to_container(self.data_config, resolve=True)
            except ModuleNotFoundError:
                # ΩConf not installed – fall back to naive conversion
                cfg_serialisable = None

            if cfg_serialisable is None:
                # Generic, best-effort deep conversion for Mapping / sequences
                def _convert(obj):
                    if isinstance(obj, Mapping):
                        return {k: _convert(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [_convert(v) for v in obj]
                    # NumPy scalars → Python scalars
                    try:
                        import numpy as _np  # local import
                        if isinstance(obj, (_np.integer, _np.floating)):
                            return obj.item()
                    except ModuleNotFoundError:
                        pass
                    return obj

                cfg_serialisable = _convert(self.data_config)

            # Finally, write the JSON file (default=str handles residual objects)
            if IS_MAIN_PROCESS:
                with open(os.path.join(trial_output_dir, "data_config.json"), "w") as fp:
                    json.dump(cfg_serialisable, fp, indent=2, default=str)
        except Exception as exc:
            # Do not interrupt hyper-parameter search if logging fails; just warn.
            logger.warning(f"Failed to save data_config.json: {exc}")

    ### overrides the one in the base class from transformers library (NOTE: Not updated yet to 4.56.0 version of the base class)
    def hyperparameter_search(
        self,
        hp_space: Optional[Callable[["optuna.Trial"], dict[str, float]]] = None,
        compute_objective: Optional[Callable[[dict[str, float]], float]] = None,
        n_trials: int = 20,
        direction: Union[str, list[str]] = "minimize",
        backend: Optional[Union["str", HPSearchBackend]] = None,
        hp_name: Optional[Callable[["optuna.Trial"], str]] = None,
        **kwargs,
    ) -> Union[BestRun, list[BestRun]]:
        """
        Launch hyperparameter search with support for nested configuration parameters.

        Parameters
        ----------
        hp_space : Optional[Callable[[optuna.Trial], Dict[str, float]]]
            Function defining the hyperparameter search space.
        compute_objective : Optional[Callable[[Dict[str, float]], float]]
            Function computing the objective to optimize.
        n_trials : int, default=20
            Number of trial runs to test.
        direction : Union[str, List[str]], default="minimize"
            If it's single objective optimization, direction is `str`, can be `"minimize"` or `"maximize"`, you
            should pick `"minimize"` when optimizing the validation loss, `"maximize"` when optimizing one or
            several metrics. If it's multi objectives optimization, direction is `List[str]`, can be List of
            `"minimize"` and `"maximize"`, you should pick `"minimize"` when optimizing the validation loss,
            `"maximize"` when optimizing one or several metrics.
        backend : Optional[Union[str, HPSearchBackend]]
            Hyperparameter search backend to use: "optuna", "ray", "sigopt", "wandb"
        hp_name : Optional[Callable[[optuna.Trial], str]]
            Function defining trial/run names.
        **kwargs
            Additional backend-specific keyword arguments.

        Returns
        -------
        Union[BestRun, List[BestRun]]
            Best run(s) information and study object.
        """
        if backend is None:
            backend = default_hp_search_backend()
        backend = HPSearchBackend(backend)
        backend_obj = ALL_HYPERPARAMETER_SEARCH_BACKENDS[backend]()
        backend_obj.ensure_available()
        self.hp_search_backend = backend
        if self.model_init is None:
            raise RuntimeError(
                "To use hyperparameter search, you need to pass your model through a model_init function."
            )

        self.hp_space = backend_obj.default_hp_space if hp_space is None else hp_space
        self.hp_name = hp_name
        self.compute_objective = default_compute_objective if compute_objective is None else compute_objective

        best_run, study = backend_obj.run(self, n_trials, direction, **kwargs)

        self.hp_search_backend = None
        return best_run, study