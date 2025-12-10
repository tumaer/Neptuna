"""
Custom Trainer Implementation.

This module provides an extended Trainer class that builds upon the HuggingFace
transformers Trainer with specialized features.

Key Features:
- Autoregressive prediction with configurable rollout steps
- Pushforward training strategy for improved temporal stability
- Residual learning with multiple configuration modes
- Custom callback system with visualization and logging
- Support for conditioning inputs and steady-state prediction
- Hyperparameter search with nested configuration support

Classes:
    Trainer: Extended trainer class with scientific computing features
    
Training Strategies:
    - Pushforward Training: Gradually increases unroll steps during training with some probabilities to introduce noise generated during rollouts of validation/testing
    - Residual Learning: Supports multiple residual computation modes
    - Autoregressive Rollout: Multi-step prediction for evaluation
    - Steady-State Prediction: Specialized handling for equilibrium problems

Evaluation Modes:
    - Window-based Loss: Computes loss per evaluation window
    - Rollout Evaluation: Multi-step autoregressive prediction
    - Memory-efficient Processing: Handles large datasets with batch accumulation

Integration Points:
    - HuggingFace transformers training infrastructure
    - Custom datasets
    - Weights & Biases logging and visualization
    - Hyperparameter optimization frameworks
"""
import torch
from torch import nn
from typing import List, Optional, Dict, Tuple, Union, Any
from transformers.trainer import *
from transformers import Trainer as Trainer_
import numpy as np
from utils.load_data import fetch_dataset
from utils.custom_callbacks import WandbCallback #custom callbacks
from utils.custom_callbacks import CallbackHandler 
from transformers.integrations.integration_utils import WandbCallback as WandbCallback_
from utils.trainer_utils import EvalPrediction
from utils.loss_utils import fetch_loss_metric
from omegaconf import OmegaConf
from collections.abc import Mapping  # locally import to avoid top-of-file change
import json
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"           
import h5py
import time


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
    pushforward_config : Dict
        Configuration for pushforward training strategy.
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
        self.loss_config = kwargs.pop("loss_config", None)

        super().__init__(**kwargs)

        #self.original_label_seq_len = self.data_config.sequence_info[1] #number of predicted timesteps from the model (#no rollout timesteps considered)
        
        self.get_prediction_loss_for_eval_windows = False #TODO: Find a way to not hardcode this.

        self.residual_config = self.data_config["residual_config"]
        # Push-forward configuration
        self.pushforward_config = self.train_config["pushforward_config"] if self.train_config is not None else None
        # Enabled flag now lives inside the pushforward_config dict (key: "enabled").
        # If the key is absent we assume push-forward should run when a config is supplied.
        self.pushforward_enabled = self.pushforward_config is not None and bool(self.pushforward_config.get("enabled", True))

        # ------------------------------------------------------------------
        # Precompute conditioning flags and a fast model-forward function to
        # avoid repeated if/else branches during every forward call.
        # ------------------------------------------------------------------
        conditioning_features = self.data_config["conditioning_features"]
        self._use_cond_input_data = conditioning_features["conditioning_in_channels"] is not None
        self._use_cond_parameters = conditioning_features["include_conditioning_parameters"]

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

        # Capture any callbacks created by the base Trainer, then re-instantiate using our custom
        # CallbackHandler class so the overridden methods are used.
        existing_callbacks = getattr(self.callback_handler, "callbacks", [])
        # Re-instantiate using our custom CallbackHandler class so the overridden methods are used.
        self.callback_handler = CallbackHandler(
            existing_callbacks,
            self.model,
            getattr(self, "tokenizer", None),
            self.optimizer,
            self.lr_scheduler,
        )

        # Initialize loss function
        if self.loss_config is not None:
            # Create a config object for the loss function
            full_cfg = OmegaConf.create({
                "loss_config": self.loss_config,
                "data_config": self.data_config
            })
            self.loss_fn = fetch_loss_metric(full_cfg)
            logger.info("Initialized composite loss function from config")

            # Move loss function to appropriate device
            device = self.args.device if hasattr(self.args, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.loss_fn = self.loss_fn.to(device)
        else:
            # Fallback to MSE if no loss config provided
            self.loss_fn = None
            logger.warning("No loss_config provided, using default MSE loss")

        # Flag to indicate if detailed losses should be collected for adaptive weighting
        self._collect_detailed_losses = getattr(self, '_collect_detailed_losses', False)

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
            pushforward_config=self.train_config["pushforward_config"] if self.train_config is not None else None,
            n_eval_rollouts=self.train_config["n_eval_rollouts"] if self.train_config is not None else None,
            n_infer_rollouts=self.infer_config["n_infer_rollouts"] if self.infer_config is not None else None,
        )
        
    ##overrides the one in the  base class from transformers library
    def get_train_dataloader(self) -> DataLoader:
        """
        Create and return the training dataloader with custom sampling strategy.

        Returns
        -------
        DataLoader
            Configured training dataloader with custom sampler and collation.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator #NOTE: Using the default collator from the base class.
        
        ## NOTE:commented out code from the base class
        #if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        #    train_dataset = self._remove_unused_columns(train_dataset, description="training")
        #else:
        #   data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler() ## here we create a custom sampler inside this function, inside this function, the sampler is set to RandomSampler if accelerator_config={"use_seedable_sampler": False} 
            #and if accelerator_config={"use_seedable_sampler": True} then SeedableRandomSampler (inside accelerate>data_loader.py, this SeedableRandomSampler is a subclass of RandomSampler and is required for distributed training). 
            #(shuffle  CANNOT BE SET to true or false (Pytorch doesnt allow it) if a custom sampler is used..  the sampler is set to RandomSampler which will provide random indices inside __get_item__ function of the dataloader), 
            # in a similar way, the _eval_sampler function has a SequentialSampler, regardless of the shuffle being true or false.
            #"From pytorch documentation:  If sampler is specified, :attr:`shuffle` must not be specified."
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params)) 
    
    ##overrides the one in the base class from transformers library
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Parameters
        ----------
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
    
        Returns
        -------
        DataLoader
            Configured evaluation dataloader with sequential sampling.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator

        ##commented out code from the base class
        # if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
        #     eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        # else:
        #     data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)
    
     ##overrides the one in the base class from transformers library
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        data_collator = self.data_collator

        ##commented out code from the base class
        # if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
        #     test_dataset = self._remove_unused_columns(test_dataset, description="test")
        # else:
        #     data_collator = self._get_collator_with_removed_columns(data_collator, description="test")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # We use the same batch_size as for eval.
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))
    ##custom function, not inside transformers library
    def _forward_model_train(self, model, inputs):
        """
        Forward pass for training with pushforward strategy support.

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
            pushforward_unroll_steps : Number of pushforward steps performed.
        """
        batch_size, _, _, *spatial_dims = inputs["input_data"].shape
        base_value = inputs["input_data"][:,-1:,]
        if not self.data_config.is_steady_state_prediction:
            #TODO: Add more training strategies here in continuation of the if statement.
            pushforward_unroll_steps = 0
            if self.pushforward_enabled and self.pushforward_config is not None:
                #########################################################
                #Pushforward trick (for training)
                #########################################################
                num_steps_in_one_epoch = self.state.max_steps//self.state.num_train_epochs
                pushforward_unroll_steps = self.select_pushforward_unroll_steps_for_training(current_epoch = self.state.global_step//num_steps_in_one_epoch)
                with torch.no_grad(): #comment this out for multi-step autoregressive training
                    for unroll_step in range(pushforward_unroll_steps):
                        #print(f"Pushforward unroll step {unroll_step+1} of {pushforward_unroll_steps}")
                        prediction = self._model_forward_fn(model, inputs)
                        prediction = prediction.reshape(batch_size, self.data_config["sequence_info"][1], len(self.data_config["filter_features"]["filter_out_channels"]), *spatial_dims)
                        
                        if self.residual_config is not None:
                            #NOTE: In all the three cases, we need the raw values to do rollout
                            raw_prediction, base_value = self._compute_raw_prediction(prediction, base_value)
                        
                        if (self.data_config.sequence_info[1] >= self.data_config.sequence_info[0]): #label_sequence length >= input_sequence length
                            inputs = {
                                        **inputs,
                                        **{ # This part replaces the "input_data" of input with the prediction of the model. 
                                            # So the new input is the output from the previous step.
                                            #prediction.shape = torch.Size([B, label_seq_len, C_labels, x_resolution, y_resolution])
                                                "input_data": (
                                                    prediction[:,(self.data_config.sequence_info[1] - self.data_config.sequence_info[0]):,].detach() #slice the outputs so as to extract the input_sequence.
                                                ) if self.residual_config is None else (
                                                    raw_prediction[:,(self.data_config.sequence_info[1] - self.data_config.sequence_info[0]):,].detach() #slice the outputs so as to extract the input_sequence.
                                                )
                                        },
                                }
                            
                        else: #input_sequence length > label_sequence length (the more usual case)
                                inputs = {
                                **inputs,
                                **{ #this part replaces the "input_data" of input with the output of the model and concatenates part of the input whch is missing to make a complete input sequence
                                    #So the new input is the output from the previous step.
                                    "input_data": (
                                        torch.cat([inputs["input_data"][:,self.data_config.sequence_info[1]:,], prediction.detach()], dim=1)
                                    ) if self.residual_config is None else (
                                        torch.cat([inputs["input_data"][:,self.data_config.sequence_info[1]:,], raw_prediction.detach()], dim=1)
                                    )
                                },
                                        }
        else: 
            #steady state prediction
            pushforward_unroll_steps = 0
            
        # NOTE:compute chain restored, the input_data is corrupted by the pushforward rollout steps (if any).
        prediction = self._model_forward_fn(model, inputs)
        
        prediction = prediction.reshape(batch_size, self.data_config["sequence_info"][1], len(self.data_config["filter_features"]["filter_out_channels"]), *spatial_dims)
        
        if self.residual_config is not None and (self.residual_config["add_predicted_value_with_raw_loss"] or self.residual_config["add_base_value_with_raw_loss"]):
            #NOTE: For the cases: add_predicted_value_with_raw_loss or add_base_value_with_raw_loss, we need the raw values before loss is computed inside compute_loss().
            raw_prediction, base_value = self._compute_raw_prediction(prediction, base_value)
            prediction = raw_prediction

        return prediction, pushforward_unroll_steps

    ##overrides the one in the base class from transformers library
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): 
        """
        Compute training loss with pushforward strategy support.

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
        # Pushforward trick (for training)
        ########################################################
        prediction, pushforward_unroll_steps = self._forward_model_train(model, inputs)
        
        # Get labels for the current rollout step
        labels = inputs["label_including_rollouts"][
            :,
            (self.data_config.sequence_info[1]*pushforward_unroll_steps):
            (self.data_config.sequence_info[1]*(pushforward_unroll_steps+1))
        ]
        
        loss, detailed = self.loss_fn(
            model=model,
            predictions=prediction,
            labels=labels,
            return_detailed=self._collect_detailed_losses
        )

        # Store detailed losses if needed
        if self._collect_detailed_losses and detailed is not None:
            if not hasattr(self, '_detailed_loss_accumulator'):
                self._detailed_loss_accumulator = {}
            
            for component_name, component_detailed in detailed.items():
                # Extract loss value
                loss_value = component_detailed.get('total', component_detailed) if isinstance(component_detailed, dict) else component_detailed
                
                # Convert to detached GPU tensor
                loss_scalar = loss_value.detach() if torch.is_tensor(loss_value) else torch.tensor(float(loss_value), device=loss.device)
                
                # Accumulate
                if component_name not in self._detailed_loss_accumulator:
                    self._detailed_loss_accumulator[component_name] = []
                self._detailed_loss_accumulator[component_name].append(loss_scalar)
            
        return (loss, prediction) if return_outputs else loss

    ##custom function, not inside transformers library
    def _forward_model_eval_or_test(self, model, inputs):
        """
        Forward pass for evaluation or testing without pushforward strategy.

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
    
    #NOTE: There are two functions for eval_loss: 
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

    def select_pushforward_unroll_steps_for_training(self, current_epoch):
        """
        Select number of pushforward unroll steps based on training epoch and configuration.

        Parameters
        ----------
        current_epoch : int
            Current training epoch.

        Returns
        -------
        int
            Number of unroll steps to perform for pushforward training.

        Raises
        ------
        ValueError
            If current epoch is before the first deciding epoch threshold.
        """
        current_epoch_tensor = torch.tensor(current_epoch)
        deciding_epochs = torch.tensor(self.train_config["pushforward_config"]["deciding_epochs"])
        max_unrolls = self.train_config["pushforward_config"]["max_allowed_unroll_steps"]
        relative_probabilities = self.train_config["pushforward_config"]["relative_probabilities"]

        assert all(deciding_epochs[i] <= deciding_epochs[i + 1] for i in range(len(deciding_epochs) - 1))

        idx = (current_epoch_tensor > deciding_epochs).sum().item()

        if idx == 0:
            raise ValueError("Training step is before first step threshold in pushforward config.")

        unroll_choices = torch.tensor(max_unrolls[:idx])
        prob_choices = torch.tensor(relative_probabilities[:idx])
        
        # Convert from relative probabilities to absolute probabilities
        prob_choices = prob_choices / prob_choices.sum()

        gen = torch.Generator().manual_seed(42+current_epoch)

        sample_idx = torch.multinomial(prob_choices, num_samples=1, generator=gen).item()
        unroll_steps = unroll_choices[sample_idx]

        # Convert the 0-D tensor to a native Python int before returning so callers can use it
        # directly in range() and arithmetic operations without needing an explicit .item() call.
        return unroll_steps.item()

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
                    loss = loss.mean().detach() #mean() is used when: self.output_all_steps = True which results in loss being a tensor of shape (num_rollout_steps+1,) and we take the mean

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
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0: #self.args.past_index = -1 by default
                        self._past = outputs[self.args.past_index - 1]

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
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
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

        RANK = int(os.environ.get("LOCAL_RANK", -1))
        IS_MAIN_PROCESS = RANK in [-1, 0]
        if IS_MAIN_PROCESS:
            logger.info(f"\n***** Running {description} *****")
            if has_length(dataloader):
                logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            else:
                logger.info("  Num examples: Unknown")
            logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_conditioning_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0
        #########################################################
        # Main evaluation loop
        #########################################################
        for step, inputs in enumerate(dataloader):
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
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function(inputs_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            #NOTE: The following is added on top of the base class
            #########################################################
            if conditioning_input_decode is not None:
                conditioning_input_decode = self.accelerator.pad_across_processes(conditioning_input_decode, dim=1, pad_index=-100)
                conditioning_input_decode = self.gather_function(conditioning_input_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_conditioning_inputs.add(conditioning_input_decode)
            #########################################################
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function(labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    #NOTE: inputs is a dict which has the input_data and conditioning_input_data
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()
                all_conditioning_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays() #all_losses.shape = torch.Size([B*(steps+1) , ]) 
        all_preds = all_preds.get_arrays() #all_preds.shape = torch.Size([B*(steps+1), n_eval_rollouts+1, label_seq_length, C_output, x_resolution, y_resolution, ...]) 
        all_labels = all_labels.get_arrays() #all_labels.shape = torch.Size([B*(steps+1), (n_eval_rollouts+1)*label_seq_length, C_output, x_resolution, y_resolution, ...]) 
        all_inputs = all_inputs.get_arrays() #all_inputs.shape = torch.Size([B*(steps+1), input_seq_length, C_input, x_resolution, y_resolution, ...]) 
        all_conditioning_inputs = all_conditioning_inputs.get_arrays() #all_conditioning_inputs.shape = torch.Size([B*(steps+1), conditioning_seq_length, C_conditioning, x_resolution, y_resolution, ...]) 

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

        # Metrics!
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
        ##NOTE: all_inputs is the additional return argument compared to the evaluation_loop() function in the base class.

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

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        #########################################################
        #NOTE: Main evaluation loop
        eval_loop_start_time = time.time()
        output, input, conditioning_input = eval_loop(
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
            # NOTE: kwargs added to be used in PlotOnEvalAndSaveCallback()
            predictions=output.predictions,
            labels=output.label_ids,
            inputs=input,
            conditioning_inputs=conditioning_input,
            eval_dataset=eval_dataset,
            data_config=self.data_config,
            train_config=self.train_config,
            output_log_config=self.output_log_config,
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)
        #NOTE: stop training if NaN is encountered in the loss and set the control flags to False.
        if self.control.should_training_stop_due_to_nan and self.state.epoch<self.train_config['num_train_epochs']:
            self.control.should_evaluate = False
            self.control.should_save = False
            self.control.should_plot = False

        return output.metrics

    ### overrides the one in the base class from transformers library
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
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

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
            test_dataloader = self.get_test_dataloader(test_dataset)
            start_time = time.time()

            eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
            output, input, conditioning_input = eval_loop(
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
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
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
            #NOTE: Implemented scientific notation for logs (improvement over base class)
            #logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            loss_value = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
            # Format the loss in scientific notation with 4-digit precision (e.g. 1.2345e-04)
            # Convert back to float so W&B logs a numeric value while retaining 4-digit scientific notation.
            #logs["loss"] = float(f"{loss_value:.4e}")
            logs["loss"] = loss_value
            if grad_norm is not None:
                # logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                grad_norm_value = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                #logs["grad_norm"] = float(f"{grad_norm_value:.4e}")
                logs["grad_norm"] = grad_norm_value
            # logs["learning_rate"] = self._get_learning_rate()
            learning_rate_value = self._get_learning_rate()
            #logs["learning_rate"] = float(f"{learning_rate_value:.4e}")
            logs["learning_rate"] = learning_rate_value

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time) #NOTE: logs into wandb for training
        
        metrics = None
        RANK = int(os.environ.get("LOCAL_RANK", -1))
        IS_MAIN_PROCESS = RANK in [-1, 0]

        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval) 
            if IS_MAIN_PROCESS:
                logger.info(f"Model checkpointing is done based on: eval_{self.args.metric_for_best_model}")
            ##NOTE: added predictions, labels, inputs as additional return arguments compared to the base class.
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

        RANK = int(os.environ.get("LOCAL_RANK", -1))
        if self.control.should_plot and (RANK == 0 or RANK == -1):  
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
        
        if self.loss_config is not None:
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
                
                # Convert to dict (keeps metric_params separate)
                loss_config_dict = OmegaConf.to_container(self.loss_config, resolve=True)
                
                # Add current weights to each component
                if self.loss_fn is not None:
                    weight_dict = self.loss_fn.get_weight_dict()
                    
                    for component in loss_config_dict['loss']['components']:
                        comp_name = component.get('name', component['type'])
                        if comp_name in weight_dict:
                            component['current_weights'] = _tensorize_for_json(weight_dict[comp_name])
                
                # Save
                with open(loss_config_path, 'w') as f:
                    json.dump(loss_config_dict, f, indent=2)
                
                logger.info(f"Saved loss_config to {loss_config_path}")
                
            except Exception as e:
                logger.warning(f"Failed to save loss_config: {e}")
    
    ### overrides the one in the base class from transformers library
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
        RANK = int(os.environ.get("LOCAL_RANK", -1))
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

def test_pushforward_unroll_steps():
    """
    Test function to verify the pushforward unroll steps selection logic.
    Tests various scenarios including:
    1. Normal case with multiple epochs
    2. Edge case at first deciding epoch
    3. Edge case at last deciding epoch
    4. Error case before first deciding epoch
    5. Progression of unroll steps across epochs
    """
    import torch
    import numpy as np
    from collections import defaultdict

    # Mock trainer class for testing
    class MockTrainer:
        def __init__(self, pushforward_config):
            self.pushforward_config = pushforward_config

        def select_pushforward_unroll_steps_for_training(self, current_epoch):
            current_epoch_tensor = torch.tensor(current_epoch)
            deciding_epochs = torch.tensor(self.pushforward_config["deciding_epochs"])
            max_unrolls = self.pushforward_config["max_allowed_unroll_steps"]
            relative_probabilities = self.pushforward_config["relative_probabilities"]

            assert all(deciding_epochs[i] <= deciding_epochs[i + 1] for i in range(len(deciding_epochs) - 1))

            idx = (current_epoch_tensor > deciding_epochs).sum().item()

            if idx == 0:
                raise ValueError("Training step is before first step threshold in pushforward config.")

            unroll_choices = torch.tensor(max_unrolls[:idx])
            prob_choices = torch.tensor(relative_probabilities[:idx])
            
            # Normalize explicitly
            prob_choices = prob_choices / prob_choices.sum()

            # Fixed local RNG with seed 42
            gen = torch.Generator().manual_seed(42+current_epoch)

            sample_idx = torch.multinomial(prob_choices, num_samples=1, generator=gen).item()
            unroll_steps = unroll_choices[sample_idx]

            # Convert the 0-D tensor to a native Python int before returning so callers can use it
            # directly in range() and arithmetic operations without needing an explicit .item() call.
            return unroll_steps.item()

    # Test configuration
    pushforward_config = {
        "deciding_epochs": [-1, 2, 50],  # Epochs where new unroll steps become available
        "max_allowed_unroll_steps": [0, 1, 2],        # Maximum unroll steps at each stage
        "relative_probabilities": [7, 2, 1]  # Relative probabilities for each stage
    }

    trainer = MockTrainer(pushforward_config)
    
    # Test 1: Normal case - multiple epochs
    print("\nTest 1: Normal case - multiple epochs")
    unroll_counts = defaultdict(int)
    n_samples = 1000
    
    # for _ in range(n_samples):
    #     unroll_steps = trainer.select_pushforward_unroll_steps_for_training(25)  # Between 20 and 30
    #     unroll_counts[unroll_steps.item()] += 1
    
    # print("Unroll step distribution:")
    # for steps, count in sorted(unroll_counts.items()):
    #     print(f"Steps {steps}: {count/n_samples:.2%}")

    # # Test 2: Edge case - at first deciding epoch
    # print("\nTest 2: Edge case - at first deciding epoch")
    # unroll_steps = trainer.select_pushforward_unroll_steps_for_training(10)
    # print(f"Unroll steps at epoch 10: {unroll_steps}")

    # # Test 3: Edge case - at last deciding epoch
    # print("\nTest 3: Edge case - at last deciding epoch")
    # unroll_steps = trainer.select_pushforward_unroll_steps_for_training(30)
    # print(f"Unroll steps at epoch 30: {unroll_steps}")

    # # Test 4: Error case - before first deciding epoch
    # print("\nTest 4: Error case - before first deciding epoch")
    # try:
    #     unroll_steps = trainer.select_pushforward_unroll_steps_for_training(5)
    # except ValueError as e:
    #     print(f"Expected error: {e}")

    # Test 5: Simple progression test
    print("\nTest 5: Epoch vs Unroll Steps")
    print("Epoch\tUnroll Steps")
    print("-" * 20)
    
    # Store results for 5 runs
    all_runs = []
    for run in range(5):
        run_results = []
        for epoch in range(500):
            try:
                unroll_steps = trainer.select_pushforward_unroll_steps_for_training(epoch)
                run_results.append(unroll_steps)
            except ValueError:
                run_results.append(-1)  # Use -1 to represent error
        all_runs.append(run_results)
    
    # Compare results across runs
    print("\nComparing results across 5 runs:")
    print("Epoch\tRun1\tRun2\tRun3\tRun4\tRun5\tAll Same?")
    print("-" * 70)
    
    for epoch in range(500):
        values = [run[epoch] for run in all_runs]
        all_same = all(v == values[0] for v in values)
        print(f"{epoch}\t{values[0]}\t{values[1]}\t{values[2]}\t{values[3]}\t{values[4]}\t{all_same}")

if __name__ == "__main__":
    test_pushforward_unroll_steps()