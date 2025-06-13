import torch
from torch import nn
from typing import List, Optional, Dict, Tuple, Union, Any
from transformers.trainer import *
from transformers import Trainer as Trainer_
from utils.plot_progress import plot_examples
import numpy as np
from utils.feature_utils import re_normalize_data
class Trainer(Trainer_):
    def __init__(self, model_config, data_config, train_config, **kwargs):
        super().__init__(**kwargs)
        self.eval_or_test_rollout_steps = None
        self.output_all_steps = False
        self.data_config = data_config
        self.model_config = model_config
        self.pushforward_config = train_config["pushforward"]
        self.plot_after_epoch = train_config["plot_after_epoch"]
        self.original_label_seq_len = self.data_config.sequence_info[1] #number of predicted timesteps from the model (#no rollout timesteps considered)
        
    ##overrides the one in the  base class from transformers library
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclassing to use the default collator from the base class.
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
    
    ##overrides the one in the  base class from transformers library
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
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
    
    ##custom function, not inside transformers library
    def _forward_model_train(self, model, inputs):  
        #########################################################
        #Pushforward trick (for training)
        #########################################################
        num_steps_in_one_epoch = self.state.max_steps//self.state.num_train_epochs
        pushforward_unroll_steps = self.select_pushforward_unroll_steps_for_training(current_epoch = self.state.global_step//num_steps_in_one_epoch)
        channel_difference = (
                self.data_config.in_channels > self.data_config.out_channels) #usually channel_difference = False
        #add a warning if channel_difference = True
        if channel_difference:
            warnings.warn("Channel difference is True, which means that the number of input and label channels are different")
        
        batch_size, _, _, *spatial_dims = inputs["input_data"].shape
        
        with torch.no_grad(): #comment this out for multi-step autoregressive training
            for unroll_step in range(pushforward_unroll_steps):
                #print(f"Pushforward unroll step {unroll_step+1} of {pushforward_unroll_steps}")
                prediction = model(inputs["input_data"])
                
                prediction = prediction.reshape(batch_size, self.data_config["sequence_info"][1], self.data_config["out_channels"], *spatial_dims)
                
                if (self.data_config.sequence_info[1] >= self.data_config.sequence_info[0]): #label_sequence length >= input_sequence length
                    inputs = {
                                **inputs,
                                **{ # This part replaces the "input_data" of input with the output of the model. 
                                    # So the new input is the output from the previous step.
                                    #prediction.shape = torch.Size([B, label_seq_len, C_labels, x_resolution, y_resolution])
                                    "input_data": (
                                        prediction[:,(self.data_config.sequence_info[1] - self.data_config.sequence_info[0]):,].detach() #slice the outputs so as to extract the input_sequence.
                                        if not channel_difference
                                        else torch.cat( 
                                            [
                                                prediction[:,(self.data_config.sequence_info[1] - self.data_config.sequence_info[0]):,].detach() ,
                                                inputs["input_data"][:,:,self.data_config.out_channels:], #adding back the channels which are not predicted like Re or Ma
                                            ],
                                            dim=2, 
                                            #concatenate along the channel dimension (dim=2) : the first dimension is the batch dimension, the second dimension is the time sequence dimension,
                                            # the third dimension is the channel dimension and the rest are the spatial dimensions.
                                        )
                                    )
                                },
                        }
                    
                else: #input_sequence length > label_sequence length (the more usual case)
                    inputs = {
                        **inputs,
                        **{ #this part replaces the "input_data" of input with the output of the model. 
                            #So the new input is the output from the previous step.
                            "input_data": (
                                torch.cat([inputs["input_data"][:,self.data_config.sequence_info[1]:,], prediction.detach()], dim=1) #slice the outputs so as to extract the input_sequence.
                                if not channel_difference
                                else torch.cat( 
                                    [
                                        torch.cat([inputs["input_data"][:,self.data_config.sequence_info[1]:,], prediction.detach()], dim=1),
                                        inputs["input_data"][:,:,self.data_config.out_channels:],
                                    ],
                                    dim=2, 
                                    #concatenate along the channel dimension (dim=2) : the first dimension is the batch dimension, the second dimension is the time sequence dimension,
                                    #the third dimension is the channel dimension and the rest are the spatial dimensions.
                                )
                            )
                        },
                    }

        prediction = model(inputs["input_data"]) #compute chain restored, the input_data is corrupted by the pushforward rollout steps.
        prediction = prediction.reshape(batch_size, self.data_config["sequence_info"][1], self.data_config["out_channels"], *spatial_dims)
        return prediction

    ##custom function, not inside transformers library
    def _forward_model_eval_or_test(self, model, inputs):  
        prediction = model(inputs["input_data"])
        batch_size, input_seq_len, input_channels, *spatial_dims = inputs["input_data"].shape
        prediction = prediction.reshape(batch_size, self.data_config["sequence_info"][1], self.data_config["out_channels"], *spatial_dims)
        return prediction
    
    ##overrides the one in the  base class from transformers library
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):   
        #return_outputs is true only when doing eval or test. By default it is false for training.
        #########################################################
        #Autoregressive prediction (for eval and test)
        #########################################################
        if self.control.should_evaluate: #should_evaluate is true for eval and test
            #here everything happens with torch.no_grad()
            channel_difference = (self.data_config.in_channels > self.data_config.out_channels) 
            #Here we assume that the channel which is not predicted in the output is the last channel in the input (like Re or Ma).
            ## inputs.keys() = dict_keys(['input_data', 'labels',])
            ## inputs['input_data'].shape = torch.Size([B, C_input, x_resolution, y_resolution, ...]) 
            ## inputs['labels'].shape = torch.Size([B, C_labels, x_resolution, y_resolution, ...])
            if self.output_all_steps: #this is set to true when self.rollout_steps is set in main.py
                losses_ = []
                predictions_ = []
            else:
                total_loss = 0
            
            for i in range(self.rollout_steps+1): #+1 because at least one bunch of outputs is always predicted and rollout_steps is added on top of that.
                #logger.debug(f"Eval/Test rollout step {i+1} of {self.rollout_steps+1}") #TODO: uncomment this later
                prediction = self._forward_model_eval_or_test(model,inputs) #prediction.shape = torch.Size([B, label_seq_length,C_labels, x_resolution, y_resolution, ...]) 
                
                loss_fn = nn.functional.mse_loss #this is the eval_loss, which is NOT used for saving the best model.
                loss = loss_fn(prediction, inputs["label_including_rollouts"][:,i*self.data_config.sequence_info[1]:(i+1)*self.data_config.sequence_info[1]]) 
                
                if self.output_all_steps:
                    predictions_.append(prediction.detach()) 
                    losses_.append(loss)
                else:
                    total_loss += loss #loss is added up across all rollout_steps and we obtain a scalar. This is divided by rollout_steps at the end of the "if" statement
                
                #recreate the inputs to be fed to the model for the next step
                if (self.data_config.sequence_info[1] >= self.data_config.sequence_info[0]): #label_sequence length > input_sequence length
                    inputs = {
                        **inputs,
                        **{ #this part replaces the "input_data" of input with the output of the model. 
                            #So the new input is the output from the previous step.
                            "input_data": (
                                prediction[:,(self.data_config.sequence_info[1] - self.data_config.sequence_info[0]):,].detach() #slice the outputs so as to extract the input_sequence.
                                if not channel_difference
                                else torch.cat( 
                                    [
                                        prediction[:,(self.data_config.sequence_info[1] - self.data_config.sequence_info[0]):,].detach() ,
                                        inputs["input_data"][:,:,self.data_config.out_channels:],
                                    ],
                                    dim=2, 
                                    #concatenate along the channel dimension (dim=2) : the first dimension is the batch dimension, the second dimension is the time sequence dimension,
                                    # the third dimension is the channel dimension and the rest are the spatial dimensions.
                                )
                            )
                        },
                }
                else: #input_sequence length > label_sequence length (the more usual case)
                    inputs = {
                        **inputs,
                        **{ #this part replaces the "input_data" of input with the output of the model. 
                            #So the new input is the output from the previous step.
                            "input_data": ( 
                                torch.cat([inputs["input_data"][:,self.data_config.sequence_info[1]:,], prediction.detach()], dim=1) #slice the outputs so as to extract the input_sequence.
                                if not channel_difference
                                else torch.cat( 
                                    [
                                        torch.cat([inputs["input_data"][:,self.data_config.sequence_info[1]:,], prediction.detach()], dim=1),
                                        inputs["input_data"][:,:,self.data_config.out_channels:],
                                    ],
                                    dim=2, 
                                    #concatenate along the channel dimension (dim=2) : the first dimension is the batch dimension, the second dimension is the time sequence dimension,
                                    #the third dimension is the channel dimension and the rest are the spatial dimensions.
                                )
                            )
                        },
                }
                    
            if self.output_all_steps:
                predictions= torch.stack(predictions_, dim=1) #predictions.shape = torch.Size([B, rollout_steps+1, label_seq_length, C_output, *spatial_resolution])
                loss = torch.stack(losses_, dim=0).mean() #mean() across the rollout steps

            else:
                loss = total_loss / (self.rollout_steps+1) #take the mean of the loss across all rollout_steps
           
        ########################################################
        # Pushforward trick (for training)
        ########################################################
        else:
            prediction = self._forward_model_train(model, inputs)
            #compute the training loss here. Assume l2 loss 
            loss_fn = nn.functional.mse_loss
            #loss is computed only for the last rollout (pushforward trick!), therefore no need to update the labels, just slice from the end of the labels_including_rollouts tensor
            loss = loss_fn(prediction, inputs["label_including_rollouts"][:,-self.data_config.sequence_info[1]:,]) 
            #the loss which is printed is rounded-off to 4 decimal places
            #printing happening inside the function: _maybe_log_save_evaluate() #TODO: Check 
            #print(f"loss of unrolled steps: {loss}")
        #return_outputs is true only when doing eval or test. By default it is false for training.
        return (loss, predictions) if return_outputs else loss

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
        
        # Convert from relative probabilities to absolute probabilities
        prob_choices = prob_choices / prob_choices.sum()

        gen = torch.Generator().manual_seed(42+current_epoch)

        sample_idx = torch.multinomial(prob_choices, num_samples=1, generator=gen).item()
        unroll_steps = unroll_choices[sample_idx]

        return unroll_steps

    ##custom function used for eval and testing, not inside transformers library
    def set_rollout_steps(self, rollout_steps=None, output_all_steps=False): 
        self.rollout_steps = rollout_steps 
        if self.rollout_steps is not None and output_all_steps:
            self.output_all_steps = True
    
    ##overrides the one in the  base class from transformers library
    def prediction_step( 
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool, #True if compute_metrics is not provided
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
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
                if has_labels or loss_without_labels: #enters here 
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss( #this has the _model_perdict() function inside it
                            model, inputs, return_outputs=True
                        ) #return_output is true only when doing eval or inference.. By default it is false
                    loss = loss.mean().detach() #mean() is used when: self.output_all_steps = True which results in loss being a tensor of shape (num_rollout_steps,) and we take the mean

                    if isinstance(outputs, dict): 
                        logits = tuple( 
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]# ignores the keys
                        ) 
                    else: # Enters here as outputs is a tensor. logits is the outputs tensor (#NOTE: in the base class it is outputs[1:] as the 0th index is the loss)
                        logits = outputs
                else: ##not sure why this 'else' is needed.
                    # print a warning saying no labels are present
                    warnings.warn("No labels are present, using the model to only generate predictions")
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = self._forward_model_eval_or_test(model, inputs) #this is the only line which is different from the base class
                        ##in the base class it is outputs = model(**inputs),but since we have the autoregressive code as well, we need to use the _forward_model_eval_or_test function
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
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
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

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0
        #########################################################
        # Main evaluation loop
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

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function(losses.repeat(batch_size))  #WHY REPEAT?
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function(inputs_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
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
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples), all_inputs
        ##NOTE: all_inputs is the additional return argument compared to the evaluation_loop() function in the base class.

    ### overrides the one in the base class from transformers library
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
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
        output, input = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
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

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics, output.predictions, output.label_ids, input 
        #NOTE: added output.predictions, output.label_ids, input as additional return arguments compared to the base class.

    ### overrides the one in the base class from transformers library
    def _evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
        metrics, predictions, labels, inputs = self.evaluate(ignore_keys=ignore_keys_for_eval)
        ##NOTE: added predictions, labels, inputs as additional return arguments compared to the base class.
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
        return metrics, predictions, labels, inputs
        ##NOTE: added predictions, labels, inputs as additional return arguments compared to the base class.

    ### overrides the one in the base class from transformers library
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
            ##TODO: Change this to use scientific notation.
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics, predictions, labels, inputs = self._evaluate(trial, ignore_keys_for_eval) 
            logger.info(f"Model checkpointing is done based on: eval_{self.args.metric_for_best_model}")
            ##NOTE: added predictions, labels, inputs as additional return arguments compared to the base class.
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            #########################################################
            ##NOTE: This plotting is not present in the base class
            ##plotting a few random examples to check the progress. 

            ##TODO: Add a if condition for plotting and to plot only after a certain number of epochs/steps.
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            
            len_eval_dataloader, num_eval_rollouts, label_seq_length, channel_dim, *spatial_dims = predictions.shape
            predictions=predictions.reshape(len_eval_dataloader, num_eval_rollouts*label_seq_length, channel_dim, *spatial_dims)

            #plot after a certain number of epochs/steps
            if self.state.epoch >= self.plot_after_epoch:
                # ------------------------------------------------------------------
                # Renormalize inputs, labels and predictions for visualization
                # ------------------------------------------------------------------
                norm_stats = self.data_config["data_normalization_stats"]
                norm_strategy = self.data_config["data_normalization_strategy"]

                # Channel ordering in the dataset 
                channel_names = getattr(self.eval_dataset, "channels", None)

                # Renormalize:
                inputs   = re_normalize_data(inputs, channel_names, norm_stats, norm_strategy)
                labels   = re_normalize_data(labels, channel_names, norm_stats, norm_strategy)
                predictions = re_normalize_data(predictions, channel_names, norm_stats, norm_strategy) 

                plot_examples(inputs, 
                            predictions, 
                            labels, 
                            channel_names,
                            ndim=self.data_config["dimension"],
                            stride=self.data_config["sequence_info"][-1],
                            extra_info= run_dir, 
                            checkpoint_step=self.state.global_step,
                            epoch=round(self.state.epoch, 3),
                            num_examples=3, #3 random examples out of len(eval_dataloader) examples will be plotted
                            save_dir=output_dir) 
            #########################################################
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            

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

            return unroll_steps

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
                run_results.append(unroll_steps.item())
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