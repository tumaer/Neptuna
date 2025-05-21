import torch
from torch import nn
from typing import List, Optional, Dict, Tuple, Union, Any
from transformers.trainer import *
from transformers import Trainer as Trainer_

class Trainer(Trainer_):
    def __init__(self, model_config, data_config, data_shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.ar_steps = None
        self.output_all_steps = False
        self.data_shuffle = data_shuffle
        self.data_config = data_config
        self.model_config = model_config


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclassing to add your own datacollator (TODO) and shuffle functionality for the dataloader.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator #NOTE: Why is this hardcoded to remove the columns?
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "shuffle": self.data_shuffle,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    def _model_forward(self, model, inputs):  ##custom function, not inside transformers library

        outputs,labels = model(**inputs)

        return outputs,labels

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):   ##overrides the one in the  base class from transformers library
                                                                                            ## TODO: Raise an issue at HF as return_output = True doesn't work.
        #########################################################
        #Autoregressive prediction (for inference)
        #########################################################
        if self.ar_steps is not None:
            channel_difference = (
                self.data_config.in_channels > self.data_config.out_channels) #Here we assume that the channel which isnot predicted in the output is the last channel in the input (like Re or Ma).
            ## inputs.keys() = dict_keys(['input_data', 'labels',])
            ## inputs['input_data'].shape = torch.Size([B, C, x_resolution, y_resolution]) 
            ## inputs['labels'].shape = torch.Size([B, C, x_resolution, y_resolution])
            if isinstance(self.ar_steps, int): 
                if self.output_all_steps: #this is set to true when self.ar_steps is set in main.py
                    loss_ = []
                    outputs_ = []
                else:
                    total_loss = 0
                
                for i in range(self.ar_steps):
                    #print(f"Autoregressive step {i+1} of {self.ar_steps}")
                    outputs, labels = self._model_forward(model,inputs) #outputs.shape = torch.Size([B, C, x_resolution, y_resolution]) , labels.shape = torch.Size([B, C, x_resolution, y_resolution])
                    #compute the loss here. Assume l2 loss (#TODO: To be changed to a generic loss function)
                    loss_fn = nn.functional.mse_loss
                    loss = loss_fn(outputs, labels) 
                    
                    if self.output_all_steps:
                        outputs_.append(outputs.detach())
                        loss_.append(loss)
                    else:
                        total_loss += loss #loss is added up across all ar_steps and we obtain a scalar. This is divided by ar_steps at the end of the "if" statement
                    
                    #recreate the inputs to be fed to the model for the next step
                    if (self.data_config.sequence_info[0][1] >= self.data_config.sequence_info[0][0]): #output_sequence length > input_sequence length
                        inputs = {
                            **inputs,
                            **{ #this part replaces the "input_data" of input with the output of the model. 
                                #So the new input is the output from the previous step.
                                "input_data": (
                                    outputs[:,(self.data_config.sequence_info[0][1] - self.data_config.sequence_info[0][0]):,].detach() #slice the outputs so as to extract the input_sequence.
                                    if not channel_difference
                                    else torch.cat( 
                                        [
                                            outputs[:,(self.data_config.sequence_info[0][1] - self.data_config.sequence_info[0][0]):,].detach() ,
                                            inputs["input_data"][:,:,self.data_config.out_channels:],
                                        ],
                                        dim=2, 
                                        #concatenate along the channel dimension (dim=2) : the first dimension is the batch dimension, the second dimension is the time sequence dimension,
                                        # the third dimension is the channel dimension and the rest are the spatial dimensions.
                                    )
                                )
                            },
                    }
                    else: #input_sequence length > output_sequence length
                        inputs = {
                            **inputs,
                            **{ #this part replaces the "input_data" of input with the output of the model. 
                                #So the new input is the output from the previous step.
                                "input_data": (
                                    torch.cat([outputs.detach(), inputs["input_data"][:,self.data_config.sequence_info[0][1]:]], dim=1) #slice the outputs so as to extract the input_sequence.
                                    if not channel_difference
                                    else torch.cat( 
                                        [
                                            torch.cat([outputs.detach(), inputs["input_data"][:,self.data_config.sequence_info[0][1]:]], dim=1),
                                            inputs["input_data"][:,:,self.data_config.out_channels:],
                                        ],
                                        dim=2, 
                                        #concatenate along the channel dimension (dim=2) : the first dimension is the batch dimension, the second dimension is the time sequence dimension,
                                        # the third dimension is the channel dimension and the rest are the spatial dimensions.
                                    )
                                )
                            },
                    }
                        
                if self.output_all_steps:
                    outputs= torch.stack(outputs_, dim=1) #shape = torch.Size([B, ar_steps, C_output, x_resolution, y_resolution])
                    loss = torch.stack(loss_, dim=0).mean() #mean() across the autoregressive steps

                else:
                    loss = total_loss / self.ar_steps #take the mean of the loss across all ar_steps

            
            else:
                raise ValueError("ar_steps is not integer")
        #########################################################
        #One-step prediction (for training)
        #########################################################
        else:
            outputs,labels = self._model_forward(model, inputs)
            #compute the loss here. Assume l2 loss
            loss_fn = nn.functional.mse_loss
            loss = loss_fn(outputs, labels) #the loss which is printed is rounded-off to 4 deicmal places
            #printing happening inside the function: _maybe_log_save_evaluate() #TODO: Check 
            #print(f"loss o: {loss}")
        
        return (loss, outputs) if return_outputs else loss
    

    def set_ar_steps(self, ar_steps=None, output_all_steps=False): ##custom function, not inside transformers library
        self.ar_steps = ar_steps
        if self.ar_steps is not None and output_all_steps:
            self.output_all_steps = True
    
    def prediction_step( ##overrides the one in the  base class from transformers library
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
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
            if is_sagemaker_mp_enabled(): #doesnt go here by default, this is for distributed inference
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
                if has_labels or loss_without_labels: #enters here (for both one step and autoregressive rollouts)
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss( #this has the _model_perdict() function inside it
                            model, inputs, return_outputs=True
                        ) 
                    loss = loss.mean().detach() #mean() is used when: self.output_all_steps = True which results in loss being a tensor of shape (num_ar_steps,) and we take the mean

                    if isinstance(outputs, dict): #enters here (outputs.keys() = dict_keys(['loss', 'output']))
                        logits = tuple( #saves the outputs['output]
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]# ignores the keys
                        ) #logits is a tuple of the outputs['output'], for CE-RP it has only one element with logits[0] = outputs['output'] 
                          #and logits[0].shape = torch.Size([B, 4, 128, 128])
                    else: #if outputs is a tensor, then logits is the slice corresponding to outputs[1:] as the 0th index is the loss
                        logits = outputs[1:]
                else: ##not sure why this 'else' is needed.
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = self._model_forward(model, inputs) #this is the only line which is different from the base class
                        ##in the base class it is outputs = model(**inputs),but since we have the autoregressive code as well, we need to use the model_forward function
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0: #self.args.past_index = -1 by default
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only: #prediction_loss_only is false by default
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1: #this is true for CE-RP, logits[0] = outputs['output'] 
            logits = logits[0] #extract the output from the tuple

        return (loss, logits, labels)