import hydra
from omegaconf import DictConfig, OmegaConf
from transformers.trainer import *
from train.trainer import Trainer
from transformers import TrainingArguments
import numpy as np
from utils.load_data import fetch_dataset
from utils.load_model import fetch_model
from utils.feature_utils import get_grid_resolution
from metrics.default_metrics import l1_error, l2_error
import time
import os
import h5py

SEED=0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

@hydra.main(version_base="1.3", config_path="./config", config_name="defaults.yaml")
def main(config: DictConfig):
    print("#" * 79, "\nStarting a benchmarking run with the following config:")
    print(OmegaConf.to_yaml(config))
    print("#" * 79)

    # Get grid resolution directly from the HDF5 file
    if config["data_config"]["grid_resolution"] is None:
        config["data_config"]["grid_resolution"] = get_grid_resolution(config["data_config"]["dataset_directory_path"])

    train_dataset, eval_dataset = fetch_dataset(dataset_name=config["data_config"]["dataset_name"],
                                dataset_directory_path=config["data_config"]["dataset_directory_path"],
                                sequence_info=config["data_config"]["sequence_info"],
                                max_pf_train_rollouts=config["train_config"]["pushforward"]["max_allowed_unroll_steps"][-1],
                                n_eval_rollouts=config["train_config"]["n_eval_rollouts"],
                                filter_frame=config["data_config"]["filter_frame"],
                                groups=config["data_config"]["filter_groups"],
                                fields=config["data_config"]["filter_fields"],
                                eval_split_ratio=config["train_config"]["eval_split_ratio"],
                                transform=None, #TODO: add transform
                                )
    


    training_arguments = TrainingArguments(
        output_dir=f"./checkpoints/{config['data_config']['dataset_name']}_{config['data_config']['dimension']}D",
        #fsdp_config=config.get("fsdp_config", None),
        overwrite_output_dir=True,  #! OVERWRITE THIS DIRECTORY IN CASE, also for resuming training
        eval_strategy="steps", #TODO: change it to epochs laterThe evaluation strategy to adopt during training (also change the save_strategy). Possible values are: no, steps, epoch
        eval_steps=5, #Number of update steps between two logs if `logging_strategy="steps", #NOTE: keep an eye on number of steps between logging 
        eval_on_start=False, #Whether to perform a evaluation step (sanity check) before the training to ensure the validation steps works correctly.
        per_device_train_batch_size=config['train_config']["batch_size"],
        per_device_eval_batch_size=config['train_config']["batch_size"],
        eval_accumulation_steps=16, #Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
                                    #left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster but
                                    #requires more memory).
        max_grad_norm=1.0, #defalt = 1.0 (set to 5.0 in poseidon)  Maximum gradient norm (for gradient clipping)
        num_train_epochs=config['train_config']["num_epochs"], 
        optim="adamw_torch", #The optimizer to use: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision or adafactor.
        learning_rate=config["scheduler_config"]["lr"], #The initial learning rate for [`AdamW`] optimizer.
        weight_decay=config["scheduler_config"]["weight_decay"], # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`] optimizer.
        adam_beta1=0.9,  # default
        adam_beta2=0.999,  # default
        adam_epsilon=1e-8,  # default
        lr_scheduler_type=config["scheduler_config"]["lr_scheduler"], #linear by default
        warmup_ratio=config["scheduler_config"]["warmup_ratio"], #Ratio of total training steps used for a linear warmup from 0 to `learning_rate`
        log_level="debug", #default #other options: debug, info, warning, error
        logging_strategy="steps", # (set to epochs later)The logging strategy to adopt during training. (either steps or epochs)
        logging_steps=1, #Number of update ste ps between two logs if `logging_strategy="steps" 
        logging_nan_inf_filter=False, #Whether to filter `nan` and `inf` losses for logging.
        save_strategy="steps", #options: no, epoch, steps, best. TODO: Change it to epoch when validation dataset is present. When 'load_best_model_at_end' set to `True`, the parameters `save_strategy` needs to be the same as `evaluation_strategy`
        save_steps=5, #Number of updates steps before two checkpoint saves if `save_strategy="steps"`#NOTE: Save steps must be the same/multiple of eval_steps. 
        save_total_limit=2, #If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in`output_dir`. #NOTE: always saves the checkpoint after performing evaluation depending on the self.state.best_global_step in _save_checkpoints in Trainer class.
        #save_only_model=False, #Whether to only save the model, or also the optimizer, scheduler & rng state.
        seed=SEED, #model_seed
        data_seed=1045, #data_seed for the sampler (for SeedableRandomSampler)
        fp16=False, # Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
        dataloader_num_workers=1,  #change to CPU_CORES later
        load_best_model_at_end=False, # TODO: Change to true later (save and eval strategy should be same for this). Whether or not to load the best model found during training at the end of training.
        metric_for_best_model="l2_error", #use this metric for checkpointing while performing evaluation
        include_for_metrics = ["inputs"],
        greater_is_better=False, #lower loss is better, therefore False
        dataloader_pin_memory=True, # Whether you want to pin memory in data loaders or not. Will default to `True`.
        gradient_checkpointing=False, #If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        auto_find_batch_size=True, #can be set to true, requires accelerate libraray
        full_determinism=False, #set to false, only required for debugging distributed training
        torch_compile=False, #check if setting it to true helps
        report_to="none", #change to wandb later  
        use_cpu=False, #Whether to not use CUDA even when it is available.
        label_names=["label_including_rollouts"],
        #accelerator_config={"use_seedable_sampler": False},  # Default is True.Setting to False disables SeedableRandomSampler and uses RandomSampler instead.
        #run_name=params.wandb_run_name, # Typically used for [wandb] and [mlflow]logging.
    )

    model = fetch_model(model_config=config["model_config"], 
                        data_config=config["data_config"],
                        )

    def compute_metrics(eval_prediction: EvalPrediction):
        predictions = eval_prediction.predictions
        len_eval_dataloader, num_eval_rollouts, label_seq_length, channel_dim, *spatial_dims = predictions.shape
        predictions=predictions.reshape(len_eval_dataloader, num_eval_rollouts*label_seq_length, channel_dim, *spatial_dims)
        targets = eval_prediction.label_ids
        
        #TODO: more metrics can be added here and checkpointing can be done based on the metrics.
        return {"l1_error": l1_error(predictions, targets),
                "l2_error": l2_error(predictions, targets)}
 
    trainer = Trainer(
        model_config=config["model_config"],
        data_config=config["data_config"],
        pushforward_config=config["train_config"]["pushforward"],
        #everything below goes to kwargs which go directly to the base trainer class of HF
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics, # The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return a dictionary string to metric values.
        #callbacks=[early_stopping],
    )

    trainer.set_rollout_steps(rollout_steps=config["train_config"]["n_eval_rollouts"],
                              output_all_steps=True)
    
    start_time = time.time()
    #trainer.train(resume_from_checkpoint=f"./checkpoints/{config['data_config']['dataset_name']}/checkpoint-30")
    trainer.train(resume_from_checkpoint=False)
    end_time = time.time()
    print(f"Total train time: {end_time - start_time:.2f} seconds")
    
if __name__=="__main__":
    main()

## NOTE: 
### How to get the number of iterations as seen in the progress bar?: number_of_training_iterations = (len(train_index_map)/batch_size) * num_epochs
### Number of eval iterations: number_of_eval_iterations = (len(eval_index_map)/batch_size) there are no epochs during eval 


##TODO:
# 3 normalize and renormalize
# 4 Inference code
# 5 Plot only after a certain number of epochs/steps
# 6 Add the model name to the checkpoint also the date and time
# also let the user specify the list of groups for validation manually in the config file