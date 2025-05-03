#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "" #set the GPU to use
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from transformers.trainer import *
from train.trainer import Trainer
from transformers import TrainingArguments
import numpy as np
from utils.load_data import fetch_dataset
from utils.load_model import fetch_model
import time

SEED=0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


@hydra.main(version_base="1.3", config_path="./config", config_name="defaults.yaml")
def main(config: DictConfig):
    print("#" * 79, "\nStarting a benchmarking run with the following config:")
    print(OmegaConf.to_yaml(config))
    print("#" * 79)

    train_dataset = fetch_dataset(dataset_name=config["data_config"]["dataset_name"],
                                dataset_directory_path=config["data_config"]["dataset_directory_path"],
                                mode="train",
                                sequence_info=config["data_config"]["sequence_info"],
                                filter_frame=config["data_config"]["filter_frame"],
                                groups=config["data_config"]["filter_groups"],
                                fields=config["data_config"]["filter_fields"],
                                transform=None, #TODO: add transform
                                )

    train_config = TrainingArguments(
        output_dir="./",
        #fsdp_config=config.get("fsdp_config", None),
        overwrite_output_dir=True,  #! OVERWRITE THIS DIRECTORY IN CASE, also for resuming training
        eval_strategy="no", #TODO: change it to epochs laterThe evaluation strategy to adopt during training (also change the save_strategy). Possible values are:
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
        log_level="passive", #default
        logging_strategy="steps", # (set to epochs later)The logging strategy to adopt during training. (either steps or epochs)
        logging_steps=5, #Number of update ste ps between two logs if `logging_strategy="steps" 
        logging_nan_inf_filter=False, #Whether to filter `nan` and `inf` losses for logging.
        save_strategy="no", #TODO: Change it to epoch when validation dataset is present. When 'load_best_model_at_end' set to `True`, the parameters `save_strategy` needs to be the same as `evaluation_strategy`
        save_total_limit=1, #If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in`output_dir`.
        seed=SEED,
        fp16=False, # Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
        dataloader_num_workers=1,  #change to CPU_CORES later
        load_best_model_at_end=True, #Whether or not to load the best model found during training at the end of training.
        metric_for_best_model="loss",
        greater_is_better=False, #lower loss is better, therefore False
        dataloader_pin_memory=True, # Whether you want to pin memory in data loaders or not. Will default to `True`.
        gradient_checkpointing=False, #If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        auto_find_batch_size=False, #can be set to true, requires accelerate libraray
        full_determinism=False, #set to false, only required for debugging distributed training
        torch_compile=False, #check if setting it to true helps
        report_to="none", #change to wandb later  
        use_cpu=False, #Whether to not use CUDA even when it is available.
        #run_name=params.wandb_run_name, # Typically used for [wandb] and [mlflow]logging.
    )

    model = fetch_model(model_config=config["model_config"], 
                        data_config=config["data_config"])

    trainer = Trainer(
        model=model,
        args=train_config,
        train_dataset=train_dataset,
        #eval_dataset=eval_dataset,
        #compute_metrics=compute_metrics, # The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return a dictionary string to metric values.
        #callbacks=[early_stopping],
    )
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"Total process time: {end_time - start_time:.2f} seconds")
    
if __name__=="__main__":
    main()
    