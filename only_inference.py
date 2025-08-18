
from omegaconf import OmegaConf
import os
import glob
from utils.load_data import fetch_dataset
from utils.plot_progress import build_info_strings
from utils.plot_progress import plot_examples, preprocess_for_plotting, plot_rollout_metrics
from metrics.default_metrics import l1_error, l2_error, compute_metrics_for_n_rollouts
from transformers.trainer import EvalPrediction
from transformers import TrainingArguments
from train.trainer import Trainer
import hydra
from omegaconf import DictConfig
import json
from utils.seed_utils import set_global_seed

def load_pretrained_model(model_config):
    """
    Factory function to load any pretrained model based on model_name in config.
    
    Args:
        model_config: Dictionary containing model configuration with 'model_name' and 'model_checkpoint_path'
    
    Returns:
        Loaded pretrained model instance
    """
    model_name = model_config.get("model_name", "").lower()
    checkpoint_path = model_config["model_checkpoint_path"]
    
    # Map model names to their corresponding classes
    model_registry = {
        "unet": ("models.UNet.unet", "UNet"),
        "fno": ("models.FNO.fno", "FNO"),
        "resnet": ("models.ResNet.resnet", "ResNet"),
        "autodeeponet": ("models.DeepONet.deeponet", "AutoDeepONet"),
        "cno": ("models.CNO.cno", "CNO"),
        "scot": ("models.ScOT.scot", "ScOT"),
        "vit": ("models.ViT.vit", "ViT"),
    }
    
    if model_name not in model_registry:
        supported_models = ", ".join(model_registry.keys())
        raise ValueError(f"Model '{model_name}' is not supported for inference loading. Supported models: {supported_models}")
    
    module_path, class_name = model_registry[model_name]
    
    # Dynamic import and model loading
    import importlib
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    


    model, loading_info = model_class.from_pretrained(
        checkpoint_path,
        output_loading_info=True,
        ignore_mismatched_sizes=False,
        local_files_only=True,
    )

    assert not loading_info["missing_keys"], f"Missing keys: {loading_info['missing_keys']}"
    assert not loading_info["unexpected_keys"], f"Unexpected keys: {loading_info['unexpected_keys']}"

    return model

def get_trainer(
    model_config,
    data_config,
    output_dir,
    train_config=None,
    scheduler_config=None,
    infer_config=None,
    output_log_config=None
):
    # Function to read train_batch_size from trainer_state.json
    def get_train_batch_size(checkpoint_path):
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if not os.path.exists(trainer_state_path):
            raise FileNotFoundError(f"trainer_state.json not found in {checkpoint_path}")
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
        return trainer_state["train_batch_size"]

    # Extract train_batch_size from trainer_state.json
    train_batch_size = get_train_batch_size(model_config["model_checkpoint_path"])

    # Seed from data config (default 0)
    seed_value = int(data_config.get("seed", 0))

    # Use train_batch_size for per_device_eval_batch_size
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=train_batch_size,
        seed=seed_value,  # model-seed
        data_seed=seed_value,  # sampler-seed for SeedableRandomSampler
        eval_accumulation_steps=16,
        dataloader_num_workers=0,
        report_to="none",
        use_cpu=False,  # use_cpu even if other devices are present
        label_names=["label_including_rollouts"],
        include_for_metrics=[
            "inputs",
        ]
        + (
            ["conditioning_inputs"]
            if data_config["conditioning_features"]["conditioning_in_channels"] is not None
            else []
        ), 
    )
    
    def compute_metrics(eval_pred: EvalPrediction):
        preds = eval_pred.predictions
        (
            len_eval_dataloader,
            num_eval_rollouts,
            label_seq_length,
            channel_dim,
            *spatial,
        ) = preds.shape
        preds = preds.reshape(
            len_eval_dataloader,
            num_eval_rollouts * label_seq_length,
            channel_dim,
            *spatial,
        )
        targets = eval_pred.label_ids
        # NOTE: more metrics to be added later here
        return {
            "l1_error": l1_error(preds, targets),
            "l2_error": l2_error(preds, targets),
        }

    # Load pretrained model using the generic factory function
    model = load_pretrained_model(model_config)

    trainer = Trainer(
        model=model,
        args=args,
        compute_metrics=compute_metrics,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        scheduler_config=scheduler_config,
        infer_config=infer_config,
        output_log_config=output_log_config,
    )
    return trainer



def find_checkpoint_path(experiment_dir):
    """
    Find the checkpoint folder within an experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Path to the checkpoint folder, or None if not found
    """
    checkpoint_pattern = os.path.join(experiment_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        return None
    
    # If multiple checkpoints exist, take the one with the highest number
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
    return checkpoint_dirs[-1]



def run_inference_for_each_experiment(experiment_dir, infer_config):
    """
    Run inference for a single experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory
        infer_config: Inference configuration
    """
    print(f"\n{'='*60}")
    print(f"Processing experiment: {os.path.basename(experiment_dir)}")
    print(f"{'='*60}")
    
    # Check if data_config.json exists
    data_config_path = os.path.join(experiment_dir, "data_config.json")
    if not os.path.exists(data_config_path):
        print(f"Warning: data_config.json not found in {experiment_dir}. Skipping...")
        return
    
    # Find checkpoint directory
    checkpoint_path = find_checkpoint_path(experiment_dir)
    if not checkpoint_path:
        print(f"Warning: No checkpoint folder found in {experiment_dir}. Skipping...")
        return
    
    # Check if config.json exists in checkpoint
    model_config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(model_config_path):
        print(f"Warning: config.json not found in {checkpoint_path}. Skipping...")
        return
    
    print(f"Data config: {data_config_path}")
    print(f"Model checkpoint: {checkpoint_path}")
    
    try:
        # Load configurations
        data_config = OmegaConf.load(data_config_path)
        model_config = OmegaConf.load(model_config_path)
        
        # Set global seed from data_config (default 0)
        seed_value = int(data_config.get("seed", 0))
        set_global_seed(seed_value)
        
        # Add the model checkpoint path to the model config
        model_config["model_checkpoint_path"] = checkpoint_path
        model_config["model_name"] = model_config["architectures"][0]
        
        # Set configs to None for inference-only mode
        train_config = None
        scheduler_config = None
        output_log_config = None
        
        # Function to create a unique solo_inference directory
        def create_unique_inference_dir(base_dir):
            solo_inference_dir = os.path.join(base_dir, "solo_inference")
            if not os.path.exists(solo_inference_dir):
                os.makedirs(solo_inference_dir)
                return solo_inference_dir

            # If solo_inference directory already exists, append a number to create a unique directory
            counter = 1
            while True:
                new_dir = f"{solo_inference_dir}_{counter}"
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                    return new_dir
                counter += 1

        # Create a unique solo_inference directory within the experiment directory
        solo_inference_dir = create_unique_inference_dir(experiment_dir)
        
        # Override filter parameters from infer_config if they are specified (not None)
        infer_filter_groups = data_config["filter_features"]["infer_filter_groups"]
        infer_filter_frames = data_config["filter_features"]["infer_filter_frames"]
        
        # Check if infer_config has filter_features and override if not None
        if "filter_features" in infer_config:
            if infer_config["filter_features"].get("infer_filter_groups") is not None:
                print(f"Original infer_filter_groups from data_config: {infer_filter_groups}")
                infer_filter_groups = infer_config["filter_features"]["infer_filter_groups"]
                print(f"Using infer_filter_groups from inference config: {infer_filter_groups}")
            
            if infer_config["filter_features"].get("infer_filter_frames") is not None:
                print(f"Original infer_filter_frames from data_config: {infer_filter_frames}")
                infer_filter_frames = infer_config["filter_features"]["infer_filter_frames"]
                print(f"Using infer_filter_frames from inference config: {infer_filter_frames}")
        
        print("Running solo inference...")
        infer_ds, infer_ds_from_ic = fetch_dataset(
                                                data_config["dataset_name"], 
                                                mode="infer",
                                                dataset_directory_path=data_config["dataset_directory_path"],
                                                sequence_info=data_config["sequence_info"],
                                                infer_filter_groups=infer_filter_groups,
                                                infer_filter_frames=infer_filter_frames,
                                                filter_in_channels=data_config["filter_features"]["filter_in_channels"],
                                                filter_out_channels=data_config["filter_features"]["filter_out_channels"],
                                                conditioning_in_channels=data_config["conditioning_features"]["conditioning_in_channels"],
                                                include_conditioning_parameters=data_config["conditioning_features"]["include_conditioning_parameters"],
                                                parameter_min_max_stats=data_config["conditioning_features"]["parameter_min_max_stats"],
                                                data_normalization_stats=data_config["data_normalization_stats"],
                                                data_normalization_strategy=data_config["data_normalization_strategy"],
                                                is_steady_state_prediction=data_config["is_steady_state_prediction"],
                                                residual_config=data_config["residual_config"],
                                                n_infer_rollouts=infer_config["n_infer_rollouts"],
                                                infer_from_random_timestep=infer_config["infer_from_random_timestep"],
                                                infer_from_ic=infer_config["infer_from_ic"],
                                                )
        
        trainer = get_trainer(
            model_config=model_config, 
            data_config=data_config,
            output_dir=solo_inference_dir,
            train_config=train_config,
            scheduler_config=scheduler_config,
            infer_config=infer_config,
            output_log_config=output_log_config
        )

        if infer_config["infer_from_random_timestep"]:
            trainer.set_eval_or_test_rollout_steps(
                rollout_steps=infer_config["n_infer_rollouts"], output_all_steps=True
            )

            predictions_obj, inputs, conditioning_inputs = trainer.predict(infer_ds, metric_key_prefix="")
            ############################################################
            # predictions_obj.predictions: the output of the model with shape (accumulated_outputs, num_rollouts, label_seq_length, channel_dim, *spatial) 
            # accumulated_outputs and accumulated_gt have the length of number of windows in the test dataset
            # predictions_obj.label_ids: the ground truth with shape (accumulated_gt, num_rollouts*label_seq_length, channel_dim, *spatial)
            # predictions_obj.metrics: the metrics computed after accumulating the outputs and ground truth
            ############################################################

            # pretty print the keys which have the word error in them
            print('Accumulated error for the whole test set (random start):')
            for key, value in predictions_obj.metrics.items():
                if "error" in key:
                    print(f"{key}: {value}")
            # ----------------------------------------------------------
            # Prepare prediction, target and input arrays
            # ----------------------------------------------------------
            preds = predictions_obj.predictions  # (N, R, T, C, *spatial)

            # Flatten rollout and label sequence dimensions if necessary
            if preds.ndim >= 5:
                n, n_rollouts, seq_len, c = preds.shape[:4]
                extra_dims = preds.shape[4:]
                preds = preds.reshape(n, n_rollouts * seq_len, c, *extra_dims) # (N, R*T, C, *spatial)

            targets = predictions_obj.label_ids  # Expected shape: (N, R*T, C, *spatial)

            # Inputs already returned by `trainer.predict`
            inp_arr = inputs  # Shape: (N, T_in, C_in, *spatial)

            # Conditioning inputs may be None
            cond_inp_arr = conditioning_inputs if conditioning_inputs is not None else None

            # ----------------------------------------------------------
            # Renormalise data and reconstruct residuals for plotting
            # ----------------------------------------------------------
            (inp_renorm,
                tgt_renorm,
                pred_renorm,
                only_input_channel_names,
                output_channel_names,
                cond_inp_renorm,
                cond_inp_channel_names) = preprocess_for_plotting(
                inputs=inp_arr,
                labels=targets,
                predictions=preds,
                data_config=data_config,
                dataset=infer_ds,
                residual_config=data_config.get("residual_config", None),
                conditioning_inputs=cond_inp_arr,
            )

            # Infer spatial dimensionality (1D / 2D / 3D)
            ndim = pred_renorm.ndim - 3  # subtract batch, time, channel dims

            # Use stride from the config if available
            stride_val = data_config.get("sequence_info", [1, 1, 1])[2]

            # Directory for saving inference plots
            plot_save_dir = os.path.join(solo_inference_dir, "inference_plots/random_start")

            # Build formatted info strings  
            model_info_str, data_info_str, train_info_str, sched_info_str = build_info_strings(
                                                                                        model_obj=trainer.model,
                                                                                        data_config=data_config,
                                                                                        model_config=model_config,
                                                                                        train_config=train_config,
                                                                                        scheduler_config=scheduler_config
                                                                                    )

            # Create rollout sample plots 
            plot_examples(
                input_array=inp_renorm,
                prediction_array=pred_renorm,
                target_array=tgt_renorm,
                only_input_channel_names=only_input_channel_names,
                output_channel_names=output_channel_names,
                conditioning_input_array=cond_inp_renorm,
                conditioning_input_channel_names=cond_inp_channel_names,
                checkpoint_step=None,
                epoch=None,
                extra_info=data_config.get("dataset_name")+"_Inference_plot_from_random_timestep",
                ndim=ndim,
                num_examples=infer_config["n_infer_plot_examples"],
                stride=stride_val,
                save_dir=plot_save_dir,
                log_to_wandb=False,
                is_best_metric=False,
                model_info=model_info_str,
                data_info=data_info_str,
                train_info=train_info_str,
                scheduler_info=sched_info_str,
            )

        if infer_config["infer_from_ic"]:
            trainer.set_eval_or_test_rollout_steps(
                rollout_steps=infer_config["n_infer_rollouts"], output_all_steps=True
            )
            # ----------------------------------------------------------
            # Prepare prediction, target and input arrays
            # ----------------------------------------------------------
            predictions_obj, inputs, conditioning_inputs = trainer.predict(infer_ds_from_ic, metric_key_prefix="")

            print('Accumulated error for the whole test set (IC start):')
            for key, value in predictions_obj.metrics.items():
                if "error" in key:
                    print(f"{key}: {value}")

            preds = predictions_obj.predictions
            targets = predictions_obj.label_ids
            inp_arr = inputs
            cond_inp_arr = conditioning_inputs if conditioning_inputs is not None else None

            # Determine outputs per rollout (T_out) before flattening, default to 1
            if preds.ndim >= 5:
                n, n_rollouts, seq_len, c = preds.shape[:4]
                outputs_per_rollout = seq_len
                extra_dims = preds.shape[4:]
                preds = preds.reshape(n, n_rollouts * seq_len, c, *extra_dims) # (N, R*T, C, *spatial)

            # Compute per-rollout errors (mean across batch) before plotting
            per_rollout_step_metrics_ic = compute_metrics_for_n_rollouts(
                preds, targets, outputs_per_rollout=outputs_per_rollout
            )
            for metric_name, values in per_rollout_step_metrics_ic.items():
                print(f"{metric_name} per-step (IC start): {values}")

            # ----------------------------------------------------------
            # Renormalise data and reconstruct residuals for plotting
            # ----------------------------------------------------------
            (inp_renorm,
                tgt_renorm,
                pred_renorm,
                only_input_channel_names,
                output_channel_names,
                cond_inp_renorm,
                cond_inp_channel_names) = preprocess_for_plotting(
                inputs=inp_arr,
                labels=targets,
                predictions=preds,
                data_config=data_config,
                dataset=infer_ds,
                residual_config=data_config.get("residual_config", None),
                conditioning_inputs=cond_inp_arr,
            )
            
            # Infer spatial dimensionality (1D / 2D / 3D)
            ndim = pred_renorm.ndim - 3  # subtract batch, time, channel dims

            # Use stride from the config if available
            stride_val = data_config.get("sequence_info", [1, 1, 1])[2]

            # Directory for saving inference plots
            plot_save_dir = os.path.join(solo_inference_dir, "inference_plots/ic_start")

            # Plot rollout metrics (per metric subplot)
            plot_rollout_metrics(
                step_metrics=per_rollout_step_metrics_ic,
                output_channel_names=output_channel_names,
                save_dir=plot_save_dir,
                title=f"Per-rollout metrics ({data_config.get('dataset_name', 'dataset')} - IC start)",
                filename="rollout_metrics.png",
            )

            model_info_str, data_info_str, train_info_str, sched_info_str = build_info_strings(
                                                                                        model_obj=trainer.model,
                                                                                        data_config=data_config,
                                                                                        model_config=model_config,
                                                                                        train_config=train_config,
                                                                                        scheduler_config=scheduler_config
                                                                                    )

            # Create rollout sample plots, these plots start from the initial condition in the test dataset
            plot_examples(
                input_array=inp_renorm,
                prediction_array=pred_renorm,
                target_array=tgt_renorm,
                only_input_channel_names=only_input_channel_names,
                output_channel_names=output_channel_names,
                conditioning_input_array=cond_inp_renorm,
                conditioning_input_channel_names=cond_inp_channel_names,
                checkpoint_step=None,
                epoch=None,
                extra_info=data_config.get("dataset_name")+"_Inference_plot_from_IC",
                ndim=ndim,
                num_examples=infer_config["n_infer_plot_examples"],
                stride=stride_val,
                save_dir=plot_save_dir,
                log_to_wandb=False,
                is_best_metric=False,
                model_info=model_info_str,
                data_info=data_info_str,
                train_info=train_info_str,
                scheduler_info=sched_info_str,
            )

        print(f"Inference completed for {os.path.basename(experiment_dir)}")
        print(f"Results saved to: {solo_inference_dir}")
        
    except Exception as e:
        print(f"Error processing {experiment_dir}: {str(e)}")
        import traceback
        traceback.print_exc()


@hydra.main(version_base="1.3", config_path="config/infer_config", config_name="only_inference")
def main(cfg: DictConfig):
    infer_config = cfg
    
    # Get all subdirectories in the inference directory
    inference_dir = infer_config.inference_directory
    if not os.path.exists(inference_dir):
        print(f"Error: Inference directory {inference_dir} does not exist!")
        return
    
    # Find all subdirectories in the inference directory
    experiment_dirs = [
        os.path.join(inference_dir, d) 
        for d in os.listdir(inference_dir) 
        if os.path.isdir(os.path.join(inference_dir, d))
    ]
    
    if not experiment_dirs:
        print(f"No experiment directories found in {inference_dir}")
        return
    
    print(f"Found {len(experiment_dirs)} experiment directories:")
    for exp_dir in experiment_dirs:
        print(f"  - {os.path.basename(exp_dir)}")
    
    # Process each experiment directory
    for experiment_dir in experiment_dirs:
        run_inference_for_each_experiment(experiment_dir, infer_config)
    
    print(f"\n{'='*60}")
    print("All inference runs completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
