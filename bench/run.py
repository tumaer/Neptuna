import time
import os
from transformers import TrainingArguments
from train.trainer import Trainer
from metrics.default_metrics import l1_error, l2_error, compute_metrics_for_n_rollouts
from transformers.trainer import EvalPrediction
from utils.load_model import fetch_model
from utils.dataset_utils import make_datasets
from utils.hp_optimization import (
    compute_objective_function,
    optuna_hp_space_factory,
    get_optuna_sampler,
)
from optuna.pruners import NopPruner
from utils.custom_callbacks import PlotOnEvalAndSaveCallback, NaNCallback
import csv
from utils.hp_optimization import trial_name_factory
from utils.loss_utils import fetch_eval_loss_config, fetch_loss_metric
from utils.plot_progress import preprocess_for_plotting, plot_rollout_metrics
from utils.plot_progress import LayoutConfig, Slice3DConfig, create_plotter
from utils.plot_progress import build_info_strings
from utils.seed_utils import set_global_seed
import psutil
from only_inference import save_errors_to_csv
import numpy as np
import torch

__all__ = ["run"]

def run(cfg):
    """Entry-point called by main.py after Hydra config is prepared."""
    RANK = int(os.environ.get("LOCAL_RANK", -1))
    IS_MAIN_PROCESS = RANK in [-1, 0]
    print(f"RANK: {RANK}")
    try:
        affinity = psutil.Process().cpu_affinity()
        CPU_CORES = len(affinity) if affinity else (psutil.cpu_count())
    except Exception:
        CPU_CORES = psutil.cpu_count()
    print(f"Detected {CPU_CORES} CPU cores")
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Global seeding
    # ------------------------------------------------------------------
    seed = int(cfg["data_config"].get("seed", 0))
    set_global_seed(seed, deterministic=True)

    # ------------------------------------------------------------------
    # Setup WANDB Project
    # ------------------------------------------------------------------
    if cfg["output_log_config"]["logging"]["wandb"] and IS_MAIN_PROCESS:
        # Set WANDB_PROJECT environment variable
        wandb_project = cfg["output_log_config"]["logging"].get(
            "wandb_project", "neptuna"
        )
        os.environ["WANDB_PROJECT"] = wandb_project
        print(f"Setting WANDB_PROJECT to: {wandb_project}")
        os.environ['WANDB_API_KEY'] = cfg["output_log_config"]["logging"]["wandb_api_key"]

    # ------------------------------------------------------------------
    # Build TrainingArguments
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        # ------------------------------------------------------------------
        # Directory & checkpointing
        # ------------------------------------------------------------------
        output_dir=cfg["output_log_config"]["logging"][
            "output_dir"
        ],  # add model name & timestamp
        overwrite_output_dir=True,  # OVERWRITE if dir exists (also used for resume)
        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        eval_strategy=cfg["train_config"]["eval_strategy"], 
        eval_steps=cfg["train_config"]["eval_steps"],  # update-steps between evaluations, has no meaning if eval_strategy is "epoch"
        # ------------------------------------------------------------------
        # Testing 
        # ------------------------------------------------------------------
        load_best_model_at_end=True, # Ensure that the save & eval strategy align
        # ------------------------------------------------------------------
        # Batching
        # ------------------------------------------------------------------
        per_device_train_batch_size=cfg["train_config"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["train_config"]["per_device_eval_batch_size"],
        # accumulate predictions on GPU before moving to CPU
        eval_accumulation_steps=cfg["train_config"]["eval_accumulation_steps"],  
        # ------------------------------------------------------------------
        # Optimiser / schedule
        # ------------------------------------------------------------------
        max_grad_norm=1.0,  # gradient clipping (default is 1.0)
        num_train_epochs=cfg["train_config"]["num_train_epochs"],
        learning_rate=cfg["scheduler_config"]["lr"],
        weight_decay=cfg["scheduler_config"]["weight_decay"],
        optim=cfg["scheduler_config"]["optim"], 
        lr_scheduler_type=cfg["scheduler_config"]["lr_scheduler"],
        warmup_ratio=cfg["scheduler_config"]["warmup_ratio"],  # linear warm-up fraction
        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        # debug / info / warning / ...
        log_level=cfg["train_config"].get("log_level", "info"),  
        logging_strategy=cfg["train_config"]["logging_strategy"],  # switch to "epoch" later if needed
        logging_steps=cfg["train_config"]["logging_steps"],  # only used if logging_strategy is "steps"
        logging_nan_inf_filter=False,  # include NaNs in logs for debugging
        # ------------------------------------------------------------------
        # Saving
        # ------------------------------------------------------------------
        save_strategy=cfg["train_config"]["save_strategy"],  # switch to "epoch" once validation present
        save_steps=cfg["train_config"]["save_steps"],  # only used if save_strategy is "steps"
        save_total_limit=cfg["train_config"]["save_total_limit"],  # keep only last N checkpoints
        push_to_hub=cfg["train_config"]["push_to_hub"],  # push to Hugging Face Hub, requires login before (run `huggingface-cli login` in terminal)
        hub_strategy=cfg["train_config"]["hub_strategy"],  # push last checkpoint to Hub (alternatives: "end", "every_save", "checkpoint", "all_checkpoints")
        # ------------------------------------------------------------------
        # Reproducibility
        # ------------------------------------------------------------------
        seed=seed,  # model-seed
        data_seed=seed,  # sampler-seed for SeedableRandomSampler
        # ------------------------------------------------------------------
        # Mixed precision training
        # ------------------------------------------------------------------
        fp16=cfg["train_config"]["mix_precision_config"]["fp16"], 
        bf16=cfg["train_config"]["mix_precision_config"]["bf16"],  
        tf32=cfg["train_config"]["mix_precision_config"]["tf32"],
        # ------------------------------------------------------------------
        # Misc runtime knobs 
        # ------------------------------------------------------------------    
        dataloader_num_workers=cfg["train_config"]["dataloader_num_workers"],
        metric_for_best_model=cfg["train_config"][
            "metric_for_best_model"
        ],  # checkpoint metric
        include_for_metrics=[
            "inputs",
        ]
        + (
            ["conditioning_inputs"]
            if cfg["data_config"]["conditioning_features"]["conditioning_in_channels"] is not None
            else []
        ),  # keep inputs and optionally conditioning_inputs for plotting
        greater_is_better=False,  # lower loss/error is better
        dataloader_pin_memory=True,
        gradient_accumulation_steps=cfg["train_config"]["gradient_accumulation_steps"],
        gradient_checkpointing=False,  # save memory, slower back-prop
        auto_find_batch_size=False,
        full_determinism=False,  # turn on for reproducible distributed training
        torch_compile=False,
        use_cpu=cfg["train_config"]["use_cpu"],  # use_cpu even if other devices are present
        label_names=["label_including_rollouts"],
        disable_tqdm=True if cfg["output_log_config"]["logging"]["wandb"] else False,
        # ------------------------------------------------------------------
        # Reporting 
        # ------------------------------------------------------------------
        report_to=(
            "wandb"
            if (
                cfg["output_log_config"]["logging"]["wandb"]
                and cfg["hyperparam_opt_config"]["optimize"] is False
            )
            else "none"
        ),
        run_name=(
            os.path.basename(
                os.path.normpath(cfg["output_log_config"]["logging"]["output_dir"])
            )
            if (cfg["hyperparam_opt_config"]["optimize"] is False)
            else "none"
        ),
        ddp_find_unused_parameters=False,
    )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_ds, eval_ds = make_datasets(cfg, mode="train")

    # ------------------------------------------------------------------
    # Model & Trainer 
    # ------------------------------------------------------------------
    if cfg["hyperparam_opt_config"]["optimize"] is False:
        model = fetch_model(cfg["model_config"], cfg["data_config"])
    else:
        model = None

    def model_init():
        return fetch_model(cfg["model_config"], cfg["data_config"])

    def compute_metrics(eval_pred: EvalPrediction):
        preds = eval_pred.predictions
        (
            len_eval_dataloader,
            num_eval_rollouts,
            label_seq_length,
            channel_dim,
            *spatial,
        ) = preds.shape

        # Flatten rollouts into time dimension
        preds = preds.reshape(
            len_eval_dataloader,
            num_eval_rollouts * label_seq_length,
            channel_dim,
            *spatial,
        )
        targets = eval_pred.label_ids

        metrics: dict[str, float] = {}

        # Always compute metrics on CPU to save GPU memory
        device = getattr(trainer, "metric_device", torch.device("cpu"))

        # Convert to CPU tensors
        if isinstance(preds, np.ndarray):
            preds_tensor = torch.from_numpy(preds).float()
        else:
            preds_tensor = (
                preds.detach().cpu()
                if torch.is_tensor(preds)
                else torch.tensor(preds, dtype=torch.float32)
            )

        if isinstance(targets, np.ndarray):
            targets_tensor = torch.from_numpy(targets).float()
        else:
            targets_tensor = (
                targets.detach().cpu()
                if torch.is_tensor(targets)
                else torch.tensor(targets, dtype=torch.float32)
            )

        # 1. Training loss metric (composite_loss for checkpointing)
        if getattr(trainer, "loss_fn", None) is not None:
            try:
                with torch.no_grad():
                    loss_fn = trainer.loss_fn.to(device)
                    composite_loss = loss_fn(
                        model=None,
                        predictions=preds_tensor.to(device),
                        labels=targets_tensor.to(device),
                        return_detailed=False,  # scalar only
                    )

                if torch.is_tensor(composite_loss):
                    metrics["composite_loss"] = composite_loss.item()
                else:
                    metrics["composite_loss"] = float(composite_loss)

            except Exception as e:
                print("[compute_metrics] composite_loss failed:", repr(e))

        # 2. Evaluation loss metrics (for logging), cached on trainer
        eval_loss_fn = getattr(trainer, "eval_loss_fn", None)
        if eval_loss_fn is not None:
            try:
                with torch.no_grad():
                    eval_loss_fn = eval_loss_fn.to(device)
                    # If you only need per-component totals, this is fine;
                    # ensure detailed only contains small tensors.
                    _, detailed = eval_loss_fn(
                        model=None,
                        predictions=preds_tensor.to(device),
                        labels=targets_tensor.to(device),
                        return_detailed=True,
                    )

                for component_name, component_detailed in detailed.items():
                    component_total = component_detailed["total"]
                    if torch.is_tensor(component_total):
                        metrics[component_name] = component_total.item()
                    else:
                        metrics[component_name] = float(component_total)

            except Exception as e:
                print("[compute_metrics] eval_loss_fn failed:", repr(e))

        return metrics

    # Build the callbacks list
    callbacks = []
    # Always add PlotOnEvalAndSaveCallback and NaNCallback
    callbacks.append(PlotOnEvalAndSaveCallback)
    callbacks.append(NaNCallback)

    loss_config = None
    if hasattr(cfg, 'loss_config'):
        loss_config = cfg.loss_config

    trainer = Trainer(
        model_config=cfg["model_config"],
        data_config=cfg["data_config"],
        train_config=cfg["train_config"],
        scheduler_config=cfg["scheduler_config"],
        infer_config=cfg["infer_config"],
        output_log_config=cfg["output_log_config"],
        # all kwargs below go directly to the base trainer class of HF
        model=model,
        model_init=model_init if cfg["hyperparam_opt_config"]["optimize"] else None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks if callbacks else None,
        loss_config=loss_config,
    )

    trainer.set_eval_or_test_rollout_steps(
        rollout_steps=cfg["train_config"]["n_eval_rollouts"], output_all_steps=True
    )

    metric_device = torch.device("cpu")
    try:
        full_eval_cfg = fetch_eval_loss_config(cfg)
        eval_loss_fn = fetch_loss_metric(full_eval_cfg)
        trainer.eval_loss_fn = eval_loss_fn.to(metric_device)
    except Exception as e:
        print("[run] Failed to initialize eval_loss_fn:", repr(e))
        trainer.eval_loss_fn = None

    trainer.metric_device = metric_device

    

    # ------------------------------------------------------------------
    # Train vs HP-search
    # ------------------------------------------------------------------
    if cfg["hyperparam_opt_config"]["optimize"] is False:
        start = time.time()
        # trainer.train(resume_from_checkpoint=f"./checkpoints/KuramotoSivashinsky_2D_ScOT_09072025_074058/checkpoint-15")
        trainer.train(resume_from_checkpoint=False)
        print(f"Total train time: {time.time() - start:.2f} s")
        if training_args.push_to_hub and IS_MAIN_PROCESS:
            print("Pushing model to Hugging Face Hub...")
            trainer.push_to_hub()
        
        # ------------------------------------------------------------------
        # Inference (Continued after training)
        # ------------------------------------------------------------------
        
        if cfg["infer_config"]["do_infer"]:
            print("Running inference...")
            inference_dir = os.path.join(cfg["output_log_config"]["logging"]["output_dir"], "inference_plots")  
            os.makedirs(inference_dir, exist_ok=True)
            infer_ds, infer_ds_from_ic = make_datasets(cfg, mode="infer")
            if cfg["infer_config"]["infer_from_random_timestep"]:
                print(" \n Running inference from random timestep...")
                trainer.set_eval_or_test_rollout_steps(
                    rollout_steps=cfg["infer_config"]["n_infer_rollouts"], output_all_steps=True
                )

                predictions_obj, inputs, conditioning_inputs = trainer.predict(infer_ds, metric_key_prefix="")
                ############################################################
                # predictions_obj.predictions: the output of the model with shape (accumulated_outputs, num_rollouts, label_seq_length, channel_dim, *spatial) 
                # accumulated_outputs and accumulated_gt have the length of number of windows in the test dataset
                # predictions_obj.label_ids: the ground truth with shape (accumulated_gt, num_rollouts*label_seq_length, channel_dim, *spatial)
                # predictions_obj.metrics: the metrics computed after accumulating the outputs and ground truth
                ############################################################

                if IS_MAIN_PROCESS:
                    # pretty print the keys which have the word error in them
                    print('Accumulated error for the whole test set:')
                    errors = {}
                    for key, value in predictions_obj.metrics.items():
                        if "error" in key:
                            print(f"{key}: {value}")
                            errors["random_start"+key] = value
                    save_errors_to_csv(errors, os.path.join(cfg["output_log_config"]["logging"]["output_dir"], "inference_plots"))
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
                        data_config=cfg["data_config"],
                        dataset=infer_ds,
                        residual_config=cfg["data_config"].get("residual_config", None),
                        conditioning_inputs=cond_inp_arr,
                    )

                    # Infer spatial dimensionality (1D / 2D / 3D)
                    ndim = pred_renorm.ndim - 3  # subtract batch, time, channel dims

                    seq_info = cfg["data_config"].get("sequence_info", [1, 1, 1])

                    stride_val = cfg["data_config"].get("sequence_info", [1, 1, 1])[2]

                    plot_save_dir = os.path.join(cfg["output_log_config"]["logging"]["output_dir"], "inference_plots/random_start")

                    model_info_str, data_info_str, train_info_str, sched_info_str = build_info_strings(model_obj=trainer.model, 
                                                                                                        data_config=cfg["data_config"],
                                                                                                        model_config=cfg["model_config"],
                                                                                                        train_config=cfg["train_config"],
                                                                                                        scheduler_config=cfg["scheduler_config"]
                                                                            )
                    # Create rollout sample plots per example in dedicated folders
                    N_examples = pred_renorm.shape[0]
                    num_plot = min(cfg["infer_config"]["n_infer_plot_examples"], N_examples)
                    np.random.seed(42)
                    chosen_example_indices = np.random.choice(N_examples, size=num_plot, replace=False)

                    for example_idx in chosen_example_indices:
                        ex_save_dir = os.path.join(plot_save_dir, f"example_{int(example_idx)}")

                        layout_config = LayoutConfig(
                            base_visual_size=3.5,
                            margin_between_plots_h=0.65,
                            margin_between_plots_v=0.65
                        )

                        slice_config = Slice3DConfig(
                            slice_axis=0,
                            num_slices=4
                        )

                        plotter = create_plotter(
                            orientation='vertical',
                            input_array=inp_renorm,
                            prediction_array=pred_renorm,
                            target_array=tgt_renorm,
                            input_channel_names=only_input_channel_names,
                            output_channel_names=output_channel_names,
                            conditioning_input_array=cond_inp_renorm,
                            conditioning_channel_names=cond_inp_channel_names,
                            checkpoint_step=None,
                            epoch=None,
                            extra_info=cfg["data_config"].get("dataset_name")+"_Inference_plot_from_random_timestep",
                            ndim=ndim,
                            slice_config=slice_config,
                            num_examples=1,
                            stride=stride_val,
                            save_dir=ex_save_dir,
                            log_to_wandb=False,
                            best_plot_at_train_end=False,
                            layout_config=layout_config,
                            include_relative_error=True,
                            model_info=model_info_str,
                            data_info=data_info_str,
                            train_info=train_info_str
                        )
                        
                        plotter.plot()

                

            if cfg["infer_config"]["infer_from_ic"]:
                print(" \n Running inference from IC...")
                trainer.set_eval_or_test_rollout_steps(
                    rollout_steps=cfg["infer_config"]["n_infer_rollouts"], output_all_steps=True
                )
                # ----------------------------------------------------------
                # Prepare prediction, target and input arrays
                # ----------------------------------------------------------
                predictions_obj, inputs, conditioning_inputs = trainer.predict(infer_ds_from_ic, metric_key_prefix="")

                if IS_MAIN_PROCESS:
                    print('Accumulated error for the whole test set (IC start):')
                    errors = {}
                    for key, value in predictions_obj.metrics.items():
                        if "error" in key:
                            print(f"{key}: {value}")
                            errors["ic_start"+key] = value
                    save_errors_to_csv(errors, os.path.join(cfg["output_log_config"]["logging"]["output_dir"], "inference_plots"))

                    preds = predictions_obj.predictions
                    targets = predictions_obj.label_ids
                    inp_arr = inputs
                    cond_inp_arr = conditioning_inputs if conditioning_inputs is not None else None

                    # Determine outputs per rollout (T_out) before flattening, default to 1
                    if preds.ndim >= 5:
                        n, n_rollouts, seq_len, c = preds.shape[:4]
                        outputs_per_rollout = seq_len
                        extra_dims = preds.shape[4:]
                        preds = preds.reshape(n, n_rollouts * seq_len, c, *extra_dims)

                    # Compute per-rollout errors (mean across batch) before plotting
                    per_rollout_step_metrics_ic = compute_metrics_for_n_rollouts(
                        preds, targets, outputs_per_rollout=outputs_per_rollout
                    )

                    errors = {}
                    for metric_name, values in per_rollout_step_metrics_ic.items():
                        print(f"{metric_name} per-step (IC start): {values}")
                        errors[metric_name] = values
                    save_errors_to_csv(errors, os.path.join(cfg["output_log_config"]["logging"]["output_dir"],"inference_plots"))

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
                        data_config=cfg["data_config"],
                        dataset=infer_ds,
                        residual_config=cfg["data_config"].get("residual_config", None),
                        conditioning_inputs=cond_inp_arr,
                    )
                    
                    # Infer spatial dimensionality (1D / 2D / 3D)
                    ndim = pred_renorm.ndim - 3  # subtract batch, time, channel dims

                    # Use stride from the config if available
                    stride_val = cfg["data_config"].get("sequence_info", [1, 1, 1])[2]

                    # Directory for saving inference plots (IC start)
                    plot_save_dir = os.path.join(cfg["output_log_config"]["logging"]["output_dir"], "inference_plots/ic_start")

                    # Plot rollout metrics (per metric subplot)
                    plot_rollout_metrics(
                        step_metrics=per_rollout_step_metrics_ic,
                        output_channel_names=output_channel_names,
                        save_dir=plot_save_dir,
                        title=f"Per-rollout step metric(s) ({cfg['data_config'].get('dataset_name', 'dataset')} - IC start)",
                        filename="rollout_metrics.png",
                        sequence_info=seq_info,
                    )

                    model_info_str, data_info_str, train_info_str, sched_info_str = build_info_strings(
                                                                                                        model_obj=trainer.model,
                                                                                                        data_config=cfg["data_config"],
                                                                                                        model_config=cfg["model_config"],
                                                                                                        train_config=cfg["train_config"],
                                                                                                        scheduler_config=cfg["scheduler_config"]
                                                                                                    )

                    # Create rollout sample plots, these plots start from the initial condition in the test dataset

                    layout_config = LayoutConfig(
                        base_visual_size=3.5,
                        margin_between_plots_h=0.65,
                        margin_between_plots_v=0.65
                    )

                    slice_config = Slice3DConfig(
                        slice_axis=0,
                        num_slices=4
                    )

                    plotter = create_plotter(
                        orientation='vertical',
                        input_array=inp_renorm,
                        prediction_array=pred_renorm,
                        target_array=tgt_renorm,
                        input_channel_names=only_input_channel_names,
                        output_channel_names=output_channel_names,
                        conditioning_input_array=cond_inp_renorm,
                        conditioning_channel_names=cond_inp_channel_names,
                        checkpoint_step=None,
                        epoch=None,
                        extra_info=cfg["data_config"].get("dataset_name")+"_Inference_plot_from_IC",
                        ndim=ndim,
                        slice_config=slice_config,
                        num_examples=cfg["infer_config"]["n_infer_plot_examples"],
                        stride=stride_val,
                        save_dir=plot_save_dir,
                        log_to_wandb=False,
                        best_plot_at_train_end=False,
                        layout_config=layout_config,
                        include_relative_error=True,
                        model_info=model_info_str,
                        data_info=data_info_str,
                        train_info=train_info_str
                    )
                    
                    plotter.plot()

                print("Inference completed")
            
    else:
        # get the sampler from the config, it could be GridSampler, RandomSampler, TPESampler etc.
        sampler = get_optuna_sampler(
            cfg["hyperparam_opt_config"]["optuna_sampler"], config=cfg
        )

        best_trial, study = trainer.hyperparameter_search(
            direction="minimize",
            backend="optuna",
            hp_space=optuna_hp_space_factory(cfg),
            n_trials=cfg["hyperparam_opt_config"]["n_trials"],
            hp_name=trial_name_factory(cfg["data_config"]),
            compute_objective=compute_objective_function(
                selected_metrics=cfg["hyperparam_opt_config"]["metric_for_tuning_hp"]
            ),
            sampler=sampler,
            pruner=NopPruner(),
        )

        # --------------------------------------------------------------
        # Save HPO results to CSV
        # --------------------------------------------------------------
        if IS_MAIN_PROCESS:
            results_dir = cfg["output_log_config"]["logging"]["output_dir"]
            os.makedirs(results_dir, exist_ok=True)
            csv_path = os.path.join(results_dir, "hp_search_results.csv")

            # Collect all parameter names across trials to build consistent header
            param_keys = set()
            for t in study.trials:
                param_keys.update(t.params.keys())
            fieldnames = ["trial_number", "value", "state"] + sorted(param_keys)

            with open(csv_path, mode="w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for t in study.trials:
                    row = {
                        "trial_number": t.number,
                        "value": t.value,
                        "state": t.state.name,
                    }
                    row.update(t.params)
                    writer.writerow(row)
            print(f"Saved hyperparameter optimisation results to {csv_path}")
