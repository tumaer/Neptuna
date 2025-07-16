import time
import os
from transformers import TrainingArguments
from train.trainer import Trainer
from metrics.default_metrics import l1_error, l2_error
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
__all__ = ["run"]


def run(cfg):
    """Entry-point called by main.py after Hydra config is prepared."""

    # ------------------------------------------------------------------
    # Setup WANDB Project
    # ------------------------------------------------------------------
    if cfg["output_log_config"]["logging"]["wandb"]:
        # Set WANDB_PROJECT environment variable
        wandb_project = cfg["output_log_config"]["logging"].get(
            "wandb_project", "neptuna"
        )
        os.environ["WANDB_PROJECT"] = wandb_project
        print(f"Setting WANDB_PROJECT to: {wandb_project}")

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
        eval_strategy="steps",  # TODO: switch to "epoch" once ready
        eval_steps=5,  # update-steps between evaluations, has no meaning if eval_strategy is "epoch"
        eval_on_start=False,  # sanity-check eval before training starts
        # ------------------------------------------------------------------
        # Batching
        # ------------------------------------------------------------------
        per_device_train_batch_size=cfg["train_config"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["train_config"]["per_device_eval_batch_size"],
        eval_accumulation_steps=cfg["train_config"][
            "eval_accumulation_steps"
        ],  # accumulate predictions on GPU before moving to CPU
        # ------------------------------------------------------------------
        # Optimiser / schedule
        # ------------------------------------------------------------------
        max_grad_norm=1.0,  # gradient clipping
        num_train_epochs=cfg["train_config"]["num_train_epochs"],
        learning_rate=cfg["scheduler_config"]["lr"],
        weight_decay=cfg["scheduler_config"]["weight_decay"],
        optim=cfg["scheduler_config"]["optim"],  # options: adamw_hf, adamw_torch, ...
        lr_scheduler_type=cfg["scheduler_config"]["lr_scheduler"],
        warmup_ratio=cfg["scheduler_config"]["warmup_ratio"],  # linear warm-up fraction
        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        log_level=cfg["train_config"].get(
            "log_level", "info"
        ),  # debug / info / warning / ...
        logging_strategy="steps",  # switch to "epoch" later if needed
        logging_steps=1,  # only used if logging_strategy is "steps"
        logging_nan_inf_filter=False,  # include NaNs in logs for debugging
        # ------------------------------------------------------------------
        # Saving
        # ------------------------------------------------------------------
        save_strategy="best",  # switch to "epoch" once validation present
        save_steps=5,  # only used if save_strategy is "steps"
        save_total_limit=2,  # keep only last N checkpoints
        push_to_hub=False,  # push to Hugging Face Hub, requires login before (run `huggingface-cli login` in terminal)
        hub_strategy="end",  # push last checkpoint to Hub (alternatives: "end", "every_save", "checkpoint", "all_checkpoints")
        # ------------------------------------------------------------------
        # Reproducibility
        # ------------------------------------------------------------------
        seed=0,  # model-seed
        data_seed=1045,  # sampler-seed for SeedableRandomSampler
        # ------------------------------------------------------------------
        # Misc runtime knobs 
        # ------------------------------------------------------------------
        fp16=cfg["train_config"]["fp16"],  # set True for mixed-precision
        dataloader_num_workers=cfg["train_config"]["dataloader_num_workers"],
        load_best_model_at_end=False,  # enable when save & eval strategy align
        metric_for_best_model=cfg["train_config"][
            "metric_for_best_model"
        ],  # checkpoint metric
        include_for_metrics=[
            "inputs",
        ]
        + (
            ["conditioning_inputs"]
            if cfg["data_config"].get("conditioning_in_channels") is not None
            else []
        ),  # keep inputs and optionally conditioning_inputs for plotting
        greater_is_better=False,  # lower loss/error is better
        dataloader_pin_memory=True,
        gradient_checkpointing=False,  # save memory, slower back-prop
        auto_find_batch_size=False,
        full_determinism=False,  # turn on for reproducible distributed training
        torch_compile=False,
        use_cpu=True,  # use_cpu even if other devices are present
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
    )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_ds, eval_ds = make_datasets(cfg)

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

    # Build the callbacks list
    callbacks = []
    # Always add PlotOnEvalAndSaveCallback and NaNCallback
    callbacks.append(PlotOnEvalAndSaveCallback)
    callbacks.append(NaNCallback)

    trainer = Trainer(
        model_config=cfg["model_config"],
        data_config=cfg["data_config"],
        train_config=cfg["train_config"],
        output_log_config=cfg["output_log_config"],
        # everything below goes to kwargs which go directly to the base trainer class of HF
        model=model,
        model_init=model_init if cfg["hyperparam_opt_config"]["optimize"] else None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks if callbacks else None,
    )

    trainer.set_eval_or_test_rollout_steps(
        rollout_steps=cfg["train_config"]["n_eval_rollouts"], output_all_steps=True
    )

    # ------------------------------------------------------------------
    # Train vs HP-search -----------------------------------------------
    # ------------------------------------------------------------------
    if cfg["hyperparam_opt_config"]["optimize"] is False:
        start = time.time()
        # {cfg['data_config']['dataset_name']}
        # trainer.train(resume_from_checkpoint=f"./checkpoints/KuramotoSivashinsky_2D_ScOT_09072025_074058/checkpoint-15")
        trainer.train(resume_from_checkpoint=False)
        print(f"Total train time: {time.time() - start:.2f} s")
        if training_args.push_to_hub:
            # Push the trained model to the Hugging Face Hub
            print("Pushing model to Hugging Face Hub...")
            trainer.push_to_hub()
    else:
        # get the sampler from the config, it could be GridSampler, RandomSampler, TPESampler
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
        # Save HPO results to CSV -------------------------------------
        # --------------------------------------------------------------
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
