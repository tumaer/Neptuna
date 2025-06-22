from __future__ import annotations

"""Orchestrates a full training or hyper-parameter-optimisation run."""

import time
from typing import Dict
from transformers import TrainingArguments
from train.trainer import Trainer
from metrics.default_metrics import l1_error, l2_error
from transformers.trainer import EvalPrediction
import copy

from utils.load_model import fetch_model
from bench.dataset_utils import make_datasets
from utils.hp_optimization import (
    compute_objective_function,
    optuna_hp_space_factory,
)
from utils.wandb_callback import WandbCallback
__all__ = ["run"]


def run(cfg):  # noqa: D401
    """Entry-point called by main.py after Hydra config is prepared."""
    # ------------------------------------------------------------------
    # Build TrainingArguments
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        # ------------------------------------------------------------------
        # Directory & checkpointing ----------------------------------------
        # ------------------------------------------------------------------
        output_dir=cfg["output_log_config"]["logging"]["output_dir"],  # add model name & timestamp
        overwrite_output_dir=True,  # OVERWRITE if dir exists (also used for resume)

        # ------------------------------------------------------------------
        # Evaluation -------------------------------------------------------
        # ------------------------------------------------------------------
        eval_strategy="steps",  # TODO: switch to "epoch" once ready
        eval_steps=5,  # update-steps between evaluations
        eval_on_start=False,  # sanity-check eval before training starts

        # ------------------------------------------------------------------
        # Batching ---------------------------------------------------------
        # ------------------------------------------------------------------
        per_device_train_batch_size=cfg["train_config"]["batch_size"],
        per_device_eval_batch_size=cfg["train_config"]["batch_size"],
        eval_accumulation_steps=16,  # accumulate predictions on GPU before moving to CPU

        # ------------------------------------------------------------------
        # Optimiser / schedule ---------------------------------------------
        # ------------------------------------------------------------------
        max_grad_norm=1.0,  # gradient clipping
        num_train_epochs=cfg["train_config"]["num_epochs"],
        optim="adamw_torch",  # options: adamw_hf, adamw_torch, ...
        learning_rate=cfg["scheduler_config"]["lr"],
        weight_decay=cfg["scheduler_config"]["weight_decay"],
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        lr_scheduler_type=cfg["scheduler_config"]["lr_scheduler"],
        warmup_ratio=cfg["scheduler_config"]["warmup_ratio"],  # linear warm-up fraction

        # ------------------------------------------------------------------
        # Logging ----------------------------------------------------------
        # ------------------------------------------------------------------
        log_level=cfg["train_config"].get("log_level", "info"),  # debug / info / warning / ...
        logging_strategy="steps",  # switch to "epoch" later if needed
        logging_steps=1,
        logging_nan_inf_filter=False,  # include NaNs in logs for debugging

        # ------------------------------------------------------------------
        # Saving -----------------------------------------------------------
        # ------------------------------------------------------------------
        save_strategy="steps",  # switch to "epoch" once validation present
        save_steps=5,  # must align with eval_steps when using steps-strategy
        save_total_limit=2,  # keep only last N checkpoints
        # save_only_model=False,  # would skip optimiser / RNG state

        # ------------------------------------------------------------------
        # Reproducibility --------------------------------------------------
        # ------------------------------------------------------------------
        seed=0,        # model-seed
        data_seed=1045,  # sampler-seed for SeedableRandomSampler

        # ------------------------------------------------------------------
        # Misc runtime knobs -----------------------------------------------
        # ------------------------------------------------------------------
        fp16=False,  # set True for mixed-precision
        dataloader_num_workers=2,  
        load_best_model_at_end=False,  # enable when save & eval strategy align
        metric_for_best_model="l2_error",  # checkpoint metric
        include_for_metrics=["inputs"],  # keep inputs for plotting
        greater_is_better=False,  # lower loss/error is better
        dataloader_pin_memory=True,
        gradient_checkpointing=False,  # save memory, slower back-prop
        auto_find_batch_size=False,
        full_determinism=False,  # turn on for reproducible distributed training
        torch_compile=False,
        use_cpu=False,  # force CPU even if CUDA present
        label_names=["label_including_rollouts"],
        disable_tqdm=True,

        # ------------------------------------------------------------------
        # Reporting --------------------------------------------------------
        # ------------------------------------------------------------------
        report_to=("wandb" if cfg["output_log_config"]["logging"]["wandb"] else "none"),
        run_name=(
            cfg["output_log_config"]["logging"].get("wandb_run_name", "none")
            if cfg["output_log_config"]["logging"]["wandb"]
            else "none"
        ),
    )

    # ------------------------------------------------------------------
    # Datasets ---------------------------------------------------------
    # ------------------------------------------------------------------
    train_ds, eval_ds = make_datasets(cfg)

    # ------------------------------------------------------------------
    # Model & Trainer --------------------------------------------------
    # ------------------------------------------------------------------
    if  cfg["hyperparam_opt_config"]["optimize"] is False:
        model = fetch_model(cfg["model_config"], cfg["data_config"])
    else:
        model = None

    def model_init():
        return fetch_model(cfg["model_config"], cfg["data_config"])

    def compute_metrics(eval_pred: EvalPrediction):
        preds = eval_pred.predictions
        len_eval_dataloader, num_eval_rollouts, label_seq_length, channel_dim, *spatial = preds.shape
        preds = preds.reshape(len_eval_dataloader, num_eval_rollouts * label_seq_length, channel_dim, *spatial)
        targets = eval_pred.label_ids
        #NOTE: more metrics to be added later here
        return {"l1_error": l1_error(preds, targets), "l2_error": l2_error(preds, targets)}

    trainer = Trainer(
        model_config=cfg["model_config"],
        data_config=cfg["data_config"],
        train_config=cfg["train_config"],
        output_log_config=cfg["output_log_config"],
        #everything below goes to kwargs which go directly to the base trainer class of HF
        model=model,
        model_init=model_init if cfg["hyperparam_opt_config"]["optimize"] else None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[WandbCallback],
    )

    trainer.set_eval_or_test_rollout_steps(rollout_steps=cfg["train_config"]["n_eval_rollouts"], output_all_steps=True)


    # Helper to give each Optuna trial a human-readable name 
    def trial_name(trial):  # noqa: D401
        """Return a short, unique name for an Optuna trial.

        We include the trial number and a couple of key sampled parameters (if
        present) to make the run list easier to read in W&B / MLflow etc.
        """

        pieces = [f"trial{trial.number}"]

        def _abbr(key: str) -> str:
            parts = key.split("_")
            # If the key has zero or one underscore (≤ two segments), keep as is.
            if len(parts) <= 2:
                return key

            # Abbreviate all but the last segment.
            prefix_abbrev = "".join(p[0] for p in parts[:-1])
            return f"{prefix_abbrev}_{parts[-1]}"

        for full_key, value in sorted(trial.params.items()):
            last = full_key.split(".")[-1]
            pieces.append(f"{_abbr(last)}={value}")

        # Join with underscores and replace any path‐unsafe characters.
        name = "_".join(pieces)
        for ch in ["/", "\\"]:
            name = name.replace(ch, "-")
        return name

    # ------------------------------------------------------------------
    # Train vs HP-search -----------------------------------------------
    # ------------------------------------------------------------------
    if  cfg["hyperparam_opt_config"]["optimize"] is False:
        start = time.time()
        #trainer.train(resume_from_checkpoint=f"./checkpoints/{config['data_config']['dataset_name']}/checkpoint-30")
        trainer.train(resume_from_checkpoint=False)
        print(f"Total train time: {time.time() - start:.2f} s")
    else:
        best_trial = trainer.hyperparameter_search(
            direction="minimize",
            backend="optuna",
            hp_space=optuna_hp_space_factory(cfg),
            n_trials=cfg["hyperparam_opt_config"].get("n_trials", 3),
            hp_name=trial_name,
            compute_objective=compute_objective_function(selected_metrics=cfg["hyperparam_opt_config"]["metric_for_tuning_hp"]),
            n_jobs=3
        ) 