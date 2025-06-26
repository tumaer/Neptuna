from transformers.integrations.integration_utils import WandbCallback as WandbCallback_
from transformers.integrations.integration_utils import rewrite_logs,is_torch_xla_available
from transformers.trainer_callback import CallbackHandler as CallbackHandler_
import wandb
from transformers.integrations.integration_utils import logger
import tempfile

########################################################################################
class CallbackHandler(CallbackHandler_):
    #NOTE: on_evaluate is modified to accept **kwargs
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics, **kwargs)

########################################################################################
class WandbCallback(WandbCallback_):
    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.
        """
        if self._wandb is None:
            return
        self._initialized = True

        from wandb.sdk.lib.config_util import ConfigError as WandbConfigError

        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_dict()}

            model_config = {}
            if hasattr(model, "config") and model.config is not None:
                model_config = model.config if isinstance(model.config, dict) else model.config.to_dict()
            if hasattr(state, "trial_params") and state.trial_params is not None:
                if isinstance(state.trial_params, dict):
                    trial_params = {
                        k.replace("model_config.", "").replace("data_config.", ""): v
                        for k, v in state.trial_params.items()
                        if k.startswith("model_config.") or k.startswith("data_config.")
                    }
                    model_config = {**model_config, **trial_params}
            combined_dict = {**model_config, **combined_dict}

            if hasattr(model, "peft_config") and model.peft_config is not None:
                peft_config = model.peft_config
                combined_dict = {**{"peft_config": peft_config}, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                init_args["name"] = trial_name
                init_args["group"] = args.run_name
            elif args.run_name is not None:
                init_args["name"] = args.run_name
                if args.run_name == args.output_dir:
                    self._wandb.termwarn(
                        "The `run_name` is currently set to the same value as `TrainingArguments.output_dir`. If this was "
                        "not intended, please specify a different run name by setting the `TrainingArguments.run_name` parameter.",
                        repeat=False,
                    )

            if self._wandb.run is None:
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    **init_args,
                )
            self._wandb.config.update(combined_dict, allow_val_change=True)

            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

            _watch_model = os.getenv("WANDB_WATCH", "false")
            if not is_torch_xla_available() and _watch_model in ("all", "parameters", "gradients"):
                self._wandb.watch(model, log=_watch_model, log_freq=max(100, state.logging_steps))
            self._wandb.run._label(code="transformers_trainer")

            try:
                self._wandb.config["model/num_parameters"] = model.num_parameters()
            except AttributeError:
                logger.info(
                    "Could not log the number of model parameters in Weights & Biases due to an AttributeError."
                )
            except WandbConfigError:
                logger.warning(
                    "A ConfigError was raised whilst setting the number of model parameters in Weights & Biases config."
                )

            if self._log_model.is_enabled:
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_name = (
                        f"model-{self._wandb.run.id}"
                        if (args.run_name is None or args.run_name == args.output_dir)
                        else f"model-{self._wandb.run.name}"
                    )
                    model_artifact = self._wandb.Artifact(
                        name=model_name,
                        type="model",
                        metadata={
                            "model_config": model.config.to_dict() if hasattr(model, "config") else None,
                            "num_parameters": self._wandb.config.get("model/num_parameters"),
                            "initial_model": True,
                        },
                    )
                    save_model_architecture_to_file(model, temp_dir)

                    for f in Path(temp_dir).glob("*"):
                        if f.is_file():
                            with model_artifact.new_file(f.name, mode="wb") as fa:
                                fa.write(f.read_bytes())
                    self._wandb.run.log_artifact(model_artifact, aliases=["base_model"])

                    badge_markdown = (
                        f'[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge'
                        f'-28.svg" alt="Visualize in Weights & Biases" width="20'
                        f'0" height="32"/>]({self._wandb.run.get_url()})'
                    )

                    modelcard.AUTOGENERATED_TRAINER_COMMENT += f"\n{badge_markdown}"
                    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]
        
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if k in single_value_scalars:
                    self._wandb.run.summary[k] = v
            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}
            # If all non-scalar log keys start with the special "plot_" prefix, we skip the standard
            # key rewrite performed by 🤗 Transformers so that these entries are logged verbatim.
            # This allows logging arbitrary plot artifacts (e.g. PNG images) without their keys being
            # altered to the train/eval namespace.
            if not non_scalar_logs or not any(k.startswith("plot_") for k in non_scalar_logs):
                non_scalar_logs = rewrite_logs(non_scalar_logs)
            # else: leave non_scalar_logs untouched to preserve the original keys.
            self._wandb.log({**non_scalar_logs, "train/global_step": state.global_step}, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        wandb.finish() if wandb.run is not None else None
########################################################################################
from utils.feature_utils import re_normalize_data
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback
import os
from utils.plot_progress import plot_examples
#from transformers.trainer import _get_output_dir

class PlotOnEvalAndSaveCallback(TrainerCallback):
    def __init__(self):
        self._should_plot = False
        self.global_step = None
        self.trial = None
    def on_evaluate(self, args, state, control, **kwargs):
        # Mark that an evaluation just happened
        self.predictions = kwargs['predictions']
        self.labels = kwargs['labels']
        self.inputs = kwargs['inputs']
        self.eval_dataset = kwargs['eval_dataset']
        self.data_config = kwargs['data_config']
        self.train_config = kwargs['train_config']
        self.output_log_config = kwargs['output_log_config']
        self.global_step = state.global_step
        self._should_plot = True

    def on_save(self, args, state, control, **kwargs):
        # Only plot if an evaluation just happened before this save
        if self._should_plot:
            ##NOTE: This plotting is not present in the base class
            part_1 = args.output_dir
            part_2 = state.trial_name or ""
            part_3 = f"checkpoint-{self.global_step}"
            output_dir = os.path.join(part_1, part_2, part_3)
            
            len_eval_dataloader, num_eval_rollouts, label_seq_length, channel_dim, *spatial_dims = self.predictions.shape
            self.predictions=self.predictions.reshape(len_eval_dataloader, num_eval_rollouts*label_seq_length, channel_dim, *spatial_dims)

            # Plot only after the configured epoch threshold (default 0 when null)
            plot_after = self.train_config.get("plot_after_epoch") or 0
            if state.epoch >= plot_after:
                # ------------------------------------------------------------------
                # Renormalize inputs, labels and predictions for visualization
                # ------------------------------------------------------------------
                norm_stats = self.data_config["data_normalization_stats"]
                norm_strategy = self.data_config["data_normalization_strategy"]

                # Channel ordering in the dataset 
                channel_names = getattr(self.eval_dataset, "channels", None)

                # Renormalize:
                inputs   = re_normalize_data(self.inputs, channel_names, norm_stats, norm_strategy)
                labels   = re_normalize_data(self.labels, channel_names, norm_stats, norm_strategy)
                predictions = re_normalize_data(self.predictions, channel_names, norm_stats, norm_strategy) 

                run_dir = os.path.join(part_1, part_2) if part_2 is not None else part_1

                fig_dict = plot_examples(
                            inputs,
                            predictions,
                            labels,
                            channel_names,
                            ndim=self.data_config["dimension"],
                            stride=self.data_config["sequence_info"][-1],
                            extra_info=run_dir,
                            checkpoint_step=state.global_step,
                            epoch=round(state.epoch, 3),
                            num_examples=self.train_config["n_plot_examples"], #NOTE: plotting is slow
                            save_dir=output_dir,
                            log_to_wandb=self.output_log_config["logging"]["wandb"]
                        )

                # If W&B logging is enabled, log the figures now.
                if self.output_log_config["logging"].get("wandb", False) and wandb.run is not None:
                    self.wandb_fig_log(fig_dict)
        # Reset the flag
        self._should_plot = False

    # ------------------------------------------------------------------
    # Utility logger so that callbacks can perform Trainer-like logging.
    # The `trainer` attribute will be injected by our custom CallbackHandler.
    # ------------------------------------------------------------------
    def wandb_fig_log(self, log_dict: dict):
        """Log `log_dict` through the Trainer if available, otherwise fall back to wandb or stdout."""
        if hasattr(self, "trainer") and self.trainer is not None:
            # Delegate to the Trainer's own logging facility. This will also trigger other callbacks' on_log.
            self.trainer.log(log_dict)
        elif wandb.run is not None:
            wandb.log(log_dict)
        else:
            print(log_dict)
