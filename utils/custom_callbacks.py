from transformers.integrations.integration_utils import WandbCallback as WandbCallback_
from transformers.integrations.integration_utils import rewrite_logs, is_torch_xla_available
from transformers.trainer_callback import CallbackHandler as CallbackHandler_
from transformers.trainer_callback import TrainerCallback as TrainerCallback_
from transformers.trainer_callback import TrainerControl as TrainerControl_
from dataclasses import dataclass
import wandb
from transformers.integrations.integration_utils import logger
from omegaconf import ListConfig
import tempfile
import numpy as np

########################################################################################
class CallbackHandler(CallbackHandler_):
    #NOTE: on_evaluate is modified to accept **kwargs
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics, **kwargs)

    # ------------------------------------------------------------------
    # New optional event that propagates custom plotting callbacks.
    # ------------------------------------------------------------------
    def on_plot(self, args, state, control, **kwargs):
        """Forward the `on_plot` event to all registered callbacks."""
        return self.call_event("on_plot", args, state, control, **kwargs)

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
            #NOTE:add trial number if available (additional to base class)
            if hasattr(args, "trial_number"):
                combined_dict["trial_number"] = args.trial_number
            model_config = {}
            #NOTE: add model_config and data_config if available (additional to base class)
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
        # Track which epoch thresholds (if any) have already triggered a plot so that we don't
        # repeatedly plot for the same threshold when `plot_after_epoch` is a list.
        self._plotted_thresholds = set()
        
    def on_evaluate(self, args, state, control, **kwargs):
        # Mark that an evaluation just happened
        self.predictions = kwargs['predictions']
        self.labels = kwargs['labels']
        self.inputs = kwargs['inputs']
        self.conditioning_inputs = kwargs['conditioning_inputs']
        self.eval_dataset = kwargs['eval_dataset']
        self.data_config = kwargs['data_config']
        self.train_config = kwargs['train_config']
        self.output_log_config = kwargs['output_log_config']
        self.global_step = state.global_step
        self.residual_config = kwargs['data_config']['residual_config']
        control.should_plot = True

    def on_plot(self, args, state, control, **kwargs):
        # Only plot if an evaluation just happened before this save
        if control.should_plot:
            ##NOTE: This plotting is not present in the base class
            part_1 = args.output_dir
            part_2 = state.trial_name or ""
            part_3 = f"plots"
            output_dir = os.path.join(part_1, part_2, part_3)
            
            len_eval_dataloader, num_eval_rollouts, label_seq_length, channel_dim, *spatial_dims = self.predictions.shape
            self.predictions=self.predictions.reshape(len_eval_dataloader, num_eval_rollouts*label_seq_length, channel_dim, *spatial_dims)

            # Plot only after the configured epoch threshold (default 0 when null)
            plot_after_epoch = self.train_config.get("plot_after_epoch") or 0

            # --------------------------------------------------------------
            # Decide whether to plot for the current epoch.
            # --------------------------------------------------------------
            should_plot = False
            # Case 1: `plot_after_epoch` is an iterable (list/tuple/set) of
            # epoch numbers. We want to plot *once* for each threshold when
            # it is first crossed.
            if isinstance(plot_after_epoch, (ListConfig, list, tuple, set)):
                for thr in plot_after_epoch:
                    if state.epoch >= thr and thr not in self._plotted_thresholds:
                        should_plot = True
                        self._plotted_thresholds.add(thr)
                        break  # Plot at most once per call
            # Case 2: `plot_after_epoch` is a single number – plot
            # continuously for every evaluation after the threshold is
            # reached.
            else:
                should_plot = state.epoch >= plot_after_epoch

            if should_plot:
                # ------------------------------------------------------------------
                # Renormalize inputs, labels and predictions for visualization
                # ------------------------------------------------------------------
                norm_stats = self.data_config["data_normalization_stats"]
                norm_strategy = self.data_config["data_normalization_strategy"]

                # Channel ordering in the dataset 
                input_channel_names = getattr(self.eval_dataset, "input_channels", None)
                output_channel_names = getattr(self.eval_dataset, "output_channels", None)
                conditioning_input_channel_names = None

                # Renormalize each channel separately
                inputs_renormed = np.copy(self.inputs)
                labels_renormed = np.copy(self.labels)
                predictions_renormed = np.copy(self.predictions)
                conditioning_inputs_renormed = None

                if self.conditioning_inputs is not None:
                    conditioning_input_channel_names = [ch_name for ch_name in input_channel_names if ch_name in getattr(self.eval_dataset, "conditioning_in_channels")]
                    conditioning_inputs_renormed = np.copy(self.conditioning_inputs)
                     # remove the conditioning_in_channels from the input_channel_names
                    only_input_channel_names = [ch_name for ch_name in input_channel_names if ch_name not in conditioning_input_channel_names]

                else:
                    only_input_channel_names = input_channel_names

                # Renormalize input channels (for inputs and conditioning_inputs if present)
                for c_idx, ch_name in enumerate(only_input_channel_names):
                    if ch_name not in norm_stats:
                        raise ValueError(f"Stats for input channel {ch_name} are unavailable.")
                    
                    stats = norm_stats[ch_name]
                    
                    if "mask" not in ch_name.lower():
                        # Renormalize inputs
                        inputs_renormed[:, :, c_idx] = re_normalize_data(
                            self.inputs[:, :, c_idx], stats, norm_strategy
                        )

                if self.conditioning_inputs is not None:
                # Renormalize conditioning_inputs
                    for c_idx, ch_name in enumerate(conditioning_input_channel_names):
                        if ch_name not in norm_stats:
                            raise ValueError(f"Stats for conditioning_input channel {ch_name} are unavailable.")
                        
                        stats = norm_stats[ch_name]

                        if "mask" not in ch_name.lower():
                            # Renormalize conditioning_inputs
                            conditioning_inputs_renormed[:, :, c_idx] = re_normalize_data(
                                self.conditioning_inputs[:, :, c_idx], stats, norm_strategy
                            )
                    
                # Renormalize output channels (for labels and predictions)
                for c_idx, ch_name in enumerate(output_channel_names):
                    if ch_name not in norm_stats:
                        raise ValueError(f"Stats for output channel {ch_name} are unavailable.")
                    
                    norm_key = ch_name if ((self.residual_config is None) or (self.residual_config["add_base_value_with_raw_loss"]) or (self.residual_config["add_predicted_value_with_raw_loss"])) else f"{ch_name}_residual"
                    stats = norm_stats[norm_key]
                    
                    # Renormalize labels and predictions
                    labels_renormed[:, :, c_idx] = re_normalize_data(
                        self.labels[:, :, c_idx], stats, norm_strategy
                    )
                    predictions_renormed[:, :, c_idx] = re_normalize_data(
                        self.predictions[:, :, c_idx], stats, norm_strategy
                    )
                #--------------------------------
                #NOTE: Comment this section out to visualize the residuals instead of the raw values
                if self.residual_config is not None and (self.residual_config["add_predicted_value_with_diff_loss"]):
                    #Create raw values for plotting
                    base_value = inputs_renormed[:, -1:, ]
                    labels_renormed = labels_renormed.cumsum(axis=1) + base_value
                    predictions_renormed = predictions_renormed.cumsum(axis=1) + base_value
                #--------------------------------
                run_dir = os.path.join(part_1, part_2) if part_2 is not None else part_1

                fig_dict = plot_examples(
                            inputs_renormed,
                            predictions_renormed,
                            labels_renormed,
                            only_input_channel_names,
                            output_channel_names,
                            conditioning_input_array=conditioning_inputs_renormed,
                            conditioning_input_channel_names=conditioning_input_channel_names,
                            ndim=self.data_config["dimension"],
                            stride=self.data_config["sequence_info"][-1],
                            extra_info=run_dir,
                            checkpoint_step=state.global_step,
                            epoch=round(state.epoch, 3),
                            num_examples=self.train_config["n_plot_examples"], #NOTE: plotting is slow
                            save_dir=output_dir,
                            log_to_wandb=self.output_log_config["logging"]["wandb"],
                            is_best_metric=kwargs["is_new_best_metric"]
                        )

                # If W&B logging is enabled, log the figures now.
                if self.output_log_config["logging"].get("wandb", False) and wandb.run is not None:
                    self.wandb_fig_log(fig_dict)
        # Reset the flag
        control.should_plot = False

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

# ------------------------------------------------------------------
# Extend the 🤗 Transformers callback interface with an optional
# `on_plot` event **without** modifying the original library files.
# We monkey-patch the base `TrainerCallback` to include a no-op
# implementation so that any callback can safely override it.
# ------------------------------------------------------------------

if not hasattr(TrainerCallback_, "on_plot"):
    def _noop_on_plot(self, args, state, control, **kwargs):  # type: ignore[unused-argument]
        """Event called after a plotting operation (user-defined)."""
        return control

    setattr(TrainerCallback_, "on_plot", _noop_on_plot)

# ------------------------------------------------------------------
# Extended TrainerControl that adds a `should_plot` flag.
# ------------------------------------------------------------------

@dataclass
class TrainerControl(TrainerControl_):
    """TrainerControl subclass with an additional `should_plot` switch."""

    should_plot: bool = False

    def _new_step(self):
        """Reset flags for a new step, including the new `should_plot`."""
        super()._new_step()
        self.should_plot = False

    # Ensure serialization/deserialization captures the new flag
    def state(self) -> dict:
        base_state = super().state()
        base_state["args"]["should_plot"] = self.should_plot
        return base_state

# Also patch the reference imported in `transformers.trainer` so that
# Trainer.__init__ uses the extended control when it instantiates.
import transformers.trainer as _tr_mod
if getattr(_tr_mod, "TrainerControl", None) is not TrainerControl:
    _tr_mod.TrainerControl = TrainerControl
