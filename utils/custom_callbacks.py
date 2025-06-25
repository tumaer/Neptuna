from transformers.integrations.integration_utils import WandbCallback as WandbCallback_
from transformers.integrations.integration_utils import rewrite_logs
from transformers.trainer_callback import CallbackHandler as CallbackHandler_
import wandb

########################################################################################
class CallbackHandler(CallbackHandler_):
    #NOTE: on_evaluate is modified to accept **kwargs
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics, **kwargs)

########################################################################################
class WandbCallback(WandbCallback_):
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
