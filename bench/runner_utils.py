import torch
from transformers.trainer import EvalPrediction
import torch.distributed as dist

class StreamingMetrics:
    def __init__(self, trainer, mode):
        self.reset()
        self.trainer = trainer
        self.mode = mode

    def reset(self):
        self.num_elements_aggregated_in_batch_dim_per_device = 0
        self.weighted_composite_train_loss_per_batch_per_device = 0
        self.weighted_eval_component_loss_sums_per_device = {}
        self.metrics = {}

    def __call__(self, eval_pred: EvalPrediction, compute_result: bool = False):
        # Called every eval batch when batch_eval_metrics=True
        device = eval_pred.predictions.device
        #print(f"[StreamingMetrics] device: {device}")
        preds = eval_pred.predictions
        (
            len_eval_dataloader,
            num_eval_rollouts,
            label_seq_length,
            channel_dim,
            *spatial,
        ) = preds.shape
        
        # Flatten rollouts into time dimension
        preds_tensor = preds.reshape(
            len_eval_dataloader,
            num_eval_rollouts * label_seq_length,
            channel_dim,
            *spatial,
        ).detach()#.to(device)
        targets_tensor = eval_pred.label_ids.detach()

        if compute_result:
            gs = self.trainer.accelerator.gradient_state
            remainder = gs.remainder  # valid global samples in last logical batch

            # Accelerate behavior: only trim when remainder > 0.
            # remainder == -1 (e.g. drop_last=True / unknown length) => no trim.
            if remainder > 0:
                local_bs = preds_tensor.shape[0]
                rank = self.trainer.accelerator.process_index

                # global gathered order is rank-concatenated:
                # rank0 chunk [0:local_bs), rank1 [local_bs:2*local_bs), ...
                start = rank * local_bs
                valid_local = max(0, min(local_bs, remainder - start))
                #print(f"[rank {RANK}] valid_local: {valid_local}")
                preds_tensor = preds_tensor[:valid_local]
                targets_tensor = targets_tensor[:valid_local]
        

        #print(f"[rank {RANK}] preds shape: {preds_tensor.shape}")
        batch_elements_per_device = preds_tensor.shape[0]
        self.num_elements_aggregated_in_batch_dim_per_device += batch_elements_per_device
        #if RANK == 0:
        
        # dist.barrier()
        # print(f"[rank {RANK}] shape={tuple(preds_tensor.shape)} "
        #     f"min={preds_tensor.min().item()} max={preds_tensor.max().item()} mean={preds_tensor.float().mean().item()}",
        #     flush=True)
        # dist.barrier()

        # 1. Create an additional composite_train_loss metric using the metrics from train_loss block.
        # "Can" be used for checkpointing.
        if batch_elements_per_device > 0 and getattr(self.trainer, "loss_fn", None) is not None:
            try:
                with torch.no_grad():
                    train_loss_fn = self.trainer.loss_fn.to(device)
                    composite_train_loss_per_device = train_loss_fn(
                        model=None,
                        predictions=preds_tensor,#.to(device),
                        labels=targets_tensor,#.to(device),
                        input_frames=None,
                        return_detailed=False,  # scalar only
                    )

                    self.weighted_composite_train_loss_per_batch_per_device += (
                        composite_train_loss_per_device * batch_elements_per_device
                    )
                    
                #metrics["composite_train_loss"] = float(composite_train_loss_per_device)

            except Exception as e:
                print("[compute_metrics] composite_train_loss failed:", repr(e))

        # 2. Evaluation loss metrics (for logging), cached on self.trainer
        if self.mode == "eval":
            eval_loss_fn = getattr(self.trainer, "eval_loss_fn", None)
        elif self.mode == "infer":
            eval_loss_fn = getattr(self.trainer, "infer_loss_fn", None)
        else:
            ValueError(f"Unsupported mode for StreamingMetrics: {self.mode}")


        if batch_elements_per_device > 0 and eval_loss_fn is not None:
            try:
                with torch.no_grad():
                    eval_loss_fn = eval_loss_fn.to(device)
                    _, detailed = eval_loss_fn(
                        model=None,
                        predictions=preds_tensor,#.to(device),
                        labels=targets_tensor,#.to(device),
                        input_frames=None,
                        return_detailed=True,
                    )

                for component_name, component_detailed in detailed.items():
                    component_total = component_detailed["total"]
                    weighted_delta = float(component_total) * batch_elements_per_device
                    self.weighted_eval_component_loss_sums_per_device[component_name] = (
                        self.weighted_eval_component_loss_sums_per_device.get(component_name, 0.0)
                        + weighted_delta
                    )

            except Exception as e:
                print("[compute_metrics] eval_loss_fn failed:", repr(e))

        if not compute_result:
            return {}  # don’t log partials

        #* last batch: return global metrics, then reset for next eval
        #dist.barrier()
        # print(f"[rank {RANK}] weighted_composite_train_loss_per_batch_per_device: {self.weighted_composite_train_loss_per_batch_per_device}", flush=True)
        # dist.barrier()
        # print(f"[rank {RANK}] num_elements_aggregated_in_batch_dim_per_device: {self.num_elements_aggregated_in_batch_dim_per_device}", flush=True)
        # dist.barrier()
        if self.num_elements_aggregated_in_batch_dim_per_device > 0:
            # print(
            #     f"[rank {RANK}] local_num_windows={self.num_elements_aggregated_in_batch_dim_per_device}",
            #     flush=True,
            # )
            global_num_windows = torch.tensor(
                float(self.num_elements_aggregated_in_batch_dim_per_device),
                device=device,
                dtype=torch.float32,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(global_num_windows, op=dist.ReduceOp.SUM)
            # print(
            #     f"[rank {RANK}] global_num_windows={global_num_windows.item()}",
            #     flush=True,
            # )

            #-------------------------------------------------------
            if getattr(self.trainer, "loss_fn", None) is not None:

                global_weighted_composite_train_loss_sum = torch.tensor(
                    float(self.weighted_composite_train_loss_per_batch_per_device),
                    device=device,
                    dtype=torch.float32,
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(global_weighted_composite_train_loss_sum, op=dist.ReduceOp.SUM)

                self.metrics["composite_train_loss"] = (
                    global_weighted_composite_train_loss_sum / global_num_windows
                ).item()

            if eval_loss_fn is not None:
                for (
                    component_name,
                    weighted_component_sum_per_device,
                ) in self.weighted_eval_component_loss_sums_per_device.items():

                    global_component_weighted_sum = torch.tensor(
                        float(weighted_component_sum_per_device),
                        device=device,
                        dtype=torch.float32,
                    )
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(global_component_weighted_sum, op=dist.ReduceOp.SUM)
                    self.metrics[component_name] = (
                        global_component_weighted_sum / global_num_windows
                    ).item()

        
        metrics_to_return = dict(self.metrics)
        self.reset()
        return metrics_to_return