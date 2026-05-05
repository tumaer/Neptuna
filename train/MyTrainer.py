from trainer import Trainer

def group_name_for_grad_debug(name: str) -> str:
    parts = name.split(".")

    # encoder.stages.0.blocks.0.attention.query.weight
    # decoder.stages.2.blocks.3.MLP.dense.weight
    if len(parts) >= 5 and parts[1] == "stages" and parts[3] == "blocks":
        return ".".join(parts[:5])   # encoder.stages.0.blocks.0

    # encoder.stages.0.downsample.reduction.weight
    # decoder.stages.1.upsample.mixup.weight
    if len(parts) >= 4 and parts[1] == "stages":
        return ".".join(parts[:4])   # encoder.stages.0.downsample

    # encoder.time_blocks.0.attention.query.weight
    # decoder.time_blocks.2.MLP.0.weight
    if len(parts) >= 3 and parts[1] == "time_blocks":
        return ".".join(parts[:3])   # encoder.time_blocks.0

    # residual_blocks_space.0.1.conv1.weight
    if parts[0] == "residual_blocks_space" and len(parts) >= 3:
        return ".".join(parts[:3])   # residual_blocks_space.0.1

    # residual_blocks_time.0.conv1.weight
    if parts[0] == "residual_blocks_time" and len(parts) >= 2:
        return ".".join(parts[:2])   # residual_blocks_time.0

    # embedder.embed.weight, recovery.projection.weight, etc.
    return parts[0]

def print_module_grad_norms(model, top_k=40):
    module_norms = {}
    module_maxabs = {}
    module_numel = {}

    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        group = group_name_for_grad_debug(name)
        g = p.grad.detach()

        grad_norm = g.norm(2).item()
        grad_max = g.abs().max().item()
        numel = g.numel()

        module_norms[group] = module_norms.get(group, 0.0) + grad_norm ** 2
        module_maxabs[group] = max(module_maxabs.get(group, 0.0), grad_max)
        module_numel[group] = module_numel.get(group, 0) + numel

    rows = []
    for group, sq_norm in module_norms.items():
        norm = math.sqrt(sq_norm)
        numel = module_numel[group]
        norm_per_sqrt_param = norm / math.sqrt(numel)
        rows.append((group, norm, module_maxabs[group], numel, norm_per_sqrt_param))

    rows = sorted(rows, key=lambda x: x[1], reverse=True)

    print("\nTop module grad norms before clipping:")
    print(f"{'module':70s} {'grad_norm':>12s} {'max_abs':>12s} {'numel':>12s} {'norm/sqrtN':>12s}")
    for group, norm, maxabs, numel, norm_per_sqrt_param in rows[:top_k]:
        print(
            f"{group:70s} "
            f"{norm:12.4e} "
            f"{maxabs:12.4e} "
            f"{numel:12d} "
            f"{norm_per_sqrt_param:12.4e}"
        )

class DebugTrainer(Trainer):
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        # super().training_step has already done backward here
        unwrapped_model = self.accelerator.unwrap_model(model)
        print_module_grad_norms(unwrapped_model)

        return loss