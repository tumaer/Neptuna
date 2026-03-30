import importlib
import glob
import os

import torch
import torch.nn.functional as F
from collections import OrderedDict
from models.DPOT.dpot_utils import dpot_load_3d_components_from_2d
MODEL_REGISTRY = {
    "unet": ("models.UNet.unet", "UNet"),
    "fno": ("models.FNO.fno", "FNO"),
    "resnet": ("models.ResNet.resnet", "ResNet"),
    "autodeeponet": ("models.DeepONet.deeponet", "AutoDeepONet"),
    "cno": ("models.CNO.cno", "CNO"),
    "scot": ("models.ScOT.scot", "ScOT"),
    "scot3D": ("models.ScOT.scot3D", "ScOT3D"),
    "vit": ("models.ViT.vit", "ViT"),
    "kfno": ("models.kFNO.kfno", "kFNO"),
    "ffno": ("models.FFNO.ffno", "FFNO"),
    "convnext": ("models.ConvNeXt.convnext", "ConvNeXt"),
    "unettransformer": ("models.UNetTransformer.unettransformer", "UNetTransformer"),
    "poseidon": ("models.Poseidon.poseidon", "Poseidon"), #entry without model size suffix needed for inference
    "poseidon_t": ("models.Poseidon.poseidon", "Poseidon"),
    "poseidon_b": ("models.Poseidon.poseidon", "Poseidon"),
    "poseidon_l": ("models.Poseidon.poseidon", "Poseidon"),
    "dpot": ("models.DPOT.dpot", "DPOT"), #entry without model size suffix needed for inference
    "dpot_t": ("models.DPOT.dpot", "DPOT"),
    "dpot_s": ("models.DPOT.dpot", "DPOT"),
    "dpot_m": ("models.DPOT.dpot", "DPOT"),
    "dpot_l": ("models.DPOT.dpot", "DPOT"),
    "dpot_h": ("models.DPOT.dpot", "DPOT"),
}


def _has_hf_checkpoint(path):
    """Check if path contains a HuggingFace-format checkpoint."""
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")) and (
        os.path.exists(os.path.join(path, "model.safetensors"))
        or os.path.exists(os.path.join(path, "pytorch_model.bin"))
    )


def _find_pth_file(checkpoint_path):
    """Resolve a .pth file from a direct path or a directory containing one."""
    if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".pth"):
        return checkpoint_path
    if os.path.isdir(checkpoint_path):
        pth_files = glob.glob(os.path.join(checkpoint_path, "*.pth"))
        if len(pth_files) == 1:
            return pth_files[0]
        if len(pth_files) > 1:
            raise ValueError(
                f"Multiple .pth files in {checkpoint_path}: {pth_files}. "
                "Set model_checkpoint_path to the exact .pth file."
            )
    return None


def _extract_state_dict(raw):
    """Extract a model state dict from common .pth checkpoint layouts."""
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict from .pth file, got {type(raw).__name__}")
    for key in ("model", "state_dict", "model_state_dict", "net"):
        if key in raw:
            return raw[key]
    if all(isinstance(v, torch.Tensor) for v in raw.values()):
        return raw
    raise ValueError(
        f"Cannot locate state dict in .pth file. Top-level keys: {list(raw.keys())}"
    )


def _remap_state_dict_keys(state_dict, model):
    """Add a child-module prefix to state_dict keys when the HF wrapper
    introduces one (e.g. raw keys 'pos_embed' -> 'dpot.pos_embed')."""
    model_keys = set(model.state_dict().keys())
    pth_keys = set(state_dict.keys())

    if pth_keys == model_keys:
        return state_dict

    for child_name, _ in model.named_children():
        prefixed = {f"{child_name}.{k}": v for k, v in state_dict.items()}
        if set(prefixed.keys()) == model_keys:
            return prefixed

    return state_dict


def _reconcile_state_dict(loaded_sd, model_sd):
    """Reconcile shape mismatches between a loaded state dict and the model's
    expected state dict.

    - Matching shapes: use loaded weights
    - pos_embed mismatch: spatially interpolate to expected size
    - Other mismatches: keep model's randomly initialized weights
    - Keys not in the model: dropped
    """
    reconciled = dict(model_sd)

    for k, v in loaded_sd.items():
        if k not in model_sd:
            print(f"  - Skipping {k} (not in current model)")
            continue

        expected = model_sd[k]
        if v.shape == expected.shape:
            reconciled[k] = v
        elif "pos_embed" in k:
            interp_mode = "bilinear" if v.ndim == 4 else "trilinear"
            reconciled[k] = F.interpolate(
                v, size=expected.shape[2:], mode=interp_mode,
            )
            print(f"  - Interpolated {k}: {list(v.shape)} -> {list(expected.shape)}")
        else:
            print(
                f"  - Skipping {k} due to size mismatch, "
                f"loaded: {list(v.shape)}, expected: {list(expected.shape)}"
            )

    return reconciled

def _load_pth_state_dict(pth_path, model_class, config):
    """Load a .pth file, extract and remap its state dict, then reconcile
    shape mismatches against a freshly initialized model."""
    raw = torch.load(pth_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(raw)

    tmp_model = model_class(config)

    # ------------------------------------------------------------------
    # DPOT 2D -> 3D fine-tuning: load compatible components and adapt
    # 2D MLP conv weights to 3D by unsqueezing the final spatial dim.
    # This is only applied when the target model is 3D and the source
    # checkpoint appears to be 2D (e.g. 4D conv weights in block MLPs).
    # ------------------------------------------------------------------
    try:
        target_dim = getattr(config, "dimension", None)
        is_target_3d = int(target_dim) == 3
    except Exception:
        is_target_3d = False

    if is_target_3d and hasattr(tmp_model, "dpot"):
        # If the source dict is already prefixed (rare), strip it back to
        # the inner-module key space expected by the helper.
        inner_sd = state_dict
        if any(isinstance(k, str) and k.startswith("dpot.") for k in inner_sd.keys()):
            inner_sd = {k[len("dpot."):]: v for k, v in inner_sd.items() if isinstance(k, str) and k.startswith("dpot.")}

        # Heuristic: if any block MLP conv weights are 4D, treat source as 2D.
        looks_2d = False
        for k, v in inner_sd.items():
            if not isinstance(k, str):
                continue
            if "blocks." in k and "mlp" in k and "weight" in k and hasattr(v, "ndim") and v.ndim == 4:
                looks_2d = True
                break

        if looks_2d:
            dpot_load_3d_components_from_2d(
                tmp_model.dpot,
                inner_sd,
                components=["blocks", "time_agg"],
            )
            state_dict = tmp_model.state_dict()
        else:
            state_dict = _remap_state_dict_keys(state_dict, tmp_model)
            state_dict = _reconcile_state_dict(state_dict, tmp_model.state_dict())
    else:
        state_dict = _remap_state_dict_keys(state_dict, tmp_model)
        state_dict = _reconcile_state_dict(state_dict, tmp_model.state_dict())
    del tmp_model

    return state_dict


def load_pretrained_model(model_config):
    """
    Factory function to load any pretrained model based on model_name in config.

    Supports two checkpoint formats:
      1. HuggingFace directory (config.json + model.safetensors / pytorch_model.bin)
         -> from_pretrained loads config and weights from the directory
      2. Raw .pth file (state dict extracted and keys remapped automatically)
         -> from_pretrained called with config= and state_dict= (path=None)

    Args:
        model_config: Dictionary with at least 'model_name' and 'model_checkpoint_path'

    Returns:
        Loaded pretrained model instance
    """
    model_name = model_config.get("model_name", "").lower()
    checkpoint_path = model_config["model_checkpoint_path"]

    if model_name not in MODEL_REGISTRY:
        supported_models = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_name}' is not supported. Supported: {supported_models}"
        )

    module_path, class_name = MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    if _has_hf_checkpoint(checkpoint_path):
        model, loading_info = model_class.from_pretrained(
            checkpoint_path,
            output_loading_info=True,
            ignore_mismatched_sizes=False,
            local_files_only=True,
        )
    else:
        pth_path = _find_pth_file(checkpoint_path)
        if pth_path is None:
            raise FileNotFoundError(
                f"No HF checkpoint (config.json + model.safetensors/pytorch_model.bin) "
                f"or .pth file found at: {checkpoint_path}"
            )
        config = model_class.config_class(**model_config)
        state_dict = _load_pth_state_dict(pth_path, model_class, config)
        model, loading_info = model_class.from_pretrained(
            pretrained_model_name_or_path=None,
            config=config,
            state_dict=state_dict,
            output_loading_info=True,
            ignore_mismatched_sizes=False,
        )

    assert not loading_info["missing_keys"], f"Missing keys: {loading_info['missing_keys']}"
    assert not loading_info["unexpected_keys"], f"Unexpected keys: {loading_info['unexpected_keys']}"

    return model
