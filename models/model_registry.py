import importlib

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
    "unettransformer": ("models.UNetTransformer.unettransformer", "UNetTransformer"),
    "poseidon": ("models.Poseidon.poseidon", "Poseidon"),
}


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

    if model_name not in MODEL_REGISTRY:
        supported_models = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_name}' is not supported for inference loading. Supported models: {supported_models}"
        )

    module_path, class_name = MODEL_REGISTRY[model_name]

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
