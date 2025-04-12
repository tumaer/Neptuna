


def fetch_model(model_name, model_directory_path):
    """
    Load the model config from the specified directory.
    """
    model_name = model_name.lower()

    if model_name == "fno":
        from models.fno import FNO
        model = FNO()
    
    return model