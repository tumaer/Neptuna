
from typing import Dict

def fetch_model(model_config: Dict,
                data_config: Dict):
    """
    Instantiate the model based on the provided model details.
    """
    model_name = model_config['model_name'].lower()


    if model_name == "fno":
        from models.fno.fno import FNO
        model = FNO(
                    dimension=model_config['dimension'],
                    in_channels=model_config['in_channels']*data_config["sequence_info"][0][0],#right now no historic steps will work
                    out_channels=model_config['out_channels']*data_config["sequence_info"][0][1], #currently only 1 step ahead prediction
                    latent_channels=model_config['latent_channels'],
                    num_fno_modes=eval(model_config['fno_modes']),
                    num_fno_layers=model_config['fno_layers'],
                    padding=model_config['padding'],
                    decoder_layers=model_config['decoder_layers'],
                    decoder_layer_size=model_config['decoder_layer_size'],
                    ) 
    
    return model