
from typing import Dict

def fetch_model(model_config: Dict,
                data_config: Dict):
    """
    Instantiate the model based on the provided model details.
    """
    model_name = model_config['model_name'].lower()


    if model_name == "fno":
        from models.FNO.fno import FNO
        model = FNO(
                    dimension=data_config['dimension'],
                    in_channels=data_config['in_channels'],
                    out_channels=data_config['out_channels'], 
                    sequence_info=data_config["sequence_info"],
                    latent_channels=model_config['latent_channels'],
                    num_fno_modes=eval(model_config['fno_modes']),
                    num_fno_layers=model_config['fno_layers'],
                    padding=model_config['padding'],
                    decoder_layers=model_config['decoder_layers'],
                    decoder_layer_size=model_config['decoder_layer_size'],
                    #TODO: more arguments like act. fn name etc to be added
                    ) 
    
    elif model_name == "unet":
        from models.UNet.unet import UNet
        model= UNet(
                    dimension=data_config['dimension'],
                    in_channels=data_config['in_channels'],
                    out_channels=data_config['out_channels'], 
                    sequence_info=data_config["sequence_info"], 
                    latent_channels=model_config['latent_channels'],
                    #TODO: more arguments to be added
        )
    
    return model