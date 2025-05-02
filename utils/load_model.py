
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
                    )         
    elif model_name == "resnet":
        from models.resnet.resnet import ResNet
        model = ResNet(
                    in_fields=data_config['in_channels'],
                    out_fields=data_config['out_channels'], 
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    num_blocks=model_config['num_blocks'],
                    block=model_config['block'],
                    hidden_channels=model_config['hidden_channels'],
                    norm=model_config['norm'],
                    )
    elif model_name == "dilresnet":
        from models.resnet.resnet import ResNet
        model = ResNet(
                    in_fields=data_config['in_channels'],
                    out_fields=data_config['out_channels'], 
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    num_blocks=model_config['num_blocks'],
                    block=model_config['block'],
                    hidden_channels=model_config['hidden_channels'],
                    norm=model_config['norm'],
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
    else:
        raise ValueError(f"Model {model_name} is not implemented yet.")                      
    
    
    return model