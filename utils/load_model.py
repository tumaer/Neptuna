
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
        from models.ResNet.resnet import ResNet
        model = ResNet(
                    in_channels=data_config['in_channels'],
                    out_channels=data_config['out_channels'], 
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    num_blocks=model_config['num_blocks'],
                    block=model_config['block'],
                    hidden_channels=model_config['hidden_channels'],
                    norm=model_config['norm'],
                    )
    elif model_name == "dilresnet":
        from models.ResNet.resnet import ResNet
        model = ResNet(
                    in_channels=data_config['in_channels'],
                    out_channels=data_config['out_channels'], 
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
    elif model_name == "deeponet_ffn":
        from models.DeepONet.deeponet import AutoDeepONet
        model = AutoDeepONet(
                    in_channels=data_config['in_channels'],
                    out_channels=data_config['out_channels'], 
                    resolution=data_config['resolution'],
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    branch_depth=model_config['branch_depth'],
                    trunk_depth=model_config['trunk_depth'],
                    width=model_config['width'],
                    branch_net = "FFN",
                    #activation_fn=model_config['activation_fn'],
                    #act_on_output=model_config['act_on_output']
        )
    elif model_name == "deeponet_cnn":
        from models.DeepONet.deeponet import AutoDeepONet
        model = AutoDeepONet(
                    in_channels=data_config['in_channels'],
                    out_channels=data_config['out_channels'], 
                    resolution=data_config['resolution'],
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    branch_depth=model_config['branch_depth'],
                    trunk_depth=model_config['trunk_depth'],
                    kernel_size=model_config['kernel_size'],
                    padding=model_config['padding'],
                    hidden_channels=model_config['hidden_channels'],
                    branch_net = "CNN",
                    width=model_config['width'],
                    #activation_fn=model_config['activation_fn'],
                    #act_on_output=model_config['act_on_output']
        )
    elif model_name == "deeponet_resnet":
        from models.DeepONet.deeponet import AutoDeepONet
        model = AutoDeepONet(
                    in_channels=data_config['in_channels'],
                    out_channels=data_config['out_channels'], 
                    resolution=data_config['resolution'],
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    branch_depth=model_config['branch_depth'],
                    trunk_depth=model_config['trunk_depth'],
                    padding=model_config['padding'],
                    hidden_channels=model_config['hidden_channels'],
                    branch_net = "ResNet",
                    width=model_config['width'],
                    #activation_fn=model_config['activation_fn'],
                    #act_on_output=model_config['act_on_output']
        )
    elif model_name == "cno":
        from models.CNO.cno import CNO
        model = CNO(
                    in_dim=data_config['in_channels'],
                    out_dim=data_config['out_channels'], 
                    size=data_config['resolution'],
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    N_layers=model_config['N_layers'],
                    N_res=model_config['N_res'],
                    N_res_neck=model_config['N_res_neck'],
                    channel_multiplier=model_config['channel_multiplier'],
                    use_bn=model_config['use_bn'],
        )
    else:
        raise ValueError(f"Model {model_name} is not implemented yet.")                      
    
    
    return model