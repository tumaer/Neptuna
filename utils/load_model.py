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
                    in_channels=len(data_config['filter_in_channels']),
                    out_channels=len(data_config['filter_out_channels']), 
                    sequence_info=data_config["sequence_info"],
                    latent_channels=model_config['latent_channels'],
                    num_fno_modes=model_config['fno_modes'],
                    num_fno_layers=model_config['n_fno_layers'],
                    padding=model_config['padding'],
                    padding_type=model_config['padding_type'],
                    decoder_layers=model_config['decoder_layers'],
                    decoder_layer_size=model_config['decoder_layer_size'],
                    decoder_activation_fn_name=model_config['decoder_activation_fn_name'],
                    activation_fn_name=model_config['activation_fn_name'],
                    )         
        
    elif model_name == "resnet":
        from models.ResNet.resnet import ResNet
        model = ResNet(
                    in_channels=len(data_config['filter_in_channels']),
                    out_channels=len(data_config['filter_out_channels']), 
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    num_blocks=model_config['num_blocks'],
                    block="BasicBlock",
                    latent_channels=model_config['latent_channels'],
                    norm=model_config['norm'],
                    n_groups=model_config['n_groups'],
                    activation_fn_name=model_config['activation_fn_name'],
                    )
    
    elif model_name == "dilresnet":
        from models.ResNet.resnet import ResNet
        model = ResNet(
                    in_channels=len(data_config['filter_in_channels']),
                    out_channels=len(data_config['filter_out_channels']), 
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    num_blocks=model_config['num_blocks'],
                    block="DilatedBasicBlock",
                    latent_channels=model_config['latent_channels'],
                    norm=model_config['norm'],
                    n_groups=model_config['n_groups'],
                    activation_fn_name=model_config['activation_fn_name'],
                    )
    
    elif model_name == "unet":
        from models.UNet.unet import UNet
        model= UNet(
                    dimension=data_config['dimension'],
                    in_channels=len(data_config['filter_in_channels']),
                    out_channels=len(data_config['filter_out_channels']), 
                    sequence_info=data_config["sequence_info"], 
                    latent_channels=model_config['latent_channels'],
                    norm=model_config['norm'],
                    n_groups=model_config['n_groups'],
                    channel_multiplier=model_config['channel_multiplier'],
                    is_attn=model_config['is_attn'],
                    mid_attn=model_config['mid_attn'],
                    n_blocks=model_config['n_blocks'],
                    use1x1=model_config['use1x1'],
                    activation_fn_name=model_config['activation_fn_name'],
                    )
    
    elif model_name == "deeponet_ffn":
        from models.DeepONet.deeponet import AutoDeepONet
        model = AutoDeepONet(
                    in_channels=len(data_config['filter_in_channels']),
                    out_channels=len(data_config['filter_out_channels']), 
                    grid_resolution=data_config['grid_resolution'],
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    branch_depth=model_config['branch_depth'],
                    trunk_depth=model_config['trunk_depth'],
                    width=model_config['width'],
                    branch_net = "FFN",
                    act_on_output=model_config['act_on_output'], #only for FFN
                    activation_fn_name=model_config['activation_fn_name'],
                    )
    
    elif model_name == "deeponet_cnn":
        from models.DeepONet.deeponet import AutoDeepONet
        model = AutoDeepONet(
                    in_channels=len(data_config['filter_in_channels']),
                    out_channels=len(data_config['filter_out_channels']), 
                    grid_resolution=data_config['grid_resolution'],
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    branch_depth=model_config['branch_depth'],
                    trunk_depth=model_config['trunk_depth'],
                    kernel_size=model_config['kernel_size'],
                    padding=model_config['padding'],
                    latent_channels=model_config['latent_channels'],
                    branch_net = "CNN",
                    width=model_config['width'],
                    activation_fn_name=model_config['activation_fn_name'],
                    )
    
    elif model_name == "deeponet_resnet":
        from models.DeepONet.deeponet import AutoDeepONet
        model = AutoDeepONet(
                    in_channels=len(data_config['filter_in_channels']),
                    out_channels=len(data_config['filter_out_channels']), 
                    grid_resolution=data_config['grid_resolution'],
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    branch_depth=model_config['branch_depth'],
                    trunk_depth=model_config['trunk_depth'],
                    padding=model_config['padding'],
                    latent_channels=model_config['latent_channels'],
                    branch_net = "ResNet",
                    width=model_config['width'],
                    activation_fn_name=model_config['activation_fn_name'],
                    ResNet_block= model_config['ResNet_block'],
                    num_blocks= model_config['num_blocks'],
                    )
    
    elif model_name == "cno":
        from models.CNO.cno import CNO
        model = CNO(
                    in_channels=len(data_config['filter_in_channels']),
                    out_channels=len(data_config['filter_out_channels']), 
                    grid_resolution=data_config['grid_resolution'], 
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    cno_depth=model_config['cno_depth'],
                    n_blocks=model_config['n_blocks'],
                    n_blocks_bottleneck=model_config['n_blocks_bottleneck'],
                    channel_multiplier=model_config['channel_multiplier'],
                    norm=model_config['norm'],
                    latent_channels=model_config['latent_channels'],
                    # Special activation function for CNO (defined in cno_utils.py)
                    )
    
    else:
        raise ValueError(f"Model {model_name} is not implemented yet.")                      
    
    return model