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
    elif model_name == 'scot':
        from models.ScOT.scot_utils import ScOTConfig
        from models.ScOT.scot import ScOT 

        if data_config['dimension'] != 2:
            raise ValueError("Model is not yet implemented for other dimension than 2")
        
        scot_config = ScOTConfig(
            resolution_x=data_config['grid_resolution'][0],
            resolution_y=data_config['grid_resolution'][1],
            patch_size=model_config['patch_size'],
            in_channels=len(data_config['filter_in_channels']),
            out_channels=len(data_config['filter_out_channels']),
            embed_dim=model_config['embed_dim'], # base dimensionality of patch embeddings (size of feature vector used to represent each patch)
            depths=model_config['depths'], #number of transformer blocks in encoder / decoder stages e.g. 4 stages each with 4 transformer blocks
            num_heads=model_config['num_heads'], # used in Swinv2SelfAttention (HF) (see ScOTEncoder: each stage has own num_heads
            # number of separate attention machanisms run in parallel; attend to different local spatial features inside each window
            skip_connections=model_config['skip_connections'], # depth of skip connections
            window_size=model_config['window_size'], # defines spatial region over which self-attention is computed in one local block instead of expensive global self-attention
            mlp_ratio=model_config['mlp_ratio'], # used in Swinv2Intermediate (HF) to expand hidden state (model gets more capacity to learn non-linear transformations
            qkv_bias=model_config['qkv_bias'], # disable / enable bias in self-attention (Q = X @ W_Q + b_Q; K = X @ W_K + b_K; V = X @ W_V + b_V) # used in Swinv2SelfAttention (HF)
            hidden_dropout_prob=model_config['hidden_dropout_prob'],  # default # for the dropout in ScOT embedding
            attention_probs_dropout_prob=model_config['attention_dropout_prob'],  # default # dropout in Swinv2SelfAttention (HF)
            drop_path_rate=model_config['drop_path_rate'], # used to create drop path for each ScOTEncodeStage in Encoder and ScOTDecodeStage in Decoder, is max. value
            hidden_act=model_config['hidden_act'], # hidden activation function in Swinv2Intermediate (HF)
            use_absolute_embeddings=model_config['use_absolute_embeddings'], # absolute position information into the patch embeddings (spatial structure of trajectory); different to time_conditioning
            initializer_range=model_config['initializer_range'], # Swinv2PreTrainedModel (HF), std of normal distribution to initialize weights
            layer_norm_eps=model_config['layer_norm_eps'], # used in layer_norm both ConditionalLayerNorm and LayerNorm; add to variance of normalization to avoid division by zero and stabilize training
            residual_model=model_config['residual_model'], # either convnext or resnet
            use_conditioning=model_config['use_conditioning'], # if True ConditionalLayerNorm is used otherwise LayerNorm
            input_steps=data_config['sequence_info'][0],
            output_steps=data_config['sequence_info'][1],
            output_hidden_states=model_config['output_hidden_states'],
            output_attentions=model_config['output_attentions'],
            coord_features=True
        )
        model = ScOT(scot_config)
    
    else:
        raise ValueError(f"Model {model_name} is not implemented yet.") 
    
    return model