from typing import Dict

def fetch_model(model_config: Dict,
                data_config: Dict):
    """
    Factory function to instantiate neural network models based on configuration.
    
    This function serves as the primary entry point for model creation, supporting
    multiple deep learning architectures. It handles model-specific configuration
    parsing and instantiation with appropriate parameter validation.
    
    Parameters
    ----------
    model_config : Dict
        Model configuration dictionary containing architecture-specific parameters.
        Must include 'model_name' key specifying the model type. Additional
        required keys depend on the selected model architecture.
        
        Common parameters across models:
        - model_name : str
            Name of the model architecture (case-insensitive)
        - latent_channels : int
            Number of latent/hidden channels in the model
        - activation_fn_name : str
            Name of activation function to use
            
    data_config : Dict
        Data configuration dictionary containing dataset-specific parameters.
        
        Required keys:
        - dimension : int
            Spatial dimension of the data (1D, 2D, or 3D)
        - filter_in_channels : List[str]
            Input channel names (length determines input channel count)
        - filter_out_channels : List[str]
            Output channel names (length determines output channel count)
        - sequence_info : List[int]
            Sequence configuration [input_len, label_len, stride]
            
        Optional keys:
        - grid_resolution : List[int]
            Spatial resolution for each dimension (required for some models)
    
    Returns
    -------
    torch.nn.Module
        Instantiated neural network model ready for training or inference.
        The specific model type depends on the 'model_name' parameter.
    
    Raises
    ------
    ValueError
        If the specified model_name is not implemented or if required
        configuration parameters are missing.
    KeyError
        If required configuration keys are missing from model_config
        or data_config dictionaries.
    
    Supported Models
    ----------------
    fno : Fourier Neural Operator
        Requires: fno_modes, n_fno_layers, padding, padding_type,
        decoder_layers, decoder_layer_size, decoder_activation_fn_name
        
    resnet : Residual Network
        Requires: num_blocks, norm, n_groups
        Uses BasicBlock architecture
        
    dilresnet : Dilated Residual Network
        Requires: num_blocks, norm, n_groups
        Uses DilatedBasicBlock architecture
        
    unet : U-Net Architecture
        Requires: norm, n_groups, channel_multiplier, is_attn,
        mid_attn, n_blocks, use1x1
        
    deeponet_ffn : DeepONet with Feed-Forward Branch Network
        Requires: branch_depth, trunk_depth, width, act_on_output
        
    deeponet_cnn : DeepONet with CNN Branch Network
        Requires: branch_depth, trunk_depth, kernel_size, padding, width
        
    deeponet_resnet : DeepONet with ResNet Branch Network
        Requires: branch_depth, trunk_depth, padding, width,
        ResNet_block, num_blocks
        
    cno : Convolutional Neural Operator
        Requires: cno_depth, n_blocks, n_blocks_bottleneck,
        channel_multiplier, norm
        
    scot : Swin Transformer for Operator Learning
        Requires: patch_size, embed_dim, depths, num_heads,
        skip_connections, window_size, mlp_ratio, qkv_bias,
        hidden_dropout_prob, attention_dropout_prob, drop_path_rate,
        hidden_act, use_absolute_embeddings, initializer_range,
        norm_layer_eps, residual_model, conditioning,
        output_hidden_states, output_attentions
        Note: Currently only supports 2D data
    
    Notes
    -----
    Model Configuration:
    - All models automatically include coordinate features (coord_features=True)
    - Input/output channel counts are determined from data_config channel lists
    - Sequence information is passed to models.
    
    Architecture Selection:
    - Model names are case-insensitive for convenience
    - Each model has specific required parameters in model_config
    - Some models have dimensional restrictions (e.g., ScOT requires 2D data)
    
    Error Handling:
    - Missing required parameters will raise KeyError with descriptive messages
    - Unsupported model names will raise ValueError with available options
    - Dimensional mismatches will raise ValueError with specific requirements
    
    The function automatically handles the import of model classes and their
    corresponding configuration classes, ensuring proper encapsulation and
    avoiding unnecessary imports for unused models.
    """
    model_name = model_config['model_name'].lower()

    if model_name == "fno":
        from models.FNO.fno import FNO
        from models.FNO.fno_utils import FNOConfig
        config = FNOConfig(
                    dimension=data_config['dimension'],
                    in_channels=len(data_config['filter_features']['filter_in_channels']),
                    out_channels=len(data_config['filter_features']['filter_out_channels']), 
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
                    coord_features=True
                    )
        model = FNO(config=config)         
        
    elif model_name == "resnet":
        from models.ResNet.resnet import ResNet
        from models.ResNet.resnet_utils import ResNetConfig
        import torch
        config = ResNetConfig(
                    in_channels=len(data_config['filter_features']['filter_in_channels']),
                    out_channels=len(data_config['filter_features']['filter_out_channels']), 
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    num_blocks=model_config['num_blocks'],
                    block="BasicBlock",
                    latent_channels=model_config['latent_channels'],
                    activation_fn_name=model_config['activation_fn_name'],
                    coord_features=True,
                    conditioning=data_config['conditioning_features']['include_conditioning_parameters'], # if True ConditionalLayerNorm is used otherwise LayerNorm
                    num_cond_params = data_config['conditioning_features']['num_cond_params'] if data_config['conditioning_features']['include_conditioning_parameters'] else 0,
                    norm_layer_eps=model_config['norm_layer_eps'], # used in norm_layer both ConditionalLayerNorm and LayerNorm; add to variance of normalization to avoid division by zero and stabilize training
                    norm=model_config['norm']
                    )
        model = ResNet(config=config)
    
    elif model_name == "dilresnet":
        from models.ResNet.resnet import ResNet
        from models.ResNet.resnet_utils import ResNetConfig
        config = ResNetConfig(
                    in_channels=len(data_config['filter_features']['filter_in_channels']),
                    out_channels=len(data_config['filter_features']['filter_out_channels']), 
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    num_blocks=model_config['num_blocks'],
                    block="DilatedBasicBlock",
                    latent_channels=model_config['latent_channels'],
                    activation_fn_name=model_config['activation_fn_name'],
                    coord_features=True,
                    conditioning=data_config['conditioning_features']['include_conditioning_parameters'], # if True ConditionalLayerNorm is used otherwise LayerNorm
                    num_cond_params = data_config['conditioning_features']['num_cond_params'] if data_config['conditioning_features']['include_conditioning_parameters'] else 0,
                    norm_layer_eps=model_config['norm_layer_eps'], # used in norm_layer both ConditionalLayerNorm and LayerNorm; add to variance of normalization to avoid division by zero and stabilize training
                    norm=model_config['norm']
                    )
        model = ResNet(config=config)
    
    elif model_name == "unet":
        from models.UNet.unet import UNet
        from models.UNet.unet_utils import UNetConfig
        config = UNetConfig(
                    dimension=data_config['dimension'],
                    in_channels=len(data_config['filter_features']['filter_in_channels']),
                    out_channels=len(data_config['filter_features']['filter_out_channels']), 
                    sequence_info=data_config["sequence_info"], 
                    latent_channels=model_config['latent_channels'],
                    channel_multiplier=model_config['channel_multiplier'],
                    is_attn=model_config['is_attn'],
                    mid_attn=model_config['mid_attn'],
                    n_blocks=model_config['n_blocks'],
                    use1x1=model_config['use1x1'],
                    activation_fn_name=model_config['activation_fn_name'],
                    coord_features=True,
                    conditioning=data_config['conditioning_features']['include_conditioning_parameters'], # if True ConditionalLayerNorm is used otherwise LayerNorm
                    num_cond_params = data_config['conditioning_features']['num_cond_params'] if data_config['conditioning_features']['include_conditioning_parameters'] else 0,
                    norm_layer_eps=model_config['norm_layer_eps'], # used in norm_layer both ConditionalLayerNorm and LayerNorm; add to variance of normalization to avoid division by zero and stabilize training
                    norm=model_config['norm']
                    )
        model= UNet(config=config)
    
    elif model_name == "deeponet_ffn":
        from models.DeepONet.deeponet import AutoDeepONet
        from models.DeepONet.deeponet_utils import DeepONetConfig
        config = DeepONetConfig(
                    in_channels=len(data_config['filter_features']['filter_in_channels']),
                    out_channels=len(data_config['filter_features']['filter_out_channels']), 
                    grid_resolution=data_config['grid_resolution'],
                    sequence_info=data_config["sequence_info"],
                    dimension=data_config['dimension'],
                    branch_depth=model_config['branch_depth'],
                    trunk_depth=model_config['trunk_depth'],
                    width=model_config['width'],
                    branch_net = "FFN",
                    act_on_output=model_config['act_on_output'], #only for FFN
                    activation_fn_name=model_config['activation_fn_name'],
                    coord_features=True
        )
        model = AutoDeepONet(config=config)
    
    elif model_name == "deeponet_cnn":
        from models.DeepONet.deeponet import AutoDeepONet
        from models.DeepONet.deeponet_utils import DeepONetConfig
        config = DeepONetConfig(                    
                    in_channels=len(data_config['filter_features']['filter_in_channels']),
                    out_channels=len(data_config['filter_features']['filter_out_channels']), 
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
                    coord_features=True
                    )
        model = AutoDeepONet(config=config)
    
    elif model_name == "deeponet_resnet":
        from models.DeepONet.deeponet import AutoDeepONet
        from models.DeepONet.deeponet_utils import DeepONetConfig
        config= DeepONetConfig(
                    in_channels=len(data_config['filter_features']['filter_in_channels']),
                    out_channels=len(data_config['filter_features']['filter_out_channels']), 
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
                    coord_features=True
        )
        model = AutoDeepONet(config=config)
    
    elif model_name == "cno":
        from models.CNO.cno_utils import CNOConfig
        from models.CNO.cno import CNO
        config = CNOConfig(
                    in_channels=len(data_config['filter_features']['filter_in_channels']),
                    out_channels=len(data_config['filter_features']['filter_out_channels']), 
                    grid_resolution=data_config['grid_resolution'], 
                    sequence_info=data_config['sequence_info'],
                    dimension=data_config['dimension'],
                    cno_depth=model_config['cno_depth'],
                    n_blocks=model_config['n_blocks'],
                    n_blocks_bottleneck=model_config['n_blocks_bottleneck'],
                    channel_multiplier=model_config['channel_multiplier'],
                    norm=model_config['norm'],
                    latent_channels=model_config['latent_channels'],
                    coord_features=True
                    # Special activation function for CNO (defined in cno_utils.py))
                    )
        model = CNO(config=config)

    elif model_name == 'scot':
        from models.ScOT.scot_utils import ScOTConfig
        from models.ScOT.scot import ScOT 

        if data_config['dimension'] != 2:
            raise ValueError("Model is not yet implemented for other dimension than 2")
        
        config = ScOTConfig(
                    patch_size=model_config['patch_size'],
                    in_channels=len(data_config['filter_features']['filter_in_channels']),
                    out_channels=len(data_config['filter_features']['filter_out_channels']),
                    grid_resolution=data_config['grid_resolution'],
                    dimension=data_config['dimension'],
                    sequence_info=data_config['sequence_info'],
                    latent_channels=model_config['latent_channels'], # base dimensionality of patch embeddings (size of feature vector used to represent each patch)
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
                    residual_model=model_config['residual_model'], # either convnext or resnet
                    output_hidden_states=model_config['output_hidden_states'],
                    output_attentions=model_config['output_attentions'],
                    coord_features=True,
                    conditioning=data_config['conditioning_features']['include_conditioning_parameters'] , # if True ConditionalLayerNorm is used otherwise LayerNorm
                    num_cond_params = data_config['conditioning_features']['num_cond_params'] if data_config['conditioning_features']['include_conditioning_parameters'] else 0,
                    norm_layer_eps=model_config['norm_layer_eps'], # used in norm_layer both ConditionalLayerNorm and LayerNorm; add to variance of normalization to avoid division by zero and stabilize training
                    norm=model_config['norm']
                    )
        model = ScOT(config)
    
    else:
        raise ValueError(f"Model {model_name} is not implemented yet.") 
    
    return model