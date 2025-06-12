
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
                    dimension=model_config['dimension'],
                    in_channels=model_config['in_channels'],
                    out_channels=model_config['out_channels'], 
                    sequence_info=data_config["sequence_info"],
                    latent_channels=model_config['latent_channels'],
                    num_fno_modes=eval(model_config['fno_modes']),
                    num_fno_layers=model_config['fno_layers'],
                    padding=model_config['padding'],
                    decoder_layers=model_config['decoder_layers'],
                    decoder_layer_size=model_config['decoder_layer_size'],
                    #TODO: more arguments like act. fn name etc to be added
                    ) 
    
    elif model_name == "unet-attn":
        from models.UNet.unet import Unet
        model= Unet(
            dimension=model_config['dimension'],
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'], 
            latent_channels=model_config['latent_channels'],
            sequence_info=data_config["sequence_info"], #TODO: more arguments to be added
        )
    
    elif model_name == 'scot':
        from models.ScOT.scot import ScOT, ScOTConfig
        scot_config = ScOTConfig(
            image_size=model_config['resolution'],
            patch_size=model_config['patch_size'],
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
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
            p=model_config['p'], # 1: l1 loss , 2: l2 loss
            channel_slice_list_normalized_loss=model_config['channel_slice_list_normalized_loss'], # if None will fall back to absolute loss otherwise normalized loss with split channels
            # divide output tensor into channel slices and compute normalized loss per slice (and then average)
            residual_model=model_config['residual_model'], # either convnext or resnet
            use_conditioning=model_config['use_conditioning'], # if True ConditionalLayerNorm is used otherwise LayerNorm
            learn_residual=model_config['learn_residual'], # can only be used if use_conditioning is True -> model trained to predict residual (difference) between input and target, rather than full output directly
            input_steps=data_config['sequence_info'][0][0],
            output_steps=data_config['sequence_info'][0][0]
        )
        model = ScOT(scot_config)
    
    return model