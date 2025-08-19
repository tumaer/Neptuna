import torch
import torch.nn as nn
from torch import Tensor
from models.CNO.cno_utils import CNOBlock, LiftProjectBlock, ResNet
from transformers import PreTrainedModel
from utils.grid_utils import oned_meshgrid, twod_meshgrid, threed_meshgrid
from .cno_utils import CNOConfig
def _div_size(size, factor):
    if isinstance(size, int):
        return size // factor
    return tuple(s // factor for s in size)

class CNO(PreTrainedModel):
    """Convolutional Neural Operator (CNO) model for learning mappings between function spaces.

    The CNO architecture consists of an encoder-decoder structure with residual blocks, designed to learn
    mappings between function spaces. It uses a hierarchical structure with multiple resolution levels
    and incorporates residual connections for better gradient flow.

    Architecture:
        - Encoder: Downsampling path with residual blocks
        - Bottleneck: Deepest residual block
        - Decoder: Upsampling path with skip connections
        - Lift/Projection: Initial feature lifting and final projection

    Args:
        in_channels (int): Number of input channels/features
        out_channels (int): Number of output channels/features
        grid_resolution (Union[int, List[int], Tuple[int]]): Input and output spatial resolution
        cno_depth (int): Number of down/up sampling blocks in the network
        dimension (int): Spatial dimension of the data (1, 2, or 3)
        sequence_info (Optional[List[int]], optional): Sequence information for input/output. Defaults to [1,1,1]
        n_blocks (int, optional): Number of residual blocks per level. Defaults to 4
        n_blocks_bottleneck (int, optional): Number of residual blocks in the bottleneck. Defaults to 4
        channel_multiplier (int, optional): Base channel multiplier for network width. Defaults to 16
        norm (bool, optional): Whether to use batch normalization. Defaults to True
        latent_channels (int, optional): Number of channels in latent space. Defaults to 64

    Shape:
        - Input: (batch_size, in_channels, *grid_resolution)
        - Output: (batch_size, out_channels, *grid_resolution)

    Example:
        >>> model = CNO(in_channels=3, out_channels=1, grid_resolution=64, cno_depth=4, dimension=2)
        >>> x = torch.randn(1, 3, 64, 64)
        >>> output = model(x)
    """
    main_input_name = "input_data"  
    conditioning_input_name = "conditioning_input_data"
    config_class = CNOConfig

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        ######## Num of channels/features - evolution ########

        self.encoder_features = [config.lift_dim] # How the features in Encoder evolve (number of features)
        for i in range(config.cno_depth):
            self.encoder_features.append(2 ** i * config.channel_multiplier)

        self.decoder_features_in = self.encoder_features[1:] # How the features in Decoder evolve (number of features)
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, config.cno_depth):
            self.decoder_features_in[i] = 2*self.decoder_features_in[i] #Pad the outputs of the resnets (we must multiply by 2 then)

        ######## Spatial sizes of channels - evolution ########
        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(config.cno_depth + 1):
            self.encoder_sizes.append(_div_size(config.grid_resolution, 2 ** i))
            self.decoder_sizes.append(_div_size(config.grid_resolution, 2 ** (config.cno_depth - i)))

        ######## Define Lift and Projection blocks ########
        self.lift   = LiftProjectBlock(
            config = config,
            in_channels = config.in_size,
            out_channels = self.encoder_features[0],
            dimension = config.dimension,
            grid_resolution = config.grid_resolution,
            latent_channels = config.latent_channels)

        self.project   = LiftProjectBlock(
            config = config,
            in_channels = self.encoder_features[0] + self.decoder_features_out[-1],
            out_channels = config.out_size,
            dimension = config.dimension,
            grid_resolution = config.grid_resolution,
            latent_channels = config.latent_channels)

        ######## Define Encoder, ED Linker and Decoder networks ########
        self.encoder         = nn.ModuleList([(CNOBlock(
                                                        config = config,
                                                        in_channels  = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i+1],
                                                        in_grid_resolution      = self.encoder_sizes[i],
                                                        out_grid_resolution     = self.encoder_sizes[i+1],
                                                        ))
                                                for i in range(config.cno_depth)])

        # After the ResNets are executed, the sizes of encoder and decoder might not match (if out_size>1)
        # We must ensure that the sizes are the same, by aplying CNO Blocks
        self.ED_expansion     = nn.ModuleList([(CNOBlock(
                                                        config = config,
                                                        in_channels = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i],
                                                        in_grid_resolution      = self.encoder_sizes[i],
                                                        out_grid_resolution     = self.decoder_sizes[config.cno_depth - i],
                                                        ))
                                                for i in range(config.cno_depth + 1)])

        self.decoder         = nn.ModuleList([(CNOBlock(
                                                        config = config,
                                                        in_channels  = self.decoder_features_in[i],
                                                        out_channels = self.decoder_features_out[i],
                                                        in_grid_resolution      = self.decoder_sizes[i],
                                                        out_grid_resolution     = self.decoder_sizes[i+1],
                                                        ))
                                                for i in range(config.cno_depth)])

        #### Define ResNets Blocks 
        # Operator UNet:
        # Outputs of the middle networks are patched (or padded) to corresponding sets of feature maps in the decoder

        self.res_nets = []
        self.n_blocks = int(config.n_blocks)
        self.n_blocks_bottleneck = int(config.n_blocks_bottleneck)

        # Define the ResNet networks (before the neck)
        for l in range(config.cno_depth):
            self.res_nets.append(
                                ResNet(
                                        config = config,
                                        channels = self.encoder_features[l],
                                        grid_resolution = self.encoder_sizes[l],
                                        num_blocks = self.n_blocks,
                                        )
                                )

        self.res_net_neck = ResNet(
                                    config = config,
                                    channels = self.encoder_features[config.cno_depth],
                                    grid_resolution = self.encoder_sizes[config.cno_depth],
                                    num_blocks = self.n_blocks_bottleneck,
                                )

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, 
                input_data: Tensor, 
                **kwargs) -> Tensor: 
        
        if "conditioning_input_data" in kwargs:
            #NOTE: Conditioning data can be passed into a conv network before concatination with input_data.
            conditioning_input_data = kwargs["conditioning_input_data"]
            input_data = torch.cat([input_data, conditioning_input_data], dim=2)
        else:
            conditioning_input_data = None                   
        
        batch, input_seq, input_channels, *spatial = input_data.shape
        input_data = input_data.reshape(batch, input_seq * input_channels, *spatial)
        
        if self.config.coord_features:
            if self.config.dimension == 1:
                coord_feat = oned_meshgrid(list(input_data.shape), input_data.device)
            elif self.config.dimension == 2:
                coord_feat = twod_meshgrid(list(input_data.shape), input_data.device)
            elif self.config.dimension == 3:
                coord_feat = threed_meshgrid(list(input_data.shape), input_data.device)
            x = torch.cat((input_data, coord_feat), dim=1)
            
        x = self.lift(x, **kwargs) #Execute Lift
        skip = []
       
        # Execute Encoder
        for i in range(self.config.cno_depth):

            #Apply ResNet & save the result
            z = self.res_nets[i](x, **kwargs)
            skip.append(z)

            # Apply (D) block
            x = self.encoder[i](x, **kwargs)
        
        # Apply the deepest ResNet (bottle neck)
        x = self.res_net_neck(x, **kwargs)

        # Execute Decode
        for i in range(self.config.cno_depth):

            # Apply (I) block (ED_expansion) & cat if needed
            if i == 0:
                x = self.ED_expansion[self.config.cno_depth - i](x, **kwargs) #BottleNeck : no cat
            else:
                x = torch.cat((x, self.ED_expansion[self.config.cno_depth - i](skip[-i], **kwargs)),1)

            # Apply (U) block
            x = self.decoder[i](x, **kwargs)

        # Cat & Execute Projetion
        x = torch.cat((x, self.ED_expansion[0](skip[0], **kwargs)),dim=1)
        x = self.project(x, **kwargs)

        return x
    
