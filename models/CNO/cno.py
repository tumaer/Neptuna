import torch
import torch.nn as nn
from torch import Tensor
from models.CNO.cno_utils import CNOBlock, LiftProjectBlock, ResNet
from typing import List, Optional, Union, Callable, Tuple

def _div_size(size, factor):
    if isinstance(size, int):
        return size // factor
    return tuple(s // factor for s in size)

class CNO(nn.Module):
    def __init__(self,
                in_dim: int,                    # Number of input channels.
                out_dim: int,                   # Number of input channels.
                size: Union[int, List[int], Tuple[int]],  # Input and Output spatial size (required )
                N_layers: int,                  # Number of (D) or (U) blocks in the network
                dimension: int,
                sequence_info: Optional[List[List[int]]] = [[1,1,1,1]],
                N_res: int = 4,                 # Number of (R) blocks per level (except the neck)
                N_res_neck: int = 4,            # Number of (R) blocks in the neck
                channel_multiplier: int = 16,   # How the number of channels evolve?
                use_bn: bool = True,             # Add BN? We do not add BN in lifting/projection layer
                ):

        super().__init__()

        self.N_layers = int(N_layers)         # Number od (D) & (U) Blocks
        self.lift_dim = channel_multiplier//2 # Input is lifted to the half of channel_multiplier dimension
        self.in_dim   = in_dim * sequence_info[0][0] 
        self.out_dim  = out_dim * sequence_info[0][1]
        self.channel_multiplier = channel_multiplier  # The growth of the channels
        self.dimension = dimension

        ######## Num of channels/features - evolution ########

        self.encoder_features = [self.lift_dim] # How the features in Encoder evolve (number of features)
        for i in range(self.N_layers):
            self.encoder_features.append(2 ** i *   self.channel_multiplier)

        self.decoder_features_in = self.encoder_features[1:] # How the features in Decoder evolve (number of features)
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2*self.decoder_features_in[i] #Pad the outputs of the resnets (we must multiply by 2 then)

        ######## Spatial sizes of channels - evolution ########

        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(_div_size(size, 2 ** i))
            self.decoder_sizes.append(_div_size(size, 2 ** (self.N_layers - i)))


        ######## Define Lift and Project blocks ########

        self.lift   = LiftProjectBlock(in_channels = self.in_dim,
                                       out_channels = self.encoder_features[0],
                                        dimension = self.dimension,
                                        size = size)

        self.project   = LiftProjectBlock(in_channels = self.encoder_features[0] + self.decoder_features_out[-1],
                                          out_channels = self.out_dim,
                                          dimension = self.dimension,
                                          size = size)

        ######## Define Encoder, ED Linker and Decoder networks ########

        self.encoder         = nn.ModuleList([(CNOBlock(in_channels  = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i+1],
                                                        in_size      = self.encoder_sizes[i],
                                                        out_size     = self.encoder_sizes[i+1],
                                                        dimension = self.dimension,
                                                        use_bn       = use_bn))
                                                for i in range(self.N_layers)])

        # After the ResNets are executed, the sizes of encoder and decoder might not match (if out_size>1)
        # We must ensure that the sizes are the same, by aplying CNO Blocks
        self.ED_expansion     = nn.ModuleList([(CNOBlock(in_channels = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i],
                                                        in_size      = self.encoder_sizes[i],
                                                        out_size     = self.decoder_sizes[self.N_layers - i],
                                                        dimension = self.dimension,
                                                        use_bn       = use_bn))
                                                for i in range(self.N_layers + 1)])

        self.decoder         = nn.ModuleList([(CNOBlock(in_channels  = self.decoder_features_in[i],
                                                        out_channels = self.decoder_features_out[i],
                                                        in_size      = self.decoder_sizes[i],
                                                        out_size     = self.decoder_sizes[i+1],
                                                        dimension = self.dimension,
                                                        use_bn       = use_bn))
                                                for i in range(self.N_layers)])

        #### Define ResNets Blocks 

        # Here, we define ResNet Blocks.

        # Operator UNet:
        # Outputs of the middle networks are patched (or padded) to corresponding sets of feature maps in the decoder

        self.res_nets = []
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)

        # Define the ResNet networks (before the neck)
        for l in range(self.N_layers):
            self.res_nets.append(ResNet(channels = self.encoder_features[l],
                                        size = self.encoder_sizes[l],
                                        num_blocks = self.N_res,
                                        dimension = self.dimension,
                                        use_bn = use_bn))

        self.res_net_neck = ResNet(channels = self.encoder_features[self.N_layers],
                                    size = self.encoder_sizes[self.N_layers],
                                    num_blocks = self.N_res_neck,
                                    dimension = self.dimension,
                                    use_bn = use_bn)

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, 
                input_data: Tensor,
                labels: Tensor) -> Tensor: #NOTE: Vimp: forward SHOULD always have the arguments EXACTLY named as "input_data" and "labels", 
                                           #else the data collator will remove them. 
                                           
        #reshape input into [batch, in_channel, grid_x, grid_y, ...]
        #NOTE: input and output fields need not be necessarily the same.
        batch, input_seq, input_fields, *spatial = input_data.shape
        x = input_data.reshape(batch, input_seq * input_fields, *spatial)
                        
        x = self.lift(x) #Execute Lift
        skip = []
       
        # Execute Encoder
        for i in range(self.N_layers):

            #Apply ResNet & save the result
            z = self.res_nets[i](x)
            skip.append(z)

            # Apply (D) block
            x = self.encoder[i](x)
        
        # Apply the deepest ResNet (bottle neck)
        x = self.res_net_neck(x)

        # Execute Decode
        for i in range(self.N_layers):

            # Apply (I) block (ED_expansion) & cat if needed
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x) #BottleNeck : no cat
            else:
                x = torch.cat((x, self.ED_expansion[self.N_layers - i](skip[-i])),1)

            # Apply (U) block
            x = self.decoder[i](x)

        # Cat & Execute Projetion
        x = torch.cat((x, self.ED_expansion[0](skip[0])),1)
        x = self.project(x)
        
         # Reshape the prediction to match the labels shape
        batch, output_seq, output_fields, *spatial = labels.shape
        y = x.reshape(batch, output_seq, output_fields, *spatial)

        return y,labels
    
