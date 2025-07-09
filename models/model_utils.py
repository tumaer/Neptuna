from transformers import PretrainedConfig as PretrainedConfig_
import json
from typing import List, Optional, Tuple, Union
from omegaconf import OmegaConf

#import smdistributed.modelparallel.torch as smp


class PretrainedConfig(PretrainedConfig_):
    """
    Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
    methods for loading/downloading/saving configurations.
     Args:
        in_channels (int): Number of input channels for the model.
        out_channels (int): Number of output channels for the model.
        dimension (int): Dimensionality of the input data (e.g., 1D, 2D, or 3D).
        grid_resolution (Union[int, List[int], Tuple[int]]): Input and output spatial size.
        sequence_info (Optional[List[int]]): Sequence information for the model. Elements of List are: input sequence length, output sequence length, stride
        coord_features (bool): Whether to include coordinate features in the model. Default is True.
        latent_channels (int): Number of latent channels in the model.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        in_channels: int = 1,       # Number of input_channels
        out_channels: int = 1,      # Number of output channels
        dimension: int = 1,
        grid_resolution: Union[int, List[int], Tuple[int]] = [160], # Input and Output spatial size (required )
        sequence_info: Optional[List[int]] = [1,1,1],
        coord_features: bool = True,
        latent_channels: int = 32,
        include_input_seq_len: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimension = dimension
        self.sequence_info = sequence_info

        if include_input_seq_len:
            self.in_size = self.in_channels * self.sequence_info[0] 
            self.out_size = self.out_channels * self.sequence_info[1]

        self.grid_resolution = grid_resolution
        self.latent_channels = latent_channels

        self.coord_features = coord_features
        # Add relative coordinate feature
        if coord_features:
            self.in_size = self.in_size + dimension


    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        # NOTE: default dumpts List objects in the PretrainedConfig to json, since only json.dumps throws an error
        def default(o):
            return OmegaConf.to_container(o, resolve=True)
        return json.dumps(config_dict, indent=2, sort_keys=True, default=default) + "\n"
