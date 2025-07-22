
from utils.model_utils import PretrainedConfig


class ViTConfig(PretrainedConfig):
    """
    Args:
        ToDo !!!
        AND: rename params
        
    """

    model_type = "vit"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        encoder_stride=16,
        pooler_output_size=None,
        pooler_act="tanh",
        interpolate_pos_encoding=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride
        self.pooler_output_size = pooler_output_size if pooler_output_size else hidden_size
        self.pooler_act = pooler_act

        self.interpolate_pos_encoding = interpolate_pos_encoding

        self.num_channels = self.in_size