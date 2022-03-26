from typing import Any, Optional, OrderedDict
import torch
from torch.nn.modules import LayerNorm
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.config.config import Config
from cptr_model.core.core_module_extension import CoreModuleExtension
from cptr_model.models.base.base_encoder_decoder_block import BaseEncoderDecoderBlock
from cptr_model.models.cptr.attention.attention import Attention
from cptr_model.models.cptr.mlp.mlp import MLP


class CPTRDecoderBlock(torch.nn.Module, BaseEncoderDecoderBlock, CoreModuleExtension):
    def __init__(self, config: Config, **kwargs) -> None:
        if config is None:
            raise ValueError('Expected instance of config object to be other than None')

        self.config = config
        self.model_config: ArchitectureConfigFileManager = self.config.cptr_specifics

        self.masked_attention_dim = self.model_config.decoder_masked_self_attention_dim
        self.cross_attention_dim = self.model_config.decoder_cross_attention_dim
        self.eps = self.config.decoder_normalization_eps

        self.mlp_dim_masked_attention = self.model_config.decoder_masked_self_attention_mlp_dim
        self.mlp_dim_cross_attention = self.model_config.decoder_cross_attention_mlp_dim
        self.mlp_dropout_masked_attention = self.model_config.decoder_masked_self_attention_mlp_dropout
        self.mlp_dropout_cross_attention = self.model_config.decoder_cross_attention_mlp_dropout
        
        super(CPTRDecoderBlock, self).__init__()
        
        attention_kwargs = {
            Attention.KEY_MASKED_ATTENTION: True,
            Attention.KEY_NUM_HEADS: self.model_config.decoder_masked_self_attention_heads,
            Attention.KEY_LATENT_DIM: self.masked_attention_dim
        }
        self.masked_self_attention = Attention(self.config, **attention_kwargs)
        self.cross_attention = Attention(self.config, **{
            Attention.KEY_MASKED_ATTENTION: True,
            Attention.KEY_NUM_HEADS: self.cross_attention_dim,
            Attention.KEY_LATENT_DIM: self.cross_attention_dim
        })
        self.cross_attention_normalization = LayerNorm(self.masked_attention_dim, eps=self.eps)
        self.ffn_normalization = LayerNorm(self.masked_attention_dim, eps=self.eps)
        self.masked_self_attention_normalization = LayerNorm(self.masked_attention_dim, eps=self.eps)
        mlp_kwargs = {
            MLP.KEY_LATENT_DIM: self.cross_attention_dim,
            MLP.KEY_MLP_DIM: self.mlp_dim_cross_attention,
            MLP.KEY_DROPOUT_RATE: self.mlp_dropout_cross_attention
        }
        self.ffn = MLP(**mlp_kwargs)

    def __verify_required_args(self) -> None:
        if not self.eps:
            raise ValueError('decoder eps value is None')

        if not self.mlp_dim_cross_attention:
            raise ValueError('mlp dim cross attention not found in config: has value None')

        if not self.mlp_dropout_cross_attention:
            raise ValueError('mlp dropout cross attention rate not found in config: has value None')

        if not self.mlp_dim_masked_attention:
            raise ValueError('mlp dim masked attention not found in config: has value None')

        if not self.mlp_dropout_masked_attention:
            raise ValueError('mlp dropout masled attention rate not found in config: has value None')

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, lookahead_mask: Optional[torch.Tensor], pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = x
        x, _ = self.masked_self_attention(x, x, x, lookahead_mask)
        x = x + h
        x = self.masked_self_attention_normalization(x)
        h = x
        x, _ = self.cross_attention(x, enc_output, enc_output, None)
        x = x + h
        x = self.cross_attention_normalization(x)
        h = x
        x = self.ffn(x)
        x = x + h
        x = self.ffn_normalization(x)

        return x

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        model_dict = self.state_dict()
        model_dict[CPTRDecoderBlock.StateKey.MASKED_SELF_ATTENTION_NORMALIZATION_WEIGHT] = weights.get(CPTRDecoderBlock.StateKey.MASKED_SELF_ATTENTION_NORMALIZATION_WEIGHT, None)
        self.cross_attention.weight_transfer_from_dict(weights[CPTRDecoderBlock.StateKey.CROSS_ATTENTION_WEIGHT])
        self.masked_self_attention.weight_transfer_from_dict(weights[CPTRDecoderBlock.StateKey.MASKED_SELF_ATTENTION_WEIGHT])
        self.ffn.weight_transfer_from_dict(weights[CPTRDecoderBlock.StateKey.FFN_WEIGHT])
        model_dict[CPTRDecoderBlock.StateKey.CROSS_ATTENTION_NORMALIZATION_WEIGHT] = weights[CPTRDecoderBlock.StateKey.CROSS_ATTENTION_NORMALIZATION_WEIGHT]
        model_dict[CPTRDecoderBlock.StateKey.FFN_NORMALIZATION_WEIGHT] = weights[CPTRDecoderBlock.StateKey.FFN_NORMALIZATION_WEIGHT]
        self.load_state_dict(model_dict)

    def bias_transfer_from_dict(self, bias: OrderedDict[str, Any]) -> None:
        model_dict = self.state_dict()
        model_dict[CPTRDecoderBlock.StateKey.MASKED_SELF_ATTENTION_NORMALIZATION_BIAS] = bias[CPTRDecoderBlock.StateKey.MASKED_SELF_ATTENTION_NORMALIZATION_BIAS]
        self.cross_attention.bias_transfer_from_dict(bias[CPTRDecoderBlock.StateKey.CROSS_ATTENTION_BIAS])
        self.masked_self_attention.bias_transfer_from_dict(bias[CPTRDecoderBlock.StateKey.MASKED_SELF_ATTENTION_BIAS])
        self.ffn.bias_transfer_from_dict(bias[CPTRDecoderBlock.StateKey.FFN_BIAS])
        model_dict[CPTRDecoderBlock.StateKey.CROSS_ATTENTION_NORMALIZATION_BIAS] = bias[CPTRDecoderBlock.StateKey.CROSS_ATTENTION_NORMALIZATION_BIAS]
        model_dict[CPTRDecoderBlock.StateKey.FFN_NORMALIZATION_BIAS] = bias[CPTRDecoderBlock.StateKey.FFN_NORMALIZATION_BIAS]
        self.load_state_dict(model_dict)

    def weight_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            CPTRDecoderBlock.StateKey.MASKED_SELF_ATTENTION_NORMALIZATION_WEIGHT: self.masked_self_attention_normalization.weight,
            CPTRDecoderBlock.StateKey.MASKED_SELF_ATTENTION_WEIGHT: self.masked_self_attention.weight_transfer_to_dict(),
            CPTRDecoderBlock.StateKey.CROSS_ATTENTION_WEIGHT: self.cross_attention.weight_transfer_to_dict(),
            CPTRDecoderBlock.StateKey.FFN_WEIGHT: self.ffn.weight_transfer_to_dict(),
            CPTRDecoderBlock.StateKey.CROSS_ATTENTION_NORMALIZATION_WEIGHT: self.cross_attention_normalization.weight,
            CPTRDecoderBlock.StateKey.FFN_NORMALIZATION_WEIGHT: self.ffn_normalization.weight
        })

    def bias_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            CPTRDecoderBlock.StateKey.MASKED_SELF_ATTENTION_NORMALIZATION_BIAS: self.masked_self_attention_normalization.bias,
            CPTRDecoderBlock.StateKey.MASKED_SELF_ATTENTION_BIAS: self.masked_self_attention.bias_transfer_to_dict(),
            CPTRDecoderBlock.StateKey.CROSS_ATTENTION_BIAS: self.cross_attention.bias_transfer_to_dict(),
            CPTRDecoderBlock.StateKey.FFN_BIAS: self.ffn.bias_transfer_to_dict(),
            CPTRDecoderBlock.StateKey.CROSS_ATTENTION_NORMALIZATION_BIAS: self.cross_attention_normalization.bias,
            CPTRDecoderBlock.StateKey.FFN_NORMALIZATION_BIAS: self.ffn_normalization.bias
        })

    class StateKey:
        MASKED_SELF_ATTENTION_NORMALIZATION_WEIGHT = 'masked_self_attention_normalization.weight'
        MASKED_SELF_ATTENTION_NORMALIZATION_BIAS = 'masked_self_attention_normalization.bias'
        MASKED_SELF_ATTENTION_WEIGHT = 'masked_self_attention.weight'
        MASKED_SELF_ATTENTION_BIAS = 'masked_self_attention.bias'
        CROSS_ATTENTION_WEIGHT = 'cross_attention.weight'
        CROSS_ATTENTION_BIAS = 'cross_attention.bias'
        CROSS_ATTENTION_NORMALIZATION_WEIGHT = 'cross_attention_normalization.weight'
        CROSS_ATTENTION_NORMALIZATION_BIAS = 'cross_attention_normalization.bias'
        FFN_WEIGHT = 'ffn.weight'
        FFN_BIAS = 'ffn.bias'
        FFN_NORMALIZATION_WEIGHT = 'ffn_normalization.weight'
        FFN_NORMALIZATION_BIAS = 'ffn_normalization.bias'