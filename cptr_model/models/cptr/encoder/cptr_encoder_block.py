from typing import Optional, OrderedDict, Any
import torch
from torch.nn.modules.normalization import LayerNorm
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.config.config import Config
from cptr_model.core.core_module_extension import CoreModuleExtension
from cptr_model.models.base.base_encoder_decoder_block import BaseEncoderDecoderBlock
from cptr_model.models.cptr.attention.attention import Attention
from cptr_model.models.cptr.mlp.mlp import MLP


class CPTREncoderBlock(torch.nn.Module, BaseEncoderDecoderBlock, CoreModuleExtension):

    def __init__(self,
                 config: Optional[Config],
                 **kwargs) -> None:
        if config is None:
            raise ValueError('Expected instance of config object to be other than None')

        self.config = config
        self.model_config: ArchitectureConfigFileManager = self.config.cptr_specifics

        self.latent_dim = self.model_config.encoder_self_attention_dim
        self.eps = self.config.encoder_normalization_eps
        self.mlp_dim = self.model_config.encoder_self_attention_mlp_dim
        self.mlp_dropout = self.model_config.encoder_self_attention_mlp_dropout

        super().__init__()

        self.eps = float(self.eps)
        self.attention_norm = LayerNorm(self.latent_dim, eps=self.eps)
        self.ffn_norm = LayerNorm(self.latent_dim, eps=self.eps)
        mlp_kwargs = {
            MLP.KEY_LATENT_DIM: self.latent_dim,
            MLP.KEY_MLP_DIM: self.mlp_dim,
            MLP.KEY_DROPOUT_RATE: self.mlp_dropout
        }
        self.mlp = MLP(**mlp_kwargs)
        attention_kwargs = {
            Attention.KEY_MASKED_ATTENTION: False,
            Attention.KEY_LATENT_DIM: self.latent_dim,
            Attention.KEY_NUM_HEADS: self.model_config.encoder_self_attention_heads
        }
        self.attention = Attention(self.config, **attention_kwargs)

    def __verify_required_args(self) -> None:
        if not self.latent_dim:
            raise ValueError('Missing dim param in config file for input embeddings')
        if not self.eps:
            raise ValueError('eps value is None')
        if not self.mlp_dim:
            raise ValueError('Missing mlp dim in config file: has value None')
        if not self.mlp_dropout:
            raise ValueError('Missing mlp dropout in config file: has value None')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        x, weights = self.attention(x)
        x = x + h
        x = self.attention_norm(x)
        h = x
        x = self.mlp(x)
        x = x + h
        x = self.ffn_norm(x)

        return x

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        model_dict = self.state_dict()
        model_dict[CPTREncoderBlock.StateKey.ATTENTION_NORM_WEIGHT] = weights[CPTREncoderBlock.StateKey.ATTENTION_NORM_WEIGHT]
        self.attention.weight_transfer_from_dict(weights[CPTREncoderBlock.StateKey.ATTENTION_WEIGHT])
        self.mlp.weight_transfer_from_dict(weights[CPTREncoderBlock.StateKey.MLP_WEIGHT])
        model_dict[CPTREncoderBlock.StateKey.FFN_NORM_WEIGHT] = weights[CPTREncoderBlock.StateKey.FFN_NORM_WEIGHT]
        # self.weight_transfer_from_dict(weights)
        self.load_state_dict(model_dict)

    def bias_transfer_from_dict(self, bias: OrderedDict[str, Any]) -> None:
        model_dict = self.state_dict()
        model_dict[CPTREncoderBlock.StateKey.ATTENTION_NORM_BIAS] = bias[CPTREncoderBlock.StateKey.ATTENTION_NORM_BIAS]
        self.attention.bias_transfer_from_dict(bias[CPTREncoderBlock.StateKey.ATTENTION_BIAS])
        self.mlp.bias_transfer_from_dict(bias[CPTREncoderBlock.StateKey.MLP_BIAS])
        model_dict[CPTREncoderBlock.StateKey.FFN_NORM_BIAS] = bias[CPTREncoderBlock.StateKey.FFN_NORM_BIAS]
        self.load_state_dict(model_dict)

    def weight_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            CPTREncoderBlock.StateKey.ATTENTION_NORM_WEIGHT: self.attention_norm.weight,
            CPTREncoderBlock.StateKey.ATTENTION_WEIGHT: self.attention.weight_transfer_to_dict(),
            CPTREncoderBlock.StateKey.MLP_WEIGHT: self.mlp.weight_transfer_to_dict(),
            CPTREncoderBlock.StateKey.FFN_NORM_WEIGHT: self.ffn_norm.weight
        })

    def bias_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            CPTREncoderBlock.StateKey.ATTENTION_NORM_BIAS: self.attention_norm.bias,
            CPTREncoderBlock.StateKey.ATTENTION_BIAS: self.attention.bias_transfer_to_dict(),
            CPTREncoderBlock.StateKey.MLP_BIAS: self.mlp.bias_transfer_to_dict(),
            CPTREncoderBlock.StateKey.FFN_NORM_BIAS: self.ffn_norm.bias
        })

    class StateKey:
        ATTENTION_NORM_WEIGHT = 'attention_norm.weight'
        ATTENTION_NORM_BIAS = 'attention_norm.bias'
        ATTENTION_WEIGHT = 'attention.weight'
        ATTENTION_BIAS = 'attention.bias'
        MLP_WEIGHT = 'mlp.weight'
        MLP_BIAS = 'mlp.bias'
        FFN_NORM_WEIGHT = 'ffn_norm.weight'
        FFN_NORM_BIAS = 'ffn_norm.bias'
