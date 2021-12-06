from typing import Optional, OrderedDict, Any
import torch
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.config.config import Config
from cptr_model.core.core_module_extension import CoreModuleExtension
from cptr_model.embeddings.position.base_position_embedding import BasePositionEmbedding
from cptr_model.factory.positional_embeddings.positional_embedding_factory import PositionalEmbeddingFactory
from cptr_model.models.base.base_encoder_decoder_block import BaseEncoderDecoderBlock
from cptr_model.models.cptr.attention.attention import Attention
from cptr_model.models.cptr.mlp.mlp import MLP


class CPTREncoderBlock(torch.nn.Module, BaseEncoderDecoderBlock, CoreModuleExtension):

    def __init__(self,
                 config: Optional[Config],
                 config_file_manager: Optional[ArchitectureConfigFileManager], **kwargs) -> None:
        if config is None or config_file_manager is None:
            raise ValueError('Expected instance of config object/config file manager object to be other than None')

        self.config = config
        self.config_file_manager = config_file_manager

        input_embedding = self.config_file_manager.get_sublayer_with_query(
            ArchitectureConfigFileManager.Model.EncoderBlock.NAME,
            ArchitectureConfigFileManager.Model.EncoderBlock.EncoderInputLayer.NAME,
            ArchitectureConfigFileManager.Model.EncoderBlock.EncoderInputLayer.SUBLAYER_PATCH_EMBEDDING
        )

        input_position_embedding = self.config_file_manager.get_sublayer_with_query(
            ArchitectureConfigFileManager.Model.EncoderBlock.NAME,
            ArchitectureConfigFileManager.Model.EncoderBlock.EncoderInputLayer.NAME,
            ArchitectureConfigFileManager.Model.EncoderBlock.EncoderInputLayer.SUBLAYER_POSITION_EMBEDDING
        )

        encoder_attention = self.config_file_manager.get_sublayer_with_query(
            ArchitectureConfigFileManager.Model.EncoderBlock.NAME,
            ArchitectureConfigFileManager.Model.EncoderBlock.EncoderAttentionLayer.NAME,
            ArchitectureConfigFileManager.Model.EncoderBlock.EncoderAttentionLayer.SUBLAYER_ATTENTION
        )

        self.latent_dim = ArchitectureConfigFileManager\
            .get_params_for_sublayer(input_embedding)\
            .get(ArchitectureConfigFileManager.Model.PatchEmbeddingParams.DIM, None)
        self.eps = self.config.encoder_normalization_eps
        self.position_embedding_type = ArchitectureConfigFileManager\
            .get_params_for_sublayer(input_position_embedding)\
            .get(ArchitectureConfigFileManager.Model.EncoderPositionEmbeddingParams.TYPE, None)

        self.mlp_dim = ArchitectureConfigFileManager.get_params_for_sublayer(encoder_attention) \
            .get(ArchitectureConfigFileManager.Model.AttentionParams.MLP_DIM, None)

        self.mlp_dropout = ArchitectureConfigFileManager.get_params_for_sublayer(encoder_attention) \
            .get(ArchitectureConfigFileManager.Model.AttentionParams.MLP_DROPOUT, None)

        super().__init__()

        self.eps = float(self.eps)
        self.attention_norm = torch.nn.LayerNorm(self.latent_dim, eps=self.eps).to(device=self.config.device)
        self.ffn_norm = torch.nn.LayerNorm(self.latent_dim, eps=self.eps).to(device=self.config.device)
        mlp_kwargs = {
            MLP.KEY_LATENT_DIM: self.latent_dim,
            MLP.KEY_MLP_DIM: self.mlp_dim,
            MLP.KEY_DROPOUT_RATE: self.mlp_dropout
        }
        self.mlp = MLP(*mlp_kwargs)
        attention_kwargs = {
            Attention.KEY_MASKED_ATTENTION: False,
            Attention.KEY_LATENT_DIM: self.latent_dim,
            Attention.KEY_NUM_HEADS: self.config_file_manager.get_params_for_sublayer(encoder_attention).get(
                ArchitectureConfigFileManager.Model.AttentionParams.HEADS, None)
        }
        self.attention = Attention(self.config, **attention_kwargs)
        # to keep the factory pattern, inject params into the position embedding instance instead of parsing it here
        # and re-packing into kwargs with the keys expected by the concrete instance. When designing the config,
        # one must make sure that the sublayer params keys match those specified in the layer class implementation
        self.position_embedding = PositionalEmbeddingFactory.get_instance(
            self.position_embedding_type,
            **self.config_file_manager.get_params_for_sublayer(input_position_embedding)
        )
        if not isinstance(self.position_embedding, (BasePositionEmbedding, torch.nn.Parameter)):
            self.register_buffer('pos_encoding', self.position_embedding.get_position_embedding_layer())
        else:
            self.pos_encoding = self.position_embedding.get_position_embedding_layer()

    def __verify_required_args(self) -> None:
        if not self.latent_dim:
            raise ValueError('Missing dim param in config file for input embeddings')
        if not self.position_embedding_type:
            raise ValueError('Missing position embedding type param in config file for position embeddings')
        if not self.eps:
            raise ValueError('eps value is None')
        if not self.mlp_dim:
            raise ValueError('Missing mlp dim in config file: has value None')
        if not self.mlp_dropout:
            raise ValueError('Missing mlp dropout in config file: has value None')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_encoding
        h = x
        x = self.attention_norm(x)
        x, weights = self.attention(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.mlp(x)
        x = x + h

        return x

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        model_dict = self.state_dict()
        model_dict[CPTREncoderBlock.StateKey.ATTENTION_NORM_WEIGHT] = weights[CPTREncoderBlock.StateKey.ATTENTION_NORM_WEIGHT]
        self.load_state_dict(model_dict)

    def bias_transfer_from_dict(self, bias: OrderedDict[str, Any]) -> None:
        model_dict = self.state_dict()
        model_dict[CPTREncoderBlock.StateKey.ATTENTION_NORM_BIAS] = bias[CPTREncoderBlock.StateKey.ATTENTION_NORM_BIAS]
        self.load_state_dict(model_dict)

    def weight_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            CPTREncoderBlock.StateKey.ATTENTION_NORM_WEIGHT: self.attention_norm.weight
        })

    def bias_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            CPTREncoderBlock.StateKey.ATTENTION_NORM_BIAS: self.attention_norm.bias
        })

    class StateKey:
        ATTENTION_NORM_WEIGHT = 'attention_norm.weight'
        ATTENTION_NORM_BIAS = 'attention_norm.bias'