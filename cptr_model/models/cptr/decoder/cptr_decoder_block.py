from typing import Any, Optional, OrderedDict
import torch.nn
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.config.config import Config
from cptr_model.core.core_module_extension import CoreModuleExtension
from cptr_model.embeddings.position.base_position_embedding import BasePositionEmbedding
from cptr_model.factory.input_embeddings.embedding_factory import EmbeddingFactory
from cptr_model.factory.positional_embeddings.positional_embedding_factory import PositionalEmbeddingFactory
from cptr_model.models.base.base_encoder_decoder_block import BaseEncoderDecoderBlock
from cptr_model.models.cptr.attention.attention import Attention
from cptr_model.models.cptr.mlp.mlp import MLP


class CPTRDecoderBlock(torch.nn.Module, BaseEncoderDecoderBlock, CoreModuleExtension):

    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'

    def __init__(self, config: Optional[Config],
                 config_file_manager: Optional[ArchitectureConfigFileManager], **kwargs) -> None:
        if config is None or config_file_manager is None:
            raise ValueError('Expected instance of config object/config file manager object to be other than None')

        self.config = config
        self.config_file_manager = config_file_manager

        input_embedding = self.config_file_manager.get_sublayer_with_query(
            ArchitectureConfigFileManager.Model.DecoderBlock.NAME,
            ArchitectureConfigFileManager.Model.DecoderBlock.DecoderInputLayer.NAME,
            ArchitectureConfigFileManager.Model.DecoderBlock.DecoderInputLayer.SUBLAYER_WORD_EMBEDDING
        )

        input_position_embedding = self.config_file_manager.get_sublayer_with_query(
            ArchitectureConfigFileManager.Model.DecoderBlock.NAME,
            ArchitectureConfigFileManager.Model.DecoderBlock.DecoderInputLayer.NAME,
            ArchitectureConfigFileManager.Model.DecoderBlock.DecoderInputLayer.SUBLAYER_WORD_EMBEDDING
        )

        decoder_attention = self.config_file_manager.get_sublayer_with_query(
            ArchitectureConfigFileManager.Model.EncoderBlock.NAME,
            ArchitectureConfigFileManager.Model.EncoderBlock.EncoderAttentionLayer.NAME,
            ArchitectureConfigFileManager.Model.EncoderBlock.EncoderAttentionLayer.SUBLAYER_ATTENTION
        )

        self.position_embedding_type = ArchitectureConfigFileManager.get_params_for_sublayer(input_position_embedding)\
            .get(ArchitectureConfigFileManager.Model.DecoderPositionEmbeddingParams.TYPE, None)

        self.dim = ArchitectureConfigFileManager.get_params_for_sublayer(input_embedding).get(
            ArchitectureConfigFileManager.Model.WordEmbeddingParams.DIM, None)

        self.eps = self.config.decoder_normalization_eps

        self.mlp_dim = ArchitectureConfigFileManager.get_params_for_sublayer(decoder_attention)\
            .get(ArchitectureConfigFileManager.Model.AttentionParams.MLP_DIM, None)

        self.mlp_dropout = ArchitectureConfigFileManager.get_params_for_sublayer(decoder_attention)\
            .get(ArchitectureConfigFileManager.Model.AttentionParams.MLP_DROPOUT, None)

        super(CPTRDecoderBlock, self).__init__()

        self.position_embedding = PositionalEmbeddingFactory().get_instance(
            self.position_embedding_type,
            **ArchitectureConfigFileManager.get_params_for_sublayer(input_position_embedding)
        )

        if not isinstance(self.position_embedding, (BasePositionEmbedding, torch.nn.Parameter)):
            self.register_buffer('pos_encoding', self.position_embedding.get_position_embedding_layer())
        else:
            self.pos_encoding = self.position_embedding.get_position_embedding_layer()

        self.words_embedding = EmbeddingFactory().get_instance(EmbeddingFactory.WORD_EMBEDDING)

        attention_kwargs = {
            Attention.KEY_MASKED_ATTENTION: True,
            Attention.KEY_NUM_HEADS: ArchitectureConfigFileManager.get_params_for_sublayer(decoder_attention).get(
                ArchitectureConfigFileManager.Model.AttentionParams.HEADS, None),
            Attention.KEY_LATENT_DIM: self.dim
        }
        self.masked_self_attention = Attention(self.config, **attention_kwargs)
        self.cross_attention = Attention(self.config, **{
            Attention.KEY_MASKED_ATTENTION: False,
            Attention.KEY_NUM_HEADS: attention_kwargs[Attention.KEY_NUM_HEADS],
            Attention.KEY_LATENT_DIM: attention_kwargs[Attention.KEY_LATENT_DIM]
        })
        self.attention_normalization = torch.nn.LayerNorm(self.dim, eps=self.eps)
        self.ffn_normalization = torch.nn.LayerNorm(self.dim, eps=self.eps)

        mlp_kwargs = {
            MLP.KEY_LATENT_DIM: self.dim,
            MLP.KEY_MLP_DIM: self.mlp_dim,
            MLP.KEY_DROPOUT_RATE: self.mlp_dropout
        }
        self.ffn = MLP(**mlp_kwargs)

    def __verify_required_args(self) -> None:
        if not self.position_embedding_type:
            raise ValueError('position embedding type not found in config file: has value None')

        if not self.dim:
            raise ValueError('word embedding dim not found in config: has value None')

        if not self.eps:
            raise ValueError('decoder eps value is None')

        if not self.mlp_dim:
            raise ValueError('mlp dim not found in config: has value None')

        if not self.mlp_dropout:
            raise ValueError('mlp dropout rate not found in config: has value None')

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor) -> Any:
        x = self.words_embedding(x) + self.pos_encoding
        h = x
        x = self.masked_self_attention(x)
        x = self.attention_normalization(x)
        x = x + h
        h = x
        x = self.cross_attention(x, enc_output, enc_output)
        x = self.attention_normalization(x)
        x = x + h
        h = x
        x = self.ffn(x)
        x = self.ffn_normalization(x)
        x = x + h

        return x

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        model_dict = self.state_dict()
        model_dict[CPTRDecoderBlock.StateKey.ATTENTION_NORMALIZATION_WEIGHT] = weights.get(CPTRDecoderBlock.StateKey.ATTENTION_NORMALIZATION_WEIGHT, None)
        self.load_state_dict(model_dict)

    def bias_transfer_from_dict(self, bias: OrderedDict[str, Any]) -> None:
        model_dict = self.state_dict()
        model_dict[CPTRDecoderBlock.StateKey.ATTENTION_NORMALIZATION_BIAS] = bias[CPTRDecoderBlock.StateKey.ATTENTION_NORMALIZATION_BIAS]
        self.attention_normalization.bias = bias.get(CPTRDecoderBlock.StateKey.ATTENTION_NORMALIZATION_BIAS, None)

    def weight_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            CPTRDecoderBlock.StateKey.ATTENTION_NORMALIZATION_WEIGHT: self.attention_normalization.weight
        })

    def bias_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            CPTRDecoderBlock.StateKey.ATTENTION_NORMALIZATION_BIAS: self.attention_normalization.bias
        })

    class StateKey:
        ATTENTION_NORMALIZATION_WEIGHT = 'attention_normalization.weight'
        ATTENTION_NORMALIZATION_BIAS = 'attention_normalization.bias'