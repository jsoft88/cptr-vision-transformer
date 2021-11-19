from typing import Any
import torch.nn
from cptr_model.embeddings.position.base_position_embedding import BasePositionEmbedding
from cptr_model.factory.input_embeddings.embedding_factory import EmbeddingFactory
from cptr_model.factory.positional_embeddings.positional_embedding_factory import PositionalEmbeddingFactory
from cptr_model.models.base.base_encoder_decoder_block import BaseEncoderDecoderBlock
from cptr_model.models.cptr.attention.attention import Attention
from cptr_model.models.cptr.mlp.mlp import MLP


class CPTRDecoderBlock(torch.nn.Module, BaseEncoderDecoderBlock):
    KEY_POSITION_EMBEDDING_TYPE = 'position-embedding-type'
    KEY_NUM_HEADS = 'num-heads'
    KEY_LATENT_DIM = 'latent-dim'

    def __init__(self, **kwargs) -> None:
        self.position_embedding_type = kwargs(CPTRDecoderBlock.KEY_POSITION_EMBEDDING_TYPE, None)
        self.enc_key = kwargs.get(CPTRDecoderBlock.KEY_ENC_KEY, None)
        self.enc_value = kwargs.get(CPTRDecoderBlock.KEY_ENC_VALUE, None)

        super(CPTRDecoderBlock, self).__init__()

        self.position_embedding = PositionalEmbeddingFactory().get_instance(self.position_embedding_type, kwargs)

        if not isinstance(self.position_embedding, (BasePositionEmbedding, torch.nn.Parameter)):
            self.register_buffer('pos_encoding', self.position_embedding.get_position_embedding_layer())
        else:
            self.pos_encoding = self.position_embedding.get_position_embedding_layer()

        self.words_embedding = EmbeddingFactory().get_instance(EmbeddingFactory.WORD_EMBEDDING)

        self.masked_self_attention = Attention(*{
            Attention.KEY_MASKED_ATTENTION: True,
            Attention.KEY_NUM_HEADS: kwargs.get(CPTRDecoderBlock.KEY_NUM_HEADS, None),
            Attention.KEY_LATENT_DIM: kwargs.get(CPTRDecoderBlock.KEY_LATENT_DIM, None)
        })
        self.cross_attention = Attention()
        self.attention_normalization = torch.nn.LayerNorm(*{
            Attention.KEY_MASKED_ATTENTION: False,
            Attention.KEY_NUM_HEADS: kwargs.get(CPTRDecoderBlock.KEY_NUM_HEADS, None),
            Attention.KEY_LATENT_DIM: kwargs.get(CPTRDecoderBlock.KEY_LATENT_DIM, None)
        })
        self.ffn_normalization = torch.nn.LayerNorm()
        self.ffn = MLP()

    def __verify_required_args(self) -> None:
        if not self.position_embedding_type:
            raise ValueError(f'{CPTRDecoderBlock.KEY_POSITION_EMBEDDING_TYPE} has value None')

        if not self.enc_value:
            raise ValueError(f'{CPTRDecoderBlock.KEY_ENC_VALUE} has value None')

        if not self.enc_key:
            raise ValueError(f'{CPTRDecoderBlock.KEY_ENC_KEY} has value None')

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

