from typing import Any
import torch
from cptr_model.embeddings.position.base_position_embedding import BasePositionEmbedding
from cptr_model.factory.positional_embeddings.positional_embedding_factory import PositionalEmbeddingFactory
from cptr_model.models.base.base_encoder_decoder_block import BaseEncoderDecoderBlock
from cptr_model.models.cptr.attention.attention import Attention
from cptr_model.models.cptr.mlp.mlp import MLP


class CPTREncoderBlock(torch.nn.Module, BaseEncoderDecoderBlock):
    KEY_PATCH_LATENT_DIM = 'patch-latent-dim'
    KEY_EPS = 'eps'
    KEY_POSITION_EMBEDDING_TYPE = 'position-embedding-type'

    def __init__(self, **kwargs) -> None:
        self.latent_dim = kwargs.get(CPTREncoderBlock.KEY_PATCH_LATENT_DIM, None)
        self.eps = kwargs.get(CPTREncoderBlock.KEY_EPS, None)
        self.position_embedding_type = kwargs.get(CPTREncoderBlock.KEY_POSITION_EMBEDDING_TYPE, None)
        self.__verify_required_args()
        self.eps = float(self.eps)
        self.attention_norm = torch.nn.LayerNorm(self.latent_dim, eps=self.eps)
        self.ffn_norm = torch.nn.LayerNorm(self.latent_dim, eps=self.eps)
        self.mlp = MLP(kwargs)
        self.attention = Attention(kwargs)
        self.position_embedding = PositionalEmbeddingFactory.get_instance(self.position_embedding_type, kwargs)
        if not isinstance(self.position_embedding, (BasePositionEmbedding, torch.nn.Parameter)):
            self.register_buffer('pos_encoding', self.position_embedding.get_position_embedding_layer())
        else:
            self.pos_encoding = self.position_embedding.get_position_embedding_layer()

        super().__init__()

    def __verify_required_args(self) -> None:
        if not self.latent_dim:
            raise ValueError(f'{CPTREncoderBlock.KEY_PATCH_LATENT_DIM} value is None')
        if not self.eps:
            raise ValueError(f'{CPTREncoderBlock.KEY_EPS} value is None')
        if not self.position_embedding_type:
            raise ValueError(f'{CPTREncoderBlock.KEY_POSITION_EMBEDDING_TYPE} value is None')

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
