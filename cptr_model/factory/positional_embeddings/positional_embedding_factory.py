from typing import List
from cptr_model.embeddings.position.base_position_embedding import BasePositionEmbedding
from cptr_model.embeddings.position.position_parameter_embedding import PositionParameterEmbedding
from cptr_model.embeddings.position.position_sin_cos_embedding import PositionSinCosEmbedding
from cptr_model.factory.base_factory import BaseFactory


class PositionalEmbeddingFactory(BaseFactory[BasePositionEmbedding]):
    _PARAMETER_EMBEDDING = 'parameter-embedding'
    _SINUSOID_EMBEDDING = 'sinusoid-embedding'

    _ALL_TYPES = [
        _PARAMETER_EMBEDDING,
        _SINUSOID_EMBEDDING
    ]

    @classmethod
    def get_instance(cls, type_str: str, **kwargs) -> BasePositionEmbedding:
        instance = {
            PositionalEmbeddingFactory._PARAMETER_EMBEDDING: PositionParameterEmbedding(kwargs),
            PositionalEmbeddingFactory._SINUSOID_EMBEDDING: PositionSinCosEmbedding(kwargs)
        }.get(type_str, None)

        if not instance:
            raise ValueError('Invalid value for type_str')

        return instance

    @classmethod
    def all_types(cls) -> List[str]:
        return ','.join(PositionalEmbeddingFactory._ALL_TYPES)
