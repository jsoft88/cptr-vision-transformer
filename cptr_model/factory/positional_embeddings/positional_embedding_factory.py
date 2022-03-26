from typing import List, Optional
from cptr_model.config.config import Config
from cptr_model.embeddings.position.base_position_embedding import BasePositionEmbedding
from cptr_model.embeddings.position.position_parameter_embedding import PositionParameterEmbedding
from cptr_model.embeddings.position.position_sin_cos_embedding import PositionSinCosEmbedding
from cptr_model.factory.base_factory import BaseFactory


class PositionalEmbeddingFactory(BaseFactory[BasePositionEmbedding]):
    _PARAMETER_EMBEDDING = 'parameter'
    _SINUSOID_EMBEDDING = 'sinusoid'

    _ALL_TYPES = [
        _PARAMETER_EMBEDDING,
        _SINUSOID_EMBEDDING
    ]

    @classmethod
    def get_instance(cls, type_str: str, config: Optional[Config], **kwargs) -> BasePositionEmbedding:
        if  type_str == PositionalEmbeddingFactory._PARAMETER_EMBEDDING:
            return PositionParameterEmbedding(config, **kwargs)
        if type_str == PositionalEmbeddingFactory._SINUSOID_EMBEDDING:
            return PositionSinCosEmbedding(config, **kwargs)
        
        raise ValueError('Invalid value for type_str')

    @classmethod
    def all_types(cls) -> List[str]:
        return ','.join(PositionalEmbeddingFactory._ALL_TYPES)
