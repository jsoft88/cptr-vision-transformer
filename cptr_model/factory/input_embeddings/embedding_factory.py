from typing import List, Union, Optional
import torch.nn
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.embeddings.input.patch_embedding import PatchEmbedding
from cptr_model.embeddings.input.word_embedding import WordEmbedding
from cptr_model.factory.base_factory import BaseFactory


class EmbeddingFactory(BaseFactory[Union[torch.nn.Module, torch.nn.Parameter]]):
    PATCH_EMBEDDING = 'patch'
    WORD_EMBEDDING = 'word'

    __ALL_TYPES = [
        PATCH_EMBEDDING,
        WORD_EMBEDDING
    ]

    @classmethod
    def get_instance(cls, type_str: str,
                     config: Optional[Config],
                     config_file_manager: Optional[ArchitectureConfigFileManager],
                     **kwargs) -> Union[torch.nn.Module, torch.nn.Parameter]:
        instance = {
            EmbeddingFactory.PATCH_EMBEDDING: PatchEmbedding(kwargs),
            EmbeddingFactory.WORD_EMBEDDING: WordEmbedding(kwargs)
        }.get(type_str, None)

        if instance:
            return instance

        raise ValueError(f'EmbeddingFactory :: Invalid instance type: {type_str}')

    @classmethod
    def all_types(cls) -> List[str]:
        return EmbeddingFactory.__ALL_TYPES
