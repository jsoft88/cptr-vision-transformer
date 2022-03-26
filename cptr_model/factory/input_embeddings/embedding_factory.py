from typing import List, Union, Optional
import torch.nn
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.embeddings.input.bert_embedding import BertEmbedding
from cptr_model.embeddings.input.patch_embedding import PatchEmbedding
from cptr_model.embeddings.input.word_embedding import WordEmbedding
from cptr_model.factory.base_factory import BaseFactory


class EmbeddingFactory(BaseFactory[Union[torch.nn.Module, torch.nn.Parameter]]):
    PATCH_EMBEDDING_CONV = 'patch-conv'
    WORD_EMBEDDING = 'word'
    BERT_EMBEDDING = 'bert'

    __ALL_TYPES = [
        PATCH_EMBEDDING_CONV,
        WORD_EMBEDDING,
        BERT_EMBEDDING
    ]

    @classmethod
    def get_instance(cls, type_str: str,
                     config: Optional[Config],
                     **kwargs) -> Union[torch.nn.Module, torch.nn.Parameter]:
        if type_str == EmbeddingFactory.PATCH_EMBEDDING_CONV:
            return PatchEmbedding(config, **kwargs)
        if type_str == EmbeddingFactory.WORD_EMBEDDING:
            return WordEmbedding(config, **kwargs)
        if type_str == EmbeddingFactory.BERT_EMBEDDING:
            return BertEmbedding(config, **kwargs)

        raise ValueError(f'EmbeddingFactory :: Invalid instance type: {type_str}')

    @classmethod
    def all_types(cls) -> List[str]:
        return EmbeddingFactory.__ALL_TYPES
