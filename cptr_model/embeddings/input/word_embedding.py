from typing import OrderedDict, Any

import torch.nn
from cptr_model.config.config import Config
from cptr_model.core.core_module_extension import CoreModuleExtension


class WordEmbedding(torch.nn.Module, CoreModuleExtension):
    KEY_DIM = 'dim'
    KEY_VOCAB_SIZE = 'vocab_size'
    KEY_PAD_IDX = 'pad_index'

    def __init__(self, config: Config, **kwargs) -> None:
        self.config = config

        self.dim = kwargs.get(WordEmbedding.KEY_DIM, None)
        self.vocab_size = kwargs.get(WordEmbedding.KEY_VOCAB_SIZE, None)
        self.pad_idx = kwargs.get(WordEmbedding.KEY_PAD_IDX, None)

        super(WordEmbedding, self).__init__()
        # self.__verify_required_args()
        # self.embedding = torch.nn.Embedding(self.vocab_size, self.dim, padding_idx=self.pad_idx).to(self.config.device)
        

    def __verify_required_args(self) -> None:
        if not self.config:
            raise ValueError('config value is None')

        if not self.dim:
            raise ValueError('dim value is None')

        if not self.pad_idx:
            raise ValueError('pad_idx value is None')

        if not self.vocab_size:
            raise ValueError('vocab_size value is None')

    def forward(self, word) -> torch.Tensor:
        return self.embedding(word)

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        self.embedding.weight = weights[WordEmbedding.StateKey.EMBEDDING_WEIGHT]

    def bias_transfer_from_dict(self, bias: OrderedDict[str, Any]) -> None:
        return

    def weight_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            WordEmbedding.StateKey.EMBEDDING_WEIGHT: self.embedding.weight
        })

    def bias_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return None

    class StateKey:
        EMBEDDING_WEIGHT = 'embedding.weight'
