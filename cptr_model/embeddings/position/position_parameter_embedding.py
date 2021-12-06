from typing import OrderedDict, Any

import torch.nn
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.config.config import Config
from cptr_model.embeddings.position.base_position_embedding import BasePositionEmbedding


class PositionParameterEmbedding(BasePositionEmbedding[torch.nn.Parameter]):
    KEY_DIMS = 'dims'

    def __init__(self, config: Config, **kwargs):
        self.config = config
        self.dims = kwargs.get(PositionParameterEmbedding.KEY_DIMS, None)
        self.param_position_embedding: torch.nn.Parameter = torch.nn.Parameter(torch.zeros(self.dims))

        super().__init__(kwargs)

    def __verify_required_args(self) -> None:
        if not self.dims:
            raise ValueError(f'PositionEmbedding:: {PositionParameterEmbedding.KEY_DIMS} is None')

    def get_position_embedding_layer(self) -> torch.nn.Parameter:
        return self.param_position_embedding

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        self.param_position_embedding = weights['param_position_embedding']

    def bias_transfer_from_dict(self, bias: OrderedDict[str, Any]) -> None:
        return

    def weight_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            'param_position_embedding': self.param_position_embedding
        })

    def bias_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return None
