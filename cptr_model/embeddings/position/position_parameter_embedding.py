from typing import List, OrderedDict, Any
import torch
from torch.nn.parameter import Parameter
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.config.config import Config
from cptr_model.embeddings.position.base_position_embedding import BasePositionEmbedding


class PositionParameterEmbedding(BasePositionEmbedding):
    KEY_DIMS = 'dims'

    def __init__(self, config: Config, **kwargs):
        self.config = config
        self.model_config = config.cptr_specifics
        self.dims: List[int] = kwargs.get(PositionParameterEmbedding.KEY_DIMS, None)
        super().__init__(config, **kwargs)
        self.param_position_embedding = Parameter(torch.zeros(*self.dims))

    def _verify_required_args(self) -> None:
        if not self.dims:
            raise ValueError(f'PositionEmbedding:: {PositionParameterEmbedding.KEY_DIMS} is None')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.param_position_embedding

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
