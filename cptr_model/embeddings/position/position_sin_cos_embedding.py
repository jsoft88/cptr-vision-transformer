from typing import List
import numpy as np
import torch.nn
from cptr_model.config.config import Config
from cptr_model.embeddings.position.base_position_embedding import BasePositionEmbedding


class PositionSinCosEmbedding(BasePositionEmbedding[torch.Tensor]):
    KEY_DIM = 'dim'
    KEY_NUM_POSITIONS = 'num-positions'

    def __init__(self, config: Config, **kwargs):
        self.config = config

        self.dim = kwargs.get(PositionSinCosEmbedding.KEY_DIM, None)
        self.num_positions = kwargs.get(PositionSinCosEmbedding.KEY_NUM_POSITIONS, None)

        super().__init__(config, **kwargs)

    def __verify_required_args(self) -> None:
        if not self.dim:
            raise ValueError(f'{PositionSinCosEmbedding.KEY_DIM} value is None')

        if not self.num_positions:
            raise ValueError(f'{PositionSinCosEmbedding.KEY_NUM_POSITIONS} value is None')

    def __get_position_angle_vec(self, position: int) -> List[float]:
        return [position / np.power(10000, 2 * (hid_j // 2) / self.dim) for hid_j in range(self.dim)]

    def get_position_embedding_layer(self) -> torch.Tensor:
        sinusoid_table = torch.tensor([self.__get_position_angle_vec(i) for i in range(self.num_positions)])
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        return sinusoid_table
