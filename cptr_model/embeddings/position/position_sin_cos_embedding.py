from typing import List
import numpy as np
import torch.nn

from cptr_model.config.config import Config
from cptr_model.config.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.embeddings.position.base_position_embedding import BasePositionEmbedding


class PositionSinCosEmbedding(BasePositionEmbedding[torch.Tensor]):
    _KEY_DIM = 'dim'
    _KEY_NUM_POSITIONS = 'num-positions'

    def __init__(self, config: Config, config_manager: ArchitectureConfigFileManager, **kwargs):
        self.config_file_manager: ArchitectureConfigFileManager = config_manager
        self.config = config
        self.config_section = self.config_file_manager\
            .get_embeddings_config_for(ArchitectureConfigFileManager.ArchitectureParts.SECTION_EMBEDDINGS_POSITION)

        super().__init__(config, config_manager, **kwargs)

        params = self.config_file_manager.get_embeddings_params(
            self.config.encoder_position_embedding_config_section, self.config_section
        )
        self.dim = self.config_file_manager.get_param_value_for(PositionSinCosEmbedding._KEY_DIM, params)
        self.num_positions = self.config_file_manager.get_param_value_for(
            PositionSinCosEmbedding._KEY_NUM_POSITIONS, params
        )

    def __verify_required_args(self) -> None:
        if not self.config_file_manager:
            raise ValueError(f'{PositionSinCosEmbedding.KEY_CONFIG_MANAGER} value is None')

        if not self.config_section:
            raise ValueError(
                f'Config section for key {ArchitectureConfigFileManager.ArchitectureParts.SECTION_EMBEDDINGS_POSITION} '
                f'is None')

    def __get_position_angle_vec(self, position: int) -> List[float]:
        return [position / np.power(10000, 2 * (hid_j // 2) / self.dim) for hid_j in range(self.dim)]

    def get_position_embedding_layer(self) -> torch.Tensor:
        sinusoid_table = torch.tensor([self.__get_position_angle_vec(i) for i in range(self.num_positions)])
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        return sinusoid_table
