import torch.nn
from cptr_model.config.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.config.config import Config
from cptr_model.embeddings.position.base_position_embedding import BasePositionEmbedding


class PositionParameterEmbedding(BasePositionEmbedding[torch.nn.Parameter]):
    _KEY_DIMS = 'dims'

    def __init__(self, config: Config, config_manager: ArchitectureConfigFileManager, **kwargs):
        self.config = config
        self.config_file_manager = config_manager

        self.config_section = self.config_file_manager.get_embeddings_config_for(
            ArchitectureConfigFileManager.ArchitectureParts.SECTION_EMBEDDINGS_POSITION)

        super().__init__(kwargs)

        params = self.config_file_manager.get_embeddings_params(
            self.config.encoder_position_embedding_config_section, self.config_section
        )

        self.dims = self.config_file_manager.get_param_value_for(PositionParameterEmbedding._KEY_DIMS, params)

    def __verify_required_args(self) -> None:
        if not self.config_section:
            raise ValueError('PositionEmbedding:: dims is None')

    def get_position_embedding_layer(self) -> torch.nn.Parameter:
        return torch.nn.Parameter(torch.zeros(self.dims))
