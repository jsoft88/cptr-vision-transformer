from typing import Any
import torch.nn
from torch.nn import Conv2d
from cptr_model.config.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.config.config import Config


class PatchEmbedding(torch.nn.Module):
    _KEY_IN_CHANNELS = 'in-channels'
    _KEY_OUT_CHANNELS = 'out-channels'
    _KEY_KERNEL_SIZE = 'kernel-size'
    _KEY_STRIDE = 'stride'

    def __init__(self, config: Config, config_file_manager: ArchitectureConfigFileManager, **kwargs) -> None:
        super().__init__()

        self.config = config
        self.config_manager = config_file_manager
        self.config_section = self.config_manager\
            .get_embeddings_config_for(ArchitectureConfigFileManager.ArchitectureParts.SECTION_EMBEDDINGS_INPUT)

        self.__verify_required_args()

        params = self.config_manager\
            .get_embeddings_params(self.config.encoder_input_embedding_config_section, self.config_section)

        self.in_channels = self.config_manager.get_param_value_for(PatchEmbedding._KEY_IN_CHANNELS, params)
        self.out_channels = self.config_manager.get_param_value_for(PatchEmbedding._KEY_OUT_CHANNELS, params)
        self.kernel_size = self.config_manager.get_param_value_for(PatchEmbedding._KEY_KERNEL_SIZE, params)
        self.stride = self.config_manager.get_param_value_for(PatchEmbedding._KEY_STRIDE, params)

        self.embedding_layer = Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride
        )

    def __verify_required_args(self) -> None:
        if not self.config:
            raise ValueError('config value is None')

        if not self.config_manager:
            raise ValueError('config file manager value is None')

        if not self.config_section:
            raise ValueError(
                f'Could not find section for {ArchitectureConfigFileManager.ArchitectureParts.SECTION_EMBEDDINGS_INPUT}'
            )

    def forward(self, x: torch.Tensor) -> Any:
        x = self.embedding_layer(x)
        return x
