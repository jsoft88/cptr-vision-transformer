import abc
from typing import Generic, TypeVar

from cptr_model.config.config import Config
from cptr_model.config.architecture_config_file_manager import ArchitectureConfigFileManager

T = TypeVar('T')


class BasePositionEmbedding(abc.ABC, Generic[T]):
    def __init__(self, config: Config, config_manager: ArchitectureConfigFileManager, **kwargs):
        self.__verify_required_args()

    def __verify_required_args(self) -> None:
        raise NotImplementedError('missing implementation of __verify_required_args')

    def get_position_embedding_layer(self) -> T:
        raise NotImplementedError('Missing implementation of get_position_embedding_layer')
