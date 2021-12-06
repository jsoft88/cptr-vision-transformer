import abc
from typing import Generic, TypeVar, OrderedDict, Any

from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.core.core_module_extension import CoreModuleExtension

T = TypeVar('T')


class BasePositionEmbedding(abc.ABC, Generic[T], CoreModuleExtension):
    def __init__(self, config: Config, **kwargs):
        self.__verify_required_args()

    def __verify_required_args(self) -> None:
        raise NotImplementedError('missing implementation of __verify_required_args')

    def get_position_embedding_layer(self) -> T:
        raise NotImplementedError('Missing implementation of get_position_embedding_layer')

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        raise NotImplementedError('Missing implementation of weight_transfer')
