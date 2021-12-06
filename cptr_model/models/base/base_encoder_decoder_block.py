from typing import Optional, OrderedDict, Any

from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.config.config import Config
from cptr_model.core.core_module_extension import CoreModuleExtension


class BaseEncoderDecoderBlock(CoreModuleExtension):
    def __init__(self,
                 config: Optional[Config],
                 config_file_manager: Optional[ArchitectureConfigFileManager], **kwargs) -> None:
        self.__verify_required_args()

    def __verify_required_args(self) -> None:
        raise NotImplementedError('__verify_required_args not implemented')

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        raise NotImplementedError(f'weight_transfer method not implemented in class {self.__class__.__name__}')