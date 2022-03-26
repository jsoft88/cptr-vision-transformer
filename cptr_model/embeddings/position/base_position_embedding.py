from typing import Generic, TypeVar, OrderedDict, Any
from torch.nn import Module
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.core.core_module_extension import CoreModuleExtension


class BasePositionEmbedding(Module, CoreModuleExtension):
    def __init__(self, config: Config, **kwargs):
        self._verify_required_args()
        super().__init__()

    def _verify_required_args(self) -> None:
        raise NotImplementedError('missing implementation of _verify_required_args')

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        raise NotImplementedError('Missing implementation of weight_transfer_from_dict')

    def bias_transfer_from_dict(self, bias: OrderedDict[str, Any]) -> None:
        raise NotImplementedError('Missing implementation of bias_transfer_from_dict')

    def weight_transfer_to_dict(self) -> OrderedDict[str, Any]:
        raise NotImplementedError('Missing implementation of weight_transfer_to_dict')

    def bias_transfer_to_dict(self) -> OrderedDict[str, Any]:
        raise NotImplementedError('Missing implementation of bias_transfer_to_dict')
