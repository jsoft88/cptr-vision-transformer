from typing import List, Optional
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.factory.base_factory import BaseFactory
from cptr_model.models.initializers.base_initializer import BaseInitializer
from cptr_model.models.initializers.cptr.init_base_vit_16_384_v2 import BaseVit16384V2


class InitializersFactory(BaseFactory[BaseInitializer]):
    _VIT_BASE_16_384 = 'vit_base_16_384_bert'

    _ALL_TYPES = [
        _VIT_BASE_16_384
    ]

    @classmethod
    def get_instance(cls,
                     type_str: str,
                     config: Optional[Config],
                     **kwargs) -> BaseInitializer:
        instance = {
            InitializersFactory._VIT_BASE_16_384: BaseVit16384V2(config, kwargs)
        }

        if instance:
            return instance

        raise ValueError(f'Invalid initializer type provided: {type_str}')

    @classmethod
    def all_types(cls) -> List[str]:
        return InitializersFactory._ALL_TYPES
