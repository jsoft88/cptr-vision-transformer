import abc
from typing import List, TypeVar, Generic, Optional

from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager

T = TypeVar('T')


class BaseFactory(abc.ABC, Generic[T]):
    @classmethod
    def get_instance(cls,
                     type_str: str,
                     config: Optional[Config],
                     config_file_manager: Optional[ArchitectureConfigFileManager],
                     **kwargs) -> T:
        raise NotImplementedError('No implementation for get_instance found')

    @classmethod
    def all_types(cls) -> List[str]:
        raise NotImplementedError('No implementation for all_types found')
