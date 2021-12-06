from typing import List, Optional
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.config.config import Config
from cptr_model.factory.base_factory import BaseFactory, T
from cptr_model.utils.file_handlers.base_file_handler import BaseFileHandler
from cptr_model.utils.file_handlers.local_fs_handler import LocalFSHandler


class FSFactory(BaseFactory[BaseFileHandler]):
    _LOCAL_FS = 'local-fs'

    _ALL_TYPES = [
        _LOCAL_FS
    ]

    @classmethod
    def get_instance(cls, type_str: str, config: Optional[Config],
                     config_file_manager: Optional[ArchitectureConfigFileManager], **kwargs) -> T:
        instance = {
            FSFactory._LOCAL_FS: LocalFSHandler(kwargs)
        }.get(type_str, None)

        if instance:
            return instance

        raise ValueError(f'Invalid file system provided: {type_str}')

    @classmethod
    def all_types(cls) -> List[str]:
        return FSFactory._ALL_TYPES
