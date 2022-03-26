from typing import List, Optional
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.config.config import Config
from cptr_model.factory.base_factory import BaseFactory
from cptr_model.utils.file_handlers.base_file_handler import BaseFileHandler
from cptr_model.utils.file_handlers.local_fs_handler import LocalFSHandler
from cptr_model.utils.file_handlers.s3_fs_handler import S3FSHandler


class FSFactory(BaseFactory[BaseFileHandler]):
    _LOCAL_FS = 'local-fs'
    _S3_FS = 's3-fs'

    _ALL_TYPES = [
        _LOCAL_FS,
        _S3_FS
    ]

    @classmethod
    def get_instance(cls, type_str: str, config: Optional[Config], **kwargs) -> BaseFileHandler:
        if type_str == FSFactory._LOCAL_FS:
            return LocalFSHandler(**kwargs)
        if type_str == FSFactory._S3_FS:
            return S3FSHandler(**kwargs)

        raise ValueError(f'Invalid file system provided: {type_str}')

    @classmethod
    def all_types(cls) -> List[str]:
        return FSFactory._ALL_TYPES
