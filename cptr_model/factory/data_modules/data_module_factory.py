from typing import List, Optional
from pytorch_lightning import LightningDataModule
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.data.data_modules.api_based.api_data import ApiData
from cptr_model.data.data_modules.batch.local_mscoco_data import LocalMSCocoData
from cptr_model.data.data_modules.batch.mscoco_data import MSCocoData
from cptr_model.factory.base_factory import BaseFactory


class DataModuleFactory(BaseFactory[LightningDataModule]):
    _BATCH_DATAMODULE = 'batch-dm'
    _API_DATAMODULE = 'api-dm'
    _LOCAL_BATCH_DATAMODULE = 'local-batch-dm'

    __ALL_TYPES = [
        _BATCH_DATAMODULE,
        _API_DATAMODULE,
        _LOCAL_BATCH_DATAMODULE
    ]

    KEY_FS = 'fs'

    @classmethod
    def get_instance(cls,
                    type_str: str,
                    config: Optional[Config],
                    **kwargs) -> LightningDataModule:

        if type_str == DataModuleFactory._API_DATAMODULE:
            return ApiData(config, kwargs.get(DataModuleFactory.KEY_FS, None), kwargs),
        if type_str == DataModuleFactory._BATCH_DATAMODULE:
            return MSCocoData(config, kwargs.get(DataModuleFactory.KEY_FS, None), kwargs)
        if type_str == DataModuleFactory._LOCAL_BATCH_DATAMODULE:
            return LocalMSCocoData(config, kwargs.get(DataModuleFactory.KEY_FS, None), kwargs)

        raise ValueError(f'Invalid data module instance type provided: {type_str}')

    @classmethod
    def all_types(cls) -> List[str]:
        return DataModuleFactory.__ALL_TYPES
