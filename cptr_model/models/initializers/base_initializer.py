from abc import ABC
from typing import Any, Optional, OrderedDict
from pytorch_lightning.core.lightning import LightningModule
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.factory.utils.fs_factory import FSFactory
from cptr_model.models.base.base_model import ModelBuilder
from cptr_model.utils.file_handlers.base_file_handler import BaseFileHandler


class BaseInitializer(ABC):
    KEY_MODEL = 'model'

    def __init__(self, config: Optional[Config],
                **kwargs) -> None:
        self.path = config.pretrained_model_path
        self.fs = FSFactory.get_instance(config.file_system_type, config, **kwargs)
        print(kwargs)
        self.model = kwargs.get(BaseInitializer.KEY_MODEL, None)
        
        if not self.model:
            raise ValueError(f'Model object is None in {self.__class__.__name__}')

    def map_state_dict_to_model(self) -> None:
        raise NotImplementedError(f'map_state_dict_to_model not implemented in {self.__class__.__name__}')
