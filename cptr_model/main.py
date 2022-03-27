from typing import List
from requests.api import post
from pytorch_lightning import Trainer
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.factory.data_modules.data_module_factory import DataModuleFactory
from cptr_model.factory.utils.fs_factory import FSFactory
from cptr_model.models.cptr.cptr import CPTRModelBuilder


class CPTRMain:
    def __init__(self, args: List[str]) -> None:
        self.config = Config(args)
        #self.config_file_manager = ArchitectureConfigFileManager(self.config.config_file)
        self.cptr_model = CPTRModelBuilder(self.config, self.config_file_manager)

    def execute(self) -> None:
        trainer = Trainer()
        dm_args = dict({DataModuleFactory.KEY_FS: FSFactory.get_instance(self.config.file_system_type, self.config, **self.config.file_system_options)})
        data_module = DataModuleFactory.get_instance(self.config.input_reader_type, self.config, **dm_args)
        
        if self.config.training:
            trainer.fit(self.cptr_model, datamodule=data_module)
        else:
            trainer.predict(self.cptr_model, return_predictions=True, datamodule=data_module)

if __name__ == '__main__':
    CPTRMain().execute()
