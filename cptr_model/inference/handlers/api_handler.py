from spacy import Config
from cptr_model.inference.handlers.base_handler import BaseHandler
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager


class ApiHandler(BaseHandler):
    def __init__(self, config: Config, config_file_manager: ArchitectureConfigFileManager) -> None:
        super().__init__()