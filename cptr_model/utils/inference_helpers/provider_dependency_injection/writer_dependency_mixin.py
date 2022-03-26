from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.utils.inference_helpers.inference_helper_mixin import InferenceHelperMixin
from cptr_model.utils.inference_helpers.request_handlers.http_manager import HttpManager


class WriterDependencyMixin:
    def write_output(self, config: Config, config_file_manager: ArchitectureConfigFileManager) -> InferenceHelperMixin:
        # TODO: Instantiate the provider via factory to be able to use other ones, such as kafka.
        return HttpManager(config.api_based_options_host, config.api_based_options_port, None, config.api_based_options_protocol)