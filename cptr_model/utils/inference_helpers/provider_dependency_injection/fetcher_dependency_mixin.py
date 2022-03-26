from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.utils.inference_helpers.inference_helper_mixin import InferenceHelperMixin
from cptr_model.utils.inference_helpers.request_handlers.http_manager import HttpManager


class FetcherDependencyMixin:
    def get_data_fetcher(self, config: Config) -> InferenceHelperMixin:
        # TODO: data fetcher to instantiate via factory to be able to request other fetchers, such as kafka
        return HttpManager(config.api_based_options_host, config.api_based_options_port, None, config.api_based_options_protocol)