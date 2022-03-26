from abc import abstractmethod, ABC
from typing import Optional, Union, Dict, List, Any, Tuple
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.config.config import Config


class BaseHandler(ABC):
    def __init__(self, config: Config, config_file_manager: ArchitectureConfigFileManager) -> None:
        pass

    def _prepare_prediction_environment(self) -> None:
        return

    def _teardown_prediction_environment(self) -> None:
        return

    def exec(self,
        obj: Union[Dict[str, List[Any]], Dict[str, Any], List[Tuple[str, Any]], List[Tuple[str, List[Any]]]]) -> BaseHandler.HandlerStatus:
        self._prepare_prediction_environment()
        status = self._push_prediction(obj)
        self._teardown_prediction_environment()
        return status

    @abstractmethod
    def _push_prediction(self,
        obj: Union[Dict[str, List[Any]], Dict[str, Any], List[Tuple[str, Any]], List[Tuple[str, List[Any]]]]) -> BaseHandler.HandlerStatus:
        raise NotImplementedError('push_prediction method needs to be implemented')

    class HandlerStatus:
        def __init__(self, success: bool, description: Optional[str], **kwargs) -> None:
            self.success = success
            self.description = description
            self.bag_of_args = kwargs