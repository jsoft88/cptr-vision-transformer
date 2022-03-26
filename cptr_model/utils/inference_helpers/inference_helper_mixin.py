from typing import Any, List, Tuple, TypeVar, Dict
from PIL.Image import Image


METADATA = TypeVar('METADATA', int, str, Any)
class InferenceHelperMixin:
    def get_input(self, batch_size: int) -> List[Tuple[METADATA, Image]]:
        raise NotImplementedError('get_input method not implemented')

    def post_prediction(self, predictions: List[Dict[Any, Any]]) -> Any:
        raise NotImplementedError('post_prediction method not implemented')