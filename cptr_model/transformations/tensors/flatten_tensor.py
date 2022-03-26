from cptr_model.transformations.base_transformation import BaseTransformation
import torch
from typing import Any, List


class FlattenTensor(BaseTransformation):

    def __init__(self, input: Any, **kwargs) -> None:
        self.list_of_tensors = input

    def _verify_required_parameters_for_transformation(self):
        if not self.list_of_tensors:
            raise ValueError('List of tensors is None')

    def _do_transformation(self) -> List[torch.Tensor]:
        return [torch.flatten(tensor) for tensor in self.list_of_tensors]
