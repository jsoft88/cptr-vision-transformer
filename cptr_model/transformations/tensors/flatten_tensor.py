from cptr_model.transformations.base_transformation import BaseTransformation
import torch
from typing import List


class FlattenTensor(BaseTransformation):
    KEY_LIST_OF_TENSORS = 'list-tensors'

    def __init__(self, **kwargs) -> None:
        self.list_of_tensors = kwargs.pop(FlattenTensor.KEY_LIST_OF_TENSORS, None)

    def _verify_required_parameters_for_transformation(self):
        if not self.list_of_tensors:
            raise ValueError('List of tensors is None')

    def __do_transformation(self) -> List[torch.Tensor]:
        return [torch.flatten(tensor) for tensor in self.list_of_tensors]
