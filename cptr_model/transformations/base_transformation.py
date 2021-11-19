import abc
from typing import List

import torch


class BaseTransformation(abc.ABC):
    def __int__(self, **kwargs) -> None:
        pass

    def _verify_required_parameters_for_transformation(self):
        raise NotImplementedError('Missing implementation for method')

    def apply(self) -> List[torch.Tensor]:
        self._verify_required_parameters_for_transformation()
        return self.__do_transformation()

    def __do_transformation(self) -> List[torch.Tensor]:
        raise NotImplementedError('Missing implementation of transformation step')
