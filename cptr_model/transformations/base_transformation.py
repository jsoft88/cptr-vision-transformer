from typing import Any, List, Union
import torch


class BaseTransformation(object):
    def __int__(self, **kwargs) -> None:
        pass

    def _verify_required_parameters_for_transformation(self):
        raise NotImplementedError('Missing implementation for method')

    def __call__(self, sample: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        self._verify_required_parameters_for_transformation()
        return self.__do_transformation(sample)

    def _do_transformation(self, sample: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        raise NotImplementedError('Missing implementation of transformation step')
