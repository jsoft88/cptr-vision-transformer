from torchvision.transforms.functional import *
from cptr_model.transformations.base_transformation import BaseTransformation


class ImageTransformation(BaseTransformation):
    KEY_IS_TENSOR = 'is-tensor'
    KEY_PATCH_SIZE = 'patch-size'

    def __init__(self, **kwargs) -> None:
        self.is_tensor = kwargs.pop(ImageTransformation.KEY_IS_TENSOR, None)
        self.patch_size = kwargs.pop(ImageTransformation.KEY_PATCH_SIZE, None)

    def _verify_required_parameters_for_transformation(self):
        if not self.input:
            raise ValueError('Input is None')
        if not self.is_tensor:
            raise ValueError('is_tensor is None')
        if not self.patch_size:
            raise ValueError('patch_size is None')

    def _do_transformation(self, sample: torch.Tensor) -> torch.Tensor:
        return self.input\
            .unfold(dimension=2, size=self.patch_size, step=self.patch_size)\
                .unfold(dimension=3, size=self.patch_size, step=self.patch_size)[:, :, -1, -1, :]\
                    .flatten(start_dim=1, end_dim=-1)
