from torchvision.transforms.functional import *
from cptr_model.transformations.base_transformation import BaseTransformation


class ImageTransformation(BaseTransformation):
    KEY_INPUT = 'input'
    KEY_IS_TENSOR = 'is-tensor'
    KEY_PATCH_SIZE = 'patch-size'

    def __init__(self, **kwargs) -> None:
        self.input = kwargs.pop(ImageTransformation.KEY_INPUT, None)
        self.is_tensor = kwargs.pop(ImageTransformation.KEY_IS_TENSOR, None)
        self.patch_size = kwargs.pop(ImageTransformation.KEY_PATCH_SIZE, None)

    def _verify_required_parameters_for_transformation(self):
        if not self.input:
            raise ValueError('Input is None')
        if not self.is_tensor:
            raise ValueError('is_tensor is None')
        if not self.patch_size:
            raise ValueError('patch_size is None')

    def __do_transformation(self) -> List[torch.Tensor]:
        return self.__split_image_in_patches_tensor()

    def __split_image_in_patches_tensor(self) -> List[torch.Tensor]:
        img = self.input if self.is_tensor else to_tensor(self.input)
        height, width = self.input.shape[1], self.input.shape[2]
        patches = [
            crop(
                img,
                self.patch_size * (i // (width - self.patch_size)),
                self.patch_size * i % (width - self.patch_size),
                self.patch_size,
                self.patch_size
            ) for i in range((height // self.patch_size) * (width // self.patch_size))
        ]
        return patches
