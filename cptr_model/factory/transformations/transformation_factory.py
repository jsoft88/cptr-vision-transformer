from typing import List, Optional
import torch
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.factory.base_factory import BaseFactory
from cptr_model.transformations.images.image_transformation import ImageTransformation
from cptr_model.transformations.tensors.flatten_tensor import FlattenTensor


class TransformationFactory(BaseFactory[torch.nn.Module]):
    PATCH_CROPPING = 'patch-cropping'
    PATCH_FLATTEN = 'patch-flatten'

    __ALL_TYPES = [
        PATCH_CROPPING,
        PATCH_FLATTEN
    ]

    @classmethod
    def get_instance(cls,
                    type_str: str,
                    config: Optional[Config],
                    config_file_manager: Optional[ArchitectureConfigFileManager], **kwargs) -> torch.nn.Module:

        instance = {
            TransformationFactory.PATCH_CROPPING: ImageTransformation(kwargs),
            TransformationFactory.PATCH_FLATTEN: FlattenTensor(kwargs)
        }.get(type_str, None)

        if instance:
            return instance
        
        raise ValueError(f'TransformationFactory:: Invalid transformation type: {type_str}')

    @classmethod
    def all_types(cls) -> List[str]:
        return TransformationFactory.__ALL_TYPES
