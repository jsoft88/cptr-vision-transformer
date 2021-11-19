from typing import List, Optional
from pytorch_lightning import LightningModule
from cptr_model.config.config import Config
from cptr_model.factory.base_factory import BaseFactory
from cptr_model.models.cptr.cptr import CPTR


class ModelFactory(BaseFactory[LightningModule]):
    CPTR_MODEL = 'cptr'

    _ALL_TYPES = [
        CPTR_MODEL
    ]

    @classmethod
    def get_instance(cls, type_str: str, config: Optional[Config], **kwargs) -> LightningModule:
        instance = {
            ModelFactory.CPTR_MODEL: CPTR(config, kwargs)
        }.get(type_str, None)

        if instance:
            return instance
        raise ValueError(f'{type_str} is not a valid model type for factory')

    @classmethod
    def all_types(cls) -> List[str]:
        return ModelFactory._ALL_TYPES
