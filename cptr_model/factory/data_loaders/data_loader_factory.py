from typing import List, Optional
from pytorch_lightning import LightningDataModule
from cptr_model.config.config import Config
from cptr_model.factory.base_factory import BaseFactory


class DataLoaderFactory(BaseFactory[LightningDataModule]):
    _ENCODER_DATA_LOADER = 'encoder-dl'
    _DECODER_DATA_LOADER = 'decoder-dl'

    __ALL_TYPES = [
        _ENCODER_DATA_LOADER,
        _DECODER_DATA_LOADER
    ]

    @classmethod
    def get_instance(cls, type_str: str, config: Optional[Config], **kwargs) -> LightningDataModule:
        pass

    @classmethod
    def all_types(cls) -> List[str]:
        return DataLoaderFactory.__ALL_TYPES
