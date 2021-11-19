from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from cptr_model.config.config import Config


class MSCocoData(LightningDataModule):
    KEY_IS_ENCODER = 'is-encoder'

    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__()
        self.is_encoder = kwargs.get(MSCocoData.KEY_IS_ENCODER, False)
        self.batch_size = config.encoder_batch_size if self.is_encoder else config.decoder_batch_size
        self.data_dir = config.encoder_data_location if self.is_encoder else config.decoder_data_location
        self.train_data = None
        self.test_data = None
        self.val_data = None

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, self.batch_size)