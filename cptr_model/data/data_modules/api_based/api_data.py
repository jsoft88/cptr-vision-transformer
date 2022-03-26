from torch.utils.data.dataloader import DataLoader
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.data.datasets.api_based.stream_dataset import StreamDataset
from pytorch_lightning import LightningDataModule


class ApiData(LightningDataModule):
    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.api_dataset = None
    
    def prepare_data(self) -> None:
        self.api_dataset = StreamDataset(self.config)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.api_dataset, batch_size=self.config.batch)