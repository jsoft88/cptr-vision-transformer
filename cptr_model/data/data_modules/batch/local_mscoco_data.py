from typing import List, Optional
from pytorch_lightning import LightningDataModule
import torch
from cptr_model.config.config import Config
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from cptr_model.data.datasets.batch.mscoco_dataset import MSCocoDataset
import os
from cptr_model.utils.file_handlers.base_file_handler import BaseFileHandler


class LocalMSCocoData(LightningDataModule):
    KEY_MAX_LEN = 'max-len'
    KEY_FS = 'fs'

    def __init__(self, config: Config, fs_handler: Optional[BaseFileHandler] = None, **kwargs) -> None:
        super().__init__()

        if not config:
            raise ValueError('config object is None')

        self.config = config
        self.train_data_dir = self.config.batch_train_metadata_location
        self.val_data_dir = self.config.batch_val_metadata_location
        self.train_data_img_dir = self.config.batch_train_img_location
        self.val_data_img_dir = self.config.batch_val_img_location
        self.train_data = None
        self.train_data_img = None
        self.val_data = None
        self.val_data_img = None
        self.max_len = kwargs.get(LocalMSCocoData.KEY_MAX_LEN, None)
        self.img_H_W = (
            self.config.cptr_specifics.encoder_input_embeddings_params_dict['height'],
            self.config.cptr_specifics.encoder_input_embeddings_params_dict['width']
        )
        self.train_dataset = None
        self.val_dataset = None
        self.fs = fs_handler

    def __parquet_reader_helper(self, pattern: str) -> pd.DataFrame:
        files = self.fs.list_files(pattern)
        ret_val = pd.concat([pd.read_parquet(f) for f in files or []])
    
        return ret_val

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.fs:
            raise ValueError('Expected filesystem object to be present, None value found')
            
        self.train_data = self.__parquet_reader_helper(f'{os.path.join(self.train_data_dir, "*.parquet")}')
        self.val_data = self.__parquet_reader_helper(f'{os.path.join(self.val_data_dir, "*.parquet")}')
        self.train_dataset = MSCocoDataset(self.config, self.train_data_dir, self.train_data_img_dir, self.train_data, self.img_H_W, self.max_len, self.fs, True)
        # self.val_dataset = MSCocoDataset(self.config, self.val_data_dir, self.val_data_img_dir, self.val_data, self.img_H_W, self.max_len, self.fs, False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True, num_workers=1, drop_last=False, batch_size=self.config.batch_size)

    def val_dataloader(self) -> DataLoader:
        pass
        #return DataLoader(self.val_dataset, shuffle=True, num_workers=1, drop_last=False)
