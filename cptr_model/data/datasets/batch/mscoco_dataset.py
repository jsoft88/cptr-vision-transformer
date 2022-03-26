import os
from typing import Tuple, Union
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
import torch
from transformers import BertTokenizer
from PIL import Image
from cptr_model.config.config import Config
from cptr_model.utils.file_handlers.base_file_handler import BaseFileHandler
from cptr_model.utils.pretrained_model_utils import PretrainedModelUtils


class MSCocoDataset(Dataset):
    def __init__(self,
                config: Config,
                data_dir: str,
                img_data_dir: str,
                loaded_data: pd.DataFrame,
                img_H_W: Tuple[int, int],
                max_len: int, 
                fs: BaseFileHandler,
                training: bool = True) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.img_data_dir = img_data_dir
        self.loaded_data = loaded_data
        self.transforms = transforms.Compose([
            transforms.Resize((img_H_W[0], img_H_W[1])),
            transforms.ToTensor()
        ])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len or self.tokenizer.max_len_single_sentence
        self.fs = fs
        self.training = training
        self.pretrained_utils = PretrainedModelUtils(config)

    def _seq_ids(self, text: str) -> torch.Tensor:
        return self.pretrained_utils.compute_seq_ids([text.lower()]).squeeze(0)
        # tokenizer_outputs = self.tokenizer(text.lower(), return_tensors='pt', max_length=self.max_len, padding='max_length')
        # return tokenizer_outputs['input_ids'].squeeze(0)

    def __len__(self) -> int:
        return self.loaded_data.shape[0]

    def __download_image(self, img_path: str) -> Image:
        img_file = self.fs.retrieve_file(img_path)
        return Image.open(img_file)

    def __getitem__(self, index) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        return self.transforms(
            self.__download_image(f'{os.path.join(self.img_data_dir, self.loaded_data.iloc[index]["filename"])}')),\
                self._seq_ids(self.loaded_data.iloc[index]['caption']),\
                self._seq_ids(self.loaded_data.iloc[index]['caption']) if self.training else \
                    self.transforms(
                        self.__download_image(f'{os.path.join(self.img_data_dir, self.loaded_data.iloc[index]["filename"])}')
                        )
