from typing import Any, Iterator, Tuple
from torch.utils.data.dataset import IterableDataset
from transformers.models.bert.tokenization_bert import BertTokenizer
from cptr_model.config.config import Config
import torch
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.utils.inference_helpers.provider_dependency_injection.fetcher_dependency_mixin import FetcherDependencyMixin
from cptr_model.utils.pretrained_model_utils import PretrainedModelUtils


class StreamDataset(IterableDataset, FetcherDependencyMixin):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.input_fetcher = self.get_data_fetcher(config)
        self.config = config
        self.model_config: ArchitectureConfigFileManager = self.config.cptr_specifics
        self.pretrained_model_utils = PretrainedModelUtils(self.config)
        self.max_len = self.model_config.decoder_max_seq_len
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __iter__(self) -> Iterator[Tuple[Any, torch.Tensor]]:
        for msg in self.input_fetcher.get_input(self.config.batch):
            yield (msg[0], msg[1], self.pretrained_model_utils.compute_seq_ids([self.bert_tokenizer.cls_token]))
