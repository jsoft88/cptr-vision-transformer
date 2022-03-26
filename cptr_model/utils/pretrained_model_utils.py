from typing import Any, Dict, List, Optional, Union
import torch
from transformers import BertConfig, BertModel, BertTokenizer
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager


class PretrainedModelUtils:
    def __init__(self, config: Config) -> None:
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', config=BertConfig())
        self.config = config
        self.model_config: ArchitectureConfigFileManager = self.config.cptr_specifics

    def compute_bert_embeddings(self, seqs: Union[Optional[List[str]], Optional[torch.Tensor]], add_special_tokens: Optional[bool] = True) -> torch.Tensor:
        tokenizer_outputs = \
            self.bert_tokenizer(
                seqs,
                return_tensors='pt',
                max_length=self.model_config.decoder_max_seq_len,
                padding='max_length', add_special_tokens=add_special_tokens) \
                if type(seqs) is List else self.prepare_bert_input_from_seq_ids(seqs)
        
        with torch.no_grad():
            outputs = self.bert_model(**tokenizer_outputs, output_hidden_states=True)
            return  torch.sum(torch.stack(list(outputs.hidden_states[-4:]), dim=0), dim=0, keepdims=False)

    def compute_seq_ids(self, seqs: List[str], add_special_tokens: Optional[bool] = True) -> torch.Tensor:
        return self.bert_tokenizer(
            seqs,
            return_tensors='pt',
            max_length=self.model_config.decoder_max_seq_len,
            padding='max_length',
            add_special_tokens=add_special_tokens)['input_ids']

    def prepare_bert_input_from_seq_ids(self, seq_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': seq_ids,
            'token_type_ids': torch.zeros_like(seq_ids),
            'attention_mask': torch.tensor(seq_ids != self.bert_tokenizer.pad_token_id, dtype=torch.int)
        }