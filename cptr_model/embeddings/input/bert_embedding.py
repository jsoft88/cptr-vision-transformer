from typing import Any, OrderedDict
import torch
from torch.nn import Module
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.core.core_module_extension import CoreModuleExtension
from cptr_model.utils.pretrained_model_utils import PretrainedModelUtils


class BertEmbedding(Module, CoreModuleExtension):
    def __init__(self, config: Config, **kwargs):
        super().__init__()
        self.config = config
        self.model_config: ArchitectureConfigFileManager = self.config.model_config
        self.pretrained_utils = PretrainedModelUtils(config)
    
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.pretrained_utils.compute_bert_embeddings(inp, add_special_tokens=self.config.training)

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        pass
    def weight_transfer_to_dict(self) -> OrderedDict[str, Any]:
        pass
    def bias_transfer_from_dict(self, bias: OrderedDict[str, Any]) -> None:
        pass
    def bias_transfer_to_dict(self) -> OrderedDict[str, Any]:
        pass