from typing import Any, OrderedDict
import torch.nn
from torch.nn import Conv2d
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.config.config import Config
from cptr_model.core.core_module_extension import CoreModuleExtension
from cptr_model.factory.input_embeddings import embedding_factory


class PatchEmbedding(torch.nn.Module, CoreModuleExtension):
    KEY_IN_CHANNELS = 'in-channels'
    KEY_OUT_CHANNELS = 'out-channels'
    KEY_KERNEL_SIZE = 'kernel-size'
    KEY_STRIDE = 'stride'

    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__()

        self.config = config

        self.in_channels = kwargs.get(PatchEmbedding.KEY_IN_CHANNELS, None)
        self.out_channels = kwargs.get(PatchEmbedding.KEY_OUT_CHANNELS, None)
        self.kernel_size = kwargs.get(PatchEmbedding.KEY_KERNEL_SIZE, None)
        self.stride = kwargs.get(PatchEmbedding.KEY_STRIDE, None)

        self.__verify_required_args()

        self.embedding_layer = Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride
        ).to(self.config.device)

    def __verify_required_args(self) -> None:
        if not self.config:
            raise ValueError('config value is None')

        if not self.in_channels:
            raise ValueError('in_channels value is None')

        if not self.out_channels:
            raise ValueError('out_channels value is None')

        if not self.stride:
            raise ValueError('stride value is None')

        if not self.kernel_size:
            raise ValueError('kernel_size value is None')

    def forward(self, x: torch.Tensor) -> Any:
        x = self.embedding_layer(x)
        return x

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        self.embedding_layer.weight = weights[PatchEmbedding.StateKey.EMBEDDING_LAYER_WEIGHT]
    
    def bias_transfer_from_dict(self, bias: OrderedDict[str, Any]) -> None:
        self.embedding_layer.bias = bias[PatchEmbedding.StateKey.EMBEDDING_LAYER_BIAS]

    def weight_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            PatchEmbedding.StateKey.EMBEDDING_LAYER_WEIGHT: self.embedding_layer.weight
        })

    def bias_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            PatchEmbedding.StateKey.EMBEDDING_LAYER_BIAS: self.embedding_layer.bias
        })

    class StateKey:
        EMBEDDING_LAYER_WEIGHT = 'embedding_layer.weight'
        EMBEDDING_LAYER_BIAS = 'embedding_layer.bias'