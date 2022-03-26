import math
from typing import Any, Optional, OrderedDict
import torch
from torch.nn.modules import Linear
from torch.nn.modules.activation import Softmax
from cptr_model.config.config import Config
from cptr_model.core.core_module_extension import CoreModuleExtension


class Attention(torch.nn.Module, CoreModuleExtension):
    KEY_NUM_HEADS = 'num-heads'
    KEY_LATENT_DIM = 'patch-latent-dim'
    KEY_MASKED_ATTENTION = 'masked-attention'

    def __init__(self, config: Config, **kwargs):
        self.num_heads = kwargs.get(Attention.KEY_NUM_HEADS, None)
        self.latent_dim = kwargs.get(Attention.KEY_LATENT_DIM, None)
        self.__verify_required_args()
        self.attention_head_size = self.latent_dim // self.num_heads
        self.all_head_size = self.attention_head_size * self.num_heads
        self.config = config

        super().__init__()
        
        self.query = Linear(self.latent_dim, self.all_head_size).to(config.device)
        self.key = Linear(self.latent_dim, self.all_head_size).to(config.device)
        self.value = Linear(self.latent_dim, self.all_head_size).to(config.device)

        self.masked_attention = kwargs.get(Attention.KEY_MASKED_ATTENTION, False)
        self.softmax = Softmax(dim=-1).to(config.device)
        self.out = Linear(self.latent_dim, self.latent_dim).to(config.device)

    def __verify_required_args(self) -> None:
        if not self.num_heads:
            raise ValueError(f'{Attention.KEY_NUM_HEADS} is None')
        if not self.latent_dim:
            raise ValueError(f'{Attention.KEY_PATCH_LATENT_DIM} is None')

    def __transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # Tensor of shape (N, P*P, D)
        new_shape_x = x.size()[: -1] + (self.num_heads, self.attention_head_size)
        # New shape is (N, P*P, h, D/h)
        x = x.view(*new_shape_x)
        # permute: (N, h, P*P, D/h)
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, k: Optional[torch.Tensor] = None, v: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> Any:
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(k if not (k is None) else x)
        mixed_value_layer = self.value(v if not (v is None) else x)

        query_layer = self.__transpose_for_scores(mixed_query_layer)
        key_layer = self.__transpose_for_scores(mixed_key_layer)
        value_layer = self.__transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if not (mask is None):
            attention_scores = attention_scores + (mask * -1e9)
        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        # restore the transpose for scores
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[: -2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        print(f'OUTPUT ATTENTION IS ==> {attention_output.shape}')
        return attention_output, attention_probs

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        model_dict = self.state_dict()
        model_dict[Attention.StateKey.ATTENTION_QUERY_WEIGHT] = weights[Attention.StateKey.ATTENTION_QUERY_WEIGHT]
        model_dict[Attention.StateKey.ATTENTION_KEY_WEIGHT] = weights[Attention.StateKey.ATTENTION_KEY_WEIGHT]
        model_dict[Attention.StateKey.ATTENTION_VALUE_WEIGHT] = weights[Attention.StateKey.ATTENTION_VALUE_WEIGHT]
        model_dict[Attention.StateKey.ATTENTION_OUT_WEIGHT] = weights[Attention.StateKey.ATTENTION_OUT_WEIGHT]

        self.load_state_dict(model_dict)

    def weight_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            Attention.StateKey.ATTENTION_QUERY_WEIGHT: self.query.weight,
            Attention.StateKey.ATTENTION_KEY_WEIGHT: self.key.weight,
            Attention.StateKey.ATTENTION_VALUE_WEIGHT: self.value.weight,
            Attention.StateKey.ATTENTION_OUT_WEIGHT: self.out.weight
        })

    def bias_transfer_from_dict(self, bias: OrderedDict[str, Any]) -> None:
        model_dict = self.state_dict()
        model_dict[Attention.StateKey.ATTENTION_QUERY_BIAS] = bias[Attention.StateKey.ATTENTION_QUERY_BIAS]
        model_dict[Attention.StateKey.ATTENTION_KEY_BIAS] = bias[Attention.StateKey.ATTENTION_KEY_BIAS]
        model_dict[Attention.StateKey.ATTENTION_VALUE_BIAS] = bias[Attention.StateKey.ATTENTION_VALUE_BIAS]
        model_dict[Attention.StateKey.ATTENTION_OUT_BIAS] = bias[Attention.StateKey.ATTENTION_OUT_BIAS]

        self.load_state_dict(model_dict)

    def bias_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            Attention.StateKey.ATTENTION_QUERY_BIAS: self.query.bias,
            Attention.StateKey.ATTENTION_KEY_BIAS: self.key.bias,
            Attention.StateKey.ATTENTION_VALUE_BIAS: self.value.bias,
            Attention.StateKey.ATTENTION_OUT_BIAS: self.out.bias
        })

    class StateKey:
        ATTENTION_QUERY_WEIGHT = 'query.weight'
        ATTENTION_KEY_WEIGHT = 'key.weight'
        ATTENTION_VALUE_WEIGHT = 'value.weight'
        ATTENTION_QUERY_BIAS = 'query.bias'
        ATTENTION_KEY_BIAS = 'key.bias'
        ATTENTION_VALUE_BIAS = 'value.bias'
        ATTENTION_OUT_WEIGHT = 'out.weight'
        ATTENTION_OUT_BIAS = 'out.bias'

