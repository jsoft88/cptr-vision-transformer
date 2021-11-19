import math
from typing import Any, Optional
import torch.nn


class Attention(torch.nn.Module):
    KEY_NUM_HEADS = 'num-heads'
    KEY_LATENT_DIM = 'patch-latent-dim'
    KEY_MASKED_ATTENTION = 'masked-attention'

    def __init__(self, **kwargs):
        self.num_heads = kwargs.get(Attention.KEY_NUM_HEADS, None)
        self.latent_dim = kwargs.get(Attention.KEY_LATENT_DIM, None)
        self.__verify_required_args()
        self.attention_head_size = self.latent_dim // self.num_heads
        self.all_head_size = self.attention_head_size * self.num_heads

        self.query = torch.nn.Linear(self.latent_dim, self.all_head_size)
        self.key = torch.nn.Linear(self.latent_dim, self.all_head_size)
        self.value = torch.nn.Linear(self.latent_dim, self.all_head_size)

        self.masked_attention = kwargs.get(Attention.KEY_MASKED_ATTENTION, False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.out = torch.nn.Linear(self.latent_dim, self.latent_dim)
        super().__init__()

    def __generate_subsequent_mask(self, *shape: int, device: torch.device) -> torch.Tensor:
        sz_b, len_s = shape
        subsequent_mask = torch.triu(
            # The input sequences are all batches, so the original 2-dimensional tensor
            # is expanded into a 3-dimensional tensor
            torch.ones((len_s, len_s), device=device, dtype=torch.uint8), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # bx ls x ls

        return subsequent_mask

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

    def forward(self, x: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]) -> Any:
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(x if not k else k)
        mixed_value_layer = self.value(x if not v else v)

        query_layer = self.__transpose_for_scores(mixed_query_layer)
        key_layer = self.__transpose_for_scores(mixed_key_layer)
        value_layer = self.__transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if self.masked_attention:
            mask = self.__generate_subsequent_mask(*x.size())
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        # restore the transpose for scores
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[: -2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)

        return attention_output, attention_probs
