from typing import Any

import torch.nn
from pytorch_lightning import LightningModule


class MLP(LightningModule):
    KEY_LATENT_DIM = 'latent-dim'
    KEY_MLP_DIM = 'mlp-dim'
    KEY_DROPOUT_RATE = 'dropout-rate'

    def __init__(self, **kwargs) -> None:
        self.latent_dim = kwargs.get(MLP.KEY_LATENT_DIM, None)
        self.mlp_dim = kwargs.get(MLP.KEY_MLP_DIM, None)
        self.dropout_rate = kwargs.get(MLP.KEY_DROPOUT_RATE, None)
        self.__verify_required_args()
        self.latent_dim = float(self.latent_dim)
        self.mlp_dim = float(self.mlp_dim)
        self.dropout_rate = float(self.dropout_rate)
        self.fc1 = torch.nn.Linear(self.latent_dim, self.mlp_dim)
        self.fc2 = torch.nn.Linear(self.mlp_dim, self.latent_dim)
        self.activation = torch.nn.GELU
        self.dropout = torch.nn.Dropout(self.dropout_rate)

        self.__init_weights()

    def __init_weights(self):
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.normal(self.fc1.bias, std=1e-6)
        torch.nn.init.normal(self.fc2.bias, std=1e-6)

    def __verify_required_args(self) -> None:
        if not self.latent_dim:
            raise ValueError(f'{MLP.KEY_LATENT_DIM} value is None')

        if not self.mlp_dim:
            raise ValueError(f'{MLP.KEY_MLP_DIM} value is None')

        if not self.dropout_rate:
            raise ValueError(f'{MLP.KEY_DROPOUT_RATE} value is None')

    def forward(self, x) -> Any:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
