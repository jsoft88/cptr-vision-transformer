from typing import Any, OrderedDict
import torch.nn
from pytorch_lightning import LightningModule
from cptr_model.core.core_module_extension import CoreModuleExtension


class MLP(torch.nn.Module, CoreModuleExtension):
    KEY_LATENT_DIM = 'latent-dim'
    KEY_MLP_DIM = 'mlp-dim'
    KEY_DROPOUT_RATE = 'dropout-rate'

    def __init__(self, **kwargs) -> None:
        self.latent_dim = kwargs.get(MLP.KEY_LATENT_DIM, None)
        self.mlp_dim = kwargs.get(MLP.KEY_MLP_DIM, None)
        self.dropout_rate = kwargs.get(MLP.KEY_DROPOUT_RATE, None)
        self.__verify_required_args()
        super(MLP, self).__init__()
        self.latent_dim = self.latent_dim
        self.mlp_dim = self.mlp_dim
        self.dropout_rate = float(self.dropout_rate)
        self.fc1 = torch.nn.Linear(self.latent_dim, self.mlp_dim)
        self.fc2 = torch.nn.Linear(self.mlp_dim, self.latent_dim)
        self.activation = torch.nn.GELU()
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

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        model_dict = self.state_dict()
        model_dict[MLP.StateKey.MLP_FC1_WEIGHT] = weights[MLP.StateKey.MLP_FC1_WEIGHT]
        model_dict[MLP.StateKey.MLP_FC2_WEIGHT] = weights[MLP.StateKey.MLP_FC2_WEIGHT]
        self.load_state_dict(model_dict)

    def bias_transfer_from_dict(self, bias: OrderedDict[str, Any]) -> None:
        model_dict = self.state_dict()
        model_dict[MLP.StateKey.MLP_FC1_BIAS] = bias[MLP.StateKey.MLP_FC1_BIAS]
        model_dict[MLP.StateKey.MLP_FC2_BIAS] = bias[MLP.StateKey.MLP_FC2_BIAS]
        self.load_state_dict(model_dict)

    def weight_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            MLP.StateKey.MLP_FC1_WEIGHT: self.fc1.weight,
            MLP.StateKey.MLP_FC2_WEIGHT: self.fc2.weight
        })

    def bias_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            MLP.StateKey.MLP_FC1_BIAS: self.fc1.bias,
            MLP.StateKey.MLP_FC2_BIAS: self.fc2.bias
        })

    class StateKey:
        MLP_FC1_WEIGHT = 'fc1.weight'
        MLP_FC1_BIAS = 'fc1.bias'
        MLP_FC2_WEIGHT = 'fc2.weight'
        MLP_FC2_BIAS = 'fc2.bias'
        MLP_DROPOUT_WEIGHT = 'dropout.weight'
        MLP_DROPOUT_BIAS = 'dropout.bias'