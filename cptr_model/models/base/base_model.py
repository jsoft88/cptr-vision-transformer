from pytorch_lightning import LightningModule
from cptr_model.config.config import Config
import torch


class ModelBuilder(LightningModule):
    def __init__(self, config: Config, **kwargs) -> None:
        super(ModelBuilder, self).__init__()
        self.config = config

    def _verify_required_args(self) -> None:
        raise NotImplementedError('__verified_required_args method not implemented in ModelBuilder')

    def build_model(self) -> None:
        self._verify_required_args()
        self._building_model_blocks()
        self.to(device=torch.device('cuda' if self.config.default_use_gpu else 'cpu'))

    def _building_model_blocks(self) -> None:
        raise NotImplementedError('_building_model_blocks not implemented')

    def load_model_state(self) -> None:
        self._building_model_blocks()
        self._assign_state_to_model()

    def _assign_state_to_model(self) -> None:
        raise NotImplementedError('_assign_state_to_model not implemented')

    def save_model_state(self) -> None:
        self._model_state_to_storage()

    def _model_state_to_storage(self) -> None:
        raise NotImplementedError('_model_state_to_storage not implemented')
