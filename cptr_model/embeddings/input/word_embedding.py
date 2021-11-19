import torch.nn
from cptr_model.config.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.config.config import Config


class WordEmbedding(torch.nn.Module):
    _KEY_DIM = 'dim'
    _KEY_VOCAB_SIZE = 'vocab_size'
    _KEY_PAD_IDX = 'pad-index'

    def __init__(self, config: Config, config_file_manager: ArchitectureConfigFileManager, **kwargs) -> None:
        self.config = config
        self.config_manager = config_file_manager
        self.config_section = self.config_manager \
            .get_embeddings_config_for(ArchitectureConfigFileManager.ArchitectureParts.SECTION_EMBEDDINGS_INPUT)

        self.__verify_required_args()

        params = self.config_manager \
            .get_embeddings_params(self.config.encoder_input_embedding_config_section, self.config_section)
        self.dim = self.config_manager.get_param_value_for(WordEmbedding._KEY_DIM, params)
        self.vocab_size = self.config_manager.get_param_value_for(WordEmbedding._KEY_VOCAB_SIZE, params)
        self.pad_idx = self.config_manager.get_param_value_for(WordEmbedding._KEY_PAD_IDX, params)

        self.__verify_required_args()
        self.embedding = torch.nn.Embedding(self.vocab_size, self.dim, padding_idx=self.pad_idx)
        super(WordEmbedding, self).__init__()

    def __verify_required_args(self) -> None:
        if not self.config:
            raise ValueError('config value is None')

        if not self.config_manager:
            raise ValueError('config file manager value is None')

        if not self.config_section:
            raise ValueError(
                f'Could not find section for {ArchitectureConfigFileManager.ArchitectureParts.SECTION_EMBEDDINGS_INPUT}'
            )

    def forward(self, word) -> torch.Tensor:
        return self.embedding(word)
