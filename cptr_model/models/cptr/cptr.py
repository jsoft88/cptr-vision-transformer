import torch.nn
from pytorch_lightning import LightningModule
from cptr_model.config.config import Config
from cptr_model.config.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.embeddings.input.patch_embedding import PatchEmbedding
from cptr_model.factory.input_embeddings.embedding_factory import EmbeddingFactory
from cptr_model.models.base.base_model import ModelBuilder
from cptr_model.models.cptr.decoder.cptr_decoder_block import CPTRDecoderBlock
from cptr_model.models.cptr.encoder.cptr_encoder_block import CPTREncoderBlock


class CPTRModelBuilder(ModelBuilder, LightningModule):

    def _assign_state_to_model(self) -> None:
        super()._assign_state_to_model()

    def __init__(self, config: Config, **kwargs):
        self.config = config
        self.num_encoder_blocks = config.num_encoder_blocks
        self.num_decoder_blocks = config.num_decoder_blocks
        self.image_patch_embeddings_dim = config.encoder_input_embedding_dim
        self.word_embeddings_dim = config.decoder_input_embedding_dim
        self.image_transformation_types = config.encoder_transformation_types
        self.image_transformations = None
        self.word_transformation_types = config.decoder_transformation_types
        self.word_transformations = None
        self.encoder_data_loader_type = config.encoder_data_loader_type
        self.decoder_data_loader_type = config.decoder_data_loader_type
        self.patch_position_embedding_type = config.encoder_position_embedding_type
        self.word_position_embedding_type = config.decoder_position_embedding_type
        self.patch_embedding_type = config.encoder_input_embedding_type
        self.word_embedding_type = config.decoder_input_embedding_type
        self.config_file_manager = ArchitectureConfigFileManager(config.config_file)

        # model building blocks
        self.encoder_input_layer = None
        self.encoder_block_layers = None
        self.encoder_output_layer = None
        self.decoder_input_layer = None
        self.decoder_block_layers = None
        self.decoder_output_layer = None
        self.linear_layer = None
        self.softmax = None

        super().__init__()

    def _verify_required_args(self) -> None:
        if not self.num_encoder_blocks:
            raise ValueError('Number of encoding blocks is None')
        if not self.num_decoder_blocks:
            raise ValueError('Number of decoding blocks is None')
        if not self.image_patch_embeddings_dim:
            raise ValueError('Dimension of patch embeddings is None')
        if not self.image_transformation_types:
            raise ValueError('Transformation types to apply to images is None')
        if not self.encoder_data_loader_type:
            raise ValueError('Encoder loader type is None')
        if not self.decoder_data_loader_type:
            raise ValueError('Decoder loader type is None')

    def _building_model_blocks(self) -> LightningModule:
        self.encoder_input_layer = EmbeddingFactory.get_instance(
            self.patch_embedding_type,
            config=self.config,
            *{
                PatchEmbedding.KEY_POSITION_EMBEDDING_TYPE: self.patch_position_embedding_type,
                PatchEmbedding.KEY_STRIDE: self.
            }
        )
        self.decoder_input_layer = EmbeddingFactory.get_instance(self.word_embedding_type)

        self.encoder_block_layers = torch.nn.ModuleList([CPTREncoderBlock() for _ in range(self.num_encoder_blocks)])
        self.decoder_block_layers = torch.nn.ModuleList([CPTRDecoderBlock() for _ in range(self.num_decoder_blocks)])

        self.linear_layer = torch.nn.Linear()
        self.softmax = torch.nn.Softmax()

        return self

    def forward(self, x_e: torch.Tensor, x_d: torch.Tensor, **kwargs) -> torch.Tensor:
        x_e = self.encoder_input_layer(x_e)
        for enc_layer in self.encoder_block_layers:
            x_e = enc_layer(x_e)
        for dec_layer in self.decoder_block_layers:
            x_d = dec_layer(x_d, x_e, x_e)
        x_d = self.linear_layer(x_d)
        x = self.softmax(x_d)

        return x
