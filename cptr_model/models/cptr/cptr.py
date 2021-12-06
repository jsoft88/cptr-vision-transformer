from typing import OrderedDict, Any
import torch.nn
from pytorch_lightning import LightningModule
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.core.core_module_extension import CoreModuleExtension
from cptr_model.embeddings.input.patch_embedding import PatchEmbedding
from cptr_model.embeddings.input.word_embedding import WordEmbedding
from cptr_model.factory.input_embeddings.embedding_factory import EmbeddingFactory
from cptr_model.factory.models.initializers_factory import InitializersFactory
from cptr_model.factory.utils.fs_factory import FSFactory
from cptr_model.models.base.base_model import ModelBuilder
from cptr_model.models.cptr.decoder.cptr_decoder_block import CPTRDecoderBlock
from cptr_model.models.cptr.encoder.cptr_encoder_block import CPTREncoderBlock
from cptr_model.models.initializers.base_initializer import BaseInitializer
from cptr_model.models.initializers.cptr.init_base_vit_16_384 import BaseVit16384
from cptr_model.utils.utils import Utils


class CPTRModelBuilder(ModelBuilder, LightningModule, CoreModuleExtension):
    ENCODER_INPUT_LAYER = 'encoder_input_layer'
    ENCODER_BLOCK_LAYERS = 'encoder_block_layers.#'
    DECODER_INPUT_LAYER = 'decoder_input_layer'
    DECODER_BLOCK_LAYERS = 'decoder_block_layers.#'
    CPTR_LINEAR = 'cptr_linear'

    def __init__(self, config: Config, **kwargs):
        if not config:
            raise ValueError('config object is None')
        if not config.config_file:
            raise ValueError('config file is None')

        self.config = config
        self.config_file_manager = ArchitectureConfigFileManager(config.config_file)

        patch_embedding_sublayer = self.config_file_manager.get_sublayer_with_query(
            ArchitectureConfigFileManager.Model.EncoderBlock.NAME,
            ArchitectureConfigFileManager.Model.EncoderBlock.EncoderInputLayer.NAME,
            ArchitectureConfigFileManager.Model.EncoderBlock.EncoderInputLayer.SUBLAYER_PATCH_EMBEDDING
        )

        word_embedding_sublayer = self.config_file_manager.get_sublayer_with_query(
            ArchitectureConfigFileManager.Model.DecoderBlock.NAME,
            ArchitectureConfigFileManager.Model.DecoderBlock.DecoderInputLayer.NAME,
            ArchitectureConfigFileManager.Model.DecoderBlock.DecoderInputLayer.SUBLAYER_WORD_EMBEDDING
        )

        encoder_attention_block = self.config_file_manager.get_sublayer_with_query(
            ArchitectureConfigFileManager.Model.EncoderBlock.NAME,
            ArchitectureConfigFileManager.Model.EncoderBlock.EncoderAttentionLayer.NAME,
            ArchitectureConfigFileManager.Model.EncoderBlock.EncoderAttentionLayer.SUBLAYER_ATTENTION
        )

        decoder_attention_block = self.config_file_manager.get_sublayer_with_query(
            ArchitectureConfigFileManager.Model.DecoderBlock.NAME,
            ArchitectureConfigFileManager.Model.DecoderBlock.DecoderAttentionLayer.NAME,
            ArchitectureConfigFileManager.Model.DecoderBlock.DecoderAttentionLayer.SUBLAYER_ATTENTION
        )

        self.image_patch_embeddings_dim = ArchitectureConfigFileManager.get_params_for_sublayer(patch_embedding_sublayer)\
            .get(ArchitectureConfigFileManager.Model.PatchEmbeddingParams.DIM, None)
        self.num_encoder_blocks = ArchitectureConfigFileManager.get_params_for_sublayer(encoder_attention_block)\
            .get(ArchitectureConfigFileManager.Model.AttentionParams.NE, None)
        self.num_decoder_blocks = ArchitectureConfigFileManager.get_params_for_sublayer(decoder_attention_block)\
            .get(ArchitectureConfigFileManager.Model.AttentionParams.ND, None)
        self.image_patch_embeddings_dim = ArchitectureConfigFileManager.get_params_for_sublayer(patch_embedding_sublayer)\
            .get(ArchitectureConfigFileManager.Model.PatchEmbeddingParams.DIM)
        self.word_embeddings_dim = ArchitectureConfigFileManager.get_params_for_sublayer(word_embedding_sublayer)\
            .get(ArchitectureConfigFileManager.Model.WordEmbeddingParams.DIM, None)
        self.vocab_size = ArchitectureConfigFileManager.get_params_for_sublayer(word_embedding_sublayer)\
            .get(ArchitectureConfigFileManager.Model.WordEmbeddingParams.VOCAB_SIZE)
        self.patch_embedding_kernel = ArchitectureConfigFileManager.get_params_for_sublayer(patch_embedding_sublayer)\
            .get(ArchitectureConfigFileManager.Model.PatchEmbeddingParams.KERNEL_SIZE)
        self.patch_embedding_stride = ArchitectureConfigFileManager.get_params_for_sublayer(patch_embedding_sublayer)\
            .get(ArchitectureConfigFileManager.Model.PatchEmbeddingParams.STRIDE)
        self.patch_embedding_channels_in = ArchitectureConfigFileManager.get_params_for_sublayer(patch_embedding_sublayer)\
            .get(ArchitectureConfigFileManager.Model.PatchEmbeddingParams.CHANNELS_IN)
        self.patch_embedding_channels_out = ArchitectureConfigFileManager.get_params_for_sublayer(patch_embedding_sublayer)\
            .get(ArchitectureConfigFileManager.Model.PatchEmbeddingParams.CHANNELS_OUT)

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
        if not self.num_decoder_blocks:
            raise ValueError('Number of decoder blocks is None')
        if not self.num_encoder_blocks:
            raise ValueError('Number of encoder blocks is None')

    def _building_model_blocks(self) -> LightningModule:
        self.encoder_input_layer = EmbeddingFactory.get_instance(
            self.patch_embedding_type,
            config=self.config,
            **{
                PatchEmbedding.KEY_STRIDE: self.patch_embedding_stride,
                PatchEmbedding.KEY_IN_CHANNELS: self.patch_embedding_channels_in,
                PatchEmbedding.KEY_OUT_CHANNELS: self.patch_embedding_channels_out,
                PatchEmbedding.KEY_KERNEL_SIZE: self.patch_embedding_kernel
            }
        )
        self.decoder_input_layer = EmbeddingFactory.get_instance(
            self.word_embedding_type,
            config=self.config,
            **{
                WordEmbedding.KEY_DIM: self.word_embeddings_dim,
                WordEmbedding.KEY_VOCAB_SIZE: self.vocab_size,
                WordEmbedding.KEY_PAD_IDX: 0 #TODO: check this how it behaves
            }
        )

        self.encoder_block_layers = torch.nn.ModuleList([
            CPTREncoderBlock(self.config, self.config_file_manager) for _ in range(self.num_encoder_blocks)
        ])
        self.decoder_block_layers = torch.nn.ModuleList([
            CPTRDecoderBlock(self.config, self.config_file_manager) for _ in range(self.num_decoder_blocks)
        ])

        self.linear_layer = torch.nn.Linear(self.word_embeddings_dim, self.word_embeddings_dim)
        self.softmax = torch.nn.Softmax(self.vocab_size)

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

    def _assign_state_to_model(self) -> None:
        if self.config.requires_model_init:
            if not self.config.model_init_path:
                raise ValueError('Initializer model value is None')
            if not self.config.model_initializer_type:
                raise ValueError('Model Initializer type value is None')

            #TODO: Allow the factory to inject whole args as opposed to filling in bag of args
            initializer_args = {
                BaseInitializer.KEY_MODEL: self,
                BaseVit16384.KEY_NUMBER_ENCODER_LAYERS: self.num_encoder_blocks,
                BaseVit16384.KEY_NUMBER_DECODER_LAYERS: self.num_decoder_blocks
            }

            initializer_instance = InitializersFactory.get_instance(
                self.config.model_initializer_type,
                self.config, self.config_file_manager, 
                initializer_args)

            initializer_instance.map_state_dict_to_model()
        elif self.config.with_pretrained_model:
            fs = FSFactory.get_instance(self.config.file_system_type, **self.config.file_system_options)
            saved_model = fs.retrieve_file(self.config.pretrained_model_path)
            pretrained_model = Utils.bytes_to_dict(saved_model)

            with torch.no_grad():
                self.state_dict.update(pretrained_model.state_dict)

    def _model_state_to_storage(self) -> None:
        if not self.config.model_save_file_system:
            raise ValueError('Attempted to save model state to storage but file system is None')

        if not self.config.model_save_path:
            raise ValueError('Attempted to save model state to storage but path is None')

        model_state: OrderedDict[str, Any] = OrderedDict[str, Any]({})
        model_state[CPTRModelBuilder.ENCODER_INPUT_LAYER] = self.encoder_input_layer.weight_transfer_to_dict()
        for idx, enc_blk in list(enumerate(len(self.encoder_block_layers))):
            model_state[f'{CPTRModelBuilder.ENCODER_BLOCK_LAYERS.replace("#", idx)}'] = enc_blk.weight_transfer_to_dict()

        for idx, dec_blk in list(enumerate(len(self.decoder_block_layers))):
            model_state[f'{CPTRModelBuilder.DECODER_BLOCK_LAYERS.replace("#", idx)}'] = dec_blk.weight_transfer_to_dict()

        model_state[CPTRModelBuilder.DECODER_INPUT_LAYER] = self.decoder_input_layer.weight_transfer_to_dict()

        model_state[f'{CPTRModelBuilder.CPTR_LINEAR}.weight'] = self.linear_layer.weight
        model_state[f'{CPTRModelBuilder.CPTR_LINEAR}.bias'] = self.linear_layer.bias

        fs = FSFactory.get_instance(self.config.model_save_file_system, **self.config.model_save_fs_options)
        fs.save_file(self.config.model_save_path, Utils.dict_to_bytes(model_state))

    def weight_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            CPTRModelBuilder.StateKey.CPTR_LINEAR_LAYER_WEIGHT: self.linear_layer.weight
        })

    def bias_transfer_to_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict({
            CPTRModelBuilder.StateKey.CPTR_LINEAR_LAYER_BIAS: self.linear_layer.bias
        })

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        self.linear_layer.weight = weights.get(CPTRModelBuilder.StateKey.CPTR_LINEAR_LAYER_WEIGHT, None)

    def bias_transfer_from_dict(self, bias: OrderedDict[str, Any]) -> None:
        self.linear_layer.bias = bias.get(CPTRModelBuilder.StateKey.CPTR_LINEAR_LAYER_BIAS, None)

    class StateKey:
        CPTR_LINEAR_LAYER_WEIGHT = 'cptr_linear_layer.weight'
        CPTR_LINEAR_LAYER_BIAS = 'cptr_linear_layer.bias'