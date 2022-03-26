from typing import Any, Optional, OrderedDict
from pytorch_lightning.core.lightning import LightningModule
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.embeddings.input.patch_embedding import PatchEmbedding
from cptr_model.embeddings.input.word_embedding import WordEmbedding
from cptr_model.models.cptr.cptr import CPTRModelBuilder
from cptr_model.models.cptr.decoder.cptr_decoder_block import CPTRDecoderBlock
from cptr_model.models.cptr.encoder.cptr_encoder_block import CPTREncoderBlock
from cptr_model.models.initializers.base_initializer import BaseInitializer
from cptr_model.utils.file_handlers.base_file_handler import BaseFileHandler
from cptr_model.utils.utils import Utils


class BaseVit16384(BaseInitializer):
    KEY_NUMBER_ENCODER_LAYERS = 'ne'
    KEY_NUMBER_DECODER_LAYERS = 'nd'
    
    def __init__(self, config: Optional[Config],
                config_file_manager: Optional[ArchitectureConfigFileManager],
                **kwargs) -> None:
        super().__init__(config, config_file_manager, kwargs)
        self.number_encoder_layers = kwargs.get(BaseVit16384.KEY_NUMBER_ENCODER_LAYERS, None)
        self.number_decoder_layers = kwargs.get(BaseVit16384.KEY_NUMBER_DECODER_LAYERS, None)

    def map_state_dict_to_model(self) -> None:
        if not self.number_encoder_layers:
            raise ValueError('Number of encoder layers is None')

        if not self.number_decoder_layers:
            raise ValueError('Number of decoder layers is None')

        model_state_bytes = self.fs.retrieve_file(self.path)
        model_dict = Utils.bytes_to_dict(model_state_bytes)
        
        self.model.encoder_input_layer.weight_transfer_from_dict(OrderedDict({
            PatchEmbedding.StateKey.EMBEDDING_LAYER_WEIGHT: model_dict.get('patch_embed.proj.weight', None)
        }))
        self.model.encoder_input_layer.bias_transfer_from_dict(OrderedDict({
            PatchEmbedding.StateKey.EMBEDDING_LAYER_BIAS: model_dict.get('patch_embed.proj.bias')
        }))
        self.model.decoder_input_layer.weight_transfer_from_dict(OrderedDict({
            WordEmbedding.StateKey.EMBEDDING_WEIGHT: model_dict.get('')
        }))
        for idx, enc_blk in list(enumerate(self.number_encoder_layers)):
            enc_blk.weight_transfer_from_dict(OrderedDict({
                CPTREncoderBlock.StateKey.ATTENTION_NORM_WEIGHT: model_dict.get(f'blocks.{idx}.norm1.weight', None)
            }))
            enc_blk.bias_transfer_from_dict(OrderedDict({
                CPTREncoderBlock.StateKey.ATTENTION_NORM_BIAS: model_dict.get(f'blocks.{idx}.norm1.bias', None)
            }))

        for idx, dec_blk in list(enumerate(self.number_decoder_layers)):
            dec_blk.weight_transfer_from_dict(OrderedDict({
                CPTRDecoderBlock.StateKey.ATTENTION_NORMALIZATION_WEIGHT: model_dict.get(f'blocks.{idx}.norm1.weight', None)
            }))
            dec_blk.bias_transfer_from_dict(OrderedDict({
                CPTRDecoderBlock.StateKey.ATTENTION_NORMALIZATION_BIAS: model_dict.get(f'blocks.{idx}.norm1.bias', None)
            }))

        self.model.weight_transfer_from_dict(OrderedDict({
            CPTRModelBuilder.StateKey.CPTR_LINEAR_LAYER_WEIGHT: model_dict['head.weight']
        }))

        self.model.bias_transfer_from_dict(OrderedDict({
            CPTRModelBuilder.StateKey.CPTR_LINEAR_LAYER_BIAS: model_dict['head.bias']
        }))

