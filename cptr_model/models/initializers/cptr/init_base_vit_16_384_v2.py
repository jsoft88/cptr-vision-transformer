from typing import Optional, OrderedDict
from transformers import EncoderDecoderModel
from transformers.models.vit.modeling_vit import ViTForImageClassification
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.models.initializers.base_initializer import BaseInitializer
from cptr_model.utils.utils import Utils


class BaseVit16384V2(BaseInitializer):
    KEY_NUMBER_ENCODER_LAYERS = 'ne'
    KEY_NUMBER_DECODER_LAYERS = 'nd'
    
    def __init__(self, config: Optional[Config],
                **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.number_encoder_layers = kwargs.get(BaseVit16384V2.KEY_NUMBER_ENCODER_LAYERS, None)
        self.number_decoder_layers = kwargs.get(BaseVit16384V2.KEY_NUMBER_DECODER_LAYERS, None)
        self.model = kwargs.get(BaseInitializer.KEY_MODEL, None)
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')
        # ViTForImageClassification('google/vit-base-patch16-384')
        self.vit.train(False)
        self.bert_encoder_decoder = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")

    def map_state_dict_to_model(self) -> None:
        if not self.model:
            raise ValueError('Missing model object to initialize.')
        
        if not self.number_encoder_layers:
            raise ValueError('Number of encoder layers is None')

        if not self.number_decoder_layers:
            raise ValueError('Number of decoder layers is None')

        # Import here to avoid circular dependencies while using global constants
        from cptr_model.embeddings.input.patch_embedding import PatchEmbedding
        from cptr_model.embeddings.input.word_embedding import WordEmbedding
        from cptr_model.models.cptr.attention.attention import Attention
        from cptr_model.models.cptr.cptr import CPTRModelBuilder
        from cptr_model.models.cptr.decoder.cptr_decoder_block import CPTRDecoderBlock
        from cptr_model.models.cptr.encoder.cptr_encoder_block import CPTREncoderBlock
        from cptr_model.models.cptr.mlp.mlp import MLP
        
        self.model.encoder_input_layer.weight_transfer_from_dict(OrderedDict({
            PatchEmbedding.StateKey.EMBEDDING_LAYER_WEIGHT: self.vit.vit.embeddings.patch_embeddings.projection.state_dict()['weight']
        }))
        self.model.encoder_input_layer.bias_transfer_from_dict(OrderedDict({
            PatchEmbedding.StateKey.EMBEDDING_LAYER_BIAS: self.vit.vit.embeddings.patch_embeddings.projection.state_dict()['bias']
        }))
        # self.model.decoder_input_layer.weight_transfer_from_dict(OrderedDict({
        #     WordEmbedding.StateKey.EMBEDDING_WEIGHT: self.bert.embeddings.word_embeddings.state_dict().get('weight')
        # }))
        
        for idx, enc_blk in enumerate(self.model.encoder_block_layers):
            enc_blk.weight_transfer_from_dict(OrderedDict({
                CPTREncoderBlock.StateKey.ATTENTION_NORM_WEIGHT: self.vit.vit.encoder.layer[idx].layernorm_before.weight,
                CPTREncoderBlock.StateKey.MLP_WEIGHT: OrderedDict({
                    MLP.StateKey.MLP_FC1_WEIGHT: self.vit.vit.encoder.layer[idx].intermediate.dense.weight,
                    MLP.StateKey.MLP_FC2_WEIGHT: self.vit.vit.encoder.layer[idx].output.dense.weight
                }),
                CPTREncoderBlock.StateKey.ATTENTION_WEIGHT: OrderedDict({
                    Attention.StateKey.ATTENTION_QUERY_WEIGHT: self.vit.vit.encoder.layer[idx].attention.attention.query.weight,
                    Attention.StateKey.ATTENTION_VALUE_WEIGHT: self.vit.vit.encoder.layer[idx].attention.attention.value.weight,
                    Attention.StateKey.ATTENTION_KEY_WEIGHT: self.vit.vit.encoder.layer[idx].attention.attention.key.weight,
                    Attention.StateKey.ATTENTION_OUT_WEIGHT: self.vit.vit.encoder.layer[idx].attention.output.dense.weight
                }),
                CPTREncoderBlock.StateKey.FFN_NORM_WEIGHT: self.vit.vit.encoder.layer[idx].layernorm_after.weight
            }))
            enc_blk.bias_transfer_from_dict(OrderedDict({
                CPTREncoderBlock.StateKey.ATTENTION_NORM_BIAS: self.vit.vit.encoder.layer[idx].layernorm_before.bias,
                CPTREncoderBlock.StateKey.MLP_BIAS: OrderedDict({
                    MLP.StateKey.MLP_FC1_BIAS: self.vit.vit.encoder.layer[idx].intermediate.dense.bias,
                    MLP.StateKey.MLP_FC2_BIAS: self.vit.vit.encoder.layer[idx].output.dense.bias
                }),
                CPTREncoderBlock.StateKey.ATTENTION_BIAS: OrderedDict({
                    Attention.StateKey.ATTENTION_QUERY_BIAS: self.vit.vit.encoder.layer[idx].attention.attention.query.bias,
                    Attention.StateKey.ATTENTION_KEY_BIAS: self.vit.vit.encoder.layer[idx].attention.attention.key.bias,
                    Attention.StateKey.ATTENTION_VALUE_BIAS: self.vit.vit.encoder.layer[idx].attention.attention.value.bias,
                    Attention.StateKey.ATTENTION_OUT_BIAS: self.vit.vit.encoder.layer[idx].attention.output.dense.bias
                }),
                CPTREncoderBlock.StateKey.FFN_NORM_BIAS: self.vit.vit.encoder.layer[idx].layernorm_after.bias
            }))

        for idx, dec_blk in enumerate(self.model.decoder_block_layers):
            dec_blk.weight_transfer_from_dict(OrderedDict({
                CPTRDecoderBlock.StateKey.MASKED_SELF_ATTENTION_NORMALIZATION_WEIGHT: self.bert_encoder_decoder.decoder.bert.encoder.layer[idx].attention.output.LayerNorm.weight,
                CPTRDecoderBlock.StateKey.MASKED_SELF_ATTENTION_WEIGHT: OrderedDict({
                    Attention.StateKey.ATTENTION_QUERY_WEIGHT: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].attention.self.query.weight,
                    Attention.StateKey.ATTENTION_VALUE_WEIGHT: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].attention.self.value.weight,
                    Attention.StateKey.ATTENTION_KEY_WEIGHT: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].attention.self.key.weight,
                    Attention.StateKey.ATTENTION_OUT_WEIGHT: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].attention.output.dense.weight
                }),
                CPTRDecoderBlock.StateKey.CROSS_ATTENTION_WEIGHT: OrderedDict({
                    Attention.StateKey.ATTENTION_QUERY_WEIGHT: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].crossattention.self.query.weight,
                    Attention.StateKey.ATTENTION_VALUE_WEIGHT: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].crossattention.self.value.weight,
                    Attention.StateKey.ATTENTION_KEY_WEIGHT: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].crossattention.self.key.weight,
                    Attention.StateKey.ATTENTION_OUT_WEIGHT: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].crossattention.output.dense.weight
                }),
                CPTRDecoderBlock.StateKey.CROSS_ATTENTION_NORMALIZATION_WEIGHT: self.bert_encoder_decoder.decoder.bert.encoder.layer[idx].crossattention.output.LayerNorm.weight,
                CPTRDecoderBlock.StateKey.FFN_WEIGHT: OrderedDict({
                    MLP.StateKey.MLP_FC1_WEIGHT: self.bert_encoder_decoder.decoder.bert.encoder.layer[idx].intermediate.dense.weight,
                    MLP.StateKey.MLP_FC2_WEIGHT: self.bert_encoder_decoder.decoder.bert.encoder.layer[idx].output.dense.weight
                }),
                CPTRDecoderBlock.StateKey.FFN_NORMALIZATION_WEIGHT: self.bert_encoder_decoder.decoder.bert.encoder.layer[idx].output.LayerNorm.weight
            }))
            dec_blk.bias_transfer_from_dict(OrderedDict({
                CPTRDecoderBlock.StateKey.MASKED_SELF_ATTENTION_NORMALIZATION_BIAS: self.bert_encoder_decoder.decoder.bert.encoder.layer[idx].attention.output.LayerNorm.bias,
                CPTRDecoderBlock.StateKey.MASKED_SELF_ATTENTION_BIAS: OrderedDict({
                    Attention.StateKey.ATTENTION_QUERY_BIAS: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].attention.self.query.bias,
                    Attention.StateKey.ATTENTION_VALUE_BIAS: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].attention.self.value.bias,
                    Attention.StateKey.ATTENTION_KEY_BIAS: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].attention.self.key.bias,
                    Attention.StateKey.ATTENTION_OUT_BIAS: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].attention.output.dense.bias
                }),
                CPTRDecoderBlock.StateKey.CROSS_ATTENTION_BIAS: OrderedDict({
                    Attention.StateKey.ATTENTION_QUERY_BIAS: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].crossattention.self.query.bias,
                    Attention.StateKey.ATTENTION_VALUE_BIAS: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].crossattention.self.value.bias,
                    Attention.StateKey.ATTENTION_KEY_BIAS: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].crossattention.self.key.bias,
                    Attention.StateKey.ATTENTION_OUT_BIAS: self.bert_encoder_decoder.decoder.bert.encoder.layer[0].crossattention.output.dense.bias
                }),
                CPTRDecoderBlock.StateKey.CROSS_ATTENTION_NORMALIZATION_BIAS: self.bert_encoder_decoder.decoder.bert.encoder.layer[idx].crossattention.output.LayerNorm.bias,
                CPTRDecoderBlock.StateKey.FFN_BIAS: OrderedDict({
                    MLP.StateKey.MLP_FC1_BIAS: self.bert_encoder_decoder.decoder.bert.encoder.layer[idx].intermediate.dense.bias,
                    MLP.StateKey.MLP_FC2_BIAS: self.bert_encoder_decoder.decoder.bert.encoder.layer[idx].output.dense.bias
                }),
                CPTRDecoderBlock.StateKey.FFN_NORMALIZATION_BIAS: self.bert_encoder_decoder.decoder.bert.encoder.layer[idx].output.LayerNorm.bias
            }))

        out_layer_1 = CPTRModelBuilder.StateKey.CPTR_OUT_WEIGHT.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, '0')
        out_layer_2 = CPTRModelBuilder.StateKey.CPTR_OUT_WEIGHT.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, '1')
        out_layer_3 = CPTRModelBuilder.StateKey.CPTR_OUT_WEIGHT.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, '2')
        self.model.weight_transfer_from_dict(OrderedDict({
            out_layer_1: self.bert_encoder_decoder.decoder.cls.predictions.transform.dense.weight,
            out_layer_2: self.bert_encoder_decoder.decoder.cls.predictions.transform.LayerNorm.weight,
            out_layer_3: self.bert_encoder_decoder.decoder.cls.predictions.decoder.weight
        }))

        out_layer_1 = CPTRModelBuilder.StateKey.CPTR_OUT_BIAS.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, '0')
        out_layer_2 = CPTRModelBuilder.StateKey.CPTR_OUT_BIAS.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, '1')
        out_layer_3 = CPTRModelBuilder.StateKey.CPTR_OUT_BIAS.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, '2')
         
        self.model.bias_transfer_from_dict(OrderedDict({
            out_layer_1: self.bert_encoder_decoder.decoder.cls.predictions.transform.dense.bias,
            out_layer_2: self.bert_encoder_decoder.decoder.cls.predictions.transform.LayerNorm.bias,
            out_layer_3: self.bert_encoder_decoder.decoder.cls.predictions.decoder.bias
        }))
