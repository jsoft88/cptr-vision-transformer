import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Union, List, Dict
from cptr_model.utils.utils import Utils


class ArchitectureConfigFileManager:
    def __init__(self) -> None:
        self.encoder_input_embedding_type = None
        self.decoder_input_embedding_type = None
        self.encoder_input_embeddings_params_dict = None
        self.patch_embedding_dim = None
        self.patch_embedding_channel_in = None
        self.patch_embedding_channel_out = None
        self.patch_embedding_height = None
        self.patch_embedding_width = None
        self.patch_embedding_kernel_size = None
        self.patch_embedding_stride = None
        self.patch_embedding_padding = None
        self.encoder_position_embedding_type = None
        # It is more useful to delegate params parsing to the position embedder
        # for more flexibility to replace during runtime
        self.encoder_position_embedding_params_dict = None
        self.encoder_self_attention_dim = None
        self.encoder_self_attention_heads = None
        self.encoder_self_attention_mlp_dim = None
        self.encoder_self_attention_mlp_dropout = None
        self.encoder_num_blocks = None
        self.decoder_input_embeddings_params_dict = None
        self.decoder_word_embedding_dim = None
        self.decoder_word_embedding_num_positions = None
        # It is more useful to delegate params parsing to the position embedder
        # for more flexibility to replace during runtime
        self.decoder_position_embedding_params_dict = None
        self.decoder_position_embedding_type = None
        self.decoder_masked_self_attention_dim = None
        self.decoder_masked_self_attention_heads = None
        self.decoder_masked_self_attention_mlp_dropout = None
        self.decoder_masked_self_attention_mlp_dim = None
        self.decoder_cross_attention_dim = None
        self.decoder_cross_attention_heads = None
        self.decoder_cross_attention_mlp_dropout = None
        self.decoder_cross_attention_mlp_dim = None
        self.decoder_num_blocks = None
        self.decoder_max_seq_len = None

    class Model:
        class EncoderBlock:
            NAME = 'encoder_block'

            class EncoderInputLayer:
                NAME = 'input'
                SUBLAYER_PATCH_EMBEDDING = 'patch_embedding'
                SUBLAYER_POSITION_EMBEDDING = 'encoder_position_embedding'

            class EncoderAttentionLayer:
                NAME = 'attention_layer'
                SUBLAYER_ATTENTION = 'attention'

        class DecoderBlock:
            NAME = 'decoder_block'

            class DecoderInputLayer:
                NAME = 'input'
                SUBLAYER_WORD_EMBEDDING = 'word_embedding'
                SUBLAYER_DECODER_POSITION_EMBEDDING = 'decoder_position_embedding'

            class DecoderAttentionLayer:
                NAME = 'attention_layer'
                SUBLAYER_ATTENTION = 'attention'

        class AttentionParams:
            DIM = 'dim'
            HEADS = 'heads'
            MLP_DIM = 'mlp_dim'
            MLP_DROPOUT = 'mlp_dropout'
            NE = 'Ne',
            ND = 'Nd'

        class PatchEmbeddingParams:
            DIM = 'dim'
            CHANNELS_IN = 'channels_in'
            HEIGHT = 'height'
            WIDTH = 'width'
            STRIDE = 'stride'
            KERNEL_SIZE = 'kernel_size'
            CHANNELS_OUT = 'channels_out'

        class EncoderPositionEmbeddingParams:
            TYPE = 'type'
            DIM = 'dim'

        class WordEmbeddingParams:
            DIM = 'dim'
            NUM_POSITIONS = 'num_positions'
            VOCAB_SIZE = 'vocab_size'

        class DecoderPositionEmbeddingParams:
            TYPE = 'type'
            DIM = 'dim'

        class Transformations:
            APPLY_TO = 'apply_to'
            TRANSFORMS = 'transforms'

            class Transform:
                TYPE = 'type'
                EXTRA_ARGS = 'extra_args'

            class ApplyTo:
                IMAGE = 'image'
                TENSOR = 'tensor'


