import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Union, List, Dict

from cptr_model.utils.utils import Utils


class ArchitectureConfigFileManager:
    def __init__(self, file_name: str) -> None:
        root = Path(__file__).parent.parent.parent.resolve()
        self.path = Path(root).joinpath("resources", "config", file_name)
        self.config_json = Utils.read_json_from_file(self.path)
        self.config_object = json.loads(self.config_json, object_hook=lambda d: SimpleNamespace(**d))

    def get_blocks(self) -> List[Any]:
        return [b for b in self.config_object.model.blocks]

    def get_layers_of(self, block: Any) -> List[Any]:
        return [l for l in block.layers]

    def get_sublayers_of(self, layer: Any) -> List[Any]:
        return [sl for sl in layer.sublayers]

    @classmethod
    def get_block_name(cls, block: Any) -> str:
        return block.name

    @classmethod
    def get_layer_name(cls, layer: Any) -> str:
        return layer.name

    @classmethod
    def get_sublayer_name(cls, sublayer: Any) -> str:
        return sublayer.name

    @classmethod
    def get_params_for_sublayer(cls, sublayer: Any) -> Dict[str, Union[str, float, List[Any], int]]:
        return {p.name: p.value for p in sublayer.params}

    def get_sublayer_with_query(self, block_name: str, layer_name: str, sublayer_name: str):
        layers = eval(f'self.config_object.{block_name}')
        layer = [l for l in layers if ArchitectureConfigFileManager.get_layer_name(l) == layer_name][0]
        sublayers = self.get_sublayers_of(layer)
        return [sl for sl in sublayers if ArchitectureConfigFileManager.get_sublayer_name(sl) == sublayer_name][0]

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


