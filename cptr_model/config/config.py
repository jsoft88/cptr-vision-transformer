from typing import List
import argparse
from cptr_model.factory.data_loaders.data_loader_factory import DataLoaderFactory
from cptr_model.factory.input_embeddings.embedding_factory import EmbeddingFactory
from cptr_model.factory.positional_embeddings.positional_embedding_factory import PositionalEmbeddingFactory


class Config:

    def __init__(self, args: List[str]) -> None:
        parser = argparse.ArgumentParser('Neural Image Captioning')
        encoder_group = parser.add_argument_group('encoder group')
        decoder_group = parser.add_argument_group('decoder group')
        general_group = parser.add_argument_group('general')

        encoder_group.add_argument(
            '--enc-input-embedding-dim',
            help='Dimension of image patches embeddings',
            dest='enc_input_embedding_dim',
            default=None,
            required=True
        )

        encoder_group.add_argument(
            '--enc-input-embedding-type',
            help=f'Type of embedding to use for patches. One of {",".join(EmbeddingFactory.all_types())}',
            dest='enc_input_embedding_type',
            default=None,
            required=True
        )

        encoder_group.add_argument(
            '--enc-input-embedding-config-section',
            help='The key used in the config file to get configuration section for the type of input embedding',
            default=None,
            dest='enc_input_embedding_config_section',
            required=True
        )

        encoder_group.add_argument(
            '--num-encoder-blocks',
            help='Number of encoding blocks to use',
            dest='num_encoder_blocks',
            default=None,
            required=True
        )

        encoder_group.add_argument(
            '--num-attention-heads',
            help='Number of attention heads to use',
            dest='num_attention_heads',
            default=None,
            required=True
        )

        encoder_group.add_argument(
            '--enc-position-embedding-type',
            help=f'''
            Type of position embedding to use in encoder.
            One of {",".join(PositionalEmbeddingFactory.all_types())}''',
            dest='enc_position_embedding_type',
            default=None,
            required=True
        )

        encoder_group.add_argument(
            '--enc-position-embedding-config-section',
            help='The key used in the config file to get configuration section for the type of position embedding',
            dest='enc_position_embedding_config_section',
            default=None,
            required=True
        )

        encoder_group.add_argument(
            '--enc-data-loader-type',
            help=f'Type of data loader to use to load data to encoder. One of {DataLoaderFactory.all_types()}',
            dest='enc_data_loader_type',
            default=None,
            required=True
        )

        encoder_group.add_argument(
            '--enc-data-location',
            help='Absolute path to the location where the encoder data lives',
            default=None,
            dest='enc_data_location',
            required=True
        )

        encoder_group.add_argument(
            '--enc-transformation-types',
            nargs='+',
            help='List of transformations to apply to input data',
            dest='enc_transformation_types',
            default=None,
            required=True
        )

        encoder_group.add_argument(
            '--enc-batch-size',
            help='Batch size to use in encoder',
            dest='enc_batch_size',
            default=None,
            required=True
        )

        decoder_group.add_argument(
            '--dec-input-embedding-dim',
            help='Dimension of word embeddings',
            dest='dec_input_embedding_dim',
            default=None,
            required=True
        )

        decoder_group.add_argument(
            '--dec-input-embedding-type',
            help=f'Type of word embedding to use. One of {",".join(EmbeddingFactory.all_types())}',
            dest='dec_input_embedding_type',
            default=None,
            required=True
        )

        decoder_group.add_argument(
            '--dec-input-embedding-config-section',
            help=f'Key used in config file to get configuration section for the input embedding type',
            dest='dec_input_embedding_config_section',
            default=None,
            required=True
        )

        decoder_group.add_argument(
            '--dec-position-embedding-type',
            help=f'''
            Type of position embedding to use in the decoder. 
            One of {",".join(PositionalEmbeddingFactory.all_types())}''',
            dest='dec_position_embedding_type',
            default=None,
            required=True
        )

        decoder_group.add_argument(
            '--dec-position-embedding-config-section',
            help='Key used in config file to get configuration section for the position embedding type',
            default=None,
            required=True,
            dest='dec_position_embedding_config_section'
        )

        decoder_group.add_argument(
            '--dec-data-loader-type',
            help=f'Type of data loader to use to load data to decoder. One of {DataLoaderFactory.all_types()}',
            dest='dec_data_loader_type',
            default=None,
            required=True
        )

        decoder_group.add_argument(
            '--dec-transformation-types',
            nargs='+',
            help='List of transformations to apply to input data',
            dest='dec_transformation_types',
            default=None,
            required=True
        )

        decoder_group.add_argument(
            '--dec-data-location',
            help='Absolute path to the location where the decoder data lives',
            default=None,
            dest='dec_data_location',
            required=True
        )

        decoder_group.add_argument(
            '--dec-batch-size',
            help='Batch size to use in decoder',
            default=None,
            dest='dec_batch_size',
            required=True
        )

        general_group.add_argument(
            '--model-output-location',
            help='Absolute path to the location where the trained model will be placed',
            dest='model_output_location',
            default=None,
            required=True
        )

        general_group.add_argument(
            '--pretrained-model-location',
            help='Absolute path to the location of a pretrained model to load',
            dest='pretrained_model_location',
            default=None,
            required=False
        )

        general_group.add_argument(
            '--config-file',
            help='Name of the config json file to use',
            default=None,
            dest='config_file',
            required=True
        )

        parsed_args = parser.parse_args(args)
        self.encoder_position_embedding_type = parsed_args.enc_position_embedding_type
        self.encoder_input_embedding_dim = parsed_args.enc_input_embedding_dim
        self.encoder_input_embedding_type = parsed_args.enc_input_embedding_type
        self.encoder_data_loader_type = parsed_args.enc_data_loader_type
        self.num_encoder_blocks = parsed_args.num_encoder_blocks
        self.encoder_transformation_types = parsed_args.enc_transformation_types
        self.encoder_batch_size = parsed_args.enc_batch_size
        self.encoder_data_location = parsed_args.enc_data_location
        self.encoder_input_embedding_config_section = parsed_args.enc_input_embedding_config_section
        self.encoder_position_embedding_config_section = parsed_args.enc_position_embedding_config_section
        self.decoder_position_embedding_type = parsed_args.dec_position_embedding_type
        self.decoder_input_embedding_dim = parsed_args.dec_input_embedding_dim
        self.decoder_input_embedding_type = parsed_args.dec_input_embedding_type
        self.decoder_data_loader_type = parsed_args.dec_data_loader_type
        self.num_decoder_blocks = parsed_args.num_decoder_blocks
        self.decoder_transformation_types = parsed_args.dec_transformation_types
        self.decoder_batch_size = parsed_args.dec_batch_size
        self.decoder_data_location = parsed_args.dec_data_location
        self.decoder_position_embedding_config_section = parsed_args.dec_position_embedding_config_section
        self.decoder_input_embedding_config_section = parsed_args.dec_input_embedding_config_section
        self.config_file = parsed_args.config_file

