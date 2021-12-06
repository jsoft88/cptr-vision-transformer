from typing import List
import argparse
import torch.cuda
from cptr_model.factory.data_loaders.data_loader_factory import DataLoaderFactory
from cptr_model.factory.input_embeddings.embedding_factory import EmbeddingFactory
from cptr_model.factory.models.initializers_factory import InitializersFactory
from cptr_model.factory.positional_embeddings.positional_embedding_factory import PositionalEmbeddingFactory
from cptr_model.factory.utils.fs_factory import FSFactory


class Config:

    def __init__(self, args: List[str]) -> None:
        parser = argparse.ArgumentParser('Neural Image Captioning')
        encoder_group = parser.add_argument_group('encoder group')
        decoder_group = parser.add_argument_group('decoder group')
        general_group = parser.add_argument_group('general')

        encoder_group.add_argument(
            '--enc-input-embedding-type',
            help=f'Type of embedding to use for patches. One of {",".join(EmbeddingFactory.all_types())}',
            dest='enc_input_embedding_type',
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

        encoder_group.add_argument(
            '--enc-norm-eps',
            help='epsilon value to use in normalization layer in encoder',
            dest='enc_norm_eps',
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
            '--dec-position-embedding-type',
            help=f'''
            Type of position embedding to use in the decoder. 
            One of {",".join(PositionalEmbeddingFactory.all_types())}''',
            dest='dec_position_embedding_type',
            default=None,
            required=True
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

        decoder_group.add_argument(
            '--dec-norm-eps',
            help='epsilon value to use in normalization layer in decoder',
            dest='dec_norm_eps',
            default=None,
            required=True
        )

        decoder_group.add_argument(
            '--dec-lang',
            help='Decoder language to use',
            choices=['en', 'es', 'it'],
            default='en',
            dest='dec_lang'
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

        general_group.add_argument(
            '--file-system',
            help=f'One of {",".join(FSFactory.all_types())}',
            default=None,
            dest='file_system',
            required=True
        )

        general_group.add_argument(
            'fs-options',
            metavar='KEY=VALUE',
            default=None,
            dest='fs_options',
            nargs='+',
            required=False,
            help='KEY=VALUE list of options to pass to the file system manager. Separate each KEY=VALUE with spaces'
        )

        general_group.add_argument(
            '--pretrained-weights-path',
            default=None,
            dest='pretrained_weights_path',
            required=False,
            help='Absolute path with protocol (if required) to the stored model'
        )

        general_group.add_argument(
            '--with-pretrained-weights',
            default=False,
            dest='with_pretrained_weights',
            action='store_true',
            help='When provided, the pretrained weights will be loaded into the model. '
        )

        general_group.add_argument(
            '--default-use-gpu',
            default=False,
            dest='default_use_gpu',
            action='store_true',
            help='By default, every tensor/layer will be create using GPU'
        )

        general_group.add_argument(
            '--model-save-file-system',
            default=None,
            dest='model_save_file_system',
            help=f'file system to use when saving model state. One of {",".join(FSFactory.all_types())}'
        )

        general_group.add_argument(
            '--model-save-fs-options',
            metavar='KEY=VALUE',
            default=None,
            dest='model_save_fs_options',
            nargs='+',
            required=False,
            help='KEY=VALUE list of options to pass to the file system manager. Separate each KEY=VALUE with spaces'
        )

        general_group.add_argument(
            '--model-save-path',
            dest='model_save_path',
            help='Absolute path where the model will be saved (with protocol included if any)',
            default=None,
            required=False
        )

        general_group.add_argument(
            '--tmp-dir',
            default='/tmp/cptr_staging_dir',
            help='Absolute path in local FS where temporary outputs are dumped. Default is /tmp/cptr_staging_dir',
            required=False,
            dest='tmp_dir'
        )

        general_group.add_argument(
            '--requires-model-init',
            dest='requires_model_init',
            action='store_true',
            default=False,
            help='Include this flag if initialization needs to be performed using an external pretrained model'
        )

        general_group.add_argument(
            '--model-init-path',
            dest='model_init_path',
            default=None,
            help='Absolute path (with protocol if required) to the model to use for initialization'
        )

        general_group.add_argument(
            '--model-initializer-type',
            dest='model_initializer_type',
            help=f'Type of model initializer to use. One of: {",".join(InitializersFactory.all_types())}',
            default=None
        )

        parsed_args = parser.parse_args(args)
        self.encoder_position_embedding_type = parsed_args.enc_position_embedding_type

        self.encoder_input_embedding_type = parsed_args.enc_input_embedding_type
        self.encoder_data_loader_type = parsed_args.enc_data_loader_type

        self.encoder_transformation_types = parsed_args.enc_transformation_types
        self.encoder_batch_size = parsed_args.enc_batch_size
        self.encoder_data_location = parsed_args.enc_data_location


        self.encoder_normalization_eps = parsed_args.enc_norm_eps
        self.decoder_position_embedding_type = parsed_args.dec_position_embedding_type

        self.decoder_input_embedding_type = parsed_args.dec_input_embedding_type
        self.decoder_data_loader_type = parsed_args.dec_data_loader_type

        self.decoder_transformation_types = parsed_args.dec_transformation_types
        self.decoder_batch_size = parsed_args.dec_batch_size
        self.decoder_data_location = parsed_args.dec_data_location
        self.dec_lang = parsed_args.dec_lang

        self.decoder_normalization_eps = parsed_args.dec_norm_eps
        self.config_file = parsed_args.config_file
        self.file_system_type = parsed_args.file_system
        self.file_system_options = dict([entry.split('=') for entry in (parsed_args.fs_options or [])])
        self.pretrained_model_path = parsed_args.pretrained_weights_path
        self.with_pretrained_model = False if not self.pretrained_model_path else parsed_args.with_pretrained_weights
        self.default_use_gpu = False if not torch.cuda.is_available() else parsed_args.default_use_gpu
        self.device = torch.device('cuda' if self.default_use_gpu else 'cpu')
        self.model_save_file_system = parsed_args.model_save_file_system
        self.model_save_fs_options = dict([entry.split('=') for entry in (parsed_args.model_save_fs_options or [])])
        self.model_save_path = parsed_args.model_save_path
        self.tmp_dir = parsed_args.tmp_dir
        self.requires_model_init = parsed_args.requires_model_init
        self.model_init_path = parsed_args.model_init_path
        self.model_initializer_type = parsed_args.model_initializer_type



