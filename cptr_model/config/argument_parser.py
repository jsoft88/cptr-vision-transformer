import argparse
from typing import Any
import sys


class ArgumentParser:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser('Neural Image Captioning')
        self.encoder_group = self.parser.add_argument_group('encoder group')
        self.decoder_group = self.parser.add_argument_group('decoder group')
        self.general_group = self.parser.add_argument_group('general')

    def parse_arguments(self, *args: str) -> Any:

        # Import here to avoid circular dependencies when accessing constants from factory
        import cptr_model.factory.utils.fs_factory as fsf
        import cptr_model.factory.data_modules.data_module_factory as dmf
        import cptr_model.factory.models.initializers_factory as mif

        self.encoder_group.add_argument(
            '--enc-norm-eps',
            help='epsilon value to use in normalization layer in encoder',
            dest='enc_norm_eps',
            default=1e-5,
            required=False
        )

        self.decoder_group.add_argument(
            '--dec-norm-eps',
            help='epsilon value to use in normalization layer in decoder',
            dest='dec_norm_eps',
            default=1e-5,
            required=False
        )

        self.general_group.add_argument(
            '--file-system',
            help=f'One of {",".join(fsf.FSFactory.all_types())}',
            default=None,
            dest='file_system',
            required=True
        )

        self.general_group.add_argument(
            '--fs-options',
            metavar='KEY=VALUE',
            default=None,
            dest='fs_options',
            nargs='+',
            required=False,
            help='KEY=VALUE list of options to pass to the file system manager. Separate each KEY=VALUE with spaces'
        )

        self.general_group.add_argument(
            '--pretrained-weights-path',
            default=None,
            dest='pretrained_weights_path',
            required=False,
            help='Absolute path with protocol (if required) to the stored model, for example, from checkpointing.'
        )

        self.general_group.add_argument(
            '--with-pretrained-weights',
            default=False,
            dest='with_pretrained_weights',
            action='store_true',
            help='When provided, the pretrained weights will be loaded into the model. '
        )

        self.general_group.add_argument(
            '--default-use-gpu',
            default=False,
            dest='default_use_gpu',
            action='store_true',
            help='By default, every tensor/layer will be create using GPU'
        )

        self.general_group.add_argument(
            '--model-save-file-system',
            default=None,
            dest='model_save_file_system',
            help=f'file system to use when saving model state. One of {",".join(fsf.FSFactory.all_types())}'
        )

        self.general_group.add_argument(
            '--model-save-fs-options',
            metavar='KEY=VALUE',
            default=None,
            dest='model_save_fs_options',
            nargs='+',
            required=False,
            help='KEY=VALUE list of options to pass to the file system manager. Separate each KEY=VALUE with spaces'
        )

        self.general_group.add_argument(
            '--model-save-path',
            dest='model_save_path',
            help='Absolute path where the model will be saved (with protocol included if any)',
            default=None,
            required=False
        )

        self.general_group.add_argument(
            '--tmp-dir',
            default='file:///tmp/cptr_staging_dir',
            help='Absolute path in local FS where temporary outputs are dumped. Default is /tmp/cptr_staging_dir',
            required=False,
            dest='tmp_dir'
        )

        self.general_group.add_argument(
            '--requires-model-init',
            dest='requires_model_init',
            action='store_true',
            default=False,
            help='Include this flag if initialization needs to be performed using an external pretrained model'
        )

        self.general_group.add_argument(
            '--model-init-path',
            dest='model_init_path',
            default=None,
            help='Absolute path (with protocol if required) to the model to use for initialization'
        )

        self.general_group.add_argument(
            '--model-initializer-type',
            dest='model_initializer_type',
            help=f'Type of model initializer to use. One of: {",".join(mif.InitializersFactory.all_types())}',
            default=None
        )
        
        self.general_group.add_argument(
            '--num-epochs',
            default=9,
            dest='num_epochs',
            help='Number of epochs to use during training'
        )

        self.general_group.add_argument(
            '--batch-size',
            default=1,
            help='Batch size to use per process',
            dest='batch_size',
            type=int
        )

        self.general_group.add_argument(
            '--lr-decay-after',
            dest='lr_decay_after',
            help='Learning rate decay after number of epochs. Default is after 1000 epochs',
            default=1000
        )

        self.general_group.add_argument(
            '--lr-decay-factor',
            default=0.5,
            dest='lr_decay_factor',
            help='The factor by which the LR is decayed after the specified --lr-decay-after. Default is 0.5'
        )

        self.general_group.add_argument(
            '--lr',
            dest='lr',
            help='Initial learning rate. Default is 3x10^-5',
            default=3e-5
        )

        self.general_group.add_argument(
            '--beam-search-size',
            default=3,
            dest='beam_search_size',
            help='Beam search size to use. Default is 3.'
        )

        self.general_group.add_argument(
            '--batch',
            dest='batch',
            action='store_true',
            default=False,
            help='When provided the dataset is of batch type as opposed to data incoming as stream'
        )

        self.general_group.add_argument(
            '--spark-master',
            default='local[*]',
            help='The master to use with spark when reading batch datasets. Default is local[*].',
            dest='spark_master'
        )

        self.general_group.add_argument(
            '--training',
            dest='training',
            help='Pass this flag when you want to enter training mode. If not included, predict mode is assumed.',
            action='store_true'
        )

        self.general_group.add_argument(
            '--Btrain-metadata-location',
            help='Absolute path to the location where the image metadata lives when using batch data reader',
            default=None,
            dest='bd_train_metadata_location',
            required=True
        )

        self.general_group.add_argument(
            '--Btrain-img-location',
            help='Absolute path to the location where images live when using batch data reader',
            default=None,
            dest='bd_train_img_location',
            required=True
        )

        self.general_group.add_argument(
            '--Bval-metadata-location',
            help='Absolute path to the location where the image metadata lives when using batch data reader',
            default=None,
            dest='bd_val_metadata_location',
            required=True
        )

        self.general_group.add_argument(
            '--Bval-img-location',
            help='Absolute path to the location where images live when using batch data reader',
            default=None,
            dest='bd_val_img_location',
            required=True
        )

        self.general_group.add_argument(
            '--Boptions',
            nargs='+',
            help='key=value [key=value...]. Provide relevant options for the batch data reader.',
            default=[],
            dest='bd_options'
        )

        self.general_group.add_argument(
            '--Shost',
            help='API Based specific arg: The host to send requests to',
            dest='sd_host',
            default=None
        )

        self.general_group.add_argument(
            '--Sport',
            help='API Based specific args: The port where the service is listening.',
            dest='sd_port',
            default=80
        )

        self.general_group.add_argument(
            '--Sprotocol',
            help='API Based specific arg: The protocol to use to connect to the API',
            default=None,
            dest='sd_protocol'
        )

        self.general_group.add_argument(
            '--input-reader-type',
            dest='input_reader_type',
            help=f'Type of input for the current execution mode. One of {",".join(dmf.DataModuleFactory.all_types())}',
            default=None
        )

        self.general_group.add_argument(
            '--prediction-writer-type',
            dest='prediction_writer_type',
            default=None,
            help='Type of writer for predictions in current execution mode'
        )

        # TODO: Consider adding args for handling different types of authentications.

        try:
            options = self.parser.parse_args(args)
            return options
        except:
            self.parser.print_usage()
            sys.exit(0)