from typing import Any, Dict, List
from hydra import compose, initialize
from omegaconf import OmegaConf
import torch.cuda
from cptr_model.config.argument_parser import ArgumentParser
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.config.specifics_mixin import SpecificsMixin


class Config(SpecificsMixin):
    DYN_LNKR_BATCH_SIZE = 'batch-size'

    def __init__(self, args: List[str]) -> None:
        # TODO: Consider adding args for handling different types of authentications.

        parsed_args = ArgumentParser().parse_arguments(*args)

        self.encoder_transformation_types = parsed_args.enc_transformation_types
        self.encoder_normalization_eps = parsed_args.enc_norm_eps
        
        self.decoder_transformation_types = parsed_args.dec_transformation_types
        self.decoder_normalization_eps = parsed_args.dec_norm_eps

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
        self.num_epochs = parsed_args.num_epochs
        self.lr_decay_after_num_epochs = parsed_args.lr_decay_after
        self.lr_decay_factor = parsed_args.lr_decay_factor
        self.lr = parsed_args.lr
        self.beam_search_size = parsed_args.beam_search_size
        self.batch = parsed_args.batch
        self.spark_master = parsed_args.spark_master
        self.batch_size = parsed_args.batch_size
        self.training = parsed_args.training
        self.batch_train_metadata_location = parsed_args.bd_train_metadata_location
        self.batch_train_img_location = parsed_args.bd_train_img_location
        self.batch_val_metadata_location = parsed_args.bd_val_metadata_location
        self.batch_val_img_location = parsed_args.bd_val_img_location
        self.batch_data_options = dict([entry.split('=') for entry in (parsed_args.bd_options or [])])
        self.api_based_options_host = parsed_args.sd_host
        self.api_based_options_port = parsed_args.sd_port
        self.api_based_options_protocol = parsed_args.sd_protocol
        self.input_reader_type = parsed_args.input_reader_type
        self.prediction_writer_type = parsed_args.prediction_writer_type

        self.__LINKERS = self.register_dynamic_linkers()
        # if self.training and self.input_reader_type ==]
        with initialize(config_path='../../resources/config', job_name='cptr'):
            model_config = compose(config_name='model_config', overrides=[])
            self.model_config = model_config
        
        self.cptr_specifics = self.load_specifics_from_file()

    def register_dynamic_linkers(self) -> Dict[str, Any]:
        return {
            Config.DYN_LNKR_BATCH_SIZE: int(self.batch_size)
        }

    def load_specifics_from_file(self) -> Any:
        if not OmegaConf.has_resolver('dynamic_linker'):
            OmegaConf.register_new_resolver("dynamic_linker", lambda x: self.__LINKERS.get(x, None))
        cptr_architecture_config = ArchitectureConfigFileManager()

        cptr_architecture_config.encoder_input_embedding_type = \
            self.model_config.model.cptr.architecture.blocks[0].encoder.layers[0].input.sublayers[0].patch_embedding.params[8].type

        cptr_architecture_config.encoder_position_embedding_type = \
            self.model_config.model.cptr.architecture.blocks[0].encoder.layers[0].input.sublayers[1].encoder_position_embedding.params[0].type
        
        cptr_architecture_config.encoder_input_embeddings_params_dict = \
            dict([
                p.items()[0] for p in
                self.model_config.model.cptr.architecture.blocks[0].encoder.layers[0].input.sublayers[0].patch_embedding.params
            ])

        cptr_architecture_config.encoder_position_embedding_params_dict = \
            dict([
                p.items()[0] for p in
                self.model_config.model.cptr.architecture.blocks[0].encoder.layers[0].input.sublayers[1].encoder_position_embedding.params
            ])

        cptr_architecture_config.encoder_position_embedding_num_positions = \
            self.model_config.model.cptr.architecture.blocks[0].encoder.layers[0].input.sublayers[1].encoder_position_embedding.params[2].num_positions

        cptr_architecture_config.encoder_self_attention_dim = \
            self.model_config.model.cptr.architecture.blocks[0].encoder.layers[1].encoder_block.sublayers[0].self_attention.params[0].dim

        cptr_architecture_config.encoder_self_attention_heads = \
            self.model_config.model.cptr.architecture.blocks[0].encoder.layers[1].encoder_block.sublayers[0].self_attention.params[1].heads

        cptr_architecture_config.encoder_self_attention_mlp_dropout = \
            self.model_config.model.cptr.architecture.blocks[0].encoder.layers[1].encoder_block.sublayers[0].self_attention.params[2].mlp_dropout

        cptr_architecture_config.encoder_self_attention_mlp_dim = \
            self.model_config.model.cptr.architecture.blocks[0].encoder.layers[1].encoder_block.sublayers[0].self_attention.params[3].mlp_dim

        cptr_architecture_config.encoder_num_blocks = \
            self.model_config.model.cptr.architecture.blocks[0].encoder.layers[1].encoder_block.num_blocks

        cptr_architecture_config.decoder_input_embedding_type = \
            self.model_config.model.cptr.architecture.blocks[1].decoder.layers[0].input.sublayers[0].word_embeddings.params[2].type

        cptr_architecture_config.decoder_position_embedding_type = \
            self.model_config.model.cptr.architecture.blocks[1].decoder.layers[0].input.sublayers[1].decoder_position_embedding.params[0].type

        cptr_architecture_config.decoder_position_embedding_params_dict = \
            dict([
                p.items()[0] for p in 
                self.model_config.model.cptr.architecture.blocks[1].decoder.layers[0].input.sublayers[1].decoder_position_embedding.params
            ])

        cptr_architecture_config.decoder_input_embeddings_params_dict = \
            dict([
                p.items()[0] for p in
                self.model_config.model.cptr.architecture.blocks[1].decoder.layers[0].input.sublayers[0].word_embeddings.params
            ])
        
        cptr_architecture_config.decoder_masked_self_attention_dim = \
            self.model_config.model.cptr.architecture.blocks[1].decoder.layers[1].decoder_block.sublayers[0].masked_self_attention.params[0].dim

        cptr_architecture_config.decoder_masked_self_attention_heads = \
            self.model_config.model.cptr.architecture.blocks[1].decoder.layers[1].decoder_block.sublayers[0].masked_self_attention.params[1].heads

        cptr_architecture_config.decoder_masked_self_attention_mlp_dropout = \
            self.model_config.model.cptr.architecture.blocks[1].decoder.layers[1].decoder_block.sublayers[0].masked_self_attention.params[2].mlp_dropout

        cptr_architecture_config.decoder_masked_self_attention_mlp_dim = \
            self.model_config.model.cptr.architecture.blocks[1].decoder.layers[1].decoder_block.sublayers[0].masked_self_attention.params[3].mlp_dim

        cptr_architecture_config.decoder_cross_attention_dim = \
            self.model_config.model.cptr.architecture.blocks[1].decoder.layers[1].decoder_block.sublayers[1].cross_attention.params[0].dim

        cptr_architecture_config.decoder_cross_attention_heads = \
            self.model_config.model.cptr.architecture.blocks[1].decoder.layers[1].decoder_block.sublayers[1].cross_attention.params[1].heads

        cptr_architecture_config.decoder_cross_attention_mlp_dropout = \
            self.model_config.model.cptr.architecture.blocks[1].decoder.layers[1].decoder_block.sublayers[1].cross_attention.params[2].mlp_dropout

        cptr_architecture_config.decoder_cross_attention_mlp_dim = \
            self.model_config.model.cptr.architecture.blocks[1].decoder.layers[1].decoder_block.sublayers[1].cross_attention.params[3].mlp_dim

        cptr_architecture_config.decoder_num_blocks = \
            self.model_config.model.cptr.architecture.blocks[1].decoder.layers[1].decoder_block.num_blocks
        
        cptr_architecture_config.decoder_max_seq_len = \
            cptr_architecture_config.decoder_position_embedding_params_dict['num_positions']
        return cptr_architecture_config
