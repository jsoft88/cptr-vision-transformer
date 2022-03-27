from typing import Dict, List, Optional, OrderedDict, Any, Tuple
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn.modules.activation import Softmax
from torch.nn.modules.container import ModuleList
from torch.nn.modules.linear import Linear
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.modules.normalization import LayerNorm
from torch.nn.parameter import Parameter
from transformers import BertTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from cptr_model.embeddings.position.base_position_embedding import BasePositionEmbedding
from cptr_model.factory.positional_embeddings.positional_embedding_factory import PositionalEmbeddingFactory
from cptr_model.models.initializers.cptr.init_base_vit_16_384_v2 import BaseVit16384V2
from cptr_model.utils.inference_helpers.provider_dependency_injection.fetcher_dependency_mixin import FetcherDependencyMixin
from cptr_model.utils.inference_helpers.provider_dependency_injection.writer_dependency_mixin import WriterDependencyMixin
from pytorch_lightning import LightningModule
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.core.core_module_extension import CoreModuleExtension
from cptr_model.factory.input_embeddings.embedding_factory import EmbeddingFactory
from cptr_model.factory.models.initializers_factory import InitializersFactory
from cptr_model.factory.utils.fs_factory import FSFactory
from cptr_model.models.base.base_model import ModelBuilder
from cptr_model.models.cptr.decoder.cptr_decoder_block import CPTRDecoderBlock
from cptr_model.models.cptr.encoder.cptr_encoder_block import CPTREncoderBlock
from cptr_model.models.initializers.base_initializer import BaseInitializer
from cptr_model.utils.utils import Utils


class CPTRModelBuilder(ModelBuilder, CoreModuleExtension, FetcherDependencyMixin, WriterDependencyMixin):
    def __init__(self, config: Config, **kwargs):
        if not config:
            raise ValueError('config object is None')
        
        self.config: Config = config
        self.model_config: ArchitectureConfigFileManager = self.config.cptr_specifics

        self.image_patch_embeddings_dim = self.model_config.patch_embedding_dim
        self.num_encoder_blocks = self.model_config.encoder_num_blocks
        self.num_decoder_blocks = self.model_config.decoder_num_blocks
        self.word_embeddings_dim = self.model_config.decoder_word_embedding_dim
        self.patch_embedding_kernel = self.model_config.patch_embedding_kernel_size
        self.patch_embedding_stride = self.model_config.patch_embedding_stride
        self.patch_embedding_channels_in = self.model_config.patch_embedding_channel_in
        self.patch_embedding_channels_out = self.model_config.patch_embedding_channel_out
        self.patch_position_embedding_type = self.model_config.encoder_position_embedding_type
        self.word_position_embedding_type = self.model_config.decoder_position_embedding_type
        self.lr_decay_factor = self.config.lr_decay_factor
        self.lr = self.config.lr
        self.lr_decay_after_num_epochs = self.config.lr_decay_after_num_epochs

        super().__init__(config)
        # model building blocks
        self.encoder_input_layer = None
        self.encoder_input_position_embeddings = None
        self.decoder_input_position_embeddings = None
        self.encoder_block_layers = None
        self.decoder_input_layer = None
        self.decoder_block_layers = None
        self.out = None
        self.softmax = None

        self.loss = CrossEntropyLoss()
        self.bert_config = BertConfig()
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.pad_tok_id = self.bert_tokenizer.pad_token_id
        self.vocab_size = self.bert_config.vocab_size

        # self.register_buffer('init_seq', torch.LongTensor([[self.bos_tok_id]]))
        # self.register_buffer('blank_seqs', torch.full((self.config.beam_search_size, self.config.max_seq_len), self.pad_tok_id, dtype=torch.long))

    def _verify_required_args(self) -> None:
        if not self.num_encoder_blocks:
            raise ValueError('Number of encoding blocks is None')
        if not self.num_decoder_blocks:
            raise ValueError('Number of decoding blocks is None')
        if not self.num_decoder_blocks:
            raise ValueError('Number of decoder blocks is None')
        if not self.num_encoder_blocks:
            raise ValueError('Number of encoder blocks is None')

    def _building_model_blocks(self) -> LightningModule:
        self.encoder_input_layer = EmbeddingFactory.get_instance(
            self.model_config.encoder_input_embedding_type,
            config=self.config,
            **self.model_config.encoder_input_embeddings_params_dict
        )

        self.decoder_input_layer = EmbeddingFactory.get_instance(
            self.model_config.decoder_input_embedding_type,
            self.config,
            **self.model_config.decoder_input_embeddings_params_dict
        )

        # self.decoder_input_layer not longer required since dataloader returns bert embeds

        self.encoder_input_position_embeddings = PositionalEmbeddingFactory.get_instance(
            self.patch_position_embedding_type,
            self.config,
            **self.model_config.encoder_position_embedding_params_dict
        )

        self.encoder_block_layers = ModuleList([
            CPTREncoderBlock(self.config) for _ in range(self.num_encoder_blocks)
        ])

        self.decoder_input_position_embeddings = PositionalEmbeddingFactory.get_instance(
            self.word_position_embedding_type,
            self.config,
            **self.model_config.decoder_position_embedding_params_dict
        )

        self.decoder_block_layers = ModuleList([
            CPTRDecoderBlock(self.config) for _ in range(self.num_decoder_blocks)
        ])

        # Based on bert pretrained encoder/decoder params
        self.out = ModuleList([
            Linear(self.model_config.decoder_input_embeddings_params_dict['dim'], self.model_config.decoder_input_embeddings_params_dict['dim']),
            LayerNorm(self.model_config.decoder_input_embeddings_params_dict['dim'], eps=1e-12, elementwise_affine=True),
            Linear(self.model_config.decoder_input_embeddings_params_dict['dim'], self.vocab_size, bias=True)
        ])
        self.softmax = Softmax(dim=-1)

        return self

    def _encoder_forward(self, x_e: torch.Tensor) -> torch.Tensor:
        x_e = self.encoder_input_layer(x_e)
        x_e = self.encoder_input_position_embeddings.forward(x_e)
        for enc_layer in self.encoder_block_layers:
            x_e = enc_layer(x_e)

        return x_e

    def _decoder_forward(self, x_e: torch.Tensor, captions_ids: torch.Tensor, pad_mask: torch.Tensor, lookahead_mask: torch.Tensor) -> torch.Tensor:
        x_d = self.decoder_input_layer(captions_ids)
        x_d = self.decoder_input_position_embeddings(x_d)

        for dec_layer in self.decoder_block_layers:
            x_d = dec_layer(x_d, x_e, lookahead_mask, pad_mask)
            
        for out_layer in self.out:
            x_d = out_layer(x_d)

        # return logits
        return x_d.view(-1, x_d.size(2)) if self.config.training else x_d
        
    def forward(self, x_e: torch.Tensor, captions_ids: torch.Tensor, seq_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        x_e = self._encoder_forward(x_e)
        pad_mask = self._get_pad_mask(seq_ids)
        lookahead_mask = self._get_lookahead_mask(seq_ids)
        x = self._decoder_forward(x_e, captions_ids, pad_mask, lookahead_mask)

        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_e, captions_ids, y = batch
        logits = self.forward(x_e, captions_ids, y) # logits shape (N*NUM_WORDS, VOCAB_SIZE)
        _, logits_idx = logits.max(1)
        y = y.contiguous().view(-1)
        loss = self.loss(logits_idx, y)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = Adam(self.parameters(), lr=self.lr)
        lr_scheduler = StepLR(optimizer, step_size=self.lr_decay_after_num_epochs, gamma=self.lr_decay_factor)
        lr_scheduler_config = {
            'scheduler': lr_scheduler,
            'interval': 'epoch',
            'frequency': self.lr_decay_after_num_epochs
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler_config
        }

    def _get_pad_mask(self, seq: torch.Tensor) -> torch.Tensor:
        pad_mask = (seq == self.pad_tok_id).bool()
        return pad_mask.unsqueeze(1)

    def _get_lookahead_mask(self, seq: torch.Tensor) -> torch.Tensor:
        _, len_seq = seq.shape
        lookahead_mask = (1 - torch.triu(torch.ones((len_seq, len_seq), device=self.config.device), diagonal=1)).bool()

        return lookahead_mask

    def _predict_initial_step(self,
                            x_e: torch.Tensor, # [BATCH_SIZE x PATCH_NUM x P^2*C]
                            init_seq_ids: torch.Tensor, # [BATCH_SIZE x MAX_LEN]
                            pad_mask: torch.Tensor, 
                            lookahead_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            
            enc_output = self._encoder_forward(x_e)
            dec_output = self._predict_model_decode(enc_output, init_seq_ids, pad_mask, lookahead_mask)
            dec_output = dec_output.unsqueeze(1)
            init_seq_ids = init_seq_ids.unsqueeze(1)
            enc_output = enc_output.unsqueeze(1)
            best_k_probs, best_k_idx = dec_output[:, :, 1, :].topk(int(self.config.beam_search_size))

            scores = torch.log(best_k_probs)
            gen_seq = init_seq_ids.clone().repeat(1, int(self.config.beam_search_size), 1) # To get [BATCH_SIZE x BEAM_WIDTH x MAX_LEN]
            gen_seq[:, :, 1] = best_k_idx
            enc_output = enc_output.repeat(1, int(self.config.beam_search_size), 1, 1)
        return enc_output, gen_seq, scores

    def _get_best_score_and_index(self,
                                gen_seq: torch.Tensor, # [BATCH_SIZE x BEAM_WIDTH x MAX_LEN]
                                dec_output: torch.Tensor, # [BATCH_SIZE x BEAM_WIDTH x MAX_LEN x VOCAB_SIZE]
                                scores: torch.Tensor, # [N x BEAM_WIDTH x BEAM_WIDTH]
                                step: int
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        best_k2_probs, best_k2_idx = dec_output[:, :, step, :].topk(int(self.config.beam_search_size))
        scores = torch.log(best_k2_probs) + scores
        scores, best_k_idx_in_k2 = scores.view(gen_seq.shape[0], -1).topk(int(self.config.beam_search_size))
         # Get the corresponding positions of the best k candidates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // int(self.config.beam_search_size), best_k_idx_in_k2 % int(self.config.beam_search_size)
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        gen_seq[:, :, :step] = gen_seq[best_k_r_idxs, :, : step]
        gen_seq[:, :, step] = best_k_idx

        return gen_seq, scores.unsqueeze(1)

    def _predict_model_decode(self, enc_output: torch.Tensor, captions_ids: torch.Tensor, pad_mask: torch.Tensor, lookahead_mask: torch.Tensor) -> torch.Tensor:
        return self.softmax(self._decoder_forward(enc_output, captions_ids, pad_mask, lookahead_mask))

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int]) -> Any:
        req_id, x_e, captions_ids, init_seq_ids = batch

        init_seq = captions_ids # torch.LongTensor([[self.bos_tok_id]]).repeat(x_e.shape[0], self.model_config.decoder_max_seq_len).unsqueeze(1)
        pad_mask = self._get_pad_mask(init_seq_ids)
        lookahead_mask = self._get_lookahead_mask(init_seq_ids)
        blank_seqs = torch.full((x_e.shape[0], int(self.config.beam_search_size), int(self.model_config.decoder_max_seq_len)), self.pad_tok_id, dtype=torch.long)
        len_map = torch.arange(1, self.model_config.decoder_max_seq_len + 1, dtype=torch.long)\
            .unsqueeze(0).unsqueeze(1).repeat(x_e.shape[0], int(self.config.beam_search_size), 1)
        

        with torch.no_grad():
            enc_output, gen_seq, scores = self._predict_initial_step(x_e, init_seq, pad_mask, lookahead_mask)

            ans_indices = torch.zeros((self.config.batch_size, 1, 1)) # default
            for step in range(2, self.model_config.decoder_max_seq_len):
                pad_mask = self._get_pad_mask(gen_seq.view(-1, gen_seq.shape[-1]))
                lookahead_mask = self._get_lookahead_mask(gen_seq.view(-1, gen_seq.shape[-1]))
                dec_output = self._predict_model_decode(
                    enc_output.view(-1, enc_output.shape[-2], enc_output.shape[-1]),
                    gen_seq.view(-1, gen_seq.shape[-1]),
                    pad_mask,
                    lookahead_mask)
                gen_seq, scores = self._get_best_score_and_index(
                    gen_seq, dec_output.view(self.config.batch_size, int(self.config.beam_search_size), *dec_output.shape[-2:]),
                    scores,
                    step)

                eos_locs = gen_seq == self.bert_tokenizer.sep_token_id
                seq_lens, _ = len_map.masked_fill(~eos_locs, self.model_config.decoder_max_seq_len).min(2)

                if eos_locs.sum().item() == self.config.batch_size * int(self.config.beam_search_size):
                    _, ans_indices = scores.div(seq_lens.permute(0, 2, 1).float() ** self.config.alpha).squeeze(1).max(-1) # [BATCH_SIZE x 1]

            selected_captions = torch.IntTensor(self.config.batch_size, self.model_config.decoder_max_seq_len)
            for i in range(self.config.batch_size):
                selected_captions[i, :] = gen_seq[i, ans_indices[i, 0].to(torch.long), :]
            
            # limit length per captions
            print(selected_captions)
            return self.bert_tokenizer.batch_decode(selected_captions, skip_special_tokens=True) #[:, seq_lens[:, ans_indices.view(-1).to(torch.long), 0]].detach().tolist())

    def _assign_state_to_model(self) -> None:
        if self.config.requires_model_init:
            if not self.config.model_init_path:
                raise ValueError('Initializer model value is None')
            if not self.config.model_initializer_type:
                raise ValueError('Model Initializer type value is None')

            #TODO: Allow the factory to inject whole args as opposed to filling in bag of args
            initializer_args = {
                BaseInitializer.KEY_MODEL: self,
                BaseVit16384V2.KEY_NUMBER_ENCODER_LAYERS: self.num_encoder_blocks,
                BaseVit16384V2.KEY_NUMBER_DECODER_LAYERS: self.num_decoder_blocks
            }

            initializer_instance = InitializersFactory.get_instance(
                self.config.model_initializer_type,
                self.config, 
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
        model_state[CPTRModelBuilder.StateKey.ENCODER_INPUT_LAYER] = self.encoder_input_layer.weight_transfer_to_dict()
        for idx, enc_blk in list(enumerate(len(self.encoder_block_layers))):
            model_state[f'{CPTRModelBuilder.StateKey.ENCODER_BLOCK_LAYERS.replace("#", str(idx))}'] = enc_blk.weight_transfer_to_dict()

        for idx, dec_blk in list(enumerate(len(self.decoder_block_layers))):
            model_state[f'{CPTRModelBuilder.StateKey.DECODER_BLOCK_LAYERS.replace("#", str(idx))}'] = dec_blk.weight_transfer_to_dict()

        model_state[CPTRModelBuilder.StateKey.DECODER_INPUT_LAYER] = self.decoder_input_layer.weight_transfer_to_dict()

        model_state[f'{CPTRModelBuilder.StateKey.CPTR_LINEAR}.weight'] = self.linear_layer.weight
        model_state[f'{CPTRModelBuilder.StateKey.CPTR_LINEAR}.bias'] = self.linear_layer.bias

        fs = FSFactory.get_instance(self.config.model_save_file_system, **self.config.model_save_fs_options)
        fs.save_file(self.config.model_save_path, Utils.dict_to_bytes(model_state))

    def weight_transfer_to_dict(self) -> OrderedDict[str, Any]:
        state_ordered_dict = OrderedDict()

        for idx, out_mod in enumerate(self.out):
            out_mod_name = CPTRModelBuilder.StateKey.CPTR_OUT_WEIGHT.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, f'{idx}')
            state_ordered_dict[out_mod_name] = out_mod.weight

        for idx, enc_layer in enumerate(self.encoder_block_layers):
            layer = CPTRModelBuilder.StateKey.CPTR_ENCODER_BLOCK_LAYERS_WEIGHT.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, f'{idx}')
            state_ordered_dict[layer] = enc_layer.weight_transfer_to_dict()

        for idx, dec_layer in enumerate(self.decoder_block_layers):
            layer = CPTRModelBuilder.StateKey.CPTR_DECODER_BLOCK_LAYERS_WEIGHT.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, f'{idx}')
            state_ordered_dict[layer] = dec_layer.weight_transfer_to_dict()

        return state_ordered_dict

    def bias_transfer_to_dict(self) -> OrderedDict[str, Any]:
        state_ordered_dict = OrderedDict()

        for idx, out_mod in enumerate(self.out):
            out_mod_name = CPTRModelBuilder.StateKey.CPTR_OUT_BIAS.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, f'{idx}')
            state_ordered_dict[out_mod_name] = out_mod.bias

        for idx, enc_layer in enumerate(self.encoder_block_layers):
            layer = CPTRModelBuilder.StateKey.CPTR_ENCODER_BLOCK_LAYERS_BIAS.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, f'{idx}')
            state_ordered_dict[layer] = enc_layer.bias_transfer_to_dict()

        for idx, dec_layer in enumerate(self.decoder_block_layers):
            layer = CPTRModelBuilder.StateKey.CPTR_DECODER_BLOCK_LAYERS_BIAS.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, f'{idx}')
            state_ordered_dict[layer] = dec_layer.bias_transfer_to_dict()

        return state_ordered_dict

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        model_dict = self.state_dict()

        # Verifying first before trying to copy from dict since this call might come from the initizalizer
        # and the initializer might have already initialized other layers, so we need to check for existence of key in dict before copying
        if CPTRModelBuilder.StateKey.CPTR_OUT_WEIGHT in weights:
            for idx, _ in enumerate(self.out):
                out_mod_name = CPTRModelBuilder.StateKey.CPTR_OUT_WEIGHT.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, f'{idx}')
                model_dict[out_mod_name] = weights.get(out_mod_name)
        
        if CPTRModelBuilder.StateKey.CPTR_ENCODER_BLOCK_LAYERS_WEIGHT.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, '0') in weights:
            for idx, enc_layer in enumerate(self.encoder_block_layers):
                layer = CPTRModelBuilder.StateKey.CPTR_ENCODER_BLOCK_LAYERS_WEIGHT.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, f'{idx}')
                model_dict[layer] = enc_layer.weight_transfer_from_dict(weights)

        if CPTRModelBuilder.StateKey.CPTR_DECODER_BLOCK_LAYERS_WEIGHT.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, '0') in weights:
            for idx, dec_layer in enumerate(self.decoder_block_layers):
                layer = CPTRModelBuilder.StateKey.CPTR_DECODER_BLOCK_LAYERS_WEIGHT.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, f'{idx}')
                model_dict[layer] = dec_layer.weight_transfer_from_dict(weights)

        self.load_state_dict(model_dict)

    def bias_transfer_from_dict(self, bias: OrderedDict[str, Any]) -> None:
        model_dict = self.state_dict()
        if CPTRModelBuilder.StateKey.CPTR_OUT_BIAS in bias:
            for idx, _ in enumerate(self.out):
                out_mod_name = CPTRModelBuilder.StateKey.CPTR_OUT_BIAS.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, f'{idx}')
                model_dict[out_mod_name] = bias.get(out_mod_name)
        
        if CPTRModelBuilder.StateKey.CPTR_ENCODER_BLOCK_LAYERS_BIAS.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, '0') in bias:
            for idx, enc_layer in enumerate(self.encoder_block_layers):
                layer = CPTRModelBuilder.StateKey.CPTR_ENCODER_BLOCK_LAYERS_BIAS.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, f'{idx}')
                model_dict[layer] = enc_layer.bias_transfer_from_dict(bias)

        if CPTRModelBuilder.StateKey.CPTR_DECODER_BLOCK_LAYERS_BIAS.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, '0') in bias:
            for idx, dec_layer in enumerate(self.decoder_block_layers):
                layer = CPTRModelBuilder.StateKey.CPTR_DECODER_BLOCK_LAYERS_BIAS.replace(CPTRModelBuilder.StateKey.CONSTANT_PLACEHOLDER, f'{idx}')
                model_dict[layer] = dec_layer.bias_transfer_from_dict(bias)

        self.load_state_dict(model_dict)

    class StateKey:
        CPTR_OUT_WEIGHT = 'out.#.weight'
        CPTR_OUT_BIAS = 'out.#.bias'
        CPTR_ENCODER_INPUT_LAYER_WEIGHT = 'cptr_encoder_input_layer.weight'
        CPTR_ENCODER_INPUT_LAYER_BIAS = 'cptr_encoder_input_layer.bias'
        CPTR_DECODER_INPUT_LAYER_WEIGHT = 'cptr_decoder_input_layer.weight'
        CPTR_DECODER_INPUT_LAYER_BIAS = 'cptr_decoder_input_layer.bias'
        CPTR_ENCODER_BLOCK_LAYERS_WEIGHT = 'encoder_block_layers.#.weight'
        CPTR_DECODER_BLOCK_LAYERS_WEIGHT = 'decoder_block_layers.#.weight'
        CPTR_ENCODER_BLOCK_LAYERS_BIAS = 'encoder_block_layers.#.bias'
        CPTR_DECODER_BLOCK_LAYERS_BIAS = 'decoder_block_layers.#.bias'
        CONSTANT_PLACEHOLDER = '#'