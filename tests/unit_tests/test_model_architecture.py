import torch
from transformers import BertConfig, BertModel, BertTokenizer
from cptr_model.config.config import Config
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.data.data_modules.batch.local_mscoco_data import LocalMSCocoData
from cptr_model.models.cptr.cptr import CPTRModelBuilder
from cptr_model.utils.file_handlers.local_fs_handler import LocalFSHandler
from tests.utils.test_fixtures.args_fixture import get_args_for_local_fs


def test_encoder_block(get_args_for_local_fs):
    config = Config(get_args_for_local_fs)
    dm = LocalMSCocoData(config, LocalFSHandler(), **{LocalMSCocoData.KEY_MAX_LEN: 510})
    dm.setup()
    cptr = CPTRModelBuilder(config)
    cptr.build_model()

    for _, batch in enumerate(dm.train_dataloader()):
        img, _, _ = batch # we don't care about the caption embeddings nor the targets here
        enc_putput = cptr._encoder_forward(img)
        model_config: ArchitectureConfigFileManager = config.cptr_specifics
        kernel_size = model_config.encoder_input_embeddings_params_dict.get('kernel_size')
        channel_out = model_config.encoder_input_embeddings_params_dict.get('channel_out')
        width = model_config.encoder_input_embeddings_params_dict.get('width')
        height = model_config.encoder_input_embeddings_params_dict.get('height')

        assert enc_putput.shape == (config.batch_size, height * width / (kernel_size[0] * kernel_size[1]) , channel_out)

def test_decoder_block(get_args_for_local_fs):
    config = Config(get_args_for_local_fs)
    dm = LocalMSCocoData(config, LocalFSHandler(), **{LocalMSCocoData.KEY_MAX_LEN: 510})
    dm.setup()
    cptr = CPTRModelBuilder(config)
    cptr.build_model()

    for _, batch in enumerate(dm.train_dataloader()):
        img, caption, target_ids = batch # we don't care about the caption embeddings nor the targets here
        enc_putput = cptr._encoder_forward(img)
        pad_mask = cptr._get_pad_mask(target_ids)
        lookahead_mask = cptr._get_lookahead_mask(target_ids)
        dec_output = cptr._decoder_forward(enc_putput, caption, pad_mask, lookahead_mask)
        assert config.training and dec_output.shape == (510, 30522)

def test_prediction(get_args_for_local_fs):
    config = Config(get_args_for_local_fs)
    config.training = False
    config.cptr_specifics.decoder_max_seq_len = 20
    dm = LocalMSCocoData(config, LocalFSHandler(), **{LocalMSCocoData.KEY_MAX_LEN: 20})
    dm.setup()
    cptr = CPTRModelBuilder(config)
    cptr.build_model()

    for _, batch in enumerate(dm.train_dataloader()):
        img, _, _ = batch # we don't care about the caption embeddings nor the targets here
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased', config=BertConfig())
        max_len = 20

        tokenizer_outputs = bert_tokenizer(bert_tokenizer.cls_token, return_tensors='pt', max_length=max_len, padding='max_length', add_special_tokens=False)
        with torch.no_grad():
            outputs = bert_model(**tokenizer_outputs, output_hidden_states=True)
            init_seq = torch.sum(torch.stack(list(outputs.hidden_states[-4:]), dim=0), dim=0, keepdims=False).squeeze(0)
        print(bert_tokenizer.batch_decode(tokenizer_outputs['input_ids']))
        predict_batch = (1, img, tokenizer_outputs['input_ids'], tokenizer_outputs.input_ids)
        out = cptr.predict_step(predict_batch, 0, 0)
        raise Exception(out)



