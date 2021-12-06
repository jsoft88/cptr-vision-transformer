from typing import List, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from cptr_model.models.cptr.decoder.cptr_decoder_block import CPTRDecoderBlock
import torchtext
import spacy
from cptr_model.config.config import Config
import pandas as pd


class MSCocoData(LightningDataModule):
    KEY_IS_ENCODER = 'is-encoder'

    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__()
        self.is_encoder = kwargs.get(MSCocoData.KEY_IS_ENCODER, False)
        self.batch_size = config.encoder_batch_size if self.is_encoder else config.decoder_batch_size
        self.data_dir = config.encoder_data_location if self.is_encoder else config.decoder_data_location

        self.data = None
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.dec_lang_model = spacy.load(config.dec_lang)
        self.dec_vocab_size = 0

    def tokenize_dec_lang(self, text: str) -> List[str]:
        return [tok.text for tok in self.dec_lang.tokenizer(text)]

    def setup(self, stage: Optional[str] = None) -> None:
        self.data = pd.read_parquet(self.data_dir)
        field = torchtext.data.Field(
            tokenize=self.tokenize_dec_lang,
            lower=True,
            pad_token=CPTRDecoderBlock.PAD_WORD,
            init_token=CPTRDecoderBlock.BOS_WORD,
            eos_token=CPTRDecoderBlock.EOS_WORD)

        preprocessed_text = self.data['caption'].apply(lambda x: field.preprocess(x))

        field.build_vocab(preprocessed_text, self.dec_lang_model)

        self.dec_vocab_size = field.vocab


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, self.batch_size)