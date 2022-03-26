from functools import partial
import io
from typing import Any, List, Optional, Tuple
import numpy as np
from pyspark import Broadcast
import pyspark.sql.functions as F
from petastorm.spark.spark_dataset_converter import make_spark_converter
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StructField, StructType
from cptr_model.config.specifics.cptr.architecture_config_file_manager import ArchitectureConfigFileManager
from cptr_model.factory.transformations.transformation_factory import TransformationFactory
from cptr_model.factory.utils.fs_factory import FSFactory
from cptr_model.transformations.base_transformation import BaseTransformation
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from cptr_model.models.cptr.decoder.cptr_decoder_block import CPTRDecoderBlock
import torchvision
from cptr_model.config.config import Config
from pyspark.sql import SparkSession
from petastorm.spark import SparkDatasetConverter
from petastorm import TransformSpec
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.tokenization_bert import BertTokenizer
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
from cptr_model.utils.file_handlers.base_file_handler import BaseFileHandler
import os


class MSCocoData(LightningDataModule):
    KEY_MAX_LEN = 'max-len'

    def __init__(self, config: Config, fs_handler: Optional[BaseFileHandler] = None, **kwargs) -> None:
        super().__init__()
        
        if not config:
            raise ValueError('config object and/or architecture config file manager is null')

        self.config = config
        self.batch_size = self.config.batch_size
        self.train_data_dir = self.config.batch_train_metadata_location
        self.val_data_dir = self.config.batch_val_metadata_location
        self.config = config
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.train_data_img = None
        self.val_data_img = None
        self.train_data_img_dir = self.config.batch_train_img_location
        self.val_data_img_dir = self.config.batch_val_img_location
        self.bert_config = BertConfig()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', config=self.bert_config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.dec_vocab_size = self.bert_config.vocab_size
        self.max_len = kwargs.get(MSCocoData.KEY_MAX_LEN, self.tokenizer.max_len_single_sentence)
        self.train_spark_converter = None
        self.val_spark_converter = None
        self.img_H_W = (
            self.config.cptr_specifics.encoder_input_embeddings_params_dict['height'],
            self.config.cptr_specifics.encoder_input_embeddings_params_dict['width']
        )
        self.fs_handler = fs_handler
    
    def prepare_data(self) -> None:
        self.spark_session = SparkSession.builder.master(self.config.spark_master).getOrCreate()
        self.spark_session.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, self.config.tmp_dir)

        self.train_data = self.spark_session.read.parquet(self.train_data_dir)\
            .select(F.col('caption').alias('seqs_ids'), F.col('caption').alias('labels'), F.col('image_id'))\
            .alias('train_df')
        self.val_data = self.spark_session.read.parquet(self.val_data_dir)\
            .select(F.col('caption').alias('seqs_ids'), F.col('caption').alias('labels'), F.col('image_id'))\
            .alias('val_df')

        self.train_data_img = self.spark_session.read.format('image').load(self.train_data_img_dir).alias('train_img_df')
        self.val_data_img = self.spark_session.read.format('image').load(self.val_data_img_dir).alias('val_img_df')

        img_embeds_df_train = self.train_data.join(self.train_data_img, on=F.col('train_img_df.image.origin').contains(F.col('train_df.image_id')), how='inner')\
            .select(
                F.col('train_img_df.image.origin'),
                F.col('train_df.seqs_ids'),
                F.col('labels')
            )

        # binary2np_schema_train = StructType(
        #     [StructField('data', ArrayType(ArrayType(ArrayType(FloatType()))))] +
        #     [f for f in img_embeds_df_train.schema.fields if f.name != 'data']
        # )

        # img_embeds_df_train =  img_embeds_df_train\
        #     .rdd\
        #     .map(
        #         lambda r: (
        #             transforms.ToTensor()(Image.frombuffer('RGB', (r.width, r.height), bytes(r.data), 'raw', 'BGR', 0, 0)).detach().numpy().tolist(),
        #             r.height,
        #             r.width,
        #             r.bert_embeddings,
        #             r.labels)
        #     ).toDF(binary2np_schema_train)
        
        img_embeds_df_val = self.val_data.join(self.val_data_img, on=F.col('val_img_df.image.origin').contains(F.col('val_df.image_id')), how='inner')\
            .select(
                F.col('val_img_df.image.origin'),
                F.col('val_df.seqs_ids'),
                F.col('val_df.labels')
            )

        # binary2np_schema_val = StructType(
        #     [StructField('data', ArrayType(ArrayType(ArrayType(FloatType()))))] +
        #     [f for f in img_embeds_df_val.schema.fields if f.name != 'data']
        # )
        # img_embeds_df_val\
        #     .rdd\
        #     .map(
        #         lambda r: (
        #             transforms.ToTensor()(Image.frombuffer('RGB', (r.width, r.height), bytes(r.data), 'raw', 'BGR', 0, 0)).detach().numpy().tolist(),
        #             r.height,
        #             r.width,
        #             r.bert_embeddings,
        #             r.labels
        #         )
        #     ).toDF(binary2np_schema_val)
        
        self.train_spark_converter = make_spark_converter(img_embeds_df_train, dtype='float32')
        self.val_spark_converter = make_spark_converter(img_embeds_df_val, dtype='float32')

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        pass

    def __prepare_image_transformations(self) -> List[BaseTransformation]:
        # all_transformations_images = self.config_file_manager.get_transforms_applicable_to(ArchitectureConfigFileManager.Model.Transformations.ApplyTo.IMAGE)
        # transform_args = None
        # image_transformations = []
        # for tt in self.config.encoder_transformation_types:
        #     transform_args = self.config_file_manager.get_args_for_transform_type(tt, all_transformations_images)
        #     image_transformations.append(TransformationFactory.get_instance(tt, self.config, self.config_file_manager, **transform_args))

        # return image_transformations
        pass

    def _image_fetch_wrapper(self, path: str) -> Image:
        loaded_img = Image.open(self.fs_handler.retrieve_file(path))
        return loaded_img

    def _transform_row(self, pd_batch: pd.DataFrame):
        transformations = transforms.Compose([
            transforms.transforms.Lambda(
                lambda x: self._image_fetch_wrapper(x)
            ),
            transforms.Resize((self.img_H_W[0], self.img_H_W[1])),
            transforms.ToTensor()
        ])

        pd_batch['img'] = pd_batch['origin'].map(lambda x: transformations(x).numpy())
        pd_batch['seqs_ids'] = pd_batch['seqs_ids'].map(lambda x: self.tokenizer(
                x.lower(),
                return_tensors='pt',
                max_length=self.max_len,
                padding='max_length', add_special_tokens=self.config.training)['input_ids']
                    .squeeze(0).detach().numpy())
        pd_batch['labels'] = pd_batch['seqs_ids']
        pd_batch = pd_batch.drop(labels=['origin'], axis=1)

        return pd_batch

    def _get_transform_spec(self) -> transforms.Compose:
        return TransformSpec(
            partial(self._transform_row),
            edit_fields=[
                ('img', np.int32, (3, self.img_H_W[0], self.img_H_W[1]), False),
                ('seqs_ids', np.int32, (self.max_len,), False),
                ('labels', np.int32, (self.max_len,), False)
            ],
            selected_fields=['img', 'seqs_ids', 'labels']
        )

    def train_dataloader(self) -> DataLoader:
        return self.train_spark_converter.make_torch_dataloader(batch_size=self.config.batch_size, transform_spec=self._get_transform_spec())

    def test_dataloader(self) -> DataLoader:
        return None

    def val_dataloader(self) -> DataLoader:
        return self.val_spark_converter.make_torch_dataloader(batch_size=self.config.batch_size, transform_spec=self._get_transform_spec())

    def teardown(self, stage: Optional[str] = None):
        self.train_spark_converter.delete()
        self.val_spark_converter.delete()