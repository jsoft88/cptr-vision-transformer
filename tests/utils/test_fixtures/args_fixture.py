import pytest
from pathlib import Path


@pytest.fixture
def get_args():
    args = [
        '--file-system', 'local-fs',
        '--with-pretrained-weights',
        '--model-save-file-system', 'local-fs',
        '--model-save-path', f'file://{Path(__file__).parent.parent.parent}/resources/cptr.pth',
        '--tmp-dir', 'file:///tmp/cptr_staging_dir',
        '--model-init-path', '/mnt/d/ds_ai_stuff/pretrained_models/jx_vit_base_patch16_384_in21k-0243c7d9.pth',
        '--requires-model-init',
        '--model-initializer-type', 'vit_base_16_384_bert',
        '--num-epochs', '1',
        '--lr-decay-after', '1',
        '--lr-decay-factor', str(float(0.5)),
        '--lr', '3e-5',
        '--beam-search-size', '1',
        '--batch-size', '1',
        '--batch',
        '--spark-master', 'local[*]',
        '--training',
        '--Btrain-metadata-location', f'file://{Path(__file__).parent.parent.parent}/resources/metadata/',
        '--Btrain-img-location', f'file://{Path(__file__).parent.parent.parent}/resources/images/',
        '--Bval-metadata-location', f'file://{Path(__file__).parent.parent.parent}/resources/metadata/',
        '--Bval-img-location', f'file://{Path(__file__).parent.parent.parent}/resources/metadata/',
        '--input-reader-type', 'batch-dm'
    ]

    return args

@pytest.fixture
def get_args_for_local_fs():
    args = [
        '--file-system', 'local-fs',
        '--with-pretrained-weights',
        '--model-save-file-system', 'local-fs',
        '--model-save-path', f'{Path(__file__).parent.parent.parent}/resources/cptr.pth',
        '--tmp-dir', 'file:///tmp/cptr_staging_dir',
        '--model-init-path', '/mnt/d/ds_ai_stuff/pretrained_models/jx_vit_base_patch16_384_in21k-0243c7d9.pth',
        '--requires-model-init',
        '--model-initializer-type', 'vit_base_16_384_bert',
        '--num-epochs', '1',
        '--lr-decay-after', '1',
        '--lr-decay-factor', str(float(0.5)),
        '--lr', '3e-5',
        '--beam-search-size', '1',
        '--batch-size', '1',
        '--batch',
        '--spark-master', 'local[*]',
        '--training',
        '--Btrain-metadata-location', f'{Path(__file__).parent.parent.parent}/resources/metadata/',
        '--Btrain-img-location', f'{Path(__file__).parent.parent.parent}/resources/images/',
        '--Bval-metadata-location', f'{Path(__file__).parent.parent.parent}/resources/metadata/',
        '--Bval-img-location', f'{Path(__file__).parent.parent.parent}/resources/metadata/',
        '--input-reader-type', 'batch-dm'
    ]

    return args
