from cptr_model.config.config import Config
from cptr_model.data.data_modules.batch.local_mscoco_data import LocalMSCocoData
from cptr_model.data.data_modules.batch.mscoco_data import MSCocoData
from cptr_model.utils.file_handlers.local_fs_handler import LocalFSHandler
from tests.utils.test_fixtures.args_fixture import get_args, get_args_for_local_fs


def test_batch_dataloader(get_args):
    config = Config(get_args)
    dm = MSCocoData(config, LocalFSHandler())

    dm.prepare_data()
    with dm.train_dataloader() as dl:
        elm = next(iter(dl))
        img, seqs_ids, targets = elm['img'], elm['seqs_ids'], elm['labels']
        assert img.shape == (1, 3, 384, 384)
        assert seqs_ids.shape == (1, 510)
        assert targets.shape == (1, 510)

def test_local_batch_dataloader(get_args_for_local_fs):
    config = Config(get_args_for_local_fs)
    dm = LocalMSCocoData(config, LocalFSHandler(), **{LocalMSCocoData.KEY_MAX_LEN: 510})
    dm.setup()

    for idx, batch in enumerate(dm.train_dataloader()):
        img, embs, targets = batch
        
        assert img.shape == (1, 3, 384, 384)
        assert embs.shape == (1, 510)
        assert targets.shape == (1, 510)