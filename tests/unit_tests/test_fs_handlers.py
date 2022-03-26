import os
from cptr_model.utils.file_handlers.local_fs_handler import LocalFSHandler
from pathlib import Path


def test_local_fs_list_files():
    local_fs = LocalFSHandler()
    print(local_fs.list_files(os.path.join(Path(__file__).parent.parent, "resources", "metadata", "*.parquet")))
    assert len(local_fs.list_files(os.path.join(Path(__file__).parent.parent, "resources", "metadata", "*.parquet"))) == 1
