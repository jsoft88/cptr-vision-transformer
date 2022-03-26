from typing import List
from cptr_model.utils.file_handlers.base_file_handler import BaseFileHandler
import s3fs


class S3FSHandler(BaseFileHandler):
    KEY_ACCESS = 'key-access'
    KEY_SECRET = 'key-secret'

    def __init__(self, **kwargs):
        if S3FSHandler.KEY_ACCESS not in kwargs:
            raise ValueError(f'{S3FSHandler.KEY_ACCESS} not present in kwargs')
        if S3FSHandler.KEY_SECRET not in kwargs:
            raise ValueError(f'{S3FSHandler.KEY_SECRET} not present in kwargs')

        self.s3fs = s3fs.S3FileSystem(key=kwargs.get(S3FSHandler.KEY_ACCESS), secret=kwargs.get(S3FSHandler.KEY_SECRET))

    def list_files(self, pattern: str) -> List[str]:
        return self.s3fs.glob(pattern)

    def save_file(self, path: str, content: bytes) -> None:
        with self.s3fs.open(path, 'wb+') as wf:
            wf.write(content)
