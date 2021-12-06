from typing import List
from cptr_model.utils.file_handlers.base_file_handler import BaseFileHandler
import io


class LocalFSHandler(BaseFileHandler):
    def list_files(self) -> List[str]:
        pass

    def retrieve_file(self, path: str) -> bytes:
        with open(path, 'rb') as f:
            file_bytes = io.BytesIO(f.read())
        return file_bytes

    def retrieve_files(self, path: str) -> List[bytes]:
        pass

    def save_file(self, path: str, content: bytes) -> None:
        pass