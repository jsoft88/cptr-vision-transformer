from typing import List
from cptr_model.utils.file_handlers.base_file_handler import BaseFileHandler
import io
import glob


class LocalFSHandler(BaseFileHandler):
    def list_files(self, pattern: str) -> List[str]:
        return glob.glob(pattern)

    def __sanitize_path(self, path: str) -> str:
        return path.replace('file://', '') if path.lower().startswith('file://') else path

    def retrieve_file(self, path: str) -> bytes:
        path = self.__sanitize_path(path)
        with open(path, 'rb') as f:
            file_bytes = io.BytesIO(f.read())
        return file_bytes

    def retrieve_files(self, path: str) -> List[bytes]:
        pass

    def save_file(self, path: str, content: bytes) -> None:
        path = self.__sanitize_path(path)
        with open(path, 'wb+') as wf:
            wf.write(content)
