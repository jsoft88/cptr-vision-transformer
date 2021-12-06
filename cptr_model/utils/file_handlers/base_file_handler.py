from abc import ABC
from typing import List


class BaseFileHandler(ABC):
    def __init__(self, **kwargs):
        pass

    def list_files(self) -> List[str]:
        raise NotImplementedError('Method list_files is not implemented')

    def retrieve_file(self, path: str) -> bytes:
        raise NotImplementedError('Method retrieve_file is not implemented')

    def retrieve_files(self, path: str) -> List[bytes]:
        raise NotImplementedError('Method retrieve_files is not implemented')

    def save_file(self, path: str, content: bytes) -> None:
        raise NotImplementedError('Method save_file is not implemented')
    