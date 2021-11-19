import json
from typing import Any


class Utils:
    @classmethod
    def read_json_from_file_to(cls, path: str) -> Any:
        with open(path, 'r') as f:
            return json.load(f)
