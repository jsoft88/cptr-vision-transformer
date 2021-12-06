import json
from typing import Any, Dict, OrderedDict


class Utils:
    @classmethod
    def read_json_from_file(cls, path: str) -> Any:
        with open(path, 'r') as f:
            return json.load(f)

    @classmethod
    def dict_to_bytes(cls, dict_to_convert: Dict[Any, Any]) -> bytes:
        return json.dumps(dict_to_convert).encode('utf-8')

    @classmethod
    def bytes_to_dict(cls, bytes_to_convert: bytes) -> Dict[Any, Any]:
        return json.loads(bytes_to_convert.decode('utf-8'))
