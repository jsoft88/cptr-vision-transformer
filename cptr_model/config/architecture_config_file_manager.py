from pathlib import Path
from typing import Any, Union, List

from cptr_model.utils.utils import Utils


class ArchitectureConfigFileManager:
    def __init__(self, file_name: str) -> None:
        root = Path(__file__).parent.parent.parent.resolve()
        self.path = Path(root).joinpath("resources", "config", file_name)
        self.config_object = Utils.read_json_from_file_to(self.path)

    def get_embeddings_config_for(self, typ: str) -> Any:
        if typ not in self.config_object['embeddings']:
            raise KeyError(f'{typ} not found in config object')

        section = self.config_object[ArchitectureConfigFileManager.ArchitectureParts.SECTION_EMBEDDINGS][typ]
        return section

    def get_embeddings_params(self, key: str, section: Any) -> Any:
        if not section:
            raise ValueError(f'{key} can not be searched in None section')
        if key not in section:
            raise KeyError(f'{key} not found in provided config section {section}')

        return section[ArchitectureConfigFileManager.ArchitectureParts.SECTION_EMBEDDINGS_PARAMS]

    def get_param_value_for(self, param: str, section: Any) -> Union[str, List[str], List[int], int, bool]:
        if param not in section:
            raise KeyError(f'Param {param} not found in config section {section}')

        return section[param]

    class ArchitectureParts:
        SECTION_EMBEDDINGS = 'embeddings'
        SECTION_EMBEDDINGS_INPUT = 'input'
        SECTION_EMBEDDINGS_POSITION = 'position'
        SECTION_EMBEDDINGS_TYPE = 'type'
        SECTION_EMBEDDINGS_PARAMS = 'params'

