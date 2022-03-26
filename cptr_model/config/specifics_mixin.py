from abc import abstractmethod
from typing import Any, Dict


class SpecificsMixin:

    @abstractmethod
    def load_specifics_from_file(self) -> Any:
        return None

    @abstractmethod
    def register_dynamic_linkers(self) -> Dict[str, Any]:
        return {}
