from typing import Any


class BaseAuthManager:
    def do_auth(self, **kwargs) -> Any:
        raise NotImplementedError('do_auth not implemented')