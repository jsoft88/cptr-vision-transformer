from typing import OrderedDict, Any


class CoreModuleExtension:

    def weight_transfer_from_dict(self, weights: OrderedDict[str, Any]) -> None:
        raise NotImplementedError(f'weight transfer from dict method for {self.__class__.__name__} not implemented')

    def bias_transfer_from_dict(self, bias: OrderedDict[str, Any]) -> None:
        raise NotImplementedError(f'bias_transfer_from_dict method for {self.__class__.__name__}')

    def bias_transfer_to_dict(self) -> OrderedDict[str, Any]:
        raise NotImplementedError(f'bias transfer to dict for {self.__class__.__name__} not implemented')

    def weight_transfer_to_dict(self) -> OrderedDict[str, Any]:
        raise NotImplementedError(f'weight transfer to dict for {self.__class__.__name__} not implemented')
