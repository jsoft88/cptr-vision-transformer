from typing import Any, Dict

class MessageUtils:

    @staticmethod
    def deserialize_value(message: Any) -> Dict[str, Any]:
        pass

    @staticmethod
    def __from_proto(message: Any) -> Any:
        kafka_pb2 = KafkaMess