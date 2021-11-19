class BaseEncoderDecoderBlock:
    def __init__(self, **kwargs) -> None:
        self.__verify_required_args()

    def __verify_required_args(self) -> None:
        raise NotImplementedError('__verify_required_args not implemented')