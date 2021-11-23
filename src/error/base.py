from abc import (
    ABC,
    abstractmethod,
    )


class BaseError(Exception, ABC):
    def __init__(self, message: str):
        self._message = message

    @staticmethod
    @abstractmethod
    def _template() -> str:
        pass

    def __str__(self) -> str:
        return self._template().format(message=self._message)
