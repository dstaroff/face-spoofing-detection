from src.error import BaseError


class ModelLoadingError(BaseError):
    def __init__(self, model_name: str):
        super(ModelLoadingError, self).__init__(message=model_name)

    @staticmethod
    def _template() -> str:
        return 'Model {message} could not be loaded.'
