from src.error import BaseError


class FaceDetectorLoadingError(BaseError):
    def __init__(self, cascade_name: str):
        super(FaceDetectorLoadingError, self).__init__(message=cascade_name)

    @staticmethod
    def _template() -> str:
        return 'Face detector "{message}" could not be loaded.'
