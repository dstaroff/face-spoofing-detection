from src.error import BaseError


class VideoCaptureOpeningError(BaseError):
    def __init__(self, device_number: int):
        super(VideoCaptureOpeningError, self).__init__(message=str(device_number))

    @staticmethod
    def _template() -> str:
        return 'Could not open video capture device #{message}'


class VideoCaptureGrabbingError(BaseError):
    def __init__(self, device_number: int):
        super(VideoCaptureGrabbingError, self).__init__(message=str(device_number))

    @staticmethod
    def _template() -> str:
        return 'Could not grab a frame from video capture device #{message}'


class AppExitError(BaseError):
    def __init__(self):
        super(AppExitError, self).__init__(message='')

    @staticmethod
    def _template() -> str:
        return 'Application exited. {message}'
