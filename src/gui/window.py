from typing import Tuple

import numpy as np
from cv2 import cv2

from src.utils import const
from .errors import (
    AppExitError,
    VideoCaptureGrabbingError,
    VideoCaptureOpeningError,
    )
from .models import ImageSize
from ..detector.face.models import (
    FaceDetection,
    )


class Window:
    def __init__(self):
        self._capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self._capture.isOpened():
            raise VideoCaptureOpeningError(0)

        self.image_size = self._get_image_size()

    def get_frame(self) -> np.ndarray:
        ret, img = self._capture.read()

        if ret is False:
            raise VideoCaptureGrabbingError(0)

        return img

    def _get_image_size(self) -> ImageSize:
        return ImageSize(width=(img := self.get_frame()).shape[1], height=img.shape[0])

    @staticmethod
    def draw_face(img: np.ndarray, face: FaceDetection, color: Tuple[int, int, int], text: str):
        cv2.rectangle(
                img,
                (face.face.x, face.face.y),
                (face.face.x + face.face.w, face.face.y + face.face.h),
                color,
                const.THICKNESS,
                )
        cv2.putText(
                img=img,
                text=text,
                org=(face.face.x, int(face.face.y - 2 * const.THICKNESS - const.FONT_SCALE)),
                fontFace=const.FONT_FACE,
                fontScale=const.FONT_SCALE,
                color=color,
                thickness=const.THICKNESS,
                lineType=cv2.LINE_AA,
                )

        return img

    @staticmethod
    def add_info_board(img: np.ndarray):
        board = np.ones((const.INFO_BOARD_HEIGHT, img.shape[1], img.shape[2]))
        return np.vstack((img / 255.0, board))

    def print_probability(self, img: np.ndarray, attack_type: str, prob: float, row: int):
        org = (
            const.THICKNESS * 2,
            int(self.image_size.height + (row * 2 + 1) * const.INFO_BOARD_HEIGHT / 4),
            )
        cv2.putText(
                img=img,
                text=f'{attack_type} spoofing attack probability = {prob:.2%}',
                org=org,
                fontFace=const.FONT_FACE,
                fontScale=const.FONT_SCALE,
                color=const.BGR_COLOR_BLACK,
                thickness=const.THICKNESS,
                )

    @staticmethod
    def does_user_wants_to_exit() -> bool:
        key = cv2.waitKey(1)
        return key == ord('q')

    def exit(self):
        self._capture.release()
        cv2.destroyAllWindows()

        raise AppExitError()
