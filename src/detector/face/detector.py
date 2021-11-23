from typing import List

import numpy as np
from cv2 import cv2

from src.utils import const
from .errors import FaceDetectorLoadingError
from .models import Face


class FaceDetector:
    def __init__(self, img_width: int, img_height: int):
        try:
            self._face_detector = cv2.CascadeClassifier(const.FACE_CASCADE_FILE)
        except Exception:
            raise FaceDetectorLoadingError(const.FACE_CASCADE_FILE)

        self._min_face_size = self._calculate_min_face_size(img_width, img_height)

    @staticmethod
    def _calculate_min_face_size(img_width: int, img_height: int) -> int:
        return int(min(img_width, img_height) * const.MIN_FACE_PERCENT)

    def detect(self, img: np.array) -> List[Face]:
        faces = self._face_detector.detectMultiScale(
                img,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self._min_face_size, self._min_face_size),
                )
        res = []
        for face in faces:
            res.append(Face(x=face[0], y=face[1], w=face[2], h=face[3]))

        return res
