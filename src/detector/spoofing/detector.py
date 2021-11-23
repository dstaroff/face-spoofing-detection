from typing import Optional

import joblib
import numpy as np
from cv2 import cv2

from src.detector.face import FaceDetector
from src.detector.face.models import FaceDetection
from src.utils import const
from .errors import ModelLoadingError
from .models import (
    SpoofingDetection,
    )


class SpoofingDetector:
    def __init__(self, face_detector: FaceDetector):
        self._face_detector = face_detector
        self.replay_attack_classifier = self._load_classifier(const.REPLAY_ATTACK_FILE)
        self.print_attack_classifier = self._load_classifier(const.PRINT_ATTACK_FILE)

    @staticmethod
    def _load_classifier(model_path: str):
        classifier = None

        # noinspection PyBroadException
        try:
            classifier = joblib.load(model_path)
        except Exception:
            raise ModelLoadingError(model_path)

        return classifier

    @staticmethod
    def _calc_histogram(img: np.ndarray, color_space_transform: Optional[int] = None) -> np.ndarray:
        img_transformed: np.ndarray
        if color_space_transform is not None:
            img_transformed = cv2.cvtColor(img.copy(), color_space_transform)
        else:
            img_transformed = img.copy()

        channel_count = img_transformed.shape[2]
        histogram = []

        for channel in range(channel_count):
            channel_histogram = cv2.calcHist([img_transformed], [channel], None, [256], [0, 256])
            normalized_channel_histogram = channel_histogram * 255.0 / channel_histogram.max()
            histogram.append(normalized_channel_histogram)

        return np.array(histogram)

    def detect(self, img: np.ndarray) -> SpoofingDetection:
        res = SpoofingDetection([])

        faces = self._face_detector.detect(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY))
        for i, face in enumerate(faces):
            roi: np.ndarray = img[face.y:face.y + face.h, face.x:face.x + face.w]

            yrb_hist = self._calc_histogram(roi, cv2.COLOR_BGR2YCR_CB)
            luv_hist = self._calc_histogram(roi, cv2.COLOR_BGR2LUV)

            feature_vector = np.append(yrb_hist.ravel(), luv_hist.ravel())
            feature_vector = feature_vector.reshape(1, len(feature_vector))

            replay_attack_prediction = self.replay_attack_classifier.predict_proba(feature_vector)[0][1]
            print_attack_prediction = self.print_attack_classifier.predict_proba(feature_vector)[0][1]

            res.faces.append(
                    FaceDetection(
                            face=face,
                            replay_attack_prediction=replay_attack_prediction,
                            print_attack_prediction=print_attack_prediction,
                            face_id=i,
                            )
                    )

        return res

    @staticmethod
    def overall_prediction(replay_attack_prediction: float, print_attack_prediction: float):
        return max(replay_attack_prediction, print_attack_prediction)
