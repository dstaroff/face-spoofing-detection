from dataclasses import dataclass
from typing import List

from src.detector.face.models import FaceDetection


@dataclass
class SpoofingDetection:
    faces: List[FaceDetection]
