from dataclasses import dataclass


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int


@dataclass
class FaceDetection:
    face: Face
    replay_attack_prediction: float
    print_attack_prediction: float
    face_id: int
