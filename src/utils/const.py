import os

from cv2 import cv2

APP_TITLE = 'Face spoofing attack detector'

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.abspath(os.path.join(PROJECT_PATH, 'data'))

FACE_CASCADE_FILE = os.path.abspath(os.path.join(DATA_PATH, 'face_detector.xml'))
PRINT_ATTACK_FILE = os.path.abspath(os.path.join(DATA_PATH, 'print_attack_classifier.pkl'))
REPLAY_ATTACK_FILE = os.path.abspath(os.path.join(DATA_PATH, 'replay_attack_classifier.pkl'))

MIN_FACE_PERCENT = 0.25
ATTACK_THRESHOLD = 0.75

BGR_COLOR_BLACK = (0, 0, 0)
BGR_COLOR_RED = (0, 0, 255)
BGR_COLOR_GREEN = (0, 255, 0)

TEXT_SPOOFING = 'Spoofing'
TEXT_GENUINE = 'Genuine'

THICKNESS = 1
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5

INFO_BOARD_HEIGHT = 64
