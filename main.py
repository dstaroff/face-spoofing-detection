import warnings

from cv2 import cv2

from src.detector import (
    FaceDetector,
    SpoofingDetector,
    )
from src.detector.face.errors import FaceDetectorLoadingError
from src.detector.spoofing.errors import ModelLoadingError
from src.gui import Window
from src.gui.errors import (
    AppExitError,
    VideoCaptureOpeningError,
    )
from src.utils import const


def warn(*args, **kwargs):
    pass


warnings.warn = warn

window: Window
face_detector: FaceDetector
spoofing_detector: SpoofingDetector


def setup():
    global window, face_detector, spoofing_detector

    try:
        window = Window()
        face_detector = FaceDetector(window.image_size.width, window.image_size.height)
        spoofing_detector = SpoofingDetector(face_detector)
    except (VideoCaptureOpeningError, FaceDetectorLoadingError, ModelLoadingError) as e:
        print(e)
        raise AppExitError()


def main():
    try:
        setup()

        while True:
            img = window.get_frame()
            spoofing_detection = spoofing_detector.detect(img)

            for face_detection in spoofing_detection.faces:
                prediction = spoofing_detector.overall_prediction(
                        face_detection.replay_attack_prediction,
                        face_detection.print_attack_prediction,
                        )

                if prediction >= const.ATTACK_THRESHOLD:
                    color = const.BGR_COLOR_RED
                    conclusion = const.TEXT_SPOOFING
                else:
                    color = const.BGR_COLOR_GREEN
                    conclusion = const.TEXT_GENUINE

                text = '#{face_id} {conclusion} [Replay: {replay:.0%}, Print: {print:.0%}]'.format(
                        face_id=face_detection.face_id,
                        conclusion=conclusion,
                        replay=face_detection.replay_attack_prediction,
                        print=face_detection.print_attack_prediction,
                        )

                img = window.draw_face(img, face_detection, color, text)

            img = window.add_info_board(img)
            if len(spoofing_detection.faces) > 0:
                window.print_probability(img, 'Replay', spoofing_detection.faces[0].replay_attack_prediction, 0)
                window.print_probability(img, 'Print', spoofing_detection.faces[0].print_attack_prediction, 1)

            cv2.imshow(const.APP_TITLE, img)

            if window.does_user_wants_to_exit():
                window.exit()
    except AppExitError as e:
        print(e)
        exit(0)


if __name__ == "__main__":
    main()
