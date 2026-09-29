import cv2
import os

from imageai.Detection import ObjectDetection


class VideoCamera:
    """
    Object detection controller.

    This class owns the selected physical camera.

    The camera index is supplied by app.py.

    Example:

        0 -> iPhone / Continuity Camera
        1 -> MacBook camera

    The actual numbers depend on macOS.
    """

    def __init__(self):

        print(
            "Initializing object detection..."
        )

        self.base_path = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.video = None

        self.camera_index = None

        self.running = False

        # ==================================================
        # LOAD YOLO
        # ==================================================

        self.detector = ObjectDetection()

        self.detector.setModelTypeAsTinyYOLOv3()

        model_path = os.path.join(
            self.base_path,
            "tiny-yolov3.pt"
        )

        print(
            "Model path:",
            model_path
        )

        if not os.path.exists(model_path):

            print(
                "WARNING: YOLO model file not found:"
            )

            print(
                model_path
            )

        self.detector.setModelPath(
            model_path
        )

        try:

            self.detector.loadModel()

            print(
                "Object detection model loaded."
            )

        except Exception as e:

            print(
                "ERROR loading object detection model:",
                e
            )

    # ======================================================
    # OPEN CAMERA
    # ======================================================

    def open_camera(
        self,
        camera_index=0
    ):

        # If already using requested camera
        if (
            self.video is not None
            and self.video.isOpened()
            and self.camera_index == camera_index
        ):

            print(
                "Object detection camera already open."
            )

            self.running = True

            return True

        # Release previous camera
        self.release_camera()

        self.camera_index = camera_index

        print(
            "Opening object detection camera:",
            camera_index
        )

        # ==================================================
        # macOS AVFOUNDATION
        # ==================================================

        self.video = cv2.VideoCapture(
            camera_index,
            cv2.CAP_AVFOUNDATION
        )

        if not self.video.isOpened():

            print(
                "AVFoundation failed."
            )

            try:
                self.video.release()
            except Exception:
                pass

            self.video = cv2.VideoCapture(
                camera_index
            )

        if not self.video.isOpened():

            print(
                "ERROR: Could not open camera:",
                camera_index
            )

            self.video = None
            self.running = False

            return False

        # ==================================================
        # CAMERA SETTINGS
        # ==================================================

        self.video.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        self.video.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        self.running = True

        print(
            "Object detection camera opened successfully:",
            camera_index
        )

        return True

    # ======================================================
    # RELEASE CAMERA
    # ======================================================

    def release_camera(self):

        self.running = False

        if self.video is not None:

            print(
                "Releasing object detection camera..."
            )

            try:

                self.video.release()

            except Exception as e:

                print(
                    "Camera release error:",
                    e
                )

            self.video = None

        self.camera_index = None

    # ======================================================
    # READ FRAME
    # ======================================================

    def read_frame(self):

        if (
            self.video is None
            or not self.video.isOpened()
        ):

            return None

        ret, frame = self.video.read()

        if not ret or frame is None:

            print(
                "ERROR: Could not read camera frame."
            )

            return None

        return frame

    # ======================================================
    # GET PROCESSED FRAME
    # ======================================================

    def get_frame(self):

        frame = self.read_frame()

        if frame is None:

            return None

        return self.process_frame(
            frame
        )

    # ======================================================
    # PROCESS FRAME
    # ======================================================

    def process_frame(
        self,
        frame
    ):

        if frame is None:

            return None

        try:

            (
                detected_image,
                detections
            ) = (
                self.detector
                .detectObjectsFromImage(

                    input_image=frame,

                    output_type="array",

                    minimum_percentage_probability=40,

                    display_percentage_probability=True,

                    display_object_name=True,

                    display_box=True
                )
            )

            # ==================================================
            # JPEG
            # ==================================================

            ret, jpeg = cv2.imencode(
                ".jpg",
                detected_image
            )

            if not ret:

                return None

            return jpeg.tobytes()

        except Exception as e:

            print(
                "Object detection error:",
                e
            )

            # ==================================================
            # FALLBACK
            # ==================================================

            ret, jpeg = cv2.imencode(
                ".jpg",
                frame
            )

            if ret:

                return jpeg.tobytes()

            return None

    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):

        print(
            "Closing object detection..."
        )

        self.release_camera()
