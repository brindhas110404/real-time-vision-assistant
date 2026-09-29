import cv2
import os
import time
import face_recognition


class Facecamera:

    """
    Face recognition controller.

    Uses the camera selected from Camera Settings.

    The selected camera is used for:
        - Face Recognition
        - Add New Face
    """

    def __init__(self):

        print("Initializing face recognition...")

        self.base_path = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.video = None
        self.camera_index = None
        self.running = False

        self.known_face_encodings = []
        self.known_face_names = []

        self.capture_dir = os.path.join(
            self.base_path,
            "static",
            "captured_faces"
        )

        os.makedirs(
            self.capture_dir,
            exist_ok=True
        )

        self.load_known_faces()

    # ======================================================
    # LOAD KNOWN FACES
    # ======================================================

    def load_known_faces(self):

        faces_dir = os.path.join(
            self.base_path,
            "faces"
        )

        os.makedirs(
            faces_dir,
            exist_ok=True
        )

        self.known_face_encodings = []
        self.known_face_names = []

        for filename in os.listdir(faces_dir):

            file_path = os.path.join(
                faces_dir,
                filename
            )

            if not os.path.isfile(file_path):
                continue

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in (
                ".jpg",
                ".jpeg",
                ".png"
            ):
                continue

            try:

                image = face_recognition.load_image_file(
                    file_path
                )

                encodings = face_recognition.face_encodings(
                    image
                )

                if len(encodings) == 0:

                    print(
                        "No face found in:",
                        filename
                    )

                    continue

                self.known_face_encodings.append(
                    encodings[0]
                )

                name = os.path.splitext(
                    filename
                )[0]

                self.known_face_names.append(
                    name
                )

            except Exception as e:

                print(
                    "Could not load face:",
                    filename,
                    e
                )

        print(
            "Known faces loaded:",
            self.known_face_names
        )

    # ======================================================
    # OPEN CAMERA
    # ======================================================

    def open_camera(
        self,
        camera_index=0
    ):

        if (
            self.video is not None
            and self.video.isOpened()
            and self.camera_index == camera_index
        ):

            print(
                "Face recognition camera already open:",
                camera_index
            )

            self.running = True

            return True

        self.release_camera()

        self.camera_index = camera_index

        print(
            "Opening camera for face recognition:",
            camera_index
        )

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

        if self.video.isOpened():

            print(
                "Camera opened successfully:",
                camera_index
            )

            self.video.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                1280
            )

            self.video.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                720
            )

            self.running = True

            return True

        print(
            "ERROR: Could not open camera:",
            camera_index
        )

        self.video = None
        self.camera_index = None
        self.running = False

        return False

    # ======================================================
    # RELEASE CAMERA
    # ======================================================

    def release_camera(self):

        self.running = False

        if self.video is not None:

            print(
                "Releasing face recognition camera..."
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

        time.sleep(0.2)

    # ======================================================
    # READ RAW FRAME
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
    # FACE DETECTION
    # ======================================================

    def Detection(self):

        frame = self.read_frame()

        if frame is None:
            return None

        try:

            # --------------------------------------------------
            # REDUCE IMAGE SIZE FOR FASTER RECOGNITION
            # --------------------------------------------------

            small_frame = cv2.resize(
                frame,
                (0, 0),
                fx=0.25,
                fy=0.25
            )

            rgb_small_frame = cv2.cvtColor(
                small_frame,
                cv2.COLOR_BGR2RGB
            )

            face_locations = face_recognition.face_locations(
                rgb_small_frame
            )

            face_encodings = face_recognition.face_encodings(
                rgb_small_frame,
                face_locations
            )

            # --------------------------------------------------
            # PROCESS EACH FACE
            # --------------------------------------------------

            for (
                face_location,
                face_encoding
            ) in zip(
                face_locations,
                face_encodings
            ):

                name = "Unknown"

                # --------------------------------------------------
                # RECOGNIZE FACE
                # --------------------------------------------------

                if self.known_face_encodings:

                    matches = face_recognition.compare_faces(
                        self.known_face_encodings,
                        face_encoding,
                        tolerance=0.5
                    )

                    face_distances = (
                        face_recognition.face_distance(
                            self.known_face_encodings,
                            face_encoding
                        )
                    )

                    if len(face_distances) > 0:

                        best_match_index = (
                            face_distances.argmin()
                        )

                        if matches[best_match_index]:

                            name = (
                                self.known_face_names[
                                    best_match_index
                                ]
                            )

                # --------------------------------------------------
                # SCALE FACE COORDINATES BACK TO ORIGINAL FRAME
                # --------------------------------------------------

                top, right, bottom, left = face_location

                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Make sure coordinates stay inside image
                top = max(0, top)
                left = max(0, left)

                right = min(
                    frame.shape[1] - 1,
                    right
                )

                bottom = min(
                    frame.shape[0] - 1,
                    bottom
                )

                # ==================================================
                # FACE BORDER
                # ==================================================
                #
                # IMPORTANT:
                # This is ONLY an outline.
                #
                # Do NOT use cv2.FILLED here.
                #

                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    (0, 255, 0),
                    2
                )

                # ==================================================
                # NAME LABEL
                # ==================================================
                #
                # Only the small area behind the name is filled.
                # The face itself remains completely visible.
                #

                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.7
                font_thickness = 1

                (
                    text_width,
                    text_height
                ), baseline = cv2.getTextSize(
                    name,
                    font,
                    font_scale,
                    font_thickness
                )

                padding_x = 8
                padding_y = 6

                label_width = (
                    text_width
                    + padding_x * 2
                )

                label_height = (
                    text_height
                    + baseline
                    + padding_y * 2
                )

                # Put label above the face if there is room
                label_bottom = top

                label_top = (
                    label_bottom
                    - label_height
                )

                # If there isn't enough room above,
                # put the label just inside the top of the box.
                if label_top < 0:

                    label_top = top

                    label_bottom = min(
                        frame.shape[0],
                        top + label_height
                    )

                label_right = min(
                    frame.shape[1],
                    left + label_width
                )

                # --------------------------------------------------
                # FILLED LABEL ONLY
                # --------------------------------------------------

                cv2.rectangle(
                    frame,
                    (left, label_top),
                    (label_right, label_bottom),
                    (0, 255, 0),
                    cv2.FILLED
                )

                # --------------------------------------------------
                # WHITE NAME TEXT
                # --------------------------------------------------

                text_x = (
                    left
                    + padding_x
                )

                text_y = (
                    label_bottom
                    - padding_y
                    - baseline
                )

                cv2.putText(
                    frame,
                    name,
                    (
                        text_x,
                        text_y
                    ),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA
                )

            # ==================================================
            # JPEG
            # ==================================================

            success, jpeg = cv2.imencode(
                ".jpg",
                frame,
                [
                    int(
                        cv2.IMWRITE_JPEG_QUALITY
                    ),
                    90
                ]
            )

            if not success:
                return None

            return jpeg.tobytes()

        except Exception as e:

            print(
                "Face recognition error:",
                e
            )

            # Fallback: return normal camera frame
            success, jpeg = cv2.imencode(
                ".jpg",
                frame
            )

            if success:
                return jpeg.tobytes()

            return None

    # ======================================================
    # CAPTURE NEW FACE
    # ======================================================

    def capture_face(self):

        print(
            "Capturing new face..."
        )

        if (
            self.video is None
            or not self.video.isOpened()
        ):

            print(
                "Camera is not open."
            )

            return None

        frame = None

        for _ in range(8):

            ret, current_frame = (
                self.video.read()
            )

            if (
                ret
                and current_frame is not None
            ):

                frame = current_frame

            time.sleep(0.05)

        if frame is None:

            print(
                "ERROR: Could not capture face image."
            )

            return None

        # ==================================================
        # JPEG ENCODING
        # ==================================================

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                95
            ]
        )

        if not success:

            print(
                "ERROR: JPEG encoding failed."
            )

            return None

        filename = (
            "capture_"
            + str(
                int(
                    time.time() * 1000
                )
            )
            + ".jpg"
        )

        file_path = os.path.join(
            self.capture_dir,
            filename
        )

        # ==================================================
        # SAVE IMAGE
        # ==================================================

        try:

            with open(
                file_path,
                "wb"
            ) as image_file:

                image_file.write(
                    encoded.tobytes()
                )

        except Exception as e:

            print(
                "ERROR: Could not save image:",
                e
            )

            return None

        # ==================================================
        # VERIFY FILE
        # ==================================================

        if not os.path.exists(file_path):

            print(
                "ERROR: Image file was not created."
            )

            return None

        if os.path.getsize(file_path) == 0:

            print(
                "ERROR: Image file is empty."
            )

            return None

        # ==================================================
        # VERIFY OPENCV
        # ==================================================

        test_image = cv2.imread(
            file_path
        )

        if test_image is None:

            print(
                "ERROR: Saved image is corrupted."
            )

            try:
                os.remove(file_path)
            except Exception:
                pass

            return None

        print(
            "Face captured successfully:"
        )

        print(
            file_path
        )

        print(
            "Image size:",
            os.path.getsize(file_path),
            "bytes"
        )

        return {
            "filename": filename,
            "path": file_path,
            "url": (
                "/static/captured_faces/"
                + filename
            )
        }

    # ======================================================
    # REGISTER FACE
    # ======================================================

    def register_face(
        self,
        image_filename,
        person_name
    ):

        person_name = person_name.strip()

        if not person_name:

            return (
                False,
                "Name is required."
            )

        # ==================================================
        # SOURCE IMAGE
        # ==================================================

        source_path = os.path.join(
            self.capture_dir,
            image_filename
        )

        if not os.path.exists(source_path):

            return (
                False,
                "Captured image not found."
            )

        image = cv2.imread(
            source_path
        )

        if image is None:

            return (
                False,
                "Captured image is corrupted."
            )

        # ==================================================
        # FACE DETECTION
        # ==================================================

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        locations = (
            face_recognition.face_locations(
                rgb_image
            )
        )

        if len(locations) == 0:

            return (
                False,
                "No face detected."
            )

        if len(locations) > 1:

            return (
                False,
                "Please capture an image "
                "with only one person."
            )

        encodings = (
            face_recognition.face_encodings(
                rgb_image,
                locations
            )
        )

        if len(encodings) == 0:

            return (
                False,
                "Could not encode face."
            )

        # ==================================================
        # SAFE FILE NAME
        # ==================================================

        safe_name = "".join(
            c
            for c in person_name
            if c.isalnum()
            or c in (
                " ",
                "_",
                "-"
            )
        ).strip()

        safe_name = safe_name.replace(
            " ",
            "_"
        )

        if not safe_name:

            return (
                False,
                "Please enter a valid name."
            )

        faces_dir = os.path.join(
            self.base_path,
            "faces"
        )

        os.makedirs(
            faces_dir,
            exist_ok=True
        )

        destination = os.path.join(
            faces_dir,
            safe_name + ".jpg"
        )

        # ==================================================
        # SAVE FACE
        # ==================================================

        saved = cv2.imwrite(
            destination,
            image
        )

        if not saved:

            return (
                False,
                "Could not save face image."
            )

        # ==================================================
        # UPDATE MEMORY
        # ==================================================

        existing_index = None

        for i, existing_name in enumerate(
            self.known_face_names
        ):

            if (
                existing_name.lower()
                == person_name.lower()
            ):

                existing_index = i
                break

        if existing_index is not None:

            self.known_face_encodings[
                existing_index
            ] = encodings[0]

            self.known_face_names[
                existing_index
            ] = person_name

        else:

            self.known_face_encodings.append(
                encodings[0]
            )

            self.known_face_names.append(
                person_name
            )

        print(
            "New face registered:",
            person_name
        )

        print(
            "Saved to:",
            destination
        )

        return (
            True,
            "Face registered successfully."
        )

    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):

        print(
            "Closing face recognition..."
        )

        self.release_camera()
