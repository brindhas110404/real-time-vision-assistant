from flask import (
    Flask,
    render_template,
    Response,
    redirect,
    request,
    url_for,
    jsonify,
    session
)

from object_camera import VideoCamera
from facecam import Facecamera

import atexit
import threading
import time
import os
import secrets


# ==========================================================
# FLASK
# ==========================================================

app = Flask(__name__)

# Needed for session
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)


# ==========================================================
# CAMERA SETTINGS
# ==========================================================

# Default camera.
#
# IMPORTANT:
#
# Camera indexes vary by device and operating system.
#
DEFAULT_CAMERA_INDEX = 0

# Number of camera indexes we will make available
MAX_CAMERA_INDEX = 5


def get_selected_camera():
    """
    Get the camera selected by the user.

    If the user has not selected one yet,
    use DEFAULT_CAMERA_INDEX.
    """

    try:

        camera_index = session.get(
            "camera_index",
            DEFAULT_CAMERA_INDEX
        )

        return int(camera_index)

    except Exception:

        return DEFAULT_CAMERA_INDEX


# ==========================================================
# CAMERA CONTROLLERS
# ==========================================================

print("Creating object detection controller...")

video_stream = VideoCamera()


print("Creating face recognition controller...")

face_video_stream = Facecamera()


# ==========================================================
# CAMERA LOCK
# ==========================================================

camera_lock = threading.Lock()


# ==========================================================
# CAMERA SETTINGS PAGE
# ==========================================================

@app.route("/camera_settings", methods=["GET", "POST"])
def camera_settings():

    if request.method == "POST":

        camera_index = request.form.get(
            "camera_index",
            ""
        ).strip()

        try:

            camera_index = int(camera_index)

            if camera_index < 0:

                raise ValueError

        except ValueError:

            return render_template(
                "camera_settings.html",
                camera_indexes=range(
                    MAX_CAMERA_INDEX + 1
                ),
                selected_camera=get_selected_camera(),
                error="Invalid camera selection."
            )

        print(
            "REQUEST: Change camera"
        )

        print(
            "Selected camera index:",
            camera_index
        )

        # Stop both cameras before changing device
        with camera_lock:

            try:
                video_stream.release_camera()
            except Exception as e:
                print(
                    "Object camera release error:",
                    e
                )

            try:
                face_video_stream.release_camera()
            except Exception as e:
                print(
                    "Face camera release error:",
                    e
                )

        # Save selection
        session["camera_index"] = camera_index

        print(
            "Camera setting saved:",
            camera_index
        )

        return redirect(
            url_for("camera_settings")
        )

    return render_template(
        "camera_settings.html",
        camera_indexes=range(
            MAX_CAMERA_INDEX + 1
        ),
        selected_camera=get_selected_camera()
    )


# ==========================================================
# OBJECT DETECTION STREAM
# ==========================================================

def gen_object():

    print(
        "Object video stream started."
    )

    while True:

        with camera_lock:

            frame = video_stream.get_frame()

        if frame is None:

            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


# ==========================================================
# FACE RECOGNITION STREAM
# ==========================================================

def gen_face():

    print(
        "Face recognition video stream started."
    )

    while True:

        with camera_lock:

            frame = face_video_stream.Detection()

        if frame is None:

            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def index():

    print(
        "REQUEST: Home"
    )

    return render_template(
        "object_index.html"
    )


# ==========================================================
# OBJECT DETECTION PAGE
# ==========================================================

@app.route("/object_recog")
def object_recog():

    print(
        "REQUEST: Switch to object detection"
    )

    with camera_lock:

        try:
            face_video_stream.release_camera()
        except Exception as e:
            print(
                "Face release error:",
                e
            )

        try:
            video_stream.release_camera()
        except Exception as e:
            print(
                "Object release error:",
                e
            )

    return render_template(
        "object_index.html"
    )


# ==========================================================
# START OBJECT DETECTION
# ==========================================================

@app.route(
    "/object_start",
    methods=["POST"]
)
def object_start():

    print(
        "REQUEST: Start object detection"
    )

    camera_index = get_selected_camera()

    print(
        "Using camera index:",
        camera_index
    )

    with camera_lock:

        # Make sure face camera is OFF
        face_video_stream.release_camera()

        time.sleep(0.2)

        # Open selected camera
        success = video_stream.open_camera(
            camera_index
        )

        if not success:

            return jsonify({
                "success": False,
                "message": (
                    "Could not open selected camera "
                    f"(index {camera_index}). "
                    "Check Camera Settings and macOS "
                    "camera permissions."
                )
            }), 500

    return jsonify({
        "success": True,
        "stream_url": url_for(
            "object_stream"
        )
    })


# ==========================================================
# OBJECT DETECTION STREAM ROUTE
# ==========================================================

@app.route("/object_stream")
def object_stream():

    return Response(
        gen_object(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ==========================================================
# STOP OBJECT DETECTION
# ==========================================================

@app.route(
    "/object_close_video",
    methods=["POST"]
)
def object_close_video():

    print(
        "REQUEST: Stop object detection"
    )

    with camera_lock:

        video_stream.release_camera()

    return jsonify({
        "success": True
    })


# ==========================================================
# FACE RECOGNITION PAGE
# ==========================================================

@app.route("/face_recog")
def face_recog():

    print(
        "REQUEST: Switch to face recognition"
    )

    with camera_lock:

        # Release object camera
        video_stream.release_camera()

        time.sleep(0.2)

        # Do not open face camera yet
        # Start button will open it

    return render_template(
        "face_index.html"
    )


# ==========================================================
# START FACE RECOGNITION
# ==========================================================

@app.route(
    "/face_start",
    methods=["POST"]
)
def face_start():

    print(
        "REQUEST: Start face recognition"
    )

    camera_index = get_selected_camera()

    print(
        "Using camera index:",
        camera_index
    )

    with camera_lock:

        # Make sure object camera is OFF
        video_stream.release_camera()

        time.sleep(0.2)

        # Open selected camera
        success = face_video_stream.open_camera(
            camera_index
        )

        if not success:

            return jsonify({
                "success": False,
                "message": (
                    "Could not open selected camera "
                    f"(index {camera_index}). "
                    "Check Camera Settings and macOS "
                    "camera permissions."
                )
            }), 500

    return jsonify({
        "success": True,
        "stream_url": url_for(
            "face_stream"
        )
    })


# ==========================================================
# FACE RECOGNITION STREAM
# ==========================================================

@app.route("/face_stream")
def face_stream():

    return Response(
        gen_face(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ==========================================================
# ADD NEW FACE
# ==========================================================

@app.route("/Add_face")
def Add_face():

    print(
        "REQUEST: Add new face"
    )

    camera_index = get_selected_camera()

    print(
        "Using camera index:",
        camera_index
    )

    with camera_lock:

        # Object camera OFF
        video_stream.release_camera()

        time.sleep(0.2)

        # Open selected camera
        if not face_video_stream.open_camera(
            camera_index
        ):

            return render_template(
                "face_index.html",
                error=(
                    "Could not open selected camera "
                    f"(index {camera_index})."
                )
            )

        # Capture one image
        captured = face_video_stream.capture_face()

    if captured is None:

        return render_template(
            "face_index.html",
            error="Could not capture a valid photo."
        )

    print(
        "Captured image:",
        captured["path"]
    )

    print(
        "Preview URL:",
        captured["url"]
    )

    return render_template(
        "capture_preview.html",
        image_url=captured["url"],
        filename=captured["filename"]
    )


# ==========================================================
# SAVE NEW FACE
# ==========================================================

@app.route(
    "/save_face",
    methods=["POST"]
)
def save_face():

    filename = request.form.get(
        "filename",
        ""
    ).strip()

    person_name = request.form.get(
        "person_name",
        ""
    ).strip()

    print(
        "REQUEST: Save face"
    )

    print(
        "Name:",
        person_name
    )

    print(
        "File:",
        filename
    )

    if not filename:

        return render_template(
            "face_index.html",
            error=(
                "No captured image was provided."
            )
        )

    if not person_name:

        return render_template(
            "capture_preview.html",
            image_url=(
                url_for(
                    "static",
                    filename=(
                        "captured_faces/"
                        + filename
                    )
                )
            ),
            filename=filename,
            error="Please enter a name."
        )

    with camera_lock:

        success, message = (
            face_video_stream.register_face(
                filename,
                person_name
            )
        )

    if not success:

        return render_template(
            "capture_preview.html",
            image_url=(
                url_for(
                    "static",
                    filename=(
                        "captured_faces/"
                        + filename
                    )
                )
            ),
            filename=filename,
            error=message
        )

    print(
        "Face successfully registered:",
        person_name
    )

    return redirect(
        url_for("face_recog")
    )


# ==========================================================
# CANCEL / RETAKE FACE
# ==========================================================

@app.route(
    "/cancel_face",
    methods=["POST"]
)
def cancel_face():

    filename = request.form.get(
        "filename",
        ""
    ).strip()

    print(
        "REQUEST: Cancel face capture"
    )

    if filename:

        file_path = os.path.join(
            face_video_stream.capture_dir,
            filename
        )

        try:

            if os.path.exists(file_path):

                os.remove(file_path)

                print(
                    "Temporary image deleted:",
                    file_path
                )

        except Exception as e:

            print(
                "Could not delete temporary image:",
                e
            )

    return redirect(
        url_for("face_recog")
    )


# ==========================================================
# STOP FACE RECOGNITION
# ==========================================================

@app.route(
    "/close_video",
    methods=["POST"]
)
def close_video():

    print(
        "REQUEST: Stop face recognition"
    )

    with camera_lock:

        face_video_stream.release_camera()

    return jsonify({
        "success": True
    })


# ==========================================================
# CLEANUP
# ==========================================================

def cleanup():

    print(
        "Cleaning up cameras..."
    )

    try:

        video_stream.close()

    except Exception as e:

        print(
            "Object camera cleanup error:",
            e
        )

    try:

        face_video_stream.close()

    except Exception as e:

        print(
            "Face camera cleanup error:",
            e
        )


atexit.register(
    cleanup
)


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    print(
        "======================================"
    )

    print(
        "Starting Virtual Assistant"
    )

    print(
        "Camera can be selected from:"
    )

    print(
        "http://127.0.0.1:5000/camera_settings"
    )

    print(
        "======================================"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )
