# ============================================================
# VIGILX-AI
# CAMERA WORKER API
# Webcam + YOLO + MJPEG Stream + Event Logging + Evidence
# Camera ON/OFF Control + CORS
# ============================================================

import time
import json
import threading
from pathlib import Path
from datetime import datetime

import cv2
from flask import Flask, Response, jsonify
from flask_cors import CORS

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# ------------------------------------------------------------
# PATHS / SETTINGS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "yolo11n.pt"
EVENT_FILE = BASE_DIR / "events" / "events.json"
ROOT_EVENT_FILE = BASE_DIR / "events.json"
EVIDENCE_DIR = BASE_DIR / "events" / "evidence"

CAMERA_INDEX = 0

CONFIDENCE = 0.70
EVENT_COOLDOWN = 5
MAX_EVENTS = 1000

HOST = "127.0.0.1"
PORT = 8000

EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# FLASK APP
# ------------------------------------------------------------

app = Flask(__name__)

# Allow React/Vite frontend to communicate with Flask
CORS(app)


# ------------------------------------------------------------
# GLOBAL CAMERA STATE
# ------------------------------------------------------------

camera = None
model = None

camera_running = False
camera_enabled = True

latest_frame = None
latest_jpeg = None

person_count = 0
severity = "LOW"

last_event_time = 0

state_lock = threading.Lock()


# ------------------------------------------------------------
# EVENT FUNCTIONS
# ------------------------------------------------------------

def load_events():

    if not EVENT_FILE.exists():
        return []

    try:

        with open(
            EVENT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, list):
            return data

    except Exception:
        pass

    return []


def save_events(events):

    try:

        for target_file in [EVENT_FILE, ROOT_EVENT_FILE]:
            with open(
                target_file,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    events[:MAX_EVENTS],
                    file,
                    indent=4,
                    ensure_ascii=False
                )

        return True

    except Exception as error:

        print(
            f"[EVENT ERROR] {error}",
            flush=True
        )

        return False


def calculate_severity(count):

    if count >= 6:
        return "SENSITIVE"

    if count >= 3:
        return "MEDIUM"

    if count >= 1:
        return "LOW"

    return "LOW"


def save_event(
    event_type,
    message,
    count,
    event_severity,
    frame
):

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    filename = (
        "evidence_"
        f"{now.strftime('%Y%m%d_%H%M%S')}_"
        f"{event_type}.jpg"
    )

    evidence_path = EVIDENCE_DIR / filename
    evidence_str = ""

    try:
        cv2.imwrite(
            str(evidence_path),
            frame
        )
        evidence_str = str(evidence_path)
        print(
            f"[EVIDENCE] {evidence_path.name}",
            flush=True
        )
    except Exception as error:
        print(
            f"[EVIDENCE ERROR] {error}",
            flush=True
        )

    event = {
        "timestamp": now_str,
        "event": event_type,
        "event_type": event_type,
        "description": message,
        "message": message,
        "severity": event_severity,
        "persons": count,
        "person_count": count,
        "vehicles": 0,
        "phones": 0,
        "source": "CAM 01",
        "evidence": evidence_str,
    }

    events = load_events()
    events.append(event)
    save_events(events)


    print(

        f"[EVENT] "
        f"{event['timestamp']} | "
        f"{event_type} | "
        f"Persons: {count} | "
        f"Severity: {event_severity}",

        flush=True
    )


# ------------------------------------------------------------
# CAMERA OPEN
# ------------------------------------------------------------

def open_camera():

    global camera
    global camera_running

    print(
        "Starting CAM 01...",
        flush=True
    )

    # Try DirectShow first
    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    # Fallback
    if not camera.isOpened():

        camera.release()

        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not camera.isOpened():

        print(
            "[ERROR] CAM 01 could not be opened.",
            flush=True
        )

        camera = None
        camera_running = False

        return False

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    try:

        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

    except Exception:
        pass

    camera_running = True

    print(
        "CAM 01 started successfully.",
        flush=True
    )

    print(
        f"Stream: http://{HOST}:{PORT}/video_feed",
        flush=True
    )

    return True


# ------------------------------------------------------------
# CAMERA CLOSE
# ------------------------------------------------------------

def close_camera():

    global camera
    global camera_running
    global latest_frame
    global latest_jpeg
    global person_count
    global severity

    camera_running = False

    if camera is not None:

        try:
            camera.release()
        except Exception:
            pass

        camera = None

    with state_lock:

        latest_frame = None
        latest_jpeg = None
        person_count = 0
        severity = "LOW"

    print(
        "CAM 01 released.",
        flush=True
    )


# ------------------------------------------------------------
# CAMERA LOOP
# ------------------------------------------------------------

def camera_loop():

    global model
    global camera_running
    global latest_frame
    global latest_jpeg
    global person_count
    global severity
    global last_event_time

    print(
        "========================================",
        flush=True
    )

    print(
        "       VIGILX-AI CAMERA SERVICE",
        flush=True
    )

    print(
        "========================================",
        flush=True
    )

    # --------------------------------------------------------
    # LOAD YOLO
    # --------------------------------------------------------

    if YOLO is not None and MODEL_PATH.exists():

        print(
            "Loading YOLO model...",
            flush=True
        )

        try:

            model = YOLO(
                str(MODEL_PATH)
            )

            print(
                "YOLO model loaded successfully.",
                flush=True
            )

        except Exception as error:

            print(
                f"[YOLO ERROR] {error}",
                flush=True
            )

            model = None

    else:

        print(
            "[WARNING] YOLO model unavailable.",
            flush=True
        )

    # --------------------------------------------------------
    # AUTO START CAMERA
    # --------------------------------------------------------

    if camera_enabled:

        open_camera()

    # --------------------------------------------------------
    # MAIN SERVICE LOOP
    # --------------------------------------------------------

    while True:

        # ----------------------------------------------------
        # CAMERA OFF
        # ----------------------------------------------------

        if not camera_enabled:

            if camera_running:

                close_camera()

            time.sleep(0.1)

            continue

        # ----------------------------------------------------
        # CAMERA ON BUT NOT RUNNING
        # ----------------------------------------------------

        if not camera_running:

            if not open_camera():

                time.sleep(2)

                continue

        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        ret, frame = camera.read()

        if not ret or frame is None:

            print(
                "[WARNING] Frame capture failed.",
                flush=True
            )

            time.sleep(0.1)

            continue

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        current_count = 0

        annotated_frame = frame.copy()

        if model is not None:

            try:

                results = model.predict(

                    source=frame,

                    conf=CONFIDENCE,

                    classes=[0],

                    imgsz=320,

                    verbose=False,

                    device="cpu"
                )

                result = results[0]

                if result.boxes is not None:

                    current_count = len(
                        result.boxes
                    )

                annotated_frame = result.plot()

            except Exception as error:

                print(
                    f"[YOLO ERROR] {error}",
                    flush=True
                )

        # ----------------------------------------------------
        # THREAT LEVEL
        # ----------------------------------------------------

        current_severity = calculate_severity(
            current_count
        )

        # ----------------------------------------------------
        # EVENT LOGGING
        # ----------------------------------------------------

        now = time.time()

        if (

            current_count > 0

            and

            now - last_event_time
            >= EVENT_COOLDOWN

        ):

            save_event(

                event_type="PERSON_DETECTED",

                message=(
                    f"Person detected: "
                    f"{current_count}"
                ),

                count=current_count,

                event_severity=current_severity,

                frame=annotated_frame
            )

            last_event_time = now

        # ----------------------------------------------------
        # VIGILX OVERLAY
        # ----------------------------------------------------

        height, width = (
            annotated_frame.shape[:2]
        )

        cv2.rectangle(

            annotated_frame,

            (0, 0),

            (width, 72),

            (0, 0, 0),

            -1
        )

        cv2.putText(

            annotated_frame,

            "VIGILX-AI | CAM 01 | LIVE",

            (15, 28),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2,

            cv2.LINE_AA
        )

        cv2.putText(

            annotated_frame,

            f"PERSONS DETECTED: {current_count}",

            (15, 56),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1,

            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # ENCODE JPEG
        # ----------------------------------------------------

        success, buffer = cv2.imencode(

            ".jpg",

            annotated_frame,

            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                80
            ]
        )

        if success:

            with state_lock:

                latest_frame = annotated_frame

                latest_jpeg = buffer.tobytes()

                person_count = current_count

                severity = current_severity

        time.sleep(0.01)


# ------------------------------------------------------------
# MJPEG STREAM
# ------------------------------------------------------------

def generate_frames():

    while True:

        with state_lock:

            frame = latest_jpeg

        if frame is None:

            time.sleep(0.05)

            continue

        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            + frame

            + b"\r\n"
        )


# ------------------------------------------------------------
# VIDEO ENDPOINT
# ------------------------------------------------------------

@app.route("/video_feed")
def video_feed():

    return Response(

        generate_frames(),

        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ------------------------------------------------------------
# STATUS ENDPOINT
# ------------------------------------------------------------

@app.route("/status")
def status():

    with state_lock:

        return jsonify({

            "camera":
                "CAM 01",

            "running":
                camera_running,

            "enabled":
                camera_enabled,

            "person_count":
                person_count,

            "severity":
                severity,

            "ai":
                model is not None,

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        })


# ------------------------------------------------------------
# CAMERA START
# ------------------------------------------------------------

@app.route(
    "/camera/start",
    methods=["POST"]
)
def start_camera():

    global camera_enabled

    camera_enabled = True

    # Camera loop will open it
    # immediately

    return jsonify({

        "success": True,

        "message":
            "CAM 01 start command received.",

        "running":
            camera_running,

        "enabled":
            camera_enabled
    })


# ------------------------------------------------------------
# CAMERA STOP
# ------------------------------------------------------------

@app.route(
    "/camera/stop",
    methods=["POST"]
)
def stop_camera():

    global camera_enabled

    camera_enabled = False

    # Release immediately
    close_camera()

    return jsonify({

        "success": True,

        "message":
            "CAM 01 stopped.",

        "running":
            False,

        "enabled":
            False
    })


# ------------------------------------------------------------
# START SERVICE
# ------------------------------------------------------------

if __name__ == "__main__":

    worker_thread = threading.Thread(

        target=camera_loop,

        daemon=True
    )

    worker_thread.start()

    print(
        "VIGILX-AI backend starting...",
        flush=True
    )

    app.run(

        host=HOST,

        port=PORT,

        debug=False,

        threaded=True,

        use_reloader=False
    )