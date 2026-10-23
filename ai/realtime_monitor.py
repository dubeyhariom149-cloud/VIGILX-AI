import cv2
import time
import threading
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

from event_logger import save_event


# =========================================================
# VIGILX-AI REALTIME MONITOR
# =========================================================

MODEL_PATH = "yolo11n.pt"
CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.60
PERSON_CONFIDENCE = 0.70
PHONE_CONFIDENCE = 0.65

EVENT_COOLDOWN = 5
IMG_SIZE = 416

EVIDENCE_DIR = Path("events/evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# YOLO COCO CLASSES
# =========================================================

PERSON = 0
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7
CELL_PHONE = 67

DETECTION_CLASSES = [
    PERSON,
    CAR,
    MOTORCYCLE,
    BUS,
    TRUCK,
    CELL_PHONE
]


# =========================================================
# SHARED DATA
# =========================================================

latest_frame = None
latest_frame_id = 0

latest_detections = []

latest_counts = {
    "person": 0,
    "car": 0,
    "motorcycle": 0,
    "bus": 0,
    "truck": 0,
    "phone": 0
}

latest_threat_level = "LOW"
latest_threat_reason = "Area clear"

detection_ready = False

frame_lock = threading.Lock()
detection_lock = threading.Lock()

stop_event = threading.Event()


# =========================================================
# EVENT STATE
# =========================================================

last_person_state = False
last_person_count = 0
last_vehicle_count = 0
last_phone_count = 0
last_event_time = 0


# =========================================================
# ALERT
# =========================================================

def add_alert(message):

    current_time = datetime.now().strftime("%H:%M:%S")

    print(
        f"[{current_time}] ALERT: {message}"
    )


# =========================================================
# EVIDENCE SNAPSHOT
# =========================================================

def save_evidence_snapshot(frame, event_type):

    try:

        if frame is None:
            return None

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H%M%S_%f"
        )[:-3]

        safe_event_type = (
            event_type
            .replace(" ", "_")
            .replace("/", "_")
        )

        filename = (
            f"{safe_event_type}_{timestamp}.jpg"
        )

        filepath = EVIDENCE_DIR / filename

        success = cv2.imwrite(
            str(filepath),
            frame
        )

        if success:

            print(
                f"[EVIDENCE] Snapshot saved: {filepath}"
            )

            return str(filepath)

        print(
            "[EVIDENCE] Failed to save snapshot."
        )

        return None

    except Exception as error:

        print(
            f"[EVIDENCE] Error: {error}"
        )

        return None


# =========================================================
# SEVERITY
# =========================================================

def calculate_severity(person_count):

    if person_count <= 2:
        return "LOW"

    elif person_count <= 5:
        return "MEDIUM"

    else:
        return "SENSITIVE"


# =========================================================
# REAL-TIME THREAT ENGINE
# =========================================================

def calculate_threat_level(
    person_count,
    vehicle_count,
    phone_count
):

    # CRITICAL
    if person_count > 5:
        return "CRITICAL", "Large crowd detected"

    if phone_count > 0 and person_count > 2:
        return (
            "CRITICAL",
            "Multiple persons + phone activity"
        )

    # HIGH
    if phone_count > 0 and person_count > 0:
        return (
            "HIGH",
            "Phone activity detected"
        )

    if person_count > 2:
        return (
            "HIGH",
            "Multiple persons detected"
        )

    # MEDIUM
    if vehicle_count > 0:
        return (
            "MEDIUM",
            "Vehicle detected"
        )

    if person_count == 2:
        return (
            "MEDIUM",
            "Multiple persons detected"
        )

    # LOW
    if person_count == 1:
        return (
            "LOW",
            "Single person detected"
        )

    return (
        "LOW",
        "Area clear"
    )


# =========================================================
# SECURITY EVENT LOGGER
# =========================================================

def log_security_event(
    event_type,
    message,
    person_count,
    frame=None
):

    global last_event_time

    current_time = time.time()

    if current_time - last_event_time < EVENT_COOLDOWN:
        return

    severity = calculate_severity(
        person_count
    )

    if frame is not None:

        save_evidence_snapshot(
            frame,
            event_type
        )

    try:

        save_event(
            event_type=event_type,
            message=message,
            person_count=person_count,
            severity=severity
        )

    except Exception as error:

        print(
            f"WARNING: Event logger error: {error}"
        )

    add_alert(
        f"{message} | Severity: {severity}"
    )

    last_event_time = current_time


# =========================================================
# YOLO DETECTION THREAD
# =========================================================

def detection_worker(model):

    global latest_frame
    global latest_frame_id
    global latest_detections
    global latest_counts
    global detection_ready

    global latest_threat_level
    global latest_threat_reason

    global last_person_state
    global last_person_count
    global last_vehicle_count
    global last_phone_count

    processed_frame_id = -1

    print(
        "[AI] Detection engine started."
    )

    while not stop_event.is_set():

        # ---------------------------------------------
        # Get latest frame
        # ---------------------------------------------

        with frame_lock:

            if (
                latest_frame is None
                or latest_frame_id == processed_frame_id
            ):

                frame_for_detection = None

            else:

                frame_for_detection = latest_frame.copy()

                current_frame_id = latest_frame_id

        if frame_for_detection is None:

            time.sleep(0.005)

            continue

        processed_frame_id = current_frame_id

        # ---------------------------------------------
        # YOLO
        # ---------------------------------------------

        try:

            results = model.predict(
                source=frame_for_detection,
                conf=CONFIDENCE_THRESHOLD,
                classes=DETECTION_CLASSES,
                imgsz=IMG_SIZE,
                verbose=False
            )

        except Exception as error:

            print(
                f"[AI] Detection error: {error}"
            )

            time.sleep(0.1)

            continue

        if not results:
            continue

        result = results[0]

        # ---------------------------------------------
        # Counters
        # ---------------------------------------------

        person_count = 0
        car_count = 0
        motorcycle_count = 0
        bus_count = 0
        truck_count = 0
        phone_count = 0

        detections = []

        # ---------------------------------------------
        # Read detections
        # ---------------------------------------------

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # PERSON
                if (
                    class_id == PERSON
                    and confidence >= PERSON_CONFIDENCE
                ):

                    person_count += 1

                    detections.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                            class_id,
                            confidence
                        )
                    )

                # CAR
                elif (
                    class_id == CAR
                    and confidence >= CONFIDENCE_THRESHOLD
                ):

                    car_count += 1

                    detections.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                            class_id,
                            confidence
                        )
                    )

                # MOTORCYCLE
                elif (
                    class_id == MOTORCYCLE
                    and confidence >= CONFIDENCE_THRESHOLD
                ):

                    motorcycle_count += 1

                    detections.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                            class_id,
                            confidence
                        )
                    )

                # BUS
                elif (
                    class_id == BUS
                    and confidence >= CONFIDENCE_THRESHOLD
                ):

                    bus_count += 1

                    detections.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                            class_id,
                            confidence
                        )
                    )

                # TRUCK
                elif (
                    class_id == TRUCK
                    and confidence >= CONFIDENCE_THRESHOLD
                ):

                    truck_count += 1

                    detections.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                            class_id,
                            confidence
                        )
                    )

                # CELL PHONE
                elif (
                    class_id == CELL_PHONE
                    and confidence >= PHONE_CONFIDENCE
                ):

                    phone_count += 1

                    detections.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                            class_id,
                            confidence
                        )
                    )

        # ---------------------------------------------
        # Vehicle count
        # ---------------------------------------------

        vehicle_count = (
            car_count
            + motorcycle_count
            + bus_count
            + truck_count
        )

        # ---------------------------------------------
        # REAL-TIME THREAT ENGINE
        # ---------------------------------------------

        threat_level, threat_reason = (
            calculate_threat_level(
                person_count,
                vehicle_count,
                phone_count
            )
        )

        print(
            f"[THREAT ENGINE] "
            f"{threat_level} | "
            f"{threat_reason}"
        )

        # ---------------------------------------------
        # Update shared detection data
        # ---------------------------------------------

        with detection_lock:

            latest_detections = detections

            latest_counts = {
                "person": person_count,
                "car": car_count,
                "motorcycle": motorcycle_count,
                "bus": bus_count,
                "truck": truck_count,
                "phone": phone_count
            }

            latest_threat_level = threat_level
            latest_threat_reason = threat_reason

            detection_ready = True

        # ---------------------------------------------
        # Security event processing
        # ---------------------------------------------

        current_person_state = (
            person_count > 0
        )

        # ---------------------------------------------
        # PERSON DETECTED
        # ---------------------------------------------

        if (
            current_person_state
            and not last_person_state
        ):

            log_security_event(
                event_type="PERSON_DETECTED",
                message=(
                    f"Person detected: "
                    f"{person_count}"
                ),
                person_count=person_count,
                frame=frame_for_detection
            )

        # ---------------------------------------------
        # MULTIPLE PERSONS
        # ---------------------------------------------

        elif (
            person_count > 1
            and last_person_count <= 1
        ):

            log_security_event(
                event_type="MULTIPLE_PERSONS",
                message=(
                    f"Multiple persons detected: "
                    f"{person_count}"
                ),
                person_count=person_count,
                frame=frame_for_detection
            )

        # ---------------------------------------------
        # VEHICLE
        # ---------------------------------------------

        if (
            vehicle_count > 0
            and last_vehicle_count == 0
        ):

            log_security_event(
                event_type="VEHICLE_DETECTED",
                message=(
                    f"Vehicle detected: "
                    f"{vehicle_count}"
                ),
                person_count=person_count,
                frame=frame_for_detection
            )

        # ---------------------------------------------
        # CELL PHONE
        # ---------------------------------------------

        if (
            phone_count > 0
            and last_phone_count == 0
        ):

            log_security_event(
                event_type="CELL_PHONE_DETECTED",
                message=(
                    f"Cell phone detected: "
                    f"{phone_count}"
                ),
                person_count=person_count,
                frame=frame_for_detection
            )

        # ---------------------------------------------
        # AREA CLEAR
        # ---------------------------------------------

        elif (
            not current_person_state
            and last_person_state
        ):

            log_security_event(
                event_type="AREA_CLEAR",
                message="No person detected",
                person_count=0,
                frame=None
            )

        # ---------------------------------------------
        # Update states
        # ---------------------------------------------

        last_person_state = (
            current_person_state
        )

        last_person_count = (
            person_count
        )

        last_vehicle_count = (
            vehicle_count
        )

        last_phone_count = (
            phone_count
        )

    print(
        "[AI] Detection engine stopped."
    )


# =========================================================
# DRAW DETECTIONS
# =========================================================

def draw_detections(frame, detections):

    names = {
        PERSON: "PERSON",
        CAR: "CAR",
        MOTORCYCLE: "MOTORCYCLE",
        BUS: "BUS",
        TRUCK: "TRUCK",
        CELL_PHONE: "PHONE"
    }

    for (
        x1,
        y1,
        x2,
        y2,
        class_id,
        confidence
    ) in detections:

        label = names.get(
            class_id,
            "OBJECT"
        )

        label = (
            f"{label} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


# =========================================================
# START SYSTEM
# =========================================================

print("========================================")
print("        VIGILX-AI REALTIME MONITOR")
print("========================================")
print()

print("Loading YOLO model...")

try:

    model = YOLO(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

except Exception as error:

    print(
        "ERROR: YOLO model could not be loaded."
    )

    print(error)

    raise SystemExit


# =========================================================
# CAMERA
# =========================================================

print(
    "Starting camera..."
)

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print(
        "ERROR: Camera could not be opened."
    )

    raise SystemExit


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

print(
    "Camera started successfully."
)

print(
    "Press Q to stop monitoring."
)

print()


# =========================================================
# START AI THREAD
# =========================================================

detection_thread = threading.Thread(
    target=detection_worker,
    args=(model,),
    daemon=True
)

detection_thread.start()


# =========================================================
# MAIN CAMERA LOOP
# =========================================================

fps_start_time = time.time()
fps_frame_count = 0
fps = 0


try:

    while True:

        # ---------------------------------------------
        # Read camera
        # ---------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Could not read camera frame."
            )

            break

        # ---------------------------------------------
        # Send newest frame to AI
        # ---------------------------------------------

        with frame_lock:

            latest_frame = frame.copy()

            latest_frame_id += 1

        # ---------------------------------------------
        # Get latest AI results
        # ---------------------------------------------

        with detection_lock:

            detections = list(
                latest_detections
            )

            counts = dict(
                latest_counts
            )

            threat_level = (
                latest_threat_level
            )

            threat_reason = (
                latest_threat_reason
            )

            ai_ready = detection_ready

        # ---------------------------------------------
        # Draw boxes on CURRENT live frame
        # ---------------------------------------------

        annotated_frame = frame.copy()

        draw_detections(
            annotated_frame,
            detections
        )

        # ---------------------------------------------
        # Counts
        # ---------------------------------------------

        person_count = counts["person"]

        vehicle_count = (
            counts["car"]
            + counts["motorcycle"]
            + counts["bus"]
            + counts["truck"]
        )

        phone_count = counts["phone"]

        # ---------------------------------------------
        # Severity
        # ---------------------------------------------

        severity = calculate_severity(
            person_count
        )

        # ---------------------------------------------
        # Status
        # ---------------------------------------------

        if not ai_ready:

            status_text = (
                "AI STATUS: INITIALIZING"
            )

        elif person_count == 0:

            status_text = "STATUS: CLEAR"

        elif person_count == 1:

            status_text = (
                "STATUS: PERSON DETECTED"
            )

        else:

            status_text = (
                "STATUS: MULTIPLE PERSONS"
            )

        # ---------------------------------------------
        # Header
        # ---------------------------------------------

        cv2.putText(
            annotated_frame,
            "VIGILX-AI | AI SURVEILLANCE",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # ---------------------------------------------
        # Counts
        # ---------------------------------------------

        cv2.putText(
            annotated_frame,
            f"PERSONS: {person_count}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"VEHICLES: {vehicle_count}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"PHONES: {phone_count}",
            (15, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2
        )

        # ---------------------------------------------
        # Status
        # ---------------------------------------------

        cv2.putText(
            annotated_frame,
            status_text,
            (15, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # ---------------------------------------------
        # Security Severity
        # ---------------------------------------------

        cv2.putText(
            annotated_frame,
            f"SECURITY: {severity}",
            (15, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # ---------------------------------------------
        # REAL-TIME THREAT
        # ---------------------------------------------

        cv2.putText(
            annotated_frame,
            f"THREAT: {threat_level}",
            (15, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                (0, 0, 255)
                if threat_level in ["HIGH", "CRITICAL"]
                else (0, 255, 255)
            ),
            2
        )

        cv2.putText(
            annotated_frame,
            f"REASON: {threat_reason}",
            (15, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # ---------------------------------------------
        # Vehicle breakdown
        # ---------------------------------------------

        breakdown = (
            f"Car:{counts['car']} "
            f"Bike:{counts['motorcycle']} "
            f"Bus:{counts['bus']} "
            f"Truck:{counts['truck']}"
        )

        cv2.putText(
            annotated_frame,
            breakdown,
            (15, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # ---------------------------------------------
        # FPS
        # ---------------------------------------------

        fps_frame_count += 1

        elapsed = (
            time.time()
            - fps_start_time
        )

        if elapsed >= 1:

            fps = (
                fps_frame_count
                / elapsed
            )

            fps_frame_count = 0
            fps_start_time = time.time()

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (15, 305),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # ---------------------------------------------
        # Display
        # ---------------------------------------------

        cv2.imshow(
            "VIGILX-AI - Realtime Monitor",
            annotated_frame
        )

        # ---------------------------------------------
        # Exit
        # ---------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print(
                "Stopping VIGILX-AI..."
            )

            break


# =========================================================
# ERROR HANDLING
# =========================================================

except KeyboardInterrupt:

    print()

    print(
        "Monitoring stopped by user."
    )


except Exception as error:

    print()

    print(
        "VIGILX-AI ERROR:"
    )

    print(error)


# =========================================================
# CLEAN SHUTDOWN
# =========================================================

finally:

    stop_event.set()

    detection_thread.join(
        timeout=2
    )

    cap.release()

    cv2.destroyAllWindows()

    print()

    print(
        "Camera released."
    )

    print(
        "VIGILX-AI closed successfully."
    )