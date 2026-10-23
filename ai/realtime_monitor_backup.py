import cv2
import time
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

# YOLO COCO classes
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
# STATE VARIABLES
# =========================================================

last_person_state = False
last_person_count = 0

last_vehicle_count = 0
last_phone_count = 0

last_event_time = 0

frame_counter = 0
last_results = None


# =========================================================
# START
# =========================================================

print("========================================")
print("        VIGILX-AI REALTIME MONITOR")
print("========================================")
print()

print("Loading YOLO model...")

try:
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")

except Exception as error:
    print("ERROR: YOLO model could not be loaded.")
    print(error)
    raise SystemExit


# =========================================================
# CAMERA
# =========================================================

print("Starting camera...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit


# Lower camera resolution = faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera started successfully.")
print("Press Q to stop monitoring.")
print()


# =========================================================
# ALERT
# =========================================================

def add_alert(message):

    current_time = datetime.now().strftime("%H:%M:%S")

    print(
        f"[{current_time}] ALERT: {message}"
    )


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
# EVENT LOGGER
# =========================================================

def log_security_event(
    event_type,
    message,
    person_count
):

    global last_event_time

    current_time = time.time()

    if current_time - last_event_time < EVENT_COOLDOWN:
        return

    severity = calculate_severity(
        person_count
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
# MAIN LOOP
# =========================================================

try:

    while True:

        # -------------------------------------------------
        # READ FRAME
        # -------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Could not read camera frame."
            )

            break


        # -------------------------------------------------
        # YOLO
        #
        # Run detection every 2nd frame for speed.
        # -------------------------------------------------

        frame_counter += 1

        if frame_counter % 2 == 0:

            last_results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                classes=DETECTION_CLASSES,
                imgsz=416,
                verbose=False
            )


        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------

        if last_results is None:

            continue


        result = last_results[0]


        # -------------------------------------------------
        # COUNTERS
        # -------------------------------------------------

        person_count = 0
        car_count = 0
        motorcycle_count = 0
        bus_count = 0
        truck_count = 0
        phone_count = 0


        # -------------------------------------------------
        # DETECTIONS
        # -------------------------------------------------

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )


                # PERSON
                if class_id == PERSON:

                    if confidence >= PERSON_CONFIDENCE:
                        person_count += 1


                # CAR
                elif class_id == CAR:

                    if confidence >= CONFIDENCE_THRESHOLD:
                        car_count += 1


                # MOTORCYCLE
                elif class_id == MOTORCYCLE:

                    if confidence >= CONFIDENCE_THRESHOLD:
                        motorcycle_count += 1


                # BUS
                elif class_id == BUS:

                    if confidence >= CONFIDENCE_THRESHOLD:
                        bus_count += 1


                # TRUCK
                elif class_id == TRUCK:

                    if confidence >= CONFIDENCE_THRESHOLD:
                        truck_count += 1


                # CELL PHONE
                elif class_id == CELL_PHONE:

                    if confidence >= PHONE_CONFIDENCE:
                        phone_count += 1


        # -------------------------------------------------
        # VEHICLES
        # -------------------------------------------------

        vehicle_count = (
            car_count
            + motorcycle_count
            + bus_count
            + truck_count
        )


        # -------------------------------------------------
        # SECURITY
        # -------------------------------------------------

        severity = calculate_severity(
            person_count
        )


        # -------------------------------------------------
        # PERSON STATE
        # -------------------------------------------------

        current_person_state = (
            person_count > 0
        )


        # =================================================
        # PERSON DETECTED
        # =================================================

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
                person_count=person_count
            )


        # =================================================
        # MULTIPLE PERSONS
        # =================================================

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
                person_count=person_count
            )


        # =================================================
        # VEHICLE
        # =================================================

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
                person_count=person_count
            )


        # =================================================
        # CELL PHONE
        # =================================================

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
                person_count=person_count
            )


        # =================================================
        # AREA CLEAR
        # =================================================

        elif (
            not current_person_state
            and last_person_state
        ):

            log_security_event(
                event_type="AREA_CLEAR",
                message="No person detected",
                person_count=0
            )


        # -------------------------------------------------
        # UPDATE STATES
        # -------------------------------------------------

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


        # =================================================
        # DRAW
        # =================================================

        annotated_frame = result.plot()


        # =================================================
        # HEADER
        # =================================================

        cv2.putText(
            annotated_frame,
            "VIGILX-AI | AI SURVEILLANCE",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        # =================================================
        # COUNTS
        # =================================================

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


        # =================================================
        # STATUS
        # =================================================

        if person_count == 0:

            status_text = "STATUS: CLEAR"
            status_color = (0, 255, 0)

        elif person_count == 1:

            status_text = "STATUS: PERSON DETECTED"
            status_color = (0, 255, 255)

        else:

            status_text = "STATUS: MULTIPLE PERSONS"
            status_color = (0, 0, 255)


        cv2.putText(
            annotated_frame,
            status_text,
            (15, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2
        )


        # =================================================
        # SECURITY
        # =================================================

        cv2.putText(
            annotated_frame,
            f"SECURITY: {severity}",
            (15, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2
        )


        # =================================================
        # VEHICLE BREAKDOWN
        # =================================================

        breakdown = (
            f"Car:{car_count} "
            f"Bike:{motorcycle_count} "
            f"Bus:{bus_count} "
            f"Truck:{truck_count}"
        )

        cv2.putText(
            annotated_frame,
            breakdown,
            (15, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )


        # =================================================
        # SHOW
        # =================================================

        cv2.imshow(
            "VIGILX-AI - Realtime Monitor",
            annotated_frame
        )


        # =================================================
        # EXIT
        # =================================================

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

    cap.release()

    cv2.destroyAllWindows()

    print()
    print("Camera released.")
    print("VIGILX-AI closed successfully.")