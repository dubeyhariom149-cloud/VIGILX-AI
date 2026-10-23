import os
import json
import csv
import time
import platform
import tempfile
import smtplib
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime
from collections import Counter

import cv2
import streamlit as st

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / '.env', override=True)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================================================
# VIGILX-AI CONFIG
# ============================================================

APP_NAME = "VIGILX-AI"
VERSION = "14.1"
OWNER = "Hariom Dubey"

BASE_DIR = Path(__file__).resolve().parent
EVENT_DIR = BASE_DIR / "events"
VIDEO_DIR = BASE_DIR / "videos"

EVENT_DIR.mkdir(exist_ok=True)
VIDEO_DIR.mkdir(exist_ok=True)

EVENT_JSON = EVENT_DIR / "events.json"
EVENT_CSV = EVENT_DIR / "events.csv"

MODEL_PATH = BASE_DIR / "yolo11n.pt"

# Class-specific confidence thresholds
CONF_PERSON = 0.45
CONF_PHONE = 0.65
CONF_VEHICLE = 0.50
MIN_CONFIDENCE = min(CONF_PERSON, CONF_PHONE, CONF_VEHICLE)

IMG_SIZE = 640
EVENT_COOLDOWN = 5

VEHICLE_IDS = {2, 3, 5, 7}
PERSON_ID = 0
PHONE_ID = 67


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="VIGILX-AI",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>
.stApp {
    background: #07111f;
}

[data-testid="stSidebar"] {
    background: #050b14;
}

[data-testid="stMetric"] {
    background: #0b1828;
    border: 1px solid #19344b;
    border-radius: 10px;
    padding: 12px;
}

.block-container {
    padding-top: 1.5rem;
}

h1, h2, h3 {
    letter-spacing: .5px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "authenticated": False,
    "monitoring": False,
    "source": "Local Video Testing",
    "uploaded_video": None,
    "video_path": "",
    "selected_dataset_video": "",
    "cap": None,
    "model": None,
    "last_event_time": 0.0,
    "page": "Command Center",
    "events_cache": None,
    "events_mtime": 0.0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# RUNTIME
# ============================================================

if "runtime" not in st.session_state:
    st.session_state.runtime = {
        "frame": None,
        "camera": "OFFLINE",
        "ai": "STANDBY",
        "source": "WEBCAM",
        "persons": 0,
        "vehicles": 0,
        "phones": 0,
        "threat": "NORMAL",
        "fps": 0.0,
        "error": "",
    }

R = st.session_state.runtime


# ============================================================
# EVENTS (OPTIMIZED IN-MEMORY CACHE + DEDUPLICATION + SYNC)
# ============================================================

def normalize_event(e):
    if not isinstance(e, dict):
        return e

    evt_type = e.get("event") or e.get("event_type", "INCIDENT")
    desc = e.get("description") or e.get("message", "")
    persons = e.get("persons") if "persons" in e else e.get("person_count", 0)
    vehicles = e.get("vehicles", 0)
    phones = e.get("phones", 0)
    source = e.get("source") or e.get("camera_source", "CAM 01")
    severity = e.get("severity", "LOW")
    evidence = e.get("evidence", "")
    ts = e.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "timestamp": ts,
        "event": evt_type,
        "event_type": evt_type,
        "description": desc,
        "message": desc,
        "severity": severity,
        "persons": persons,
        "person_count": persons,
        "vehicles": vehicles,
        "phones": phones,
        "source": source,
        "evidence": evidence,
    }


def load_events(force_reload=False):
    if not EVENT_JSON.exists():
        st.session_state.events_cache = []
        st.session_state.events_mtime = 0.0
        return []

    try:
        current_mtime = EVENT_JSON.stat().st_mtime
    except OSError:
        current_mtime = 0.0

    # In-memory cache hit: avoids reading disk multiple times per frame
    if (
        not force_reload
        and st.session_state.get("events_cache") is not None
        and st.session_state.get("events_mtime", 0.0) == current_mtime
    ):
        return st.session_state.events_cache

    try:
        with open(EVENT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            events = [normalize_event(e) for e in data if isinstance(e, dict)]
        else:
            events = []
    except Exception:
        events = st.session_state.get("events_cache") or []

    st.session_state.events_cache = events
    st.session_state.events_mtime = current_mtime
    return events


def save_events(events):
    normalized = [normalize_event(e) for e in events if isinstance(e, dict)]

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for e in normalized:
        key = (e.get("timestamp"), e.get("event"))
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    # Sort chronologically by timestamp
    deduped.sort(key=lambda x: x.get("timestamp", ""))

    try:
        # Write to authoritative events/events.json
        with open(EVENT_JSON, "w", encoding="utf-8") as f:
            json.dump(deduped, f, indent=2, ensure_ascii=False)

        # Mirror to root events.json for unified access across workers
        root_json = BASE_DIR / "events.json"
        with open(root_json, "w", encoding="utf-8") as f:
            json.dump(deduped, f, indent=2, ensure_ascii=False)
    except Exception as e:
        R["error"] = f"Event file write error: {e}"

    if deduped:
        fields = sorted(
            set().union(
                *(event.keys() for event in deduped)
            )
        )
        for target_csv in [EVENT_CSV, BASE_DIR / "events.csv"]:
            try:
                with open(
                    target_csv,
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=fields,
                    )
                    writer.writeheader()
                    for event in deduped:
                        writer.writerow(event)
            except Exception:
                pass

    # Update in-memory session cache immediately
    st.session_state.events_cache = deduped
    try:
        st.session_state.events_mtime = EVENT_JSON.stat().st_mtime
    except OSError:
        st.session_state.events_mtime = time.time()

    return deduped


def add_event(
    event_type,
    severity,
    persons,
    vehicles,
    phones,
    source,
    description,
    evidence="",
):

    events = load_events()

    event = {
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "event": event_type,
        "event_type": event_type,
        "severity": severity,
        "persons": persons,
        "person_count": persons,
        "vehicles": vehicles,
        "phones": phones,
        "source": source,
        "description": description,
        "message": description,
        "evidence": evidence,
    }

    events.append(event)

    save_events(events)

    return event


# ============================================================
# THREAT
# ============================================================

def get_threat(persons, vehicles, phones):

    if persons >= 6:
        return "HIGH"

    if persons >= 3:
        return "MEDIUM"

    if persons > 0 or vehicles > 0 or phones > 0:
        return "LOW"

    return "NORMAL"


# ============================================================
# EVIDENCE
# ============================================================

def save_evidence(frame, event_type):

    filename = (
        "evidence_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        + "_"
        + event_type
        + ".jpg"
    )

    path = EVENT_DIR / filename

    try:
        cv2.imwrite(str(path), frame)
        return str(path)
    except Exception:
        return ""


def evidence_count():

    return len(
        list(EVENT_DIR.glob("evidence_*.jpg"))
    )


# ============================================================
# EMAIL ALERT
# ============================================================

def get_email_config():
    smtp_host = os.getenv("VIGILX_SMTP_HOST", "")
    smtp_port = int(os.getenv("VIGILX_SMTP_PORT", "587"))
    smtp_user = os.getenv("VIGILX_SMTP_USER", "")
    smtp_password = os.getenv("VIGILX_SMTP_PASSWORD", "")
    alert_to = os.getenv("VIGILX_ALERT_TO", "")

    # Fallback to .streamlit/secrets.toml if env vars are unset
    try:
        if not smtp_user and "EMAIL_ADDRESS" in st.secrets:
            smtp_user = str(st.secrets["EMAIL_ADDRESS"])
            if not smtp_host:
                smtp_host = "smtp.gmail.com"
        if not smtp_password and "EMAIL_APP_PASSWORD" in st.secrets:
            smtp_password = str(st.secrets["EMAIL_APP_PASSWORD"])
        if not alert_to and smtp_user:
            alert_to = smtp_user
    except Exception:
        pass

    configured = bool(smtp_host and smtp_user and smtp_password and alert_to)
    return {
        "host": smtp_host,
        "port": smtp_port,
        "user": smtp_user,
        "password": smtp_password,
        "alert_to": alert_to,
        "configured": configured,
    }


def send_email_alert(event):
    cfg = get_email_config()
    if not cfg["configured"]:
        return False

    try:
        message = EmailMessage()
        message["Subject"] = (
            f"🚨 VIGILX-AI HIGH THREAT ALERT - "
            f"{event.get('event', 'INCIDENT')}"
        )
        message["From"] = cfg["user"]
        message["To"] = cfg["alert_to"]

        body = f"""VIGILX-AI
AI-Powered Intelligent Video Surveillance and Monitoring System

HIGH THREAT ALERT

Time:
{event.get('timestamp', '')}

Event:
{event.get('event', '')}

Severity:
{event.get('severity', '')}

Persons:
{event.get('persons', 0)}

Vehicles:
{event.get('vehicles', 0)}

Phones:
{event.get('phones', 0)}

Source:
{event.get('source', '')}

Description:
{event.get('description', '')}

Evidence:
{event.get('evidence', '')}

Operator:
{OWNER}

VIGILX-AI v{VERSION}
"""
        message.set_content(body)

        evidence_path = event.get("evidence", "")
        if evidence_path and os.path.isfile(evidence_path):
            try:
                with open(evidence_path, "rb") as ef:
                    img_data = ef.read()
                message.add_attachment(
                    img_data,
                    maintype="image",
                    subtype="jpeg",
                    filename=Path(evidence_path).name,
                )
            except Exception:
                pass

        with smtplib.SMTP(
            cfg["host"],
            cfg["port"],
            timeout=10,
        ) as server:
            server.starttls()
            server.login(
                cfg["user"],
                cfg["password"],
            )
            server.send_message(message)

        return True

    except Exception as e:
        R["error"] = f"Email alert error: {e}"
        return False



# ============================================================
# MODEL
# ============================================================

def load_model():

    if YOLO is None:

        raise RuntimeError(
            "Ultralytics is not installed."
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


# ============================================================
# OPEN SOURCE
# ============================================================

def open_source():

    source = st.session_state.source

    # ---------------- WEBCAM ----------------

    if source == "Webcam Testing":

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        R["source"] = "WEBCAM"

        return cap

    # ---------------- UPLOADED VIDEO ----------------

    if source == "Local Video Testing":

        video_path = st.session_state.video_path

        if not video_path:
            raise RuntimeError(
                "Please upload a CCTV video first."
            )

        R["source"] = "UPLOADED CCTV"

        cap = cv2.VideoCapture(
            video_path
        )

        return cap

    # ---------------- OFFICIAL DATASET ----------------

    if source == "Official Dataset CCTV Video":

        selected_clip = st.session_state.get(
            "selected_dataset_video",
            "",
        )

        target_path = None
        if selected_clip and (VIDEO_DIR / selected_clip).exists():
            target_path = VIDEO_DIR / selected_clip
        else:
            videos = sorted([
                p
                for p in VIDEO_DIR.iterdir()
                if p.suffix.lower()
                in {
                    ".mp4",
                    ".avi",
                    ".mov",
                    ".mkv",
                }
            ])

            if not videos:
                raise RuntimeError(
                    "No CCTV video found inside videos folder."
                )

            target_path = videos[0]
            st.session_state.selected_dataset_video = target_path.name

        R["source"] = f"DATASET ({target_path.stem[:14]})"

        return cv2.VideoCapture(
            str(target_path)
        )

    # ---------------- RTSP ----------------

    if source == "Government / Police CCTV RTSP/IP":

        url = st.session_state.get(
            "rtsp_url",
            "",
        )

        if not url:

            raise RuntimeError(
                "Enter RTSP/IP URL."
            )

        R["source"] = "RTSP/IP"

        return cv2.VideoCapture(url)

    raise RuntimeError(
        "Unknown camera source."
    )


# ============================================================
# START
# ============================================================

def start_monitoring():

    try:

        if st.session_state.cap is not None:

            try:
                st.session_state.cap.release()
            except Exception:
                pass

        if st.session_state.model is None:

            st.session_state.model = load_model()

        cap = open_source()

        if not cap.isOpened():

            raise RuntimeError(
                "Unable to open selected video/camera."
            )

        ok, first_frame = cap.read()

        if not ok or first_frame is None:
            cap.release()
            raise RuntimeError(
                "Camera opened but no video frame could be read."
            )

        st.session_state.cap = cap
        st.session_state.monitoring = True

        R["frame"] = first_frame
        R["camera"] = "ONLINE"
        R["ai"] = "ACTIVE"
        R["error"] = ""

        return True

    except Exception as e:

        st.session_state.monitoring = False

        R["camera"] = "ERROR"
        R["ai"] = "ERROR"
        R["error"] = str(e)

        return False


# ============================================================
# STOP
# ============================================================

def stop_monitoring():

    if st.session_state.cap is not None:

        try:
            st.session_state.cap.release()
        except Exception:
            pass

    st.session_state.cap = None
    st.session_state.monitoring = False

    R["camera"] = "OFFLINE"
    R["ai"] = "STANDBY"
    R["frame"] = None
    R["persons"] = 0
    R["vehicles"] = 0
    R["phones"] = 0
    R["threat"] = "NORMAL"


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def process_frame():

    cap = st.session_state.cap

    if cap is None:
        return False

    ok, frame = cap.read()

    if not ok:

        # Uploaded/local video reached end.
        if st.session_state.source in [
            "Local Video Testing",
            "Official Dataset CCTV Video",
        ]:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0,
            )

            ok, frame = cap.read()

        if not ok:

            stop_monitoring()

            return False

    model = st.session_state.model

    start = time.time()

    results = model.predict(
        frame,
        conf=0.40,
        imgsz=IMG_SIZE,
        verbose=False,
    )

    persons = 0
    vehicles = 0
    phones = 0

    annotated = frame

    if results:

        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:

            keep_indices = []

            for i, box in enumerate(result.boxes):

                class_id = int(box.cls[0].item())
                conf_val = float(box.conf[0].item())

                if class_id == PERSON_ID and conf_val >= CONF_PERSON:
                    persons += 1
                    keep_indices.append(i)

                elif class_id in VEHICLE_IDS and conf_val >= CONF_VEHICLE:
                    vehicles += 1
                    keep_indices.append(i)

                elif class_id == PHONE_ID and conf_val >= CONF_PHONE:
                    phones += 1
                    keep_indices.append(i)

            try:
                if len(keep_indices) == len(result.boxes):
                    annotated = result.plot()
                elif len(keep_indices) > 0:
                    annotated = result[keep_indices].plot()
                else:
                    annotated = frame
            except Exception:
                annotated = frame

    elapsed = time.time() - start

    R["fps"] = (
        1 / elapsed
        if elapsed > 0
        else 0
    )

    R["frame"] = annotated
    R["persons"] = persons
    R["vehicles"] = vehicles
    R["phones"] = phones

    level = get_threat(
        persons,
        vehicles,
        phones,
    )

    R["threat"] = level

    # --------------------------------------------------------
    # EVENT
    # --------------------------------------------------------

    now = time.time()

    detected = (
        persons > 0
        or vehicles > 0
        or phones > 0
    )

    if (
        detected
        and now - st.session_state.last_event_time
        >= EVENT_COOLDOWN
    ):

        if persons > 0:

            event_type = "PERSON_DETECTED"

            description = (
                f"Person detected: {persons}"
            )

        elif vehicles > 0:

            event_type = "VEHICLE_DETECTED"

            description = (
                f"Vehicle detected: {vehicles}"
            )

        else:

            event_type = "OBJECT_DETECTED"

            description = (
                f"Phone/object detected: {phones}"
            )

        severity = (
            "HIGH"
            if level == "HIGH"
            else "MEDIUM"
            if level == "MEDIUM"
            else "LOW"
        )

        evidence = save_evidence(
            annotated,
            event_type,
        )

        event = add_event(
            event_type,
            severity,
            persons,
            vehicles,
            phones,
            R["source"],
            description,
            evidence,
        )

        # Email ONLY for HIGH
        if severity == "HIGH":

            send_email_alert(event)

        st.session_state.last_event_time = now

    return True


# ============================================================
# LOGIN
# ============================================================

def login_page():
    st.markdown(
        """
        <style>
        .login-founder {
            position: fixed;
            top: 24px;
            right: 35px;
            font-size: 15px;
            font-weight: 600;
            z-index: 9999;
        }

        .login-title {
            text-align: center;
            margin-top: 90px;
            font-size: 32px;
            font-weight: 800;
        }

        .login-subtitle {
            text-align: center;
            font-size: 20px;
            font-weight: 600;
            margin-top: 8px;
        }

        .login-security {
            text-align: center;
            font-size: 15px;
            margin-top: 12px;
            opacity: 0.85;
        }
        </style>

        <div class="login-founder">
            Founder: Hariom Dubey
        </div>

        <div class="login-title">
            🛡️ VIGILX-AI
        </div>

        <div class="login-subtitle">
            AI-Powered Intelligent Video Surveillance and Monitoring System
        </div>

        <div class="login-security">
            🔐 Secure operator authentication required.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    with st.form("login_form"):
        username = st.text_input("Operator Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("LOGIN")

        if submitted:
            expected_user = os.getenv("VIGILX_LOGIN_USER", "admin")
            expected_pass = os.getenv("VIGILX_LOGIN_PASSWORD", "admin")

            if username == expected_user and password == expected_pass:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password.")

def render_sidebar():

    with st.sidebar:

        st.title("🛡️ VIGILX-AI")

        st.caption(
            "AI-Powered Intelligent Video Surveillance and Monitoring System"
        )

        st.divider()

        st.session_state.page = st.radio(
            "SYSTEM MODULE",
            [
                "Command Center",
                "Event History",
                "Evidence",
                "Analytics",
                "Detection Settings",
                "Email Alerts",
                "System Health",
            ],
        )

        st.divider()

        st.subheader(
            "📡 Camera Source"
        )

        source = st.selectbox(
            "Input Source",
            [
                "Local Video Testing",
                "Webcam Testing",
                "Official Dataset CCTV Video",
                "Government / Police CCTV RTSP/IP",
            ],
        )

        st.session_state.source = source

        # ====================================================
        # UPLOAD CCTV
        # ====================================================

        if source == "Local Video Testing":

            uploaded = st.file_uploader(
                "ðŸ“¹ Upload CCTV Video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv",
                ],
            )

            if uploaded:

                upload_dir = (
                    BASE_DIR / "uploaded_videos"
                )

                upload_dir.mkdir(
                    exist_ok=True
                )

                save_path = (
                    upload_dir
                    / uploaded.name
                )

                # Save only if changed/new
                if (
                    not save_path.exists()
                    or save_path.stat().st_size
                    != uploaded.size
                ):

                    with open(
                        save_path,
                        "wb",
                    ) as f:

                        f.write(
                            uploaded.getbuffer()
                        )

                st.session_state.video_path = str(
                    save_path
                )

                st.success(
                    f"Loaded: {uploaded.name}"
                )

                st.caption(
                    f"{uploaded.size / 1024 / 1024:.2f} MB"
                )

        # ====================================================
        # OFFICIAL DATASET CCTV VIDEO
        # ====================================================

        if source == "Official Dataset CCTV Video":

            dataset_files = sorted(
                [
                    p.name
                    for p in VIDEO_DIR.iterdir()
                    if p.suffix.lower()
                    in {
                        ".mp4",
                        ".avi",
                        ".mov",
                        ".mkv",
                    }
                ]
            )

            if dataset_files:

                current_selected = st.session_state.get(
                    "selected_dataset_video",
                    dataset_files[0],
                )
                if current_selected not in dataset_files:
                    current_selected = dataset_files[0]

                selected_file = st.selectbox(
                    "ðŸ“ Select Dataset CCTV Clip",
                    dataset_files,
                    index=dataset_files.index(current_selected),
                    key="dataset_clip_selector",
                )

                st.session_state.selected_dataset_video = selected_file
                st.caption(f"Active Clip: {selected_file}")

            else:

                st.warning("No video files found in videos/ folder.")

        # ====================================================
        # RTSP
        # ====================================================

        if source == "Government / Police CCTV RTSP/IP":

            st.session_state.rtsp_url = st.text_input(
                "RTSP / IP URL",
                type="password",
                placeholder="rtsp://...",
            )

        st.divider()

        if not st.session_state.monitoring:

            if st.button(
                "â–¶ START MONITORING",
                type="primary",
                use_container_width=True,
            ):

                if start_monitoring():

                    st.success(
                        "Monitoring started."
                    )

                    st.rerun()

                else:

                    st.error(
                        R["error"]
                    )

        else:

            if st.button(
                "â¹ STOP MONITORING",
                use_container_width=True,
            ):

                stop_monitoring()

                st.rerun()

        st.divider()

        st.caption(
            "CAM 01 PRIMARY"
        )

        st.caption(
            "CAM 02 / CAM 03 RTSP READY"
        )

        st.caption(
            f"Operator: {OWNER}"
        )


# ============================================================
# HEADER
# ============================================================

def header():

    st.title("🛡️ VIGILX-AI")

    st.subheader(
        "AI-Powered Intelligent Video Surveillance and Monitoring System"
    )

    a, b, c = st.columns(3)

    with a:

        if st.session_state.monitoring:
            st.success(
                "● SYSTEM ACTIVE"
            )
        else:
            st.info(
                "● SYSTEM STANDBY"
            )

    with b:

        st.write(
            "📅 "
            + datetime.now().strftime(
                "%d %b %Y • %H:%M:%S"
            )
        )

    with c:

        st.write(
            f"Operator: **{OWNER}**"
        )

    st.divider()


# ============================================================
# KPI
# ============================================================

def kpis():

    events = load_events()

    a, b, c, d, e = st.columns(5)

    with a:
        st.metric(
            "CAMERA STATUS",
            R["camera"],
            "CAM 01 • PRIMARY",
        )

    with b:
        st.metric(
            "SOURCE",
            R["source"],
            "Current input",
        )

    with c:
        st.metric(
            "PERSONS",
            R["persons"],
            "YOLO detections",
        )

    with d:
        st.metric(
            "VEHICLES",
            R["vehicles"],
            "Cars / bikes / bus / truck",
        )

    with e:
        st.metric(
            "EVENTS",
            len(events),
            "Preserved history",
        )


# ============================================================
# CAMERA GRID
# ============================================================

def camera_grid():

    st.subheader(
        "📡 LIVE THREAT MONITORING • CAMERA GRID"
    )

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            "### CAM 01 • PRIMARY"
        )

        st.write(
            f"Status: **{R['camera']}**"
        )

        if R["frame"] is not None:

            display_frame = cv2.cvtColor(
                R["frame"],
                cv2.COLOR_BGR2RGB,
            )

            st.image(
                display_frame,
                use_container_width=True,
            )

        else:

            st.info(
                "📷 CAMERA STANDBY\n\n"
                "Upload/select a source and press "
                "START MONITORING."
            )

        st.caption(
            f"SOURCE: {R['source']}"
        )

    with b:

        st.markdown(
            "### CAM 02 • SECONDARY"
        )

        st.warning(
            "● STANDBY"
        )

        st.info(
            "📡 RTSP / IP READY\n\n"
            "Government / Police CCTV architecture\n"
            "Source configurable"
        )

        st.caption(
            "SIGNAL READY"
        )

    with c:

        st.markdown(
            "### CAM 03 • SECONDARY"
        )

        st.warning(
            "● STANDBY"
        )

        st.info(
            "📡 RTSP / IP READY\n\n"
            "Government / Police CCTV architecture\n"
            "Source configurable"
        )

        st.caption(
            "SIGNAL READY"
        )


# ============================================================
# INTELLIGENCE
# ============================================================

def intelligence():

    st.subheader(
        "🧠 AI INTELLIGENCE • DETECTION • RESPONSE"
    )

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            "### THREAT LEVEL"
        )

        st.metric(
            "Current Threat",
            R["threat"],
        )

        st.write(
            f"Persons: **{R['persons']}**"
        )

        st.write(
            f"Vehicles: **{R['vehicles']}**"
        )

        st.write(
            f"Phones: **{R['phones']}**"
        )

    with b:

        st.markdown(
            "### LIVE ACTIVITY"
        )

        events = load_events()

        if events:

            event = events[-1]

            st.write(
                f"**{event.get('event', 'UNKNOWN')}**"
            )

            st.write(
                event.get(
                    "description",
                    "",
                )
            )

            st.caption(
                event.get(
                    "timestamp",
                    "",
                )
            )

        else:

            st.info(
                "NO ACTIVE INCIDENT\n\n"
                "Event engine is ready."
            )

    with c:

        st.markdown(
            "### AI STATUS"
        )

        st.write(
            "Engine: **READY**"
        )

        st.write(
            f"Model: **{MODEL_PATH.name}**"
        )

        st.write(
            f"AI Status: **{R['ai']}**"
        )

        st.write(
            f"Camera: **{R['camera']}**"
        )

        st.write(
            f"Platform: **{platform.system()}**"
        )

        st.write(
            f"FPS: **{R['fps']:.1f}**"
        )


# ============================================================
# RESPONSE
# ============================================================

def response_center():

    st.subheader(
        "🚨 ALERT & RESPONSE CENTER"
    )

    events = load_events()

    a, b, c = st.columns(3)

    with a:

        st.info(
            "📧 EMAIL ALERT SYSTEM"
        )

        if os.getenv(
            "VIGILX_SMTP_USER",
            "",
        ):

            st.success(
                "CONFIGURED"
            )

        else:

            st.warning(
                "STANDBY"
            )

        st.caption(
            "HIGH threat â†’ automatic email"
        )

    with b:

        st.success(
            "💾 EVIDENCE STORAGE"
        )

        st.metric(
            "Snapshots",
            evidence_count(),
        )

    with c:

        st.success(
            "🗄️ EVENT DATABASE"
        )

        st.metric(
            "Records",
            len(events),
        )


# ============================================================
# TIMELINE
# ============================================================

def timeline():

    st.subheader(
        "📋 INCIDENT TIMELINE"
    )

    events = load_events()

    if not events:

        st.info(
            "No incidents recorded yet."
        )

        return

    for event in reversed(events[-8:]):

        with st.expander(
            f"{event.get('timestamp', '')} • "
            f"{event.get('event', '')} • "
            f"{event.get('severity', '')}"
        ):

            st.write(
                event.get(
                    "description",
                    "",
                )
            )

            st.write(
                f"Persons: {event.get('persons', 0)}"
            )

            st.write(
                f"Vehicles: {event.get('vehicles', 0)}"
            )

            st.write(
                f"Phones: {event.get('phones', 0)}"
            )

            st.write(
                f"Source: {event.get('source', '')}"
            )


# ============================================================
# COMMAND CENTER
# ============================================================

def command_center():

    header()

    if not st.session_state.monitoring:

        st.info(
            "🧪 Local CCTV/video source is ready. "
            "Press START MONITORING to begin AI analysis."
        )

    kpis()

    st.divider()

    camera_grid()

    st.divider()

    intelligence()

    st.divider()

    response_center()

    st.divider()

    timeline()

    st.divider()

    st.caption(
        f"VIGILX-AI • AI-Powered Intelligent Video Surveillance and Monitoring System • "
        f"AI-POWERED INTELLIGENT SURVEILLANCE & "
        f"SECURITY SYSTEM • {OWNER} • v{VERSION}"
    )


# ============================================================
# EVENT HISTORY
# ============================================================

def event_history():

    st.title("📋 Event History")

    events = load_events()

    st.metric(
        "Preserved Events",
        len(events),
    )

    for event in reversed(events):

        with st.expander(
            f"{event.get('timestamp', '')} • "
            f"{event.get('event', '')}"
        ):

            st.write(
                f"Severity: **{event.get('severity', '')}**"
            )

            st.write(
                event.get(
                    "description",
                    "",
                )
            )

            st.write(
                f"Persons: {event.get('persons', 0)}"
            )

            st.write(
                f"Vehicles: {event.get('vehicles', 0)}"
            )

            st.write(
                f"Phones: {event.get('phones', 0)}"
            )


# ============================================================
# EVIDENCE PAGE
# ============================================================

def evidence_page():

    st.title("💾 Evidence Storage")

    files = sorted(
        EVENT_DIR.glob(
            "evidence_*.jpg"
        ),
        reverse=True,
    )

    st.metric(
        "Snapshots",
        len(files),
    )

    if not files:

        st.info(
            "No evidence snapshots available."
        )

        return

    cols = st.columns(3)

    for i, file in enumerate(files[:60]):

        with cols[i % 3]:

            st.image(
                str(file),
                caption=file.name,
                use_container_width=True,
            )


# ============================================================
# ANALYTICS
# ============================================================

def analytics_page():

    st.title("ðŸ“Š Surveillance Analytics")

    events = load_events()

    if not events:

        st.info(
            "No event data available."
        )

        return

    counter = Counter(
        event.get(
            "event",
            "UNKNOWN",
        )
        for event in events
    )

    severity = Counter(
        event.get(
            "severity",
            "UNKNOWN",
        )
        for event in events
    )

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "Total Events",
            len(events),
        )

    with b:
        st.metric(
            "Person Events",
            counter.get(
                "PERSON_DETECTED",
                0,
            ),
        )

    with c:
        st.metric(
            "HIGH Events",
            severity.get(
                "HIGH",
                0,
            ),
        )

    st.bar_chart(counter)

    st.subheader(
        "Severity Distribution"
    )

    st.bar_chart(severity)


# ============================================================
# SETTINGS
# ============================================================

def settings_page():

    st.title("âš™ï¸ Detection Settings")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("PERSON CONFIDENCE", f"{CONF_PERSON:.2f}")
    with c2:
        st.metric("PHONE CONFIDENCE", f"{CONF_PHONE:.2f}")
    with c3:
        st.metric("VEHICLE CONFIDENCE", f"{CONF_VEHICLE:.2f}")

    st.write(
        f"Inference Image Size: **{IMG_SIZE}x{IMG_SIZE}**"
    )

    st.write(
        f"Event Cooldown: **{EVENT_COOLDOWN}s**"
    )

    st.divider()

    st.markdown("### ðŸŽ¯ Monitored Target Classes")
    st.write("• **Person** — COCO ID 0 (Threshold: 0.70)")
    st.write("• **Car** — COCO ID 2 (Threshold: 0.50)")
    st.write("• **Motorcycle** — COCO ID 3 (Threshold: 0.50)")
    st.write("• **Bus** — COCO ID 5 (Threshold: 0.50)")
    st.write("• **Truck** — COCO ID 7 (Threshold: 0.50)")
    st.write("• **Cell Phone** — COCO ID 67 (Threshold: 0.65)")


# ============================================================
# EMAIL PAGE
# ============================================================

def email_page():

    st.title("📧 Email Alert System")

    cfg = get_email_config()

    if cfg["configured"]:

        st.success(
            "● SMTP CONFIGURED & READY"
        )

        st.write(
            f"SMTP Host: **{cfg['host']}** | Port: **{cfg['port']}**"
        )
        st.write(
            f"Sender: **{cfg['user']}**"
        )
        st.write(
            f"Recipient: **{cfg['alert_to']}**"
        )
        st.info(
            "HIGH threat incidents automatically trigger real-time email notifications with snapshot evidence."
        )

        if st.button("ðŸ“¤ SEND TEST ALERT", use_container_width=True):
            test_evt = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "event": "TEST_ALERT",
                "severity": "HIGH",
                "persons": 1,
                "vehicles": 0,
                "phones": 0,
                "source": "OPERATOR TEST",
                "description": "Manual operator alert diagnostic test",
                "evidence": "",
            }
            if send_email_alert(test_evt):
                st.success("Test alert email dispatched successfully!")
            else:
                st.error(f"Failed to dispatch test alert: {R.get('error', 'Unknown error')}")

    else:

        st.warning(
            "● SMTP NOT CONFIGURED"
        )

        st.code("""
$env:VIGILX_SMTP_HOST="smtp.gmail.com"
$env:VIGILX_SMTP_PORT="587"
$env:VIGILX_SMTP_USER="yourgmail@gmail.com"
$env:VIGILX_SMTP_PASSWORD="your-16-character-app-password"
$env:VIGILX_ALERT_TO="receiver@gmail.com"
""")

        st.caption(
            "Credentials can also be provided securely in .streamlit/secrets.toml (EMAIL_ADDRESS / EMAIL_APP_PASSWORD)."
        )



# ============================================================
# HEALTH
# ============================================================

def health_page():

    st.title("ðŸ–¥ï¸ System Health")

    st.write(
        f"Application: **{APP_NAME}**"
    )

    st.write(
        f"Version: **{VERSION}**"
    )

    st.write(
        f"Platform: **{platform.system()}**"
    )

    st.write(
        f"Python: **{platform.python_version()}**"
    )

    st.write(
        f"YOLO: **{'AVAILABLE' if YOLO else 'MISSING'}**"
    )

    st.write(
        f"Model: **{'FOUND' if MODEL_PATH.exists() else 'MISSING'}**"
    )

    st.write(
        f"Model path: `{MODEL_PATH}`"
    )

    if R["error"]:

        st.error(
            R["error"]
        )


# ============================================================
# MAIN
# ============================================================

if not st.session_state.authenticated:

    login_page()

else:

    render_sidebar()

    if st.session_state.monitoring:
        process_frame()

    if st.session_state.page == "Command Center":

        command_center()

    elif st.session_state.page == "Event History":

        event_history()

    elif st.session_state.page == "Evidence":

        evidence_page()

    elif st.session_state.page == "Analytics":

        analytics_page()

    elif st.session_state.page == "Detection Settings":

        settings_page()

    elif st.session_state.page == "Email Alerts":

        email_page()

    elif st.session_state.page == "System Health":

        health_page()

# ============================================================
# LIVE VIDEO REFRESH
# ============================================================

if (
    st.session_state.authenticated
    and st.session_state.monitoring
):

    time.sleep(0.05)
    st.rerun()


