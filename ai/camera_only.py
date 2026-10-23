from ultralytics import YOLO
import cv2

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("Model loaded successfully.")
print("Reading camera frame...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit

ret, frame = cap.read()

if not ret:
    print("ERROR: Could not read camera frame.")
    cap.release()
    raise SystemExit

print("Camera frame received.")

results = model(frame, verbose=False)

annotated_frame = results[0].plot()

cv2.imshow("VIGILX-AI - YOLO Person Detection", annotated_frame)

print("YOLO detection complete.")
print("Saved: yolo_test.jpg")

cap.release()