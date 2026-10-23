import cv2
from ultralytics import YOLO

VIDEO_PATH = r"C:\Users\Hariom\Documents\VIGILX-AI\videos\vidssave.com kidnap real shorts , captured in CCTV camera. #shorts 🙏 make it viral #viral 720P.mp4"
MODEL_PATH = r"C:\Users\Hariom\Documents\VIGILX-AI\yolo11n.pt"

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: CCTV video open nahi ho raha!")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("CCTV video started successfully!")
print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")
print("AI detection started.")
print("Press Q to stop.")

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print(f"CCTV video finished. Frames processed: {frame_count}")
        break

    frame_count += 1

    results = model.predict(
        frame,
        conf=0.35,
        verbose=False
    )

    output = results[0].plot()

    h, w = output.shape[:2]

    if h > 800:
        scale = 800 / h
        output = cv2.resize(
            output,
            (int(w * scale), 800)
        )

    cv2.imshow("VIGILX-AI | CCTV AI Detection", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Monitoring manually stopped.")
        break

cap.release()
cv2.destroyAllWindows()

print("CCTV monitoring stopped.")