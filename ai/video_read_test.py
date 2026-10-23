import cv2

VIDEO_PATH = r"C:\Users\Hariom\Documents\VIGILX-AI\videos\vidssave.com kidnap real shorts , captured in CCTV camera. #shorts 🙏 make it viral #viral 720P.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

print("Opened:", cap.isOpened())
print("Width :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS   :", cap.get(cv2.CAP_PROP_FPS))
print("Frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))

for i in range(5):
    ret, frame = cap.read()
    print("Frame", i + 1, ":", ret, "Shape:", None if frame is None else frame.shape)

cap.release()
