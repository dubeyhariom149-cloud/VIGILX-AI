import cv2
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk


print("Loading YOLO model...")
model = YOLO("yolo11n.pt")
print("Model loaded successfully!")


root = tk.Tk()
root.title("VIGILX-AI - AI Camera Detection")
root.geometry("1000x700")


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera open nahi ho raha!")
    root.destroy()
    exit()

print("Camera started successfully!")
print("AI detection started.")

video_label = tk.Label(root)
video_label.pack(fill="both", expand=True)


def update_camera():

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Camera frame nahi mil raha!")
        root.after(100, update_camera)
        return

    # YOLO detection
    results = model.predict(
        frame,
        conf=0.35,
        verbose=False
    )

    # Detection boxes
    frame = results[0].plot()

    # BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # PIL
    image = Image.fromarray(frame)

    # Window size
    image = image.resize((960, 640))

    photo = ImageTk.PhotoImage(image=image)

    video_label.config(image=photo)
    video_label.image = photo

    root.after(30, update_camera)


def close_camera():

    print("Stopping camera...")

    camera.release()

    cv2.destroyAllWindows()

    root.destroy()

    print("Camera stopped.")


root.protocol("WM_DELETE_WINDOW", close_camera)

update_camera()

root.mainloop()