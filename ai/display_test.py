import cv2
import numpy as np

print("Opening test window...")

img = np.zeros((400, 600, 3), dtype=np.uint8)

cv2.putText(
    img,
    "VIGILX-AI DISPLAY TEST",
    (80, 200),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

cv2.imshow("VIGILX-AI Test", img)

print("Window created. Press Q to close.")

while True:
    key = cv2.waitKey(100) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()