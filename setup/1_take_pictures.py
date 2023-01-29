# Adding the parent directory to the path so that the _helper module can be imported
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from _helper import get_config

import cv2

CONFIG = get_config()

print(f"[WARNING] this program will overwrite the images in the {CONFIG['images_for_calc_intrinsics_folder_location']} folder")

cap = cv2.VideoCapture(CONFIG["camera"]["port"])
print(f"[INFO] start VideoCapture on USB {CONFIG['camera']['port']}")

cnt = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (CONFIG["camera"]["size"]["width"], CONFIG["camera"]["size"]["height"]))
    key = cv2.waitKey(1)
    cv2.imshow("frame", frame)
    if key == ord('q'):
        break
    if key == ord(' '):
        fileName = f"{CONFIG['images_for_calc_intrinsics_folder_location']}/{cnt}.jpg"
        print(f"[INFO] took picture {fileName}")
        cv2.imwrite(fileName, frame)
        cnt += 1

cap.release()
print("[INFO] stop VideoCapture")
