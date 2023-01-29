# Adding the parent directory to the path so that the _helper module can be imported
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from _helper import get_config

import cv2

########################################################################################################
# ISSUE: the camera properties are not set correctly even though the cap.set functions return true
#        (when using the WPILibPi web, the camera properties are set correctly using the settings menu)
#        (except for the width & height, which are cropped each frame in the other scripts)
# WORKAROUND: use the v4l2-ctl command to set the camera properties
# v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=1
print("[WARNING] This script is not working as expected. See the code for more information.")
from time import sleep
for x in range(5):
    print(5-x)
    sleep(1)
########################################################################################################

CONFIG = get_config()

# start a videocapture device at the given usb port
cap = cv2.VideoCapture(CONFIG["camera"]["port"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera"]["size"]["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera"]["size"]["height"])
cap.set(cv2.CAP_PROP_FPS, CONFIG["camera"]["properties"]["fps"])
cap.set(cv2.CAP_PROP_BRIGHTNESS, CONFIG["camera"]["properties"]["brightness"])
cap.set(cv2.CAP_PROP_CONTRAST, CONFIG["camera"]["properties"]["contrast"])
cap.set(cv2.CAP_PROP_SATURATION, CONFIG["camera"]["properties"]["saturation"])
cap.set(cv2.CAP_PROP_GAIN, CONFIG["camera"]["properties"]["gain"])
cap.set(cv2.CAP_PROP_EXPOSURE, CONFIG["camera"]["properties"]["exposure_absolute"])

# check if the camera has accepted the settings
print("[INFO] camera properties:")
print(f"width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"fps: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
print(f"contrast: {cap.get(cv2.CAP_PROP_CONTRAST)}")
print(f"saturation: {cap.get(cv2.CAP_PROP_SATURATION)}")
print(f"gain: {cap.get(cv2.CAP_PROP_GAIN)}")
print(f"exposure_absolute: {cap.get(cv2.CAP_PROP_EXPOSURE)}")

# show the current camera image
print(f"[INFO] start VideoCapture on USB {CONFIG['camera']['port']}")
while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
print("[INFO] stop VideoCapture")