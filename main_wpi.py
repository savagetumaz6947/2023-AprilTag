from cscore import CameraServer
import json
from ntcore import NetworkTableInstance
import numpy as np
import time
import cv2
from detection import detect_apriltags

# use the settings from the WPILib web interface
with open('/boot/frc.json') as f:
    config = json.load(f)
camera = config['cameras'][0]

width = camera['width']
height = camera['height']

CameraServer.enableLogging()

# start the camera server
camera = CameraServer.startAutomaticCapture()
camera.setResolution(width, height)

input_stream = CameraServer.getVideo()
orig_output_stream = CameraServer.putVideo("Original", width, height)
marked_output_stream = CameraServer.putVideo("Marked", width, height)

# get the network table instance
inst = NetworkTableInstance.getDefault()
inst.startServer()
vision_nt = inst.getTable("RPiVision")

xPub = vision_nt.getDoubleTopic("target_x").publish()
yPub = vision_nt.getDoubleTopic("target_y").publish()

# Allocating new images is very expensive, always try to preallocate
img = np.zeros(shape=(1024, 576, 3), dtype=np.uint8)

# Wait for NetworkTable to start
time.sleep(0.5)

while True:
    start_time = time.time()
    frame_time, input_img = input_stream.grabFrame(img)

    if frame_time == 0: # There is an error
        orig_output_stream.notifyError(input_stream.getError())
        marked_output_stream.notifyError(input_stream.getError())
        continue

    marked_frame, tags = detect_apriltags(input_img, draw_tags=True, draw_tag_dists=True)

    processing_time = time.time() - start_time
    fps = 1 / processing_time
    print(tags)
    # put the fps count on the marked_frame
    cv2.putText(marked_frame, str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    orig_output_stream.putFrame(input_img)
    marked_output_stream.putFrame(marked_frame)

    # publish to NetworkTables
    # xPub.set(<double>) or do more calculations
