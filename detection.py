import pupil_apriltags as atag
import cv2
import numpy as np
from _helper import get_config
from math import cos, sin, radians
from scipy.spatial.transform import Rotation as R

CONFIG = get_config()

MTX = np.load(CONFIG["calc_intrinsics"]["filenames"]["mtx"])
DIST = np.load(CONFIG["calc_intrinsics"]["filenames"]["dist"])

TAG_SIZE = CONFIG["apriltag"]["tag_size"]

def detect_apriltags(input_frame: cv2.Mat, draw_tags=True, draw_tag_dists=True):
    frame = input_frame.copy()

    # convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # get the dectector with the family of tags we want to detect
    detector = atag.Detector(families=CONFIG["apriltag"]["tag_family"])

    # detect the apriltags in the image
    results = detector.detect(gray_frame, True, (MTX[0,0], MTX[1,1], MTX[0,2], MTX[1,2]), TAG_SIZE)

    return_list = []

    for r in results:
        if not (r.decision_margin > 10 and r.hamming <= 1 and 1 <= r.tag_id <= 8):
            continue
        # draw r.center and r.corners in the image
        cv2.circle(frame, (int(r.center[0]), int(r.center[1])), 5, (0, 0, 255), -1)
        # draw r.corners as a polygon
        cv2.polylines(frame, np.int32([r.corners]), True, (0, 255, 0), 2)
        # translate that to yaw, pitch, roll in degrees
        rot = R.from_matrix(r.pose_R)
        yaw, pitch, roll = rot.as_euler('yxz', degrees=True)
        
        return_list.append({
            "id": r.tag_id,
            "dist": tuple(np.squeeze(r.pose_t)), # x, y, z distances
            "rot": (yaw, pitch, roll), # yaw, pitch, roll
        })

    return frame, return_list

def draw_on_field(results):
    # open an image called field.png
    img = cv2.imread("field.png")
    # the tags' extrinsic data
    TAGS = CONFIG["field_data"]["tags"]
    tag_used = [False]*9
    # if there are tags detected
    if len(results) > 0:
        # two variables storing the estimated location of the camera on the field
        avgX = 0
        avgY = 0
        for r in results:
            # this id has been used to estimate the location of the camera -> will light up in red
            tag_used[r["id"]] = True
            # get the values
            x = r["dist"][0]
            z = r["dist"][2]
            yaw = radians(r["rot"][0])
            # split into different cases (calculate in meters)
            # TODO: find a more elegant catch-all solution
            if yaw > 0 and x > 0:
                x_Trans = - (z * cos(yaw) + x * sin(yaw))
                y_Trans = z * sin(yaw) - x * cos(yaw)
            elif yaw < 0 and x < 0:
                yaw = abs(yaw)
                x = abs(x)
                x_Trans = - (z * cos(yaw) + x * sin(yaw))
                y_Trans = - (z * sin(yaw) - x * cos(yaw))
            elif yaw > 0 and x < 0:
                x = abs(x)
                x_Trans = - (z * cos(yaw) - x * sin(yaw))
                y_Trans = z * sin(yaw) + x * cos(yaw)
            else:
                yaw = abs(yaw)
                x_Trans = - (z * cos(yaw) - x * sin(yaw))
                y_Trans = - (z * sin(yaw) + x * cos(yaw))
            # add to the average location of the camera
            # the tag's original location + the translation converted to pixels
            avgX += TAGS[r["id"]]["x"] + x_Trans * CONFIG["field_data"]["m_to_px"]
            avgY += TAGS[r["id"]]["y"] + y_Trans * CONFIG["field_data"]["m_to_px"]
        # average out the estimations
        avgX /= len(results)
        avgY /= len(results)
        # draw a dot on the estimated location of the camera
        # TODO: turn that into a rectangle with the rotation of the robot
        cv2.circle(img, (int(avgX), int(avgY)), 5, (0, 255, 0), -1)
    for point in TAGS:
        if point["id"] == 0:
            continue
        # draw a circle on the image at the tag point (if used, it will light up in red)
        cv2.circle(img, (point["x"], point["y"]), 5, (0, 0, 255) if tag_used[point["id"]] else (0, 0, 0), -1)
        # write the id of the point on the image
        cv2.putText(img, str(point["id"]), (point["x"], point["y"]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    cap = cv2.VideoCapture(CONFIG["camera"]["port"])
    print(f"[INFO] start VideoCapture on USB {CONFIG['camera']['port']}")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (CONFIG["camera"]["size"]["width"], CONFIG["camera"]["size"]["height"]))
        drawn_frame, tags = detect_apriltags(frame)
        field = draw_on_field(tags)
        cv2.imshow("field", field)
        cv2.imshow("drawn_frame", drawn_frame)
        cv2.imshow("frame", frame)
        print(tags)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cap.release()
    print("[INFO] stop VideoCapture")
