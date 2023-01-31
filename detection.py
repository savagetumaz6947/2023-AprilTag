import pupil_apriltags as atag
import cv2
import numpy as np
from _helper import get_config
from math import cos, sin, radians, atan2, sqrt
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

    # create a list of the things we want to return
    return_list = []

    # loop over the detected AprilTags
    for r in results:
        if not 1 <= r.tag_id <= 8:
            # not a tag used in FRC2023, may be a false detection
            continue
        # extract the bounding box (x, y)-coordinates for the AprilTag & center and undistort them
        # using the camera matrix and distortion coefficients
        corners = np.array(cv2.undistortImagePoints(r.corners, MTX, DIST)).squeeze()
        r = r._replace(corners=corners)
        center = np.array(cv2.undistortImagePoints(r.center, MTX, DIST)).squeeze()
        r = r._replace(center=center)

        # unpack the corners
        (ptA, ptB, ptC, ptD) = corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        if draw_tags:
            # draw the bounding box of the AprilTag detection on the original, distorted and colored frame
            cv2.line(frame, ptA, ptB, (255, 0, 0), 2)
            cv2.line(frame, ptB, ptC, (255, 0, 0), 2)
            cv2.line(frame, ptC, ptD, (255, 0, 0), 2)
            cv2.line(frame, ptD, ptA, (255, 0, 0), 2)

            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(center[0]), int(center[1]))
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

            # draw the id number on the frame
            tagId = f"ID {r.tag_id}"
            cv2.putText(frame, tagId, (ptA[0], ptA[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # detect the pose of the apriltag
        pose, e0, e1 = detector.detection_pose(r, (MTX[0,0], MTX[1,1], MTX[0,2], MTX[1,2]), TAG_SIZE)

        # create the 4x4 pose matrix that stores the pixel coordinates of the corners
        ncorners = np.array([[-TAG_SIZE/2, TAG_SIZE/2, 0, 1],
                        [TAG_SIZE/2, TAG_SIZE/2, 0, 1],
                        [TAG_SIZE/2, -TAG_SIZE/2, 0, 1],
                        [-TAG_SIZE/2, -TAG_SIZE/2, 0, 1],
                        [0, 0, 0, 1]])

        # transform the pixel coordinate corners to the real-world coordinate system based on the pose (linear algebra)
        corners_trans = np.matmul(pose, np.transpose(ncorners))

        if draw_tag_dists:
            # draw the real-world distance from the camera to the tag
            for i in range(3):
                cv2.putText(frame, f"{chr(ord('x')+i)}: {round(corners_trans[i][4], 5)}", (cX+15, cY+(15*i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        rot = R.from_matrix(pose[:3, :3])
        # TODO: need to double check, this seems to be incorrect
        yaw, pitch, roll = rot.as_euler('xyz', degrees=True)

        return_list.append({
            "id": r.tag_id,
            "dist": (corners_trans[0][4], corners_trans[1][4], corners_trans[2][4]), # x, y, z distances
            "rot": (yaw, pitch, roll), # yaw, pitch, roll
        })

    return frame, return_list

def draw_on_field(results):
    # open an image called field.png
    img = cv2.imread("field.png")
    TAGS = CONFIG["field_data"]["tags"]
    tag_used = [False]*9
    if len(results) > 0:
        avgX = 0
        avgY = 0
        for r in results:
            tag_used[r["id"]] = True
            x = r["dist"][0]
            z = r["dist"][2]
            yaw = radians(r["rot"][0])
            if yaw < 0 and x > 0:
                yaw = abs(yaw)
                l1 = z * sin(yaw)
                l2 = x * sin(yaw)
                y_Trans = - (l1 + l2)
                x_Trans = - (z * cos(yaw))
            elif yaw < 0 and x < 0:
                yaw = abs(yaw)
                x = abs(x)
                l1 = z * sin(yaw)
                l2 = (1 / cos(yaw)) * x
                y_Trans = - (l1 - l2)
                x_Trans = - (z * sin(yaw))
            elif yaw > 0 and x > 0:
                l1 = z * sin(yaw)
                l2 = (1 / cos(yaw)) * x
                y_Trans = l1 - l2
                x_Trans = - (z * cos(yaw))
            else:
                x = abs(x)
                l1 = x * cos(yaw)
                l2 = z * sin(yaw)
                y_Trans = l1 + l2
                x_Trans = - (z * cos(yaw))

        #     # imageX = -r["dist"][2] * cos(radians(90 - r["rot"][0]))
        #     # imageY = (r["dist"][0] * cos(radians(r["rot"][0])) + r["dist"][2] * sin(radians(90 - r["rot"][0])))
        #     # # print(f"Tag {r['id']} at ({imageX}, {imageY})")
            # avgX += TAGS[r["id"]]["x"] + imageX * CONFIG["field_data"]["m_to_px"]
            # avgY += TAGS[r["id"]]["y"] + imageY * CONFIG["field_data"]["m_to_px"]
        #     l1 = (-r["dist"][0]) * cos(radians(r["rot"][0])) # x cos yaw
        #     l2 = r["dist"][2] * sin(radians(r["rot"][0])) # z sin yaw
        #     y_Trans = l1 + l2
        #     y_Trans = 0
        #     x_Trans = - (r["dist"][2] * sin(radians(r["rot"][0])))
        #     print(r["dist"][2], r["rot"][0], sin(radians(r["rot"][0])), x_Trans)
            avgX += TAGS[r["id"]]["x"] + x_Trans * CONFIG["field_data"]["m_to_px"]
            avgY += TAGS[r["id"]]["y"] + y_Trans * CONFIG["field_data"]["m_to_px"]
        # #     avgX += TAGS[r["id"]]["x"] - (r["dist"][2] * CONFIG["field_data"]["m_to_px"])
        # #     avgY += TAGS[r["id"]]["y"] - (r["dist"][0] * CONFIG["field_data"]["m_to_px"])
        avgX /= len(results)
        avgY /= len(results)
        cv2.circle(img, (int(avgX), int(avgY)), 5, (0, 255, 0), -1)
    for point in TAGS:
        if point["id"] == 0:
            continue
        # draw a circle on the image at the point
        cv2.circle(img, (point["x"], point["y"]), 5, (0, 0, 255) if tag_used[point["id"]] else (0, 0, 0), -1)
        # draw the id of the point on the image
        cv2.putText(img, str(point["id"]), (point["x"], point["y"]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    cap = cv2.VideoCapture(CONFIG["camera"]["port"])
    print(f"[INFO] start VideoCapture on USB {CONFIG['camera']['port']}")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (CONFIG["camera"]["size"]["width"], CONFIG["camera"]["size"]["height"]))
        # detect_apriltags(frame)
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
