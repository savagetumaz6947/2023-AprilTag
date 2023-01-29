# Adding the parent directory to the path so that the _helper module can be imported
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from _helper import get_config

import numpy as np
import cv2
import glob


CONFIG = get_config()
CB_WIDTH = CONFIG["calc_intrinsics"]["checkerboard"]["width"]
CB_HEIGHT = CONFIG["calc_intrinsics"]["checkerboard"]["height"]

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CB_WIDTH*CB_HEIGHT, 3), np.float32)
objp[:,:2] = np.mgrid[0:CB_WIDTH, 0:CB_HEIGHT].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob(f"{CONFIG['images_for_calc_intrinsics_folder_location']}/*.jpg")

# loop over all images to be used for calculation
for fname in images:
    print(f"[INFO] Processing {fname}")
    img = cv2.imread(fname)
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (CB_WIDTH, CB_HEIGHT), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (CB_WIDTH, CB_HEIGHT), corners2, ret)
        cv2.imshow("img", img)
        # display the image for 500ms (0.5s)
        key = cv2.waitKey(500)

cv2.destroyAllWindows()

# calculate camera matrix and distortion coefficients
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# save the camera matrix and distortion coefficients into numpy files
np.save(CONFIG["calc_intrinsics"]["filenames"]["mtx"], mtx)
np.save(CONFIG["calc_intrinsics"]["filenames"]["dist"], dist)
