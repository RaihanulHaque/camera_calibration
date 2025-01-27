import cv2
import numpy as np


def calibrate_camera(img):
    # Chessboard parameters
    chessboardSize = (7, 7)
    size_of_chessboard_squares_mm = 24

    found, corners = cv2.findChessboardCorners(img, chessboardSize, cv2.CALIB_CB_ADAPTIVE_THRESH)

    print(found)

    cv2.drawChessboardCorners(img, chessboardSize, corners, found)

    cv2.imshow('img', img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    # Perform camera calibration

# path = '/Users/rahi/Code/camera_calibration/cameraCalibration/images/image_20250127_183643_0.jpg'
path = 'cameraCalibration/images/image_20250127_183740_21.jpg'

img = cv2.imread(path)
calibrate_camera(img)