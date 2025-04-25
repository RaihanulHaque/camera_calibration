import numpy as np
import cv2 as cv
import glob
import pickle

# Chessboard dimensions (number of inner corners per row and column)
chessboardSize = (7, 7)
frameSize = (1920, 1080)

# Size of each square in the chessboard (in millimeters)
size_of_chessboard_squares_mm = 24

# Termination criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in real-world space)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
objp *= size_of_chessboard_squares_mm

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load calibration images
images = glob.glob('cameraCalibration/images/*.jpg')

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('Chessboard Corners', img)
        cv.waitKey(500)

cv.destroyAllWindows()

# Perform camera calibration
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Save the calibration results
with open("calibration_1.pkl", "wb") as f:
    pickle.dump((cameraMatrix, dist), f)


print("Done calibrating the camera")

# Load a test image for undistortion
test_image_path = '/Users/rahi/Code/camera_calibration/cameraCalibration/images_2/image_20250127_211905_3.jpg'
img = cv.imread(test_image_path)
h, w = img.shape[:2]

# Get the optimal new camera matrix
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

# Undistort the image
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# Crop the undistorted image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)

# Undistort using remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# Crop the remapped image
dst = dst[y:y+h, x:x+w]
cv.imwrite('After.jpg', dst)

# Calculate reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"Total reprojection error: {mean_error / len(objpoints)}")

# Function to draw colored lines on the chessboard
def draw_colored_lines(img, corners, chessboard_size):
    corners = corners.reshape(-1, 2)
    for i in range(chessboard_size[1]):
        for j in range(chessboard_size[0] - 1):
            pt1 = tuple(corners[i * chessboard_size[0] + j].astype(int))
            pt2 = tuple(corners[i * chessboard_size[0] + j + 1].astype(int))
            cv.line(img, pt1, pt2, (0, 255, 0), 2)  # Green horizontal lines

    for i in range(chessboard_size[0]):
        for j in range(chessboard_size[1] - 1):
            pt1 = tuple(corners[j * chessboard_size[0] + i].astype(int))
            pt2 = tuple(corners[(j + 1) * chessboard_size[0] + i].astype(int))
            cv.line(img, pt1, pt2, (0, 0, 255), 2)  # Red vertical lines

    return img

# Load an image and draw colored lines on the detected chessboard
test_image = cv.imread(test_image_path)
gray = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

if ret:
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    test_image_with_lines = draw_colored_lines(test_image.copy(), corners2, chessboardSize)
    cv.imshow('Chessboard with Colored Lines', test_image_with_lines)
    cv.waitKey(0)
    cv.imwrite('chessboard_with_lines.png', test_image_with_lines)

cv.destroyAllWindows()