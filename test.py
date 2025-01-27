import cv2
import numpy as np
import glob

# Chessboard dimensions (number of inner corners per row and column)
chessboardSize = (7, 7)  # Change this to match your chessboard
frameSize = (1920, 1080)   # Change this to match your image size

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in real-world space)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load calibration images
images = glob.glob('cameraCalibration/images/*.jpg')  # Change the path and extension if needed

if len(images) == 0:
    print("Error: No images found in the specified directory.")
    exit()

for image in images:
    img = cv2.imread(image)
    if img is None:
        print(f"Error: Unable to load image {image}")
        continue

    # Color-segmentation to get binary mask
    lwr = np.array([0, 0, 143])
    upr = np.array([179, 61, 252])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(hsv, lwr, upr)

    # Extract chess-board
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dlt = cv2.dilate(msk, krn, iterations=5)
    res = 255 - cv2.bitwise_and(dlt, msk)

    # Convert to grayscale for corner detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(res, chessboardSize, None)

    if ret:
        print(f"Chessboard corners found in {image}")
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)
    else:
        print(f"No chessboard corners found in {image}")

cv2.destroyAllWindows()

# Check if any valid images were found
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("Error: No valid images with chessboard corners found. Calibration cannot proceed.")
    exit()

# Perform camera calibration
ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

if ret:
    print("Calibration successful!")
    print("Camera matrix:\n", cameraMatrix)
    print("Distortion coefficients:\n", dist)
else:
    print("Error: Calibration failed.")

# Save the calibration results
np.savez("calibration_data.npz", cameraMatrix=cameraMatrix, dist=dist)

# Undistort a test image
test_image = cv2.imread(images[0])
h, w = test_image.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

# Undistort
dst = cv2.undistort(test_image, cameraMatrix, dist, None, newCameraMatrix)

# Crop the undistorted image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# Display the undistorted image
cv2.imshow("Undistorted Image", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()