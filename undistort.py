import cv2
import numpy as np
import pickle

def undisort_frame(frame, cameraMatrix, dist):
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

    # Undistort the image
    undistorted_frame = cv2.undistort(frame, cameraMatrix, dist, None, new_camera_matrix)

    # Crop the image
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y + h, x:x + w]

    return undistorted_frame


# Load the calibration data
with open("calibration_1.pkl", "rb") as f:
    cameraMatrix, dist = pickle.load(f)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    # Undistort the frame
    undistorted_frame = undisort_frame(frame, cameraMatrix, dist)

    # Display the frame
    cv2.imshow('Undistorted Frame', undistorted_frame)

    # Check for 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break