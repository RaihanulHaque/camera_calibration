import cv2
import os
from datetime import datetime

def create_images_folder():
    # Create 'images' folder if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')
        print("'images' folder created successfully")

def capture_images():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Create images folder
    create_images_folder()
    
    image_count = 0
    
    print("Press 'c' to capture an image")
    print("Press 'q' to quit")
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Display the frame
        cv2.imshow('Camera', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # If 'c' is pressed, save the image
        if key == ord('c'):
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"images/image_{timestamp}_{image_count}.jpg"
            
            # Save the image
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
            image_count += 1
        
        # If 'q' is pressed, quit the program
        elif key == ord('q'):
            print("Quitting...")
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images()