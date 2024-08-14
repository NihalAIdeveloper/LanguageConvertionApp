import cv2
import os

# Set the directory where images will be stored
directory = 'SignImage48x48/'
print("Current working directory:", os.getcwd())

# Create the main directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Create subdirectories for each letter and 'blank'
for i in range(65, 91):  # ASCII values for A-Z
    letter = chr(i)
    if not os.path.exists(f'{directory}/{letter}'):
        os.makedirs(f'{directory}/{letter}')
if not os.path.exists(f'{directory}/blank'):
    os.makedirs(f'{directory}/blank')

# Start video capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Create a dictionary to keep track of image count in each folder
    count = {chr(i): len(os.listdir(directory + f"/{chr(i)}")) for i in range(65, 91)}
    count['blank'] = len(os.listdir(directory + "/blank"))

    # Draw a rectangle on the frame to define the ROI (Region of Interest)
    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    cv2.imshow("Data Collection", frame)

    # Define the region of interest (ROI) for capturing the hand sign
    roi = frame[40:300, 0:300]
    cv2.imshow("ROI", roi)

    # Convert ROI to grayscale and resize it
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, (48, 48))

    # Wait for a key press and save the image based on the key pressed
    interrupt = cv2.waitKey(10) & 0xFF
    if interrupt in range(97, 123):  # For keys 'a' to 'z'
        letter = chr(interrupt).upper()
        filepath = os.path.join(directory, letter, f'{count[letter]}.jpg')
        cv2.imwrite(filepath, roi_resized)
        print(f"Captured {letter} image: {filepath}")

    if interrupt == ord('.'):  # For 'blank' images
        filepath = os.path.join(directory, 'blank', f'{count["blank"]}.jpg')
        cv2.imwrite(filepath, roi_resized)
        print(f"Captured blank image: {filepath}")

    if interrupt == ord('q'):  # Press 'q' to quit
        print("Exiting data collection...")
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
