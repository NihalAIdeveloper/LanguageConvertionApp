import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Load the Keras model
model = load_model('my_final_model.keras')

# Define the labels used in your model
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']

def preprocess_image(image):
    # Convert to grayscale, resize, and normalize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))  # Resize to match model input
    image = image.reshape(1, 48, 48, 1)  # Reshape for the model
    return image / 255.0

def real_time_detection():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    # Create or open a text file to write detected symbols
    with open('recognized_text.txt', 'a') as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Define region of interest
            cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
            roi = frame[40:300, 0:300]

            # Preprocess the image
            processed_image = preprocess_image(roi)

            # Make prediction
            pred = model.predict(processed_image)
            prediction_label = labels[np.argmax(pred)]

            # Display results
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
            if prediction_label == 'blank':
                cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                accuracy = "{:.2f}".format(np.max(pred) * 100)
                cv2.putText(frame, f'{prediction_label} {accuracy}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show the output frame
            cv2.imshow("Output", frame)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to break the loop
                break
            elif key == 9:  # Tab key to write the symbol to the text file
                if prediction_label != 'blank':
                    f.write(prediction_label)
                    f.flush()  # Ensure the text is written to the file immediately
            elif key == 32:  # Space bar key to add a space to the text file
                f.write(' ')
                f.flush()  # Ensure the space is written to the file immediately

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def start_detection():
    # Disable the Convert button to prevent multiple instances
    convert_button.config(state=tk.DISABLED)
    real_time_detection()
    convert_button.config(state=tk.NORMAL)

def exit_app():
    # Confirm before exiting
    if messagebox.askokcancel("Exit", "Do you really want to exit?"):
        root.destroy()

def show_help():
    help_text = (
        "This Real-Time Detection Application allows you to detect and recognize hand gestures or symbols in real-time.\n\n"
        "How it works:\n"
        "1. Click the 'Convert' button to start the video capture and real-time detection process.\n"
        "2. The application will display the detected symbols along with their accuracy on the video feed.\n"
        "3. You can press the 'Tab' key to save the detected symbol to the 'recognized_text.txt' file.\n"
        "4. Press the 'Space' key to add a space to the text file.\n"
        "5. Click 'Exit' to close the application.\n"
        "\nNote: Ensure your camera is properly connected."
    )
    messagebox.showinfo("Help", help_text)

# Create the Tkinter window
root = tk.Tk()
root.title("Real-Time Detection Application")
root.geometry("650x450")
root.config(bg="#383C45")

image_path = r"C:\Users\nihal\Downloads\ASL-cover-image.png"
image_image = Image.open(image_path)
image_resized = image_image.resize((650, 450), Image.BILINEAR)
image_tk = ImageTk.PhotoImage(image_resized)
image_label = tk.Label(root, image=image_tk, bg="#6B71FF")
image_label.image = image_tk
image_label.place(x=4, y=-1)

# Create Convert button
convert_button = tk.Button(root, text="Convert", command=start_detection, width=15)
convert_button.place(x=275, y=405)

# Create Exit button
exit_button = tk.Button(root, text="Exit", command=exit_app, width=15)
exit_button.place(x=475, y=405)

# Create Help button
help_button = tk.Button(root, text="Help", command=show_help, width=15)
help_button.place(x=75, y=405)

# Start the Tkinter event loop
root.mainloop()
