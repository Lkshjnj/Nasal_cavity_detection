import cv2
import dlib
import numpy as np

# Load the shape predictor
predictor = dlib.shape_predictor("C:\\Use")

# Load the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    _, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = dlib.get_frontal_face_detector()(gray, 1)

    # Loop through the detected faces
    for face in faces:
        # Predict the landmarks for each face
        landmarks = predictor(gray, face)

        # Get the coordinates of the left eyebrow
        left_eyebrow_x = landmarks.part(17).x
        left_eyebrow_y = landmarks.part(17).y
        right_eyebrow_x = landmarks.part(26).x
        right_eyebrow_y = landmarks.part(26).y

        # Get the coordinates of both eyes
        left_eye_x = landmarks.part(36).x
        left_eye_y = landmarks.part(36).y
        right_eye_x = landmarks.part(45).x
        right_eye_y = landmarks.part(45).y

        # Get the coordinates of the nose
        nose_x = landmarks.part(30).x
        nose_y = landmarks.part(30).y

        # Calculate the center point for the eyebrows and eyes
        center_x = int((left_eyebrow_x + right_eyebrow_x + left_eye_x + right_eye_x) / 4)
        center_y = int((left_eyebrow_y + right_eyebrow_y + left_eye_y + right_eye_y) / 4)

        # Draw a circle at the center point
        cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), -1)

        # Draw a line between the nose and center point
        cv2.line(frame, (center_x, center_y), (nose_x, nose_y), (255, 0, 0), 2)

    # Display the frame with the landmarks and center point
    cv2.imshow("Distance between eyes and nose", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

