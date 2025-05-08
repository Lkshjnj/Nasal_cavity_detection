import cv2
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over each face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect the nasal bridge using the Haar cascade
        nasal_bridge = gray[y + int(h/4):y + int(h/2), x:x + w]
        nasal_bridge_cascade = cv2.CascadeClassifier("NasalBridge.xml")
        nasal_bridge_rects = nasal_bridge_cascade.detectMultiScale(nasal_bridge, 1.3, 5)

        # Calculate the average depth of the nasal bridge
        depth = 0
        for (nb_x, nb_y, nb_w, nb_h) in nasal_bridge_rects:
            depth += nb_y + nb_h/2
        if len(nasal_bridge_rects) > 0:
            depth /= len(nasal_bridge_rects)

        # Display the depth on the frame
        cv2.putText(frame, "Depth: {:.2f}".format(depth), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Nasal Bridge Detection", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()
