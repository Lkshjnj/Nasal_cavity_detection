import cv2
import numpy as np

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces and measure the distance between the eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml").detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eye_center = (x + ex + ew // 2, y + ey + eh // 2)
            cv2.circle(frame, eye_center, 5, (0, 0, 255), -1)
            if len(eyes) >= 2:
                eye1, eye2 = eyes[:2]
                eye1_center = (x + eye1[0] + eye1[2] // 2, y + eye1[1] + eye1[3] // 2)
                eye2_center = (x + eye2[0] + eye2[2] // 2, y + eye2[1] + eye2[3] // 2)
                distance = np.sqrt((eye2_center[0] - eye1_center[0]) ** 2 + (eye2_center[1] - eye1_center[1]) ** 2)
                cv2.line(frame, eye1_center, eye2_center, (255, 0, 0), 2)
                cv2.putText(frame, str(int(distance)), (int((eye1_center[0]+eye2_center[0])/2), int((eye1_center[1]+eye2_center[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()
