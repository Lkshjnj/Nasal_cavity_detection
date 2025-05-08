import cv2
import dlib
import numpy as np
import math

# Load the facial landmark predictor
predictor = dlib.shape_predictor("C:\\Users\\Lakshya\\Desktop\\Nasal_cavity_detection\\Final\\shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# Function to compute Euclidean distance
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Key points
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        left_eyebrow = (landmarks.part(17).x, landmarks.part(17).y)
        right_eyebrow = (landmarks.part(26).x, landmarks.part(26).y)
        nose = (landmarks.part(30).x, landmarks.part(30).y)

        # Eye center
        center_eyes = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        # Eyebrow center
        center_brows = ((left_eyebrow[0] + right_eyebrow[0]) // 2, (left_eyebrow[1] + right_eyebrow[1]) // 2)
        # Combined center (eyes + brows)
        center_point = ((center_eyes[0] + center_brows[0]) // 2, (center_eyes[1] + center_brows[1]) // 2)

        # Draw key points
        cv2.circle(frame, center_point, 3, (0, 255, 255), -1)
        cv2.circle(frame, nose, 3, (0, 0, 255), -1)
        cv2.line(frame, center_point, nose, (255, 0, 0), 2)

        # Compute distances
        nasal_depth = abs(nose[1] - center_point[1])
        eye_distance = euclidean(left_eye, right_eye)

        # Show distances
        cv2.putText(frame, f"Nasal Depth: {nasal_depth:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Eye Distance: {eye_distance:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Determine anomaly based on relative proportion
        # (nasal depth > 0.8 * eye distance) --> anomaly
        if nasal_depth > 0.8 * eye_distance:
            status = "Anomaly Detected"
            color = (0, 0, 255)
        else:
            status = "Healthy"
            color = (0, 255, 0)

        cv2.putText(frame, status, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Facial Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
