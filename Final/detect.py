import cv2
import dlib
import numpy as np

# Load the predictor and the face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# Load the image
img_path = input("Enter image path: ")
img = cv2.imread(img_path)

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the face
faces = detector(gray)

# Get the facial landmarks for the first face detected
for face in faces:
    landmarks = predictor(gray, face)
    left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
    right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
    nose = np.array([landmarks.part(28).x, landmarks.part(28).y])
    point_22 = np.array([landmarks.part(22).x, landmarks.part(22).y])
    point_23 = np.array([landmarks.part(23).x, landmarks.part(23).y])
    
    # Calculate distances between points 22 and 28, and points 23 and 28
    dist_22_to_28 = np.linalg.norm(point_22 - nose)
    dist_23_to_28 = np.linalg.norm(point_23 - nose)
    
    # Calculate center point of both eyes
    center_eyes = np.mean([left_eye, right_eye], axis=0)
    
    # Calculate center point of triangle formed by points 22, 23, and 28
    triangle_center = np.mean([point_22, point_23, nose], axis=0)
    
    # Check if triangle center is above center of eyes and point 28
    if triangle_center[1] < center_eyes[1] and triangle_center[1] < nose[1]:
        print("Anomaly detected.")
    else:
        print("No anomaly detected.")
    
    # Print distances
    print("Distance from point 22 to 28:", dist_22_to_28)
    print("Distance from point 23 to 28:", dist_23_to_28)

# Display the image with facial landmarks
cv2.imshow("Facial Landmarks", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
