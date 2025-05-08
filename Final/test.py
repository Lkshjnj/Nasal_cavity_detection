import cv2
import dlib
import numpy as np

# Load the facial landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the input image
image_path = input("Enter the path of the input image: ")
img = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the face in the image
detector = dlib.get_frontal_face_detector()
faces = detector(gray)

# Loop over each detected face
for face in faces:
    # Get the facial landmarks
    landmarks = predictor(gray, face)
    
    # Get the coordinates of the left and right eyes and the nose
    left_eye = np.array([landmarks.part(22).x, landmarks.part(22).y])
    right_eye = np.array([landmarks.part(27).x, landmarks.part(27).y])
    nose = np.array([landmarks.part(28).x, landmarks.part(28).y])
    
    # Calculate the distances from the eyes to the nose
    dist_left_eye_nose = np.linalg.norm(left_eye - nose)
    dist_right_eye_nose = np.linalg.norm(right_eye - nose)
    
    # Calculate the center points of the eyes and check if they are above or below the nose
    center_left_eye = (left_eye + np.array([landmarks.part(23).x, landmarks.part(23).y])) / 2
    center_right_eye = (right_eye + np.array([landmarks.part(24).x, landmarks.part(24).y])) / 2
    center_eyes = (center_left_eye + center_right_eye) / 2
    if center_eyes[1] < nose[1]:
        print("Center point of eyes is above the nose.")
    elif center_eyes[1] > nose[1]:
        print("Center point of eyes is below the nose.")
    else:
        print("Center point of eyes is on the nose.")
    
    # Calculate the center point of the triangle formed by points 22, 23, and 28
    center_triangle = (left_eye + np.array([landmarks.part(23).x, landmarks.part(23).y]) + nose) / 3
    
    # Check if the center point of the triangle is above or below the center point of the eyes
    if center_triangle[1] < center_eyes[1]:
        print("Center point of triangle is above the center point of eyes.")
    elif center_triangle[1] > center_eyes[1]:
        print("Center point of triangle is below the center point of eyes.")
    else:
        print("Center point of triangle is on the center point of eyes.")
    
    # Print the distances
    print("Distance from point 22 to 28:", np.linalg.norm(left_eye - nose))
    print("Distance from point 23 to 28:", np.linalg.norm(np.array([landmarks.part(23).x, landmarks.part(23).y]) - nose))
