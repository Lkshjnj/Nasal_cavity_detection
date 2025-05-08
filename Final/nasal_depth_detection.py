import cv2
import dlib
import numpy as np

# Load the detector
detector = dlib.get_frontal_face_detector()
# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Read the input image
img_path = input("Enter the path of the input image: ")
img = cv2.imread(img_path)

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = detector(gray)

# Loop through each face
for face in faces:
    # Get the landmarks/parts for the face in box d.
    landmarks = predictor(gray, face)

    # Define points 22, 23, and 28
    point_22 = np.array([landmarks.part(22).x, landmarks.part(22).y])
    point_23 = np.array([landmarks.part(23).x, landmarks.part(23).y])
    point_28 = np.array([landmarks.part(28).x, landmarks.part(28).y])

    # Calculate distance from point 22 to 28 and 23 to 28
    distance_22_28 = np.linalg.norm(point_22 - point_28)
    distance_23_28 = np.linalg.norm(point_23 - point_28)
    
    # Define points for both eyes
    left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
    right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])

    # Calculate the center point of both eyes
    center_eyes = (left_eye + right_eye) // 2

    # Check if center point is on or above/below point 28
    if center_eyes[1] <= point_28[1]:
        position = "above"
    else:
        position = "below"

    # Make a triangle from points 22, 23, and 28 and find the center of the triangle
    triangle_points = np.array([point_22, point_23, point_28])
    triangle_center = np.mean(triangle_points, axis=0)

    # Check if center point of the triangle is above the center point of both eyes
    if triangle_center[1] <= center_eyes[1]:
        triangle_position = "above"
    else:
        triangle_position = "below"

    # Draw lines connecting the facial landmarks and display the image
    img = cv2.line(img, left_eye, right_eye, (0, 255, 0), 2)
    img = cv2.line(img, left_eye, point_28, (0, 255, 0), 2)
    img = cv2.line(img, right_eye, point_28, (0, 255, 0), 2)
    img = cv2.line(img, point_22, point_23, (0, 255, 0), 2)
    img = cv2.line(img, point_23, point_28, (0, 255, 0), 2)
    img = cv2.line(img, point_28, point_22, (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Output", img)
    print(f"Distance from point 22 to 28: {distance_22_28:.2f}")
    print(f"Distance from point 23 to 28: {distance_23_28:.2f}")
    print(f"Center point of both eyes: {center_eyes:.2f}")
    print(f"Center point of triangle: {triangle_center:.2f}")


