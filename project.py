import cv2
import dlib
import numpy as np
import os
import csv
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw
import time
import threading
from typing import Dict, Any, Optional, Tuple

# Load the predictor and face detector
predictor = dlib.shape_predictor("D:\\projects\\ML\\gemini\\Nasal-Depth-Detection\\Final\\shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
csv_file = "D:\\projects\\ML\\gemini\\Nasal-Depth-Detection\\photos\\nasal.csv"
csv_summary_file = "D:\\projects\\ML\\gemini\\Nasal-Depth-Detection\\photos\\summary_report.csv"

# Initialize counters
total_images = 0
anomaly_count = 0
normal_count = 0

# Prompt user for folder
folder_path = input("Enter the folder path containing the images: ").strip()
if not os.path.exists(folder_path):
    print(f"Error: Folder '{folder_path}' not found.")
    exit()

# Supported image formats
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

# Prepare CSV logging
file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
with open(csv_file, mode="a", newline="") as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Image", "Distance_22_28", "Distance_23_28", "Anomaly_Status"])

    # Process each image
    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot load image {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
            right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
            nose = np.array([landmarks.part(28).x, landmarks.part(28).y])
            point_22 = np.array([landmarks.part(22).x, landmarks.part(22).y])
            point_23 = np.array([landmarks.part(23).x, landmarks.part(23).y])

            dist_22_to_28 = np.linalg.norm(point_22 - nose)
            dist_23_to_28 = np.linalg.norm(point_23 - nose)

            center_eyes = np.mean([left_eye, right_eye], axis=0)
            triangle_center = np.mean([point_22, point_23, nose], axis=0)

            # Check anomaly
            if triangle_center[1] < center_eyes[1] and triangle_center[1] < nose[1]:
                status_text = "Anomaly Detected"
                anomaly_count += 1
            else:
                status_text = "No Anomaly"
                normal_count += 1

            writer.writerow([img_name, f"{dist_22_to_28:.2f}", f"{dist_23_to_28:.2f}", status_text])
            total_images += 1

# Save summary
summary_file_exists = os.path.exists(csv_summary_file) and os.path.getsize(csv_summary_file) > 0
with open(csv_summary_file, mode="a", newline="") as summary_file:
    summary_writer = csv.writer(summary_file)
    if not summary_file_exists:
        summary_writer.writerow(["Total Processed Images", "Anomalies Detected", "Normal Detected", "Anomalies Percentage"])

    if total_images > 0:
        anomalies_percentage = (anomaly_count / total_images) * 100
        summary_writer.writerow([total_images, anomaly_count, normal_count, f"{anomalies_percentage:.2f}%"])
    else:
        summary_writer.writerow([0, 0, 0, "0%"])

# Print final summary
print("\n--- Final Report ---")
print(f"Total images processed: {total_images}")
print(f"Anomalies detected    : {anomaly_count}")
print(f"Normal cases detected : {normal_count}")
if total_images > 0:
    print(f"Anomaly percentage    : {anomaly_count / total_images * 100:.2f}%")
else:
    print("No images were processed.")
