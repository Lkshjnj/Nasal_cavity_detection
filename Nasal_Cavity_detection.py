import cv2
import dlib
import numpy as np
from csv import writer
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw
import time
import threading
import os
from typing import Dict, Any, Optional, Tuple
import math

# Load the predictor and face detector
predictor = dlib.shape_predictor("C:\\Users\\Lakshya\\Desktop\\Nasal_cavity_detection\\Final\\shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
# csv_file = "D:\\projects\\ML\\gemini\\Nasal-Depth-Detection\\photos\\nasal.csv"
# csv_summary_file = "D:\\projects\\ML\\gemini\\Nasal-Depth-Detection\\photos\\summary_report.csv"

class NasalCavityDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nasal Cavity Detection AI")
        self.root.geometry("900x600")
        self.root.configure(bg="#f5f7fa")
        
        # State variables
        self.patient_data = {
            "patientId": "",
            "age": "",
            "heightWeight": ""
        }
        self.uploaded_image = None
        self.display_image = None
        self.show_landmarks = True
        self.show_measurements = True
        self.results = None
        self.is_analyzing = False
        
        # Create the main layout
        self.create_header()
        self.create_main_content()
        
    def create_header(self):
        """Create the application header"""
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        header_frame.pack(fill=tk.X)
        
        header_label = tk.Label(
            header_frame, 
            text="Nasal Cavity Detection AI", 
            font=('Arial', 18), 
            bg="#2c3e50", 
            fg="white",
            pady=10
        )
        header_label.pack(side=tk.LEFT, padx=20)
    
    def create_main_content(self):
        """Create the main content area with panels"""
        main_content = tk.Frame(self.root, bg="#f5f7fa", padx=20, pady=20)
        main_content.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel
        self.create_left_panel(main_content)
        
        # Create right content area
        self.create_content_area(main_content)
    
    def create_left_panel(self, parent):
        """Create the left panel with patient data and controls"""
        left_panel = tk.Frame(
            parent, 
            bg="white", 
            width=260, 
            padx=20, 
            pady=20,
            highlightbackground="#ddd",
            highlightthickness=1,
            bd=0
        )
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)  # Prevent shrinking
        
        # Patient Data Section
        tk.Label(
            left_panel, 
            text="Patient Data", 
            font=('Arial', 12, 'bold'), 
            bg="white", 
            fg="#2c3e50"
        ).pack(anchor=tk.W, pady=(0, 15))
        
        # Patient ID Input
        tk.Label(
            left_panel, 
            text="Patient ID:", 
            font=('Arial', 10), 
            bg="white", 
            fg="#2c3e50"
        ).pack(anchor=tk.W)
        
        patient_id_entry = tk.Entry(
            left_panel, 
            font=('Arial', 10),
            bg="#eef2f7",
            fg="#2c3e50",
            bd=1,
            relief=tk.SOLID
        )
        patient_id_entry.pack(fill=tk.X, pady=(0, 10))
        patient_id_entry.bind("<KeyRelease>", lambda e: self.update_patient_data("patientId", patient_id_entry.get()))
        
        # Age Input
        tk.Label(
            left_panel, 
            text="Age:", 
            font=('Arial', 10), 
            bg="white", 
            fg="#2c3e50"
        ).pack(anchor=tk.W)
        
        age_entry = tk.Entry(
            left_panel, 
            font=('Arial', 10),
            bg="#eef2f7",
            fg="#2c3e50",
            bd=1,
            relief=tk.SOLID
        )
        age_entry.pack(fill=tk.X, pady=(0, 10))
        age_entry.bind("<KeyRelease>", lambda e: self.update_patient_data("age", age_entry.get()))
        
        # Height/Weight Input
        tk.Label(
            left_panel, 
            text="Height/Weight:", 
            font=('Arial', 10), 
            bg="white", 
            fg="#2c3e50"
        ).pack(anchor=tk.W)
        
        height_weight_entry = tk.Entry(
            left_panel, 
            font=('Arial', 10),
            bg="#eef2f7",
            fg="#2c3e50",
            bd=1,
            relief=tk.SOLID
        )
        height_weight_entry.pack(fill=tk.X, pady=(0, 10))
        height_weight_entry.bind("<KeyRelease>", lambda e: self.update_patient_data("heightWeight", height_weight_entry.get()))
        
        # Image Upload Section
        tk.Label(
            left_panel, 
            text="Image Upload", 
            font=('Arial', 12, 'bold'), 
            bg="white", 
            fg="#2c3e50"
        ).pack(anchor=tk.W, pady=(20, 15))
        
        self.upload_frame = tk.Frame(
            left_panel, 
            bg="#eef2f7", 
            height=100,
            highlightbackground="#ccc",
            highlightthickness=1,
            highlightcolor="#ccc",
            bd=0,
            cursor="hand2"
        )
        self.upload_frame.pack(fill=tk.X)
        self.upload_frame.pack_propagate(False)
        
        upload_label = tk.Label(
            self.upload_frame, 
            text="CAPTURE IMAGE", 
            font=('Arial', 10), 
            bg="#eef2f7", 
            fg="#95a5a6"
        )
        upload_label.pack(expand=True)
        
        # self.upload_frame.bind("<Button-1>", self.browse_files)
        self.upload_frame.bind("<Button-1>",self._run_analysis)
        
        # Controls Section
        tk.Label(
            left_panel, 
            text="Controls", 
            font=('Arial', 12, 'bold'), 
            bg="white", 
            fg="#2c3e50"
        ).pack(anchor=tk.W, pady=(20, 15))
        
        # Toggle Landmarks Button
        self.landmark_btn = tk.Button(
            left_panel, 
            text="Hide Landmarks", 
            bg="#3498db", 
            fg="white", 
            font=('Arial', 10),
            bd=0,
            padx=5,
            pady=5,
            cursor="hand2",
            command=self.toggle_landmarks
        )
        self.landmark_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Toggle Measurements Button
        self.measurement_btn = tk.Button(
            left_panel, 
            text="Hide Measurements", 
            bg="#3498db", 
            fg="white", 
            font=('Arial', 10),
            bd=0,
            padx=5,
            pady=5,
            cursor="hand2",
            command=self.toggle_measurements
        )
        self.measurement_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Analyze Button (by video)
        self.analyze_btn = tk.Button(
            left_panel, 
            text="Live Analysis", 
            bg="#27ae60", 
            fg="white", 
            font=('Arial', 11, 'bold'),
            bd=0,
            padx=5,
            pady=8,
            cursor="hand2",
            command=self.analyze_image
        )
        self.analyze_btn.pack(fill=tk.X, pady=(10, 0))

        # Analyze Button 2 (by image)
        # self.analyze_btn = tk.Button(
        #     left_panel, 
        #     text="Live Analysis", 
        #     bg="#27ae60", 
        #     fg="white", 
        #     font=('Arial', 11, 'bold'),
        #     bd=0,
        #     padx=5,
        #     pady=8,
        #     cursor="hand2",
        #     command=self.analyze_image_by_upload,
        #     state=tk.DISABLED
        # )
        # self.analyze_btn.pack(fill=tk.X, pady=(10, 0))
    
    def create_content_area(self, parent):
        """Create the main content area with image display and results"""
        content_frame = tk.Frame(parent, bg="#f5f7fa")
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(20, 0))
        
        # Image display area
        self.image_frame = tk.Frame(
            content_frame, 
            bg="white", 
            highlightbackground="#ddd",
            highlightthickness=1,
            bd=0,
            height=320
        )
        self.image_frame.pack(fill=tk.X, pady=(0, 20))
        self.image_frame.pack_propagate(False)
        
        # Canvas for image display
        self.canvas = tk.Canvas(
            self.image_frame, 
            bg="white", 
            bd=0, 
            highlightthickness=0,
            width=480, 
            height=320
        )
        self.canvas.pack(expand=True)
        
        # Placeholder text
        self.placeholder_text = self.canvas.create_text(
            240, 160, 
            text="Upload an image to begin analysis", 
            font=('Arial', 12),
            fill="#95a5a6"
        )
        
        # Results panel
        self.results_frame = tk.Frame(
            content_frame, 
            bg="white",
            padx=20,
            pady=20,
            highlightbackground="#ddd",
            highlightthickness=1,
            bd=0
        )
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(
            self.results_frame, 
            text="Analysis Results", 
            font=('Arial', 12, 'bold'), 
            bg="white", 
            fg="#2c3e50"
        ).pack(anchor=tk.W, pady=(0, 20))
        
        # Container for results content
        self.results_content_frame = tk.Frame(self.results_frame, bg="white")
        self.results_content_frame.pack(fill=tk.BOTH, expand=True)
        
        # No results placeholder
        self.no_results_label = tk.Label(
            self.results_content_frame, 
            text="No analysis results yet", 
            font=('Arial', 12), 
            bg="white", 
            fg="#95a5a6"
        )
        self.no_results_label.pack(expand=True)
        
        # Results metrics frame (hidden initially)
        self.metrics_frame = tk.Frame(self.results_content_frame, bg="white")
        
        # Create a parent frame for the two columns
        results_columns = tk.Frame(self.metrics_frame, bg="white")
        results_columns.pack(fill=tk.BOTH, expand=True)
        
        # Left column: Metrics
        metrics_col = tk.Frame(
            results_columns, 
            bg="#eef2f7", 
            padx=15, 
            pady=15,
            highlightbackground="#ccc",
            highlightthickness=1,
            bd=0
        )
        metrics_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Nasal Depth Ratio
        metric_frame1 = tk.Frame(metrics_col, bg="#eef2f7")
        metric_frame1.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            metric_frame1, 
            text="Nasal Depth Ratio:", 
            font=('Arial', 10), 
            bg="#eef2f7", 
            fg="#2c3e50",
            anchor=tk.W
        ).pack(side=tk.LEFT)
        
        self.nasal_depth_value = tk.Label(
            metric_frame1, 
            text="--", 
            font=('Arial', 10, 'bold'), 
            bg="#eef2f7", 
            fg="#2c3e50",
            anchor=tk.E
        )
        self.nasal_depth_value.pack(side=tk.RIGHT)
        
        # Eye-Nose Distance
        metric_frame2 = tk.Frame(metrics_col, bg="#eef2f7")
        metric_frame2.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            metric_frame2, 
            text="Eye-Nose Distance:", 
            font=('Arial', 10), 
            bg="#eef2f7", 
            fg="#2c3e50",
            anchor=tk.W
        ).pack(side=tk.LEFT)
        
        self.eye_nose_value = tk.Label(
            metric_frame2, 
            text="--", 
            font=('Arial', 10, 'bold'), 
            bg="#eef2f7", 
            fg="#2c3e50",
            anchor=tk.E
        )
        self.eye_nose_value.pack(side=tk.RIGHT)
        
        # Nasal Width
        metric_frame3 = tk.Frame(metrics_col, bg="#eef2f7")
        metric_frame3.pack(fill=tk.X)
        
        tk.Label(
            metric_frame3, 
            text="Nasal Width:", 
            font=('Arial', 10), 
            bg="#eef2f7", 
            fg="#2c3e50",
            anchor=tk.W
        ).pack(side=tk.LEFT)
        
        self.nasal_width_value = tk.Label(
            metric_frame3, 
            text="--", 
            font=('Arial', 10, 'bold'), 
            bg="#eef2f7", 
            fg="#2c3e50",
            anchor=tk.E
        )
        self.nasal_width_value.pack(side=tk.RIGHT)
        
        # Right column: Assessment
        assessment_col = tk.Frame(
            results_columns, 
            bg="#eef2f7", 
            padx=15, 
            pady=15,
            highlightbackground="#ccc",
            highlightthickness=1,
            bd=0
        )
        assessment_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
        
        tk.Label(
            assessment_col, 
            text="Assessment:", 
            font=('Arial', 10), 
            bg="#eef2f7", 
            fg="#2c3e50"
        ).pack(anchor=tk.W, pady=(0, 10))
        
        self.status_frame = tk.Frame(
            assessment_col, 
            bg="#d5f5e3", 
            padx=10,
            pady=5,
            bd=0
        )
        self.status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = tk.Label(
            self.status_frame, 
            text="Normal Range", 
            font=('Arial', 10, 'bold'), 
            bg="#d5f5e3", 
            fg="#27ae60"
        )
        self.status_label.pack()
        
        tk.Label(
            assessment_col, 
            text="Reference: 0.65 - 0.85", 
            font=('Arial', 8), 
            bg="#eef2f7", 
            fg="#7f8c8d"
        ).pack(anchor=tk.W)
    
    def update_patient_data(self, field: str, value: str):
        """Update patient data state"""
        self.patient_data[field] = value
    
    # def browse_files(self, event=None):
    #     """Open file dialog to select an image"""
    #     file_path = filedialog.askopenfilename(
    #         initialdir=os.getcwd(),
    #         title="Select Image File",
    #         filetypes=(
    #             ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
    #             ("All files", "*.*")
    #         )
    #     )
        
    #     if file_path:
    #         self.load_image(file_path)
    
    def load_image(self, file_path: str):
        """Load and display the selected image"""
        try:
            # Load the image using PIL
            image = Image.open(file_path)
            
            # Store original image
            self.uploaded_image = image
            
            # Display the image
            self.display_image_with_landmarks()
            
            # Enable the analyze button
            self.analyze_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            print(f"Error loading image: {e}")
    
    def display_image_with_landmarks(self):
        """Display the image with facial landmarks if enabled"""
        if not self.uploaded_image:
            return
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Calculate aspect ratio
        img_width, img_height = self.uploaded_image.size
        aspect_ratio = img_width / img_height
        
        # Calculate dimensions to maintain aspect ratio
        if canvas_width / aspect_ratio <= canvas_height:
            draw_width = canvas_width
            draw_height = int(canvas_width / aspect_ratio)
        else:
            draw_height = canvas_height
            draw_width = int(canvas_height * aspect_ratio)
        
        # Resize the image
        resized_image = self.uploaded_image.resize((draw_width, draw_height), Image.LANCZOS)
        
        # Create a copy for drawing
        draw_image = resized_image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        # Hide placeholder text
        self.canvas.itemconfigure(self.placeholder_text, state='hidden')
        
        # Calculate facial feature positions
        center_x = draw_width // 2
        center_y = draw_height // 2
        face_width = int(draw_width * 0.6)
        face_height = int(draw_height * 0.8)
        
        # Draw facial landmarks if enabled
        if self.show_landmarks:
            # Eye positions
            eye_y = center_y - int(face_height * 0.15)
            eye_distance = int(face_width * 0.3)
            eye_width = int(face_width * 0.08)
            eye_height = int(face_height * 0.03)
            
            # Left eye coordinates
            left_eye_x = center_x - eye_distance // 2
            
            # Right eye coordinates
            right_eye_x = center_x + eye_distance // 2
            
            # Nose landmarks
            nose_top = (center_x, eye_y + int(face_height * 0.05))
            nose_bottom = (center_x, eye_y + int(face_height * 0.25))
            nose_left = (center_x - int(face_width * 0.1), eye_y + int(face_height * 0.15))
            nose_right = (center_x + int(face_width * 0.1), eye_y + int(face_height * 0.15))
            
            # Draw face outline (ellipse)
            for i in range(2):  # Draw slightly thicker line
                bbox = [
                    center_x - face_width // 2, 
                    center_y - face_height // 2,
                    center_x + face_width // 2, 
                    center_y + face_height // 2
                ]
                draw.ellipse(bbox, outline="#95a5a6")
            
            # Draw eyes (ellipses)
            for i in range(2):  # Draw slightly thicker line
                # Left eye
                bbox_left = [
                    left_eye_x - eye_width, 
                    eye_y - eye_height,
                    left_eye_x + eye_width, 
                    eye_y + eye_height
                ]
                draw.ellipse(bbox_left, outline="#3498db")
                
                # Right eye
                bbox_right = [
                    right_eye_x - eye_width, 
                    eye_y - eye_height,
                    right_eye_x + eye_width, 
                    eye_y + eye_height
                ]
                draw.ellipse(bbox_right, outline="#3498db")
            
            # Draw nose lines
            for i in range(2):  # Draw slightly thicker line
                draw.line([nose_top, nose_bottom], fill="#e74c3c")
                draw.line([nose_left, nose_right], fill="#e74c3c")
            
            # Draw nose landmark points
            for point in [nose_top, nose_bottom, nose_left, nose_right]:
                draw.ellipse([point[0]-3, point[1]-3, point[0]+3, point[1]+3], fill="#e74c3c")
            
            # Draw mouth (arc)
            mouth_y = center_y + int(face_height * 0.2)
            mouth_left = (center_x - int(face_width * 0.2), mouth_y)
            mouth_right = (center_x + int(face_width * 0.2), mouth_y)
            mouth_control = (center_x, mouth_y + int(face_height * 0.08))
            
            # Simulate a curve using multiple line segments
            curve_points = []
            steps = 20
            for i in range(steps + 1):
                t = i / steps
                x = (1-t)**2 * mouth_left[0] + 2*(1-t)*t*mouth_control[0] + t**2*mouth_right[0]
                y = (1-t)**2 * mouth_left[1] + 2*(1-t)*t*mouth_control[1] + t**2*mouth_right[1]
                curve_points.append((x, y))
            
            for i in range(len(curve_points) - 1):
                draw.line([curve_points[i], curve_points[i+1]], fill="#95a5a6")
            
            # Draw measurements if enabled
            if self.show_measurements:
                # Draw dashed lines for measurements
                self._draw_dashed_line(draw, nose_left, nose_right, "#27ae60")
                self._draw_dashed_line(draw, nose_top, nose_bottom, "#27ae60")
        
        # Convert to PhotoImage for tkinter
        self.tk_image = ImageTk.PhotoImage(draw_image)
        
        # Clear existing canvas content and display the new image
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2, 
            canvas_height // 2, 
            image=self.tk_image
        )
    
    def _draw_dashed_line(self, draw, start, end, color, dash_length=3, gap_length=3):
        """Draw a dashed line on PIL ImageDraw"""
        # Calculate line length and angle
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = (dx**2 + dy**2) ** 0.5
        
        if length == 0:
            return
        
        # Normalize direction vector
        dx, dy = dx/length, dy/length
        
        # Draw dashes
        pos = 0
        drawing = True
        
        while pos < length:
            segment_length = min(dash_length if drawing else gap_length, length - pos)
            end_pos = pos + segment_length
            
            if drawing:
                draw.line(
                    [
                        (start[0] + dx * pos, start[1] + dy * pos),
                        (start[0] + dx * end_pos, start[1] + dy * end_pos)
                    ],
                    fill=color
                )
            
            pos = end_pos
            drawing = not drawing
    
    def toggle_landmarks(self):
        """Toggle the visibility of facial landmarks"""
        self.show_landmarks = not self.show_landmarks
        
        # Update button text
        if self.show_landmarks:
            self.landmark_btn.config(text="Hide Landmarks")
        else:
            self.landmark_btn.config(text="Show Landmarks")
        
        # Redraw the image
        self.display_image_with_landmarks()
    
    def toggle_measurements(self):
        """Toggle the visibility of measurements"""
        self.show_measurements = not self.show_measurements
        
        # Update button text
        if self.show_measurements:
            self.measurement_btn.config(text="Hide Measurements")
        else:
            self.measurement_btn.config(text="Show Measurements")
        
        # Redraw the image
        self.display_image_with_landmarks()
    
    def analyze_image(self):
        """Analyze the video""" 
        # Set analyzing state
        self.is_analyzing = True
        self.analyze_btn.config(text="Analyzing...", state=tk.DISABLED)
        
        # Run analysis in a separate thread to avoid UI freezing
        threading.Thread(target=self._run_analysis).start()
    
    def _run_analysis(self):
        # Start webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,620)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
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
                 self.analyze_btn.config(state=tk.NORMAL,text="Live Analysis")
                 break

        cap.release()
        cv2.destroyAllWindows()
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

if __name__ == "__main__":
    root = tk.Tk()
    app = NasalCavityDetectionApp(root)
    
    # Update the canvas when the window is resized
    def on_resize(event):
        if hasattr(app, 'uploaded_image') and app.uploaded_image:
            app.display_image_with_landmarks()
    
    root.bind("<Configure>", on_resize)
    
    root.mainloop()