from flask import Flask, render_template, Response
import cv2
import cvzone
import math
import pyttsx3
import numpy as np
import time
import os
from ultralytics import YOLO

# Initialize the Flask app
app = Flask(__name__)

# Load the YOLO model
model = YOLO('../Yolo-Weights/yolov8n.pt')

# Text-to-speech engine initialization
engine = pyttsx3.init()

# Class names for YOLO
classNames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
              'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
              'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
              'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
              'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
              'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
              'teddy bear', 'hair drier', 'toothbrush']

# Function to apply CLAHE
def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    return cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

# Function to apply gamma correction
def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Function to estimate distance
def estimate_distance(x1, y1, x2, y2, known_object_height=1.7):
    object_pixel_height = y2 - y1
    focal_length = 700
    if object_pixel_height > 0:
        distance = (known_object_height * focal_length) / object_pixel_height
        return round(distance, 2)
    return None

# Logging detected objects for post-analysis
def log_detection(cls, conf, x1, y1, x2, y2):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    file_exists = os.path.isfile('detections_log.csv')
    
    try:
        with open('detections_log.csv', 'a') as log_file:
            if not file_exists:
                log_file.write('Class,Confidence,X1,Y1,X2,Y2,Timestamp\n')  # Write header
            log_file.write(f'{classNames[cls]},{conf},{x1},{y1},{x2},{y2},{timestamp}\n')
    except Exception as e:
        print(f"Error logging detection: {e}")

# Function to check if the object is too close
def is_too_close(distance, threshold=2):
    return distance and distance < threshold

# Function to trigger voice alert
def trigger_audio_alert(message):
    engine.say(message)
    engine.runAndWait()

# Generalized function to process video frames
def gen_frames(video_source):  
    cap = cv2.VideoCapture(video_source)  

    while True:
        success, img = cap.read()
        if not success:
            break

        # Apply CLAHE and gamma correction
        img_clahe = apply_clahe(img)
        img_gamma_corrected = adjust_gamma(img_clahe, gamma=1.5)

        # Run YOLO detection on the image
        results = model(img_gamma_corrected, stream=True)
        
        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                
                # Confidence and class
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                # Estimate distance
                distance = estimate_distance(x1, y1, x2, y2)

                # Check if the object is too close
                if classNames[cls] in ['person', 'car', 'bicycle'] and conf > 0.5:
                    if is_too_close(distance):
                        trigger_audio_alert(f'{classNames[cls]} is too close! Distance: {distance} meters')
                        log_detection(cls, conf, x1, y1, x2, y2)  # Log the detection

                # Display class, confidence, and distance on the frame
                cvzone.putTextRect(img, f'{classNames[cls]} {conf} Dist: {distance}m', (max(0, x1), max(35, y1)), scale=1, thickness=2)

        # Encode the frame to a format suitable for the browser
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask route to render the web page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for front video streaming
@app.route('/front_video_feed')
def front_video_feed():
    return Response(gen_frames("D:\\object_detection_web\\videos\\bikes.mp4"), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route for back video streaming
@app.route('/back_video_feed')
def back_video_feed():
    return Response(gen_frames("D:\\object_detection_web\\videos\\video-1.mp4"), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
