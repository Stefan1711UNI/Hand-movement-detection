import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Video Capture
import cv2
import numpy as np
import time

#Imports for CSV, thread-safety and filename timestamping
import csv
import threading
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo  
except Exception:
    ZoneInfo = None
import os


#Kalman filter module
from kalman_filter_test import Kalman3D

model_path = "hand_landmarker.task"

#Options for task
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Shared container for the latest HandLandmarkerResult
shared = {
    "result": None,       
    "timestamp_ms": None
}
shared_lock = threading.Lock()

#Kalman filter globals 
kf = None
kf_lock = threading.Lock()

#Initialize Kalman filter 
kf = Kalman3D(initial_time=None, q=0.02)

#print_data()

# Create a hand landmarker instance with the live stream mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_data)

# Initialize the hand landmarker
landmarker = HandLandmarker.create_from_options(options)

frame_timestamp_ms = 0

#RUn 
try:
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        #Flip horizontally (mirror for user)
        frame = cv2.flip(frame, 1)
        # Convert the frame to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # Get the current timestamp in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)
        # Perform hand landmark detection asynchronously
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        #Get last hand landmark detection
        with shared_lock:
                result = shared.get("result", None)
        # Show the frame
        cv2.imshow("Hand Tracking (press q to quit)", frame)
        # handle keys: 'q' quit 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release the video capture object and close all OpenCV windows
    camera.release()
    # cv2.destroyAllWindows()
    landmarker.close()
    print("Video capture released and landmarker closed.")