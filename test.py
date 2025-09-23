import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Video Capture
import cv2
import numpy as np
import time

model_path = "hand_landmarker.task"

# 0 means webcam
video_source = 0

cap = cv2.VideoCapture(video_source)

# Check if the video source opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Video source opened: {video_source}")
print(f"Resolution: {frame_width}x{frame_height}")
print(f"FPS: {fps}")

#Options for task
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

# Create a hand landmarker instance with the live stream mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# Initialize the hand landmarker
landmarker = HandLandmarker.create_from_options(options)

frame_timestamp_ms = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Get the current timestamp in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)

        # Perform hand landmark detection asynchronously
        landmarker.detect_async(mp_image, frame_timestamp_ms)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release the video capture object and close all OpenCV windows
    cap.release()
    # cv2.destroyAllWindows()
    landmarker.close()
    print("Video capture released and landmarker closed.")