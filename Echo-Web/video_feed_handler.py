# pip install mediapipe opencv-python numpy
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Video Capture
import cv2
import numpy as np
import time

#Kalman filter module
#from kalman_filter import Kalman3D

model_path = "other/hand_landmarker.task"

#Options for task
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


#Kalman filter globals 
kf = None
#kf_lock = threading.Lock()

#Initialize Kalman filter 
#kf = Kalman3D(initial_time=None, q=0.02)

def print_data(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    try:
        # safe extraction of wrist world landmark (index 0)
        # if result has attribute then world_con.. becomes result, but if it is empty/or not have then it becoms None instead of rasing an AttributeError
        world_container = getattr(result, "hand_world_landmarks", None)

        # if none then show "N/A"
        if not world_container:
                print(f"{timestamp_ms}: wrist_world: N/A")
                return

        # gets the first hand detected. 1 represents second hand if it is tracked
        first_hand = world_container[0]

        # Normalize to a plain list of landmarks
        if hasattr(first_hand, "landmark"):
            lm_list = first_hand.landmark
        elif isinstance(first_hand, (list, tuple)):
            lm_list = first_hand
        else:
            # Try converting to list as a last resort
            try:
                lm_list = list(first_hand)
            except Exception:
                print(f"{timestamp_ms}: wrist_world: N/A")
                return

        # Wrist is index 0 in MediaPipe hand model
        if len(lm_list) > 0:
            w = lm_list[0]

            #Raw measurements 
            raw_x, raw_y, raw_z = float(w.x), float(w.y), float(w.z)

            #Print values to terminal
            print(f"{timestamp_ms}: wrist_world: raw=({raw_x:.6f},{raw_y:.6f},{raw_z:.6f})")
            
        else:
            print(f"{timestamp_ms}: wrist_world: N/A")
    
    # Print error
    except Exception as e:
        print(f"{timestamp_ms}: wrist_world: error: {e}")

# Create a hand landmarker instance with the live stream mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_data)

# Initialize the hand landmarker
landmarker = HandLandmarker.create_from_options(options)

frame_timestamp_ms = 0


def generate_frames1():
    camera = cv2.VideoCapture(0)
    seq = 0

    while True:
        success, frame = camera.read()

        if not success:
            break

        # Convert the frame to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # Get the current timestamp in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)
        # Perform hand landmark detection asynchronously
        landmarker.detect_async(mp_image, frame_timestamp_ms)


         # Encode the frame as JPEG and yield for MJPEG stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # yield as multipart/x-mixed-replace
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

