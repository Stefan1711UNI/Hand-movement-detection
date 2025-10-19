# pip install mediapipe opencv-python numpy
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Video Capture
import cv2
import numpy as np
import time

import threading

import mediapipe as mp
from types import SimpleNamespace

#Kalman filter module
#from kalman_filter import Kalman3D

model_path = "other/hand_landmarker.task"

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
#kf_lock = threading.Lock()

#Initialize Kalman filter 
#kf = Kalman3D(initial_time=None, q=0.02)


#--------------DRAWING HELPERS-----------------------

mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
MP_HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS


def to_landmark_list(hand_landmarks):
    """
    Accepts either:
      - an object with .landmark (NormalizedLandmarkList)
      - or a plain Python list of landmarks
    Returns a list-like sequence of landmarks where each landmark has .x and .y
    """
    if hand_landmarks is None:
        return None
    # If it's already a list (or tuple), return as-is
    if isinstance(hand_landmarks, (list, tuple)):
        return hand_landmarks
    # If it has .landmark attribute, return that
    if hasattr(hand_landmarks, "landmark"):
        return hand_landmarks.landmark
    # Fallback: try to treat it as iterable
    try:
        return list(hand_landmarks)
    except Exception:
        return None

# Drawing helper that works for both shapes
def draw_landmarks_adaptive(frame, landmark_list, image_w, image_h, connections):
    """
    landmark_list: list of landmarks where each has .x and .y
    connections: e.g., mp.solutions.hands.HAND_CONNECTIONS
    """
    if landmark_list is None:
        return
    # Draw points
    for i, lm in enumerate(landmark_list):
        try:
            cx = int(lm.x * image_w)
            cy = int(lm.y * image_h)
        except Exception:
            # If landmark uses different attribute names, skip
            continue
        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

    # Draw connections (lines) using the index pairs in connections
    # connections are a list of tuples like (start_idx, end_idx)
    for conn in connections:
        start_idx = conn[0]
        end_idx = conn[1]
        # bounds-check indices
        if start_idx < len(landmark_list) and end_idx < len(landmark_list):
            s = landmark_list[start_idx]
            e = landmark_list[end_idx]
            try:
                sx = int(s.x * image_w); sy = int(s.y * image_h)
                ex = int(e.x * image_w); ey = int(e.y * image_h)
                cv2.line(frame, (sx, sy), (ex, ey), (0, 200, 200), 2)
            except Exception:
                continue


def print_data(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    try:
        # safe extraction of wrist world landmark (index 0)
        # if result has attribute then world_con.. becomes result, but if it is empty/or not have then it becoms None instead of rasing an AttributeError
        world_container = getattr(result, "hand_world_landmarks", None)

        # if none then show "N/A"
        if not world_container:
                #print(f"{timestamp_ms}: wrist_world: N/A")
                return

        # gets the first hand detected. 1 represents second hand if it is tracked
        first_hand = world_container[0]

        # Normalize to a plain list of landmarks
        if hasattr(first_hand, "landmark"):
            #print("First case")
            lm_list = first_hand.landmark
        elif isinstance(first_hand, (list, tuple)):
            #print("Second case")
            lm_list = first_hand
            #print(lm_list)
        else:
            # Try converting to list as a last resort
            try:
                #print("third case")
                lm_list = list(first_hand)
            except Exception:
                #print(f"{timestamp_ms}: wrist_world: N/A")
                return

        #PUTS THE RESULTS IN SHARED  
        with shared_lock:
            shared["result"] = result
            shared["timestamp_ms"] = timestamp_ms
        

        # Wrist is index 0 in MediaPipe hand model
        if len(lm_list) > 0:
            w = lm_list[0]

            #Raw measurements 
            raw_x, raw_y, raw_z = float(w.x), float(w.y), float(w.z)

            #Print values to terminal
           # print(f"{timestamp_ms}: wrist_world: raw=({raw_x:.6f},{raw_y:.6f},{raw_z:.6f})")
            
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


def generate_frames():
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

        #Draw hand overlay
        try:
            frame_height, frame_width = frame.shape[:2]

            if result is not None and result.hand_landmarks:
                # Get the first detected hand (adaptive conversion)
                raw_hand = result.hand_landmarks[0]            # may be list or object-with-.landmark
                lm_list = to_landmark_list(raw_hand)           # now a plain list

                if lm_list:
                    # draw landmarks & connections robustly
                    draw_landmarks_adaptive(frame, lm_list, frame_width, frame_height, mp.solutions.hands.HAND_CONNECTIONS)

        except Exception:
            print("Overlay printing failed, skipping this frame")

        cv2.circle(frame, (10,10), 5, (0,255,0), -1)

         # Encode the frame as JPEG and yield for MJPEG stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # yield as multipart/x-mixed-replace
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

