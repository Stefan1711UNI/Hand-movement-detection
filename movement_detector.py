import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Video Capture
import cv2
import numpy as np
import time

#Mmports for CSV, thread-safety and filename timestamping
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

#CSV helpers
csv_file = None
csv_writer = None
csv_lock = threading.Lock()

#Kalman filter globals 
kf = None
kf_lock = threading.Lock()

#Returns a timestamp string.
def make_timestamp_string(use_local=True, fmt="%Y-%m-%d_%H-%M-%S"):

    if use_local and ZoneInfo is not None:
        try:
            now = datetime.now(tz=ZoneInfo("Europe/Amsterdam"))
            return now.strftime(fmt)
        
        except Exception:
            pass
    # fallback to UTC
    return datetime.now(tz=timezone.utc).strftime(fmt)


#Create an output CSV with a timestamped filename.
def init_csv(base_name="hand_log", out_dir=".", append=False,
             use_local_time=True, timestamp_format="%Y-%m-%d_%H-%M-%S"):

    global csv_file, csv_writer
    os.makedirs(out_dir, exist_ok=True)

    ts = make_timestamp_string(use_local=use_local_time, fmt=timestamp_format)
    filename = f"{base_name}_{ts}.csv"
    path = os.path.join(out_dir, filename)

    #If append=True and file exists, open in append mode; otherwise create new file
    mode = "a" if append and os.path.exists(path) else "w"
    first_write = not (append and os.path.exists(path))

    csv_file = open(path, mode, newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    if first_write:
        csv_writer.writerow([
            "timestamp_ms",
            "timestamp_iso_utc",
            "landmark",
            "x_m",
            "y_m",
            "z_m"
        ])
        csv_file.flush()
    print(f"CSV logging -> {path}")
    return path

#Writes a single row to CSV
def write_row_csv(timestamp_ms, landmark, x, y, z, do_fsync=True):
    global csv_file, csv_writer, csv_lock
    if csv_writer is None:
        return
    dt_utc = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
    iso_utc = dt_utc.isoformat()
    
    row = [timestamp_ms, iso_utc, landmark, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]
    with csv_lock:
        csv_writer.writerow(row)
        csv_file.flush()
        if do_fsync:
            try:
                os.fsync(csv_file.fileno())
            except Exception:
                pass

    
def close_csv():
    global csv_file, csv_writer
    if csv_file:
        try:
            csv_file.close()
        except Exception:
            pass
    csv_file = None
    csv_writer = None


# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

#Updated to include kalman filter
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
            #Log raw measurements
            write_row_csv(timestamp_ms, "wrist_world", raw_x, raw_y, raw_z)

            #Kalman filter Update ------------------

            #timestamp_ms is converted to seconds for filter
            ts_s = timestamp_ms / 1000.0
            filtered_accepted = True
            mahal = 0.0
            if kf is not None:
                with kf_lock:
                    # run predict+update on the filter
                    filtered_accepted, mahal = kf.step(ts_s, (raw_x, raw_y, raw_z), gating_threshold=16.0)
                    pos_f, vel_f = kf.get_state()
            else:
                # If kf not yet initialized, just copy raw -> filtered
                pos_f = np.array([raw_x, raw_y, raw_z], dtype=float)
                vel_f = np.array([0.0, 0.0, 0.0], dtype=float)

            #Logs filtered measurements 
            write_row_csv(timestamp_ms, "wrist_world_filtered", float(pos_f[0]), float(pos_f[1]), float(pos_f[2]))

            #Print values to terminal
            print(f"{timestamp_ms}: wrist_world: raw=({raw_x:.6f},{raw_y:.6f},{raw_z:.6f}) "
                  f"filtered=({pos_f[0]:.6f},{pos_f[1]:.6f},{pos_f[2]:.6f}) mahal={mahal:.2f}")
            
        else:
            print(f"{timestamp_ms}: wrist_world: N/A")
    
    # Print error
    except Exception as e:
        print(f"{timestamp_ms}: wrist_world: error: {e}")


#Initialize CSV
csv_path = init_csv(base_name="hand_log", out_dir="./logs", append=False, use_local_time=True)

#Initialize Kalman filter ---------------
kf = Kalman3D(initial_time=None, q=0.02)

# Create a hand landmarker instance with the live stream mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_data)

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
    cap.release()
    # cv2.destroyAllWindows()
    landmarker.close()
     # Close CSV file safely
    close_csv()
    print("Video capture released and landmarker closed.")