# pip install mediapipe opencv-python numpy
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import numpy as np
import time
from collections import deque
import threading

# ========== USER TUNABLES ==========
model_path = "hand_landmarker.task"
video_source = 0
target_fps = 15            # visualization refresh (set lower to "slow down" display)
trail_length = 50         # how many previous positions to keep for the trail
smoothing_alpha = 0.6     # EMA smoothing (0 = full smoothing, 1 = no smoothing)
# ===================================

# Setup video capture
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolution: {frame_width}x{frame_height}")

# MediaPipe Tasks options (same pattern as your code)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Shared container for callback -> main thread communication
shared = {
    "result": None,       # will hold the latest HandLandmarkerResult
    "timestamp_ms": None
}
shared_lock = threading.Lock()

# Callback called by MediaPipe when an async result is ready
def print_and_store_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # You previously printed the raw result; we still keep that optional:
    # print('hand landmarker result:', result)
    # Store into shared container (thread-safe)
    with shared_lock:
        shared["result"] = result
        shared["timestamp_ms"] = timestamp_ms

# Create hand landmarker (LIVE_STREAM) with callback
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_and_store_result
)
landmarker = HandLandmarker.create_from_options(options)

# Helper: normalize landmark container to a plain list of landmarks
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


# helper: compute palm center in image pixels from 2D normalized landmarks
def palm_center_pixel(hand_landmarks, w, h):
    # average some palm/wrist landmarks for more stable palm center:
    idxs = [0, 5, 9, 13, 17]  # wrist + MCP joints
    xs = [hand_landmarks.landmark[i].x for i in idxs]
    ys = [hand_landmarks.landmark[i].y for i in idxs]
    norm_x = float(np.mean(xs))
    norm_y = float(np.mean(ys))
    px = int(norm_x * w)
    py = int(norm_y * h)
    return px, py

# Data for drawing/motion trail
trail = deque(maxlen=trail_length)
smoothed_pos = None
prev_time = None

# For drawing hand landmarks we can use the drawing utils from mediapipe.solutions
mp_draw = mp.solutions.drawing_utils
mp_hands_conns = mp.solutions.hands.HAND_CONNECTIONS

print("Starting visualization. Press 'q' to quit, 'p' to pause/resume display.")
paused = False

try:
    while cap.isOpened():
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Optionally pause display (detector still runs because detect_async uses callback)
        # Press 'p' to toggle - handled later via waitKey
        if not paused:
            # Convert to RGB and wrap into mp.Image as required by Tasks API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # send frame to detector asynchronously (timestamp in ms)
            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            # Read the latest result (if any)
            with shared_lock:
                result = shared.get("result", None)

            #add here
            if result is not None and result.hand_landmarks:
                # Get the first detected hand (adaptive conversion)
                raw_hand = result.hand_landmarks[0]            # may be list or object-with-.landmark
                lm_list = to_landmark_list(raw_hand)           # now a plain list

                if lm_list:
                    # draw landmarks & connections robustly
                    draw_landmarks_adaptive(frame, lm_list, frame_width, frame_height, mp.solutions.hands.HAND_CONNECTIONS)

                    # compute palm center using indices safely
                    idxs = [0, 5, 9, 13, 17]
                    xs = []
                    ys = []

                    for i in idxs:
                        if i < len(lm_list):
                            xs.append(lm_list[i].x)
                            ys.append(lm_list[i].y)
                    if xs and ys:
                        norm_x = float(np.mean(xs))
                        norm_y = float(np.mean(ys))
                        px = int(norm_x * frame_width)
                        py = int(norm_y * frame_height)

                        # smoothing as before...
                        cur_pos = np.array([px, py], dtype=float)

                        if smoothed_pos is None:
                            smoothed_pos = cur_pos.copy()
                        else:
                            smoothed_pos = smoothing_alpha * cur_pos + (1 - smoothing_alpha) * smoothed_pos

                # compute velocity (pixels/sec)
                now = time.time()
                if prev_time is None:
                    vel = np.array([0.0, 0.0])
                else:
                    dt = now - prev_time
                    vel = (smoothed_pos - prev_smoothed_pos) / dt if dt > 1e-6 else np.array([0.0, 0.0])

                prev_smoothed_pos = smoothed_pos.copy()
                prev_time = now

                # push to trail (use integer pixels for drawing)
                trail.appendleft((int(smoothed_pos[0]), int(smoothed_pos[1])))

                # Draw palm center and velocity arrow
                cv2.circle(frame, (int(smoothed_pos[0]), int(smoothed_pos[1])), 6, (0, 255, 0), -1)
                # Draw arrow scaled down for visibility
                arrow_tip = (int(smoothed_pos[0] + vel[0] * 0.02), int(smoothed_pos[1] + vel[1] * 0.02))
                cv2.arrowedLine(frame, (int(smoothed_pos[0]), int(smoothed_pos[1])), arrow_tip, (255, 0, 0), 2, tipLength=0.3)

                # Draw trail (older points are lighter)
                for i, (tx, ty) in enumerate(trail):
                    alpha = 1.0 - (i / len(trail)) if len(trail) > 0 else 1.0
                    radius = max(1, 4 - int(i / max(1, len(trail) // 10)))
                    color = (0, int(255 * alpha), int(255 * alpha * 0.4))
                    cv2.circle(frame, (tx, ty), radius, color, -1)

                # Overlay some numeric info
                cv2.putText(frame, f"pos: ({int(smoothed_pos[0])},{int(smoothed_pos[1])})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"vel: ({vel[0]:.1f},{vel[1]:.1f}) px/s", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                # No hand detected: optionally decay trail slowly
                if len(trail) > 0:
                    # pop oldest so trail fades even if loss of detection
                    trail.pop()

        # Show the frame
        cv2.imshow("Hand Tracking (press q to quit)", frame)

        # handle keys: 'q' quit, 'p' pause/resume display
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            paused = not paused

        # Simple throttling to target_fps for visualization (do not block detection)
        elapsed = time.time() - loop_start
        min_frame_time = 1.0 / max(1, target_fps)
        if elapsed < min_frame_time:
            time.sleep(min_frame_time - elapsed)

except Exception as e:
    print("Exception:", e)

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Closed.")
