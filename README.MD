# Echo Arm - Hand Movement Detection Project

This project uses MediaPipe's Hand Landmarker model to detect user hand movements via a webcam. A web-based UI displays the camera feed with landmark overlays and provides real-time wrist position data. This part focuses on the computer vision aspect for controlling a future robot arm.

## Prerequisites

* **Python 3.x:** Make sure you have Python installed. You can download it from [python.org](https://www.python.org/).
* **Webcam:** A webcam is required to capture hand movements.
* **Git:** Needed to clone the repository.

## Setup Instructions

1.  **Clone the Repository:**
    Open your terminal or command prompt and run:
    ```bash
    git clone https://github.com/Stefan1711UNI/Hand-movement-detection.git
    cd hand-movement-detection
    ```

2.  **Set up a Virtual Environment (Recommended):**
    This isolates project dependencies.
    ```bash
    # Create the virtual environment (named 'venv' here)
    python -m venv venv

    # Activate it:
    # On Windows (cmd.exe):
    venv\Scripts\activate.bat
    # On Windows (PowerShell):
    venv\Scripts\Activate.ps1
    # On macOS/Linux:
    source venv/bin/activate
    ```
    You should see `(venv)` at the beginning of your terminal prompt.

3.  **Install Dependencies:**
    Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Start the Flask Web Server:**
    Navigate to the project's root directory in your terminal (if you aren't already there) and run:
    ```bash
    python Echo-Web/app.py
    ```

2.  **Access the Web UI:**
    Open your web browser and go to:
    * `http://127.0.0.1:5000/` (or `http://localhost:5000/`) to see the main camera feed.
    * `http://127.0.0.1:5000/demo` (or `http://localhost:5000/demo`) to see the live wrist visualization page.

3.  **Stopping the Application:**
    Press `Ctrl + C` in the terminal where the server is running.

## Project Notes

* The MediaPipe hand landmark model (`other/hand_landmarker.task`) is included in the repository.
* The application uses Flask to serve the web UI and video stream.
* `video_feed_handler.py` contains the main logic for capturing video, running the MediaPipe model, applying a Kalman filter for smoothing, and drawing landmarks.
* Make sure your webcam is connected and accessible by the system.