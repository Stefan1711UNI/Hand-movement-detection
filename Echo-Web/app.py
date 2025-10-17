# Import necessary modules
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import json
import logging
import time

from video_feed_handler import generate_frames1

# Create a Flask app instance
app = Flask(__name__, static_url_path='/static')

# Set to keep track of RTCPeerConnection instances
pcs = set()

# Function to generate video frames from the camera
def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        start_time = time.time()
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Concatenate frame and yield for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
            elapsed_time = time.time() - start_time
            logging.debug(f"Frame generation time: {elapsed_time} seconds")

# Route to render the HTML template
@app.route('/')
def index():
    return render_template('index.html')


# Route to stream video frames
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')