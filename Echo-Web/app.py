# Import necessary modules
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import json
import logging
import time

from video_feed_handler import generate_frames

# Create a Flask app instance
app = Flask(__name__, static_url_path='/static')


# Route to render the HTML template
@app.route('/')
def index():
    return render_template('index.html')


# Route to stream video frames
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')