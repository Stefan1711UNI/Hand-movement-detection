# Import necessary modules
from flask import Flask, render_template, Response, request, jsonify

from video_feed_handler import generate_frames, pose_lock, latest_pose


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

@app.route('/demo')
def movement_demo():
    return render_template('movement.html')


@app.route('/pose')
def pose():
    # return latest pose as JSON
    with pose_lock:
        # make a shallow copy to avoid races while flask serializes
        copy = {
            "t": latest_pose["t"],
            "landmark": latest_pose["landmark"],
            "pos": latest_pose["pos"],
            "vel": latest_pose["vel"],
            "valid": latest_pose["valid"]
        }

    return jsonify(copy)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')