# Import necessary modules
from flask import Flask, render_template, Response, jsonify

from video_feed_handler import generate_frames, pose_lock
    
from robot_controll import robot_state

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


@app.route("/emergency_stop")
def eme_stop():
    pass

@app.route('/pose')
def pose():
    # return latest pose as JSON
    with pose_lock:
        # make a shallow copy to avoid races while flask serializes
        copy = {
            "robot": robot_state
        }

    return jsonify(copy)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')