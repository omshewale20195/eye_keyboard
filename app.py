import cv2
import time
import pyttsx3
import logging
from flask import Flask, render_template, Response, jsonify, request
from gaze_tracking import GazeTracking
from collections import deque
import numpy as np

app = Flask(__name__)

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the GazeTracking object and webcam
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    logging.error("Error: Unable to access the webcam. Please check your camera connection.")
    exit(1)

# Initialize pyttsx3 for text-to-speech
def initialize_tts():
    try:
        engine = pyttsx3.init()
        return engine
    except Exception as e:
        logging.error(f"Error initializing pyttsx3: {e}")
        return None

engine = initialize_tts()

# Function to convert text to speech
def speak_text(text):
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logging.error(f"Error during text-to-speech conversion: {e}")
    else:
        logging.warning("TTS engine not initialized.")

# Store gaze state and buffer for smoothing
current_gaze_state = ""
gaze_buffer = deque(maxlen=5)  # Buffer for smoothing gaze transitions

# Function to clean up resources
def release_resources():
    if webcam.isOpened():
        webcam.release()
    cv2.destroyAllWindows()

# Function to get gaze state with smoothing
def get_gaze_state():
    global current_gaze_state
    text = ""
    state = ""

    try:
        if gaze.is_blinking():
            # print("blinking iiiiii")
            state = "blinking"
            text = "Blinking"
        elif gaze.is_right() and gaze.pupil_right_coords():
            state = "right"
            text = "Looking right"
        elif gaze.is_left() and gaze.pupil_left_coords():
            state = "left"
            text = "Looking left"
        elif gaze.is_center():
            state = "center"
            text = "Looking center"

        # Add current state to buffer for smoothing
        if state:
            gaze_buffer.append(state)
            # Use majority vote for a stable state
            stable_state = max(set(gaze_buffer), key=gaze_buffer.count)
            if stable_state != current_gaze_state:
                current_gaze_state = stable_state
    except Exception as e:
        logging.error(f"Error determining gaze state: {e}")

    return text

# Function to generate webcam frames with gaze tracking
def generate_frames():
    while True:
        try:
            ret, frame = webcam.read()
            if not ret:
                logging.warning("No frame captured from webcam.")
                continue

            gaze.refresh(frame)
            frame = gaze.annotated_frame()

            # Update and display gaze state
            text = get_gaze_state()
            cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

            # Display pupil coordinates
            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()
            cv2.putText(frame, f"Left pupil: {left_pupil}", (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, f"Right pupil: {right_pupil}", (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        except Exception as e:
            logging.error(f"Error generating frames: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/eye_tracking')
def eye_tracking():
    return render_template('eye_tracking.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gaze_state')
def gaze_state():
    return jsonify({'gaze_state': current_gaze_state})

@app.route('/speak_text', methods=['POST'])
def speak():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if text:
            speak_text(text)  # Convert the text to speech
            return jsonify({"status": "success", "message": "Text is being spoken"})
        return jsonify({"status": "failure", "message": "No text provided"}), 400
    except Exception as e:
        logging.error(f"Error in /speak_text endpoint: {e}")
        return jsonify({"status": "failure", "message": "Server error"}), 500
    


@app.route('/status')
def status():
    webcam_status = "active" if webcam.isOpened() else "inactive"
    tts_status = "initialized" if engine else "unavailable"
    return jsonify({"webcam_status": webcam_status, "tts_status": tts_status})




if __name__ == '__main__':
    try:
        app.run(debug=True)
    except KeyboardInterrupt:
        logging.info("Shutting down the application...")
    finally:
        release_resources()
