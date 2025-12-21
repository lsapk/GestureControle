import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import json
import threading
from gui import launch_gui
from config import DEFAULT_CONFIG

# Create a HandLandmarker object.
class KalmanFilter:
    def __init__(self, dt=0.1, u=0, std_acc=1, std_meas=0.1):
        self.dt = dt
        self.u = u
        self.A = np.array([[1, self.dt], [0, 1]])
        self.B = np.array([[(self.dt**2)/2], [self.dt]])
        self.H = np.array([[1, 0]])
        self.Q = np.array([[(self.dt**4)/4, (self.dt**3)/2], [(self.dt**3)/2, self.dt**2]]) * std_acc**2
        self.R = np.array([[std_meas**2]])
        self.P = np.zeros((2, 2))
        self.x = np.zeros((2, 1))

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        self.P = self.P - np.dot(K, np.dot(S, K.T))
        return self.x

def detect_gesture(landmarks, config, lock):
    with lock:
        click_threshold = config['CLICK_THRESHOLD']
        fist_threshold = config['FIST_THRESHOLD']

    if not landmarks:
        return 'no_hand'

    thumb_tip = landmarks[4]
    index_finger_tip = landmarks[8]
    middle_finger_tip = landmarks[12]
    ring_finger_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]

    # Click gesture
    click_distance = np.sqrt((thumb_tip.x - index_finger_tip.x)**2 + (thumb_tip.y - index_finger_tip.y)**2)
    if click_distance < click_threshold:
        return 'click'

    # Fist gesture
    fist_distance = (np.sqrt((middle_finger_tip.x - wrist.x)**2 + (middle_finger_tip.y - wrist.y)**2) +
                     np.sqrt((ring_finger_tip.x - wrist.x)**2 + (ring_finger_tip.y - wrist.y)**2) +
                     np.sqrt((pinky_tip.x - wrist.x)**2 + (pinky_tip.y - wrist.y)**2)) / 3
    if fist_distance < fist_threshold:
        return 'fist'

    return 'open_palm'

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def load_config(lock):
    with lock:
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            config = DEFAULT_CONFIG
        return config

# Gesture variables
config_lock = threading.Lock()
config = load_config(config_lock)
last_click_time = 0
CLICK_COOLDOWN = 0.5
is_dragging = False

# Motion smoothing variables
frame_timestamp_ms = 0
kf_x = KalmanFilter(dt=0.1, std_acc=1, std_meas=0.1)
kf_y = KalmanFilter(dt=0.1, std_acc=1, std_meas=0.1)

# Launch the GUI in a separate thread
gui_thread = threading.Thread(target=launch_gui, args=(config, config_lock))
gui_thread.daemon = True
gui_thread.start()

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Convert the image to a mediapipe Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Detect hand landmarks from the input image.
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # Draw the hand annotations on the image.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gesture = 'no_hand'
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            # Get the coordinates of key landmarks
            index_finger_tip = hand_landmarks[8] # INDEX_FINGER_TIP

            with config_lock:
                sensitivity = config['SENSITIVITY']
                motion_scale = config['MOTION_SCALE']

            # Convert normalized coordinates to screen coordinates
            effective_scale = sensitivity * motion_scale
            x = int((index_finger_tip.x - 0.5) * effective_scale * screen_width + screen_width / 2)
            y = int((index_finger_tip.y - 0.5) * effective_scale * screen_height + screen_height / 2)

            # Clamp coordinates to screen boundaries
            x = np.clip(x, 1, screen_width - 1)
            y = np.clip(y, 1, screen_height - 1)

            # Apply Kalman Filter for smoothing
            kf_x.predict()
            smoothed_x = int(kf_x.update(x)[0, 0])

            kf_y.predict()
            smoothed_y = int(kf_y.update(y)[0, 0])

            # Move the mouse cursor to the smoothed position
            pyautogui.moveTo(smoothed_x, smoothed_y)

            # Gesture recognition
            gesture = detect_gesture(hand_landmarks, config, config_lock)
            current_time = time.time()

            if gesture == 'click' and (current_time - last_click_time) > CLICK_COOLDOWN:
                pyautogui.click()
                last_click_time = current_time
            elif gesture == 'fist':
                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
            elif gesture == 'open_palm':
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
        else:
            # Reset dragging state if no hand is detected
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False


        # Display gesture guide
        cv2.putText(image, f"Gesture: {gesture}", (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "Gestures: click, fist, open_palm", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('GestureControl OS', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
