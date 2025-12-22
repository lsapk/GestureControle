import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import json
import threading
import queue
from gui import launch_gui
from config import DEFAULT_CONFIG

class KalmanFilter:
    # ... (KalmanFilter class implementation remains the same)
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

class VideoCaptureThread(threading.Thread):
    # ... (VideoCaptureThread class implementation remains the same)
    def __init__(self, camera_index=0, max_retries=10, queue_size=5):
        super().__init__()
        self.daemon = True
        self.camera_index = camera_index
        self.max_retries = max_retries
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.cap = None

    def connect_camera(self):
        for i in range(self.max_retries):
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                print("Camera connected successfully.")
                return True
            print(f"Failed to connect camera. Retry {i+1}/{self.max_retries}")
            time.sleep(1)
        return False

    def run(self):
        if not self.connect_camera():
            print("Could not open video source. Stopping thread.")
            return

        failure_count = 0
        while not self.stopped:
            if not self.cap.isOpened():
                print("Camera disconnected. Attempting to reconnect...")
                if not self.connect_camera():
                    self.stop()
                    continue
                failure_count = 0

            success, frame = self.cap.read()
            if success:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                failure_count = 0
            else:
                failure_count += 1
                if failure_count > self.max_retries:
                    print("Too many consecutive frame read failures. Re-initializing camera.")
                    self.cap.release()
                    self.connect_camera()
                    failure_count = 0

        if self.cap:
            self.cap.release()
        print("Video capture thread stopped.")

    def stop(self):
        self.stopped = True


class GestureController:
    def __init__(self):
        self.config_lock = threading.Lock()
        self.config = self.load_config()

        # Kalman filters
        self.kf_x = KalmanFilter(dt=0.1, std_acc=1, std_meas=0.1)
        self.kf_y = KalmanFilter(dt=0.1, std_acc=1, std_meas=0.1)
        self.screen_width, self.screen_height = pyautogui.size()

        # Video capture thread
        self.video_thread = VideoCaptureThread()

        # Gesture state variables for hysteresis
        self.is_dragging = False
        self.pinch_start_time = None
        self.is_pinching = False
        self.fist_start_time = None
        self.is_fist_closed = False
        self.PINCH_CONFIRM_TIME = 0.1  # 100ms
        self.FIST_CONFIRM_TIME = 0.2   # 200ms


    def load_config(self):
        with self.config_lock:
            try:
                with open("config.json", "r") as f:
                    config = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                config = DEFAULT_CONFIG
            return config

    def detect_gesture(self, landmarks):
        with self.config_lock:
            click_threshold = self.config['CLICK_THRESHOLD']
            fist_threshold = self.config['FIST_THRESHOLD']

        if not landmarks:
            return 'no_hand'

        thumb_tip = landmarks[4]
        index_finger_tip = landmarks[8]
        middle_finger_tip = landmarks[12]
        ring_finger_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]

        click_distance = np.sqrt((thumb_tip.x - index_finger_tip.x)**2 + (thumb_tip.y - index_finger_tip.y)**2)
        if click_distance < click_threshold:
            return 'pinch'

        fist_distance = (np.sqrt((middle_finger_tip.x - wrist.x)**2 + (middle_finger_tip.y - wrist.y)**2) +
                         np.sqrt((ring_finger_tip.x - wrist.x)**2 + (ring_finger_tip.y - wrist.y)**2) +
                         np.sqrt((pinky_tip.x - wrist.x)**2 + (pinky_tip.y - wrist.y)**2)) / 3
        if fist_distance < fist_threshold:
            return 'fist'

        return 'open_palm'

    def handle_gestures(self, gesture):
        current_time = time.time()

        # --- Pinch Gesture for Clicking ---
        if gesture == 'pinch':
            if self.pinch_start_time is None:
                self.pinch_start_time = current_time

            # If pinch is held long enough and we haven't already clicked
            if (current_time - self.pinch_start_time) >= self.PINCH_CONFIRM_TIME and not self.is_pinching:
                pyautogui.click()
                self.is_pinching = True # Lock state to prevent multiple clicks
        else:
            # Reset when pinch is released
            self.pinch_start_time = None
            self.is_pinching = False

        # --- Fist Gesture for Dragging ---
        if gesture == 'fist':
            if self.fist_start_time is None:
                self.fist_start_time = current_time

            # If fist is held long enough and we are not already dragging
            if (current_time - self.fist_start_time) >= self.FIST_CONFIRM_TIME and not self.is_dragging:
                pyautogui.mouseDown()
                self.is_dragging = True
                self.is_fist_closed = True
        elif gesture == 'open_palm':
            # Release drag only if it was active
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False
            self.fist_start_time = None
            self.is_fist_closed = False

        # If no hand is detected, ensure drag is released
        if gesture == 'no_hand' and self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False
            self.fist_start_time = None
            self.is_fist_closed = False


    def run(self):
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        BaseOptions = mp.tasks.BaseOptions

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO)

        self.video_thread.start()

        print("Waiting for first frame...")
        try:
            first_frame = self.video_thread.frame_queue.get(timeout=10)
        except queue.Empty:
            print("Could not get first frame. Exiting.")
            self.video_thread.stop()
            self.video_thread.join()
            return

        frame_height, frame_width, _ = first_frame.shape

        # --- Aspect Ratio Correction ---
        self.screen_aspect_ratio = self.screen_width / self.screen_height
        self.camera_aspect_ratio = frame_width / frame_height
        self.aspect_ratio_correction = self.screen_aspect_ratio / self.camera_aspect_ratio

        gui_thread = threading.Thread(target=launch_gui, args=(self.config, self.config_lock))
        gui_thread.daemon = True
        gui_thread.start()

        frame_timestamp_ms = 0

        with HandLandmarker.create_from_options(options) as landmarker:
            while self.video_thread.is_alive():
                try:
                    image = self.video_thread.frame_queue.get(timeout=1)
                except queue.Empty:
                    if not self.video_thread.is_alive():
                        break
                    continue

                frame_timestamp_ms += 33

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                gesture = 'no_hand'

                if detection_result.hand_landmarks:
                    hand_landmarks = detection_result.hand_landmarks[0] # Assuming one hand

                    for landmark in hand_landmarks:
                        x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

                    index_finger_tip = hand_landmarks[8]
                    with self.config_lock:
                        sensitivity = self.config['SENSITIVITY']
                        motion_scale = self.config['MOTION_SCALE']

                    effective_scale = sensitivity * motion_scale

                    # Apply aspect ratio correction
                    corrected_x = (index_finger_tip.x - 0.5) * self.aspect_ratio_correction

                    x = int(corrected_x * effective_scale * self.screen_width + self.screen_width / 2)
                    y = int((index_finger_tip.y - 0.5) * effective_scale * self.screen_height + self.screen_height / 2)

                    x = np.clip(x, 1, self.screen_width - 1)
                    y = np.clip(y, 1, self.screen_height - 1)

                    self.kf_x.predict()
                    smoothed_x = int(self.kf_x.update(x)[0, 0])
                    self.kf_y.predict()
                    smoothed_y = int(self.kf_y.update(y)[0, 0])

                    pyautogui.moveTo(smoothed_x, smoothed_y)
                    gesture = self.detect_gesture(hand_landmarks)
                    self.handle_gestures(gesture)
                else:
                    self.handle_gestures('no_hand')

                cv2.putText(image, f"Gesture: {gesture}", (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "Gestures: pinch, fist, open_palm", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('GestureControl OS', image)

                if cv2.waitKey(5) & 0xFF == 27:
                    self.video_thread.stop()
                    break

            self.video_thread.join()
            cv2.destroyAllWindows()
