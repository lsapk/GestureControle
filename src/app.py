import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import threading
from gui import launch_gui
from config import load_config, save_config

class KalmanFilter:
    def __init__(self, dt=0.01, u=0, std_acc=1, std_meas=0.1): # Start with a smaller dt
        self.dt = dt
        self.u = u
        self.A = np.array([[1, self.dt], [0, 1]])
        self.B = np.array([[(self.dt**2) / 2], [self.dt]])
        self.H = np.array([[1, 0]])
        self.Q = np.array([[(self.dt**4) / 4, (self.dt**3) / 2], [(self.dt**3) / 2, self.dt**2]]) * std_acc**2
        self.R = np.array([[std_meas**2]])
        self.P = np.zeros((2, 2))
        self.x = np.zeros((2, 1))

    def predict(self, dt):
        self.dt = dt
        self.A = np.array([[1, self.dt], [0, 1]])
        self.B = np.array([[(self.dt**2) / 2], [self.dt]])
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        self.P = self.P - np.dot(K, np.dot(S, K.T))
        return self.x

class GestureController:
    def __init__(self):
        self.config = load_config()
        self.config_lock = threading.Lock()
        self.screen_width, self.screen_height = pyautogui.size()
        self.cap = cv2.VideoCapture(0)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.kf_x = KalmanFilter()
        self.kf_y = KalmanFilter()

        self.last_click_time = 0
        self.click_armed = True
        self.running = True
        self.cursor_x, self.cursor_y = pyautogui.position()
        self.prev_tip_x = None
        self.prev_tip_y = None

        self.prev_time = time.time()
        self.fps = 0

        # MediaPipe HandLandmarker setup
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO)
        self.landmarker = HandLandmarker.create_from_options(options)

    def detect_gesture(self, landmarks):
        with self.config_lock:
            click_threshold = self.config['CLICK_THRESHOLD']

        if not landmarks:
            return 'no_hand'

        thumb_tip = landmarks[4]
        index_finger_tip = landmarks[8]
        # Click gesture (thumb + index pinch)
        click_distance = np.sqrt((thumb_tip.x - index_finger_tip.x)**2 + (thumb_tip.y - index_finger_tip.y)**2)
        if click_distance < click_threshold:
            return 'click'

        return 'move'

    def run(self):
        gui_thread = threading.Thread(target=launch_gui, args=(self.config, self.config_lock, self.stop))
        gui_thread.daemon = True
        gui_thread.start()

        while self.running and self.cap.isOpened():
            # Calculate dynamic dt
            current_time = time.time()
            dt = current_time - self.prev_time
            self.prev_time = current_time
            self.fps = 1 / dt if dt > 0 else 0

            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame_timestamp_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detection_result = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gesture = 'no_hand'

            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]
                gesture = self.detect_gesture(hand_landmarks)
                freeze_cursor = gesture == 'click'
                self.process_landmarks(hand_landmarks, image, dt, freeze_cursor)
                self.handle_gestures(gesture)

            else:
                self.prev_tip_x = None
                self.prev_tip_y = None
                self.click_armed = True

            self.draw_guides(image, gesture)

            cv2.imshow('GestureControl OS', image)
            if cv2.waitKey(5) & 0xFF == 27:
                self.stop()
                break

        self.shutdown()

    def process_landmarks(self, hand_landmarks, image, dt, freeze_cursor=False):
        # Draw landmarks
        for landmark in hand_landmarks:
            x, y = int(landmark.x * self.frame_width), int(landmark.y * self.frame_height)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        # Move cursor with relative hand motion + adaptive acceleration.
        # This makes edge access easier without sacrificing fine precision.
        index_finger_tip = hand_landmarks[8]
        with self.config_lock:
            sensitivity = self.config['SENSITIVITY']
            motion_scale = self.config['MOTION_SCALE']

        if self.prev_tip_x is None or self.prev_tip_y is None:
            self.prev_tip_x = index_finger_tip.x
            self.prev_tip_y = index_finger_tip.y
            return

        delta_x = index_finger_tip.x - self.prev_tip_x
        delta_y = index_finger_tip.y - self.prev_tip_y
        self.prev_tip_x = index_finger_tip.x
        self.prev_tip_y = index_finger_tip.y

        if freeze_cursor:
            return

        base_scale = sensitivity * motion_scale
        motion_speed = np.sqrt(delta_x**2 + delta_y**2)
        acceleration = 1.0 + min(3.0, motion_speed * 18.0)
        effective_scale = base_scale * acceleration

        self.cursor_x += delta_x * self.screen_width * effective_scale
        self.cursor_y += delta_y * self.screen_height * effective_scale

        x = int(self.cursor_x)
        y = int(self.cursor_y)

        x = np.clip(x, 1, self.screen_width - 1)
        y = np.clip(y, 1, self.screen_height - 1)
        self.cursor_x = x
        self.cursor_y = y

        self.kf_x.predict(dt)
        smoothed_x = int(self.kf_x.update(x)[0, 0])
        self.kf_y.predict(dt)
        smoothed_y = int(self.kf_y.update(y)[0, 0])

        pyautogui.moveTo(smoothed_x, smoothed_y)

    def handle_gestures(self, gesture):
        current_time = time.time()
        CLICK_COOLDOWN = 0.25

        if gesture == 'click' and self.click_armed and (current_time - self.last_click_time) > CLICK_COOLDOWN:
            pyautogui.click()
            self.last_click_time = current_time
            self.click_armed = False
        elif gesture == 'move':
            self.click_armed = True

    def draw_guides(self, image, gesture):
        # Display FPS
        cv2.putText(image, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Gesture: {gesture}", (10, self.frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "Gesture: click (thumb + index)", (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def stop(self):
        self.running = False

    def shutdown(self):
        self.cap.release()
        cv2.destroyAllWindows()
        save_config(self.config)

def main():
    controller = GestureController()
    controller.run()

if __name__ == '__main__':
    main()
