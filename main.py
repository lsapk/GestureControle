import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Create a HandLandmarker object.
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

# Gesture variables
CLICK_THRESHOLD = 0.05
FIST_THRESHOLD = 0.1
last_click_time = 0
CLICK_COOLDOWN = 0.5
is_dragging = False

# Motion smoothing variables
alpha = 0.5  # Smoothing factor (0 < alpha < 1)
prev_x, prev_y = 0, 0
frame_timestamp_ms = 0

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
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            # Get the coordinates of key landmarks
            index_finger_tip = hand_landmarks[8] # INDEX_FINGER_TIP
            thumb_tip = hand_landmarks[4] # THUMB_TIP
            middle_finger_tip = hand_landmarks[12] # MIDDLE_FINGER_TIP
            ring_finger_tip = hand_landmarks[16] # RING_FINGER_TIP
            pinky_tip = hand_landmarks[20] # PINKY_TIP
            wrist = hand_landmarks[0] # WRIST

            # Convert normalized coordinates to screen coordinates
            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)

            # Apply EMA for smoothing
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            smoothed_x = int(alpha * x + (1 - alpha) * prev_x)
            smoothed_y = int(alpha * y + (1 - alpha) * prev_y)

            # Move the mouse cursor to the smoothed position
            pyautogui.moveTo(smoothed_x, smoothed_y)
            prev_x, prev_y = smoothed_x, smoothed_y

            # Click gesture
            click_distance = np.sqrt((thumb_tip.x - index_finger_tip.x)**2 + (thumb_tip.y - index_finger_tip.y)**2)
            current_time = time.time()
            if click_distance < CLICK_THRESHOLD and (current_time - last_click_time) > CLICK_COOLDOWN:
                pyautogui.click()
                last_click_time = current_time

            # Drag and drop gesture (fist)
            fist_distance = (np.sqrt((middle_finger_tip.x - wrist.x)**2 + (middle_finger_tip.y - wrist.y)**2) +
                             np.sqrt((ring_finger_tip.x - wrist.x)**2 + (ring_finger_tip.y - wrist.y)**2) +
                             np.sqrt((pinky_tip.x - wrist.x)**2 + (pinky_tip.y - wrist.y)**2)) / 3

            if fist_distance < FIST_THRESHOLD:
                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
            else:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False

        else:
            # Reset previous position and dragging state if no hand is detected
            prev_x, prev_y = 0, 0
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False


        cv2.imshow('GestureControl OS', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
