# GestureControl OS

GestureControl OS is a Python application that allows you to control your computer's cursor using hand gestures. It uses OpenCV to capture video from your webcam and MediaPipe to detect hand landmarks.

## Features

- **Cursor Control:** Move the mouse cursor by moving your hand.
- **Clicking:** Perform a pinching gesture with your thumb and index finger to click.
- **Dragging:** Make a fist to drag and drop.
- **Smooth Motion:** A Kalman filter is used to smooth the cursor's movement.
- **GUI for Settings:** A simple graphical user interface allows you to adjust sensitivity and other parameters in real-time.
- **Robust Video Stream:** The application can handle camera disconnections and frame drops without crashing.
- **Improved Gesture Recognition:** Hysteresis logic prevents accidental clicks and drags caused by hand jitter.

## Requirements

- Python 3.x
- `opencv-python`
- `mediapipe`
- `numpy`
- `pyautogui`
- `tkinter` (usually included with Python)

## Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download the Model:** Download the `hand_landmarker.task` model from [here](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) and place it in the root directory of the project.

3. **Run the Application:**
   ```bash
   python main.py
   ```

## Configuration

All settings can be adjusted through the graphical user interface that launches with the application. The settings are saved in `config.json`.

- **SENSITIVITY:** Controls the overall speed of the cursor.
- **MOTION_SCALE:** Adjusts the mapping of hand movement to screen space. A higher value means smaller hand movements are needed.
- **CLICK_THRESHOLD:** Defines how close the thumb and index finger must be to trigger a pinch gesture.
- **FIST_THRESHOLD:** Defines how closed the hand must be to be recognized as a fist.
