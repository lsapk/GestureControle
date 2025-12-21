# GestureControl OS

GestureControl OS is a Python application that allows you to control your computer's cursor using hand gestures. It uses OpenCV to capture video from your webcam and MediaPipe to detect hand landmarks.

## Features

- **Cursor Control:** Move the mouse cursor by moving your hand.
- **Clicking:** Perform a pinching gesture with your thumb and index finger to click.
- **Dragging:** Make a fist to drag and drop.
- **Smooth Motion:** A Kalman filter is used to smooth the cursor's movement.

## Advanced Features

- **Sensitivity Control:** Adjust the `SENSITIVITY` variable in `main.py` to change the cursor's speed.
- **Motion Scaling:** Modify the `MOTION_SCALE` variable in `main.py` to control the mapping of hand movement to the screen, reducing the need for large arm movements.

## Requirements

- Python 3.x
- `opencv-python`
- `mediapipe`
- `numpy`
- `pyautogui`

## Usage

1. Install the required dependencies: `pip install -r requirements.txt`
2. Download the `hand_landmarker.task` model from [here](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) and place it in the root of the project.
3. Run the application: `python main.py`
