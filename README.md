# GestureControl OS

GestureControl OS is a Python application that allows you to control your computer's cursor using hand gestures. It uses OpenCV to capture video from your webcam and MediaPipe to detect hand landmarks.

## Features

- **Cursor Control:** Move the mouse cursor by moving your hand.
- **Clicking:** Perform a pinching gesture with your thumb and index finger to click.
- **Dragging:** Make a fist to drag and drop.
- **Smooth Motion:** A Kalman filter with dynamic time step calculation is used to smooth the cursor's movement.
- **Modern UI:** A `ttkbootstrap`-based GUI for adjusting settings.
- **Performance Monitoring:** Real-time FPS (Frames Per Second) display.

## Requirements

- Python 3.x
- See `requirements.txt` for a full list of dependencies.

## Usage

1. **Installation:**
   - Clone the repository.
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Download the Hand Landmarker Model:**
   - Download the `hand_landmarker.task` model from [here](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task).
   - Place the downloaded file in the root directory of the project.

3. **Run the Application:**
   ```bash
   python main.py
   ```
   - A settings window will appear alongside the main video feed.
   - You can adjust sensitivity, motion scale, and gesture thresholds in the settings window.
   - Press `Esc` to exit the application.
