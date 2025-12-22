from src.app import GestureController
import pyautogui

def main():
    pyautogui.FAILSAFE = True
    app = GestureController()
    app.run()

if __name__ == "__main__":
    main()
