import tkinter as tk
from tkinter import ttk
import json
from config import DEFAULT_CONFIG

class SettingsGUI:
    def __init__(self, root, config, lock):
        self.root = root
        self.config = config
        self.lock = lock
        self.root.title("Settings")

        self.sliders = {}
        self.create_widgets()

    def create_widgets(self):
        slider_frame = ttk.Frame(self.root, padding="10")
        slider_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        row = 0
        for key, value in self.config.items():
            label = ttk.Label(slider_frame, text=key)
            label.grid(row=row, column=0, sticky=tk.W)

            min_val, max_val = 0, 3.0
            if "THRESHOLD" in key:
                min_val, max_val = 0, 1.0

            slider = ttk.Scale(
                slider_frame,
                from_=min_val,
                to=max_val,
                orient=tk.HORIZONTAL,
                value=value,
                command=lambda v, k=key: self.update_setting(k, v),
            )
            slider.grid(row=row, column=1, sticky=(tk.W, tk.E))
            self.sliders[key] = slider

            value_label = ttk.Label(slider_frame, text=f"{value:.2f}")
            value_label.grid(row=row, column=2)
            self.sliders[key].value_label = value_label

            row += 1

        save_button = ttk.Button(self.root, text="Save", command=self.save_settings)
        save_button.grid(row=1, column=0, pady=10)

    def update_setting(self, key, value):
        float_value = float(value)
        with self.lock:
            self.config[key] = float_value
        self.sliders[key].value_label.config(text=f"{float_value:.2f}")

    def save_settings(self):
        with self.lock:
            with open("config.json", "w") as f:
                json.dump(self.config, f, indent=2)

def launch_gui(config, lock):
    root = tk.Tk()
    app = SettingsGUI(root, config, lock)
    root.mainloop()

if __name__ == "__main__":
    import threading
    lock = threading.Lock()
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = DEFAULT_CONFIG
    launch_gui(config, lock)
