import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from src.config import save_config, reset_config

class SettingsGUI:
    def __init__(self, root, config, lock, on_close_callback):
        self.root = root
        self.config = config
        self.lock = lock
        self.on_close_callback = on_close_callback
        self.root.title("Settings")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.sliders = {}
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=BOTH, expand=True)

        slider_frame = ttk.Labelframe(main_frame, text="Adjustments", padding="10")
        slider_frame.pack(fill=X, pady=5)

        row = 0
        for key, value in self.config.items():
            label = ttk.Label(slider_frame, text=key)
            label.grid(row=row, column=0, sticky=W, padx=5, pady=5)

            min_val, max_val = 0, 3.0
            if "THRESHOLD" in key:
                min_val, max_val = 0, 1.0

            slider = ttk.Scale(
                slider_frame,
                from_=min_val,
                to=max_val,
                orient=HORIZONTAL,
                value=value,
                command=lambda v, k=key: self.update_setting(k, v),
            )
            slider.grid(row=row, column=1, sticky=EW, padx=5)
            self.sliders[key] = slider

            value_label = ttk.Label(slider_frame, text=f"{value:.2f}", width=5)
            value_label.grid(row=row, column=2, padx=5)
            self.sliders[key].value_label = value_label

            row += 1

        slider_frame.columnconfigure(1, weight=1)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=X, pady=10)

        save_button = ttk.Button(button_frame, text="Save", command=self.save_settings, bootstyle=SUCCESS)
        save_button.pack(side=LEFT, padx=5, expand=True, fill=X)

        reset_button = ttk.Button(button_frame, text="Reset to Default", command=self.reset_settings, bootstyle=WARNING)
        reset_button.pack(side=LEFT, padx=5, expand=True, fill=X)

    def update_setting(self, key, value):
        float_value = float(value)
        with self.lock:
            self.config[key] = float_value
        self.sliders[key].value_label.config(text=f"{float_value:.2f}")

    def save_settings(self):
        with self.lock:
            save_config(self.config)

    def reset_settings(self):
        with self.lock:
            self.config = reset_config()

        for key, slider in self.sliders.items():
            slider.set(self.config[key])
            slider.value_label.config(text=f"{self.config[key]:.2f}")

    def on_close(self):
        self.on_close_callback()
        self.root.destroy()

def launch_gui(config, lock, on_close_callback):
    root = ttk.Window(themename="cyborg")
    app = SettingsGUI(root, config, lock, on_close_callback)
    root.mainloop()

if __name__ == "__main__":
    import threading
    from src.config import load_config

    config = load_config()
    lock = threading.Lock()

    def on_close_stub():
        print("GUI closed")

    launch_gui(config, lock, on_close_stub)
