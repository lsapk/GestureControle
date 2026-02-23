import json

DEFAULT_CONFIG = {
    "SENSITIVITY": 1.5,
    "MOTION_SCALE": 1.2,
    "CLICK_THRESHOLD": 0.05,
    "MOVEMENT_DEADZONE": 0.003,
    "MAX_ACCELERATION": 3.0,
}

def load_config():
    """Loads the configuration from config.json, falling back to defaults."""
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            return {key: config.get(key, value) for key, value in DEFAULT_CONFIG.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_CONFIG.copy()

def save_config(config):
    """Saves the configuration to config.json."""
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)

def reset_config():
    """Resets the configuration to the default settings."""
    return DEFAULT_CONFIG.copy()
