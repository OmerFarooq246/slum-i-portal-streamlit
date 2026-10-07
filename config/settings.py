import os

TMP_DIR = "tmp"
FIXED_ZOOM_LEVEL = 19
TILE_SIZE = 512

os.makedirs(TMP_DIR, exist_ok=True)

PRESET_LOCATIONS: dict[str, dict[str, object]] = {
    "Lahore": {
        "center": [31.5497, 74.3436],
        "zoom": 12,
        "description": "Lahore metropolitan area",
    },
    "Islamabad": {
        "center": [33.6844, 73.0479],
        "zoom": 12,
        "description": "Islamabad capital territory",
    },
}
