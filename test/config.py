"""
Configuration File
All parameters for the tracking system
"""

import torch

# =============================================================================
# DETECTION & TRACKING CONFIGURATION
# =============================================================================
MODEL_PATH = "yolov8n.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Detection
DETECT_WIDTH = 640
DETECT_EVERY_N = 1
CONF_THRESH = 0.45
NMS_IOU_THRESH = 0.5
ONLY_PERSONS = True

# DeepSort Tracker
MAX_COSINE_DISTANCE = 0.4
TRACKER_MAX_AGE = 50
TRACKER_N_INIT = 3
NN_BUDGET = 100

# =============================================================================
# VISUALIZATION & UI
# =============================================================================
MAX_PATH_LENGTH = 300
PATH_WINDOW_SIZE = (900, 700)
TRAIL_LENGTH = 30

# Colors (BGR)
COLOR_UID_ASSIGNED = (255, 0, 255)    # Magenta
COLOR_SELECTED = (255, 255, 0)        # Cyan
COLOR_LOW_DENSITY = (0, 255, 0)       # Green
COLOR_MEDIUM_DENSITY = (0, 255, 255)  # Yellow
COLOR_HIGH_DENSITY = (0, 165, 255)    # Orange
COLOR_VERY_HIGH = (0, 0, 255)         # Red

# =============================================================================
# RECORDING
# =============================================================================
OUTPUT_VIDEO_CODEC = 'mp4v'
OUTPUT_VIDEO_FPS = 20.0
