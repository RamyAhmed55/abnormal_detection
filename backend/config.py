"""
config.py — Central configuration for the Fire & Smoke Detection System.
All tunable constants live here. Change values here, never inside modules.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# YOLO model weights
# best5.pt = copy of "best (5).pt" renamed to avoid Windows path issues
# To switch to best.pt: change the filename below
YOLO_WEIGHTS = str(BASE_DIR / "weights" / "best5.pt")

# SQLite database file
DB_PATH = str(BASE_DIR / "data" / "detection.db")

# Directory where alert packages (clip + JSON + message) are saved
ALERTS_DIR = str(BASE_DIR / "alerts_output")

# Temp directory for rolling-buffer clip exports
TEMP_DIR = str(BASE_DIR / "data" / "temp")

# Directory where cropped object images are saved for audit
CROPS_OUTPUT_DIR = str(BASE_DIR / "saved_crops")

# Directory where grouped object clusters are stored
GROUPED_CROPS_DIR = str(BASE_DIR / "grouped_crops")

# ─────────────────────────────────────────────────────────────────────────────
# YOLO Detection
# ─────────────────────────────────────────────────────────────────────────────
# Minimum confidence score to accept a YOLO detection
YOLO_CONF_THRESHOLD = 0.45

# Target inference size (pixels). Lower = faster, higher = more accurate
YOLO_IMG_SIZE = 640

# Class names as defined during YOLO training
YOLO_CLASS_NAMES = ["fire", "smoke"]

# ─────────────────────────────────────────────────────────────────────────────
# Event Stability & Kalman Re-ID Tracking Parameters
# ─────────────────────────────────────────────────────────────────────────────
# Seconds of continuous detection required to trigger the VLM pipeline
DETECTION_STABILITY_SECONDS = 3.0

# Maximum time (seconds) a track can remain undetected before permanently lost
MAX_MISSED_SECONDS = 10.0

# Maximum distance (pixels) between new detection and Kalman-predicted position
REID_MAX_DISTANCE_PX = 60.0

# Assumed FPS used for Kalman filter prediction step
ASSUMED_FPS = 20.0

# Margin padding ratio around detection boxes for feature extraction & crop saving
CROP_PADDING_RATIO = 0.15

# After an event is triggered, ignore further detections for this many seconds
EVENT_COOLDOWN_SECONDS = 30.0

# IoU threshold for matching frame detections to active tracks
TRACKER_IOU_THRESHOLD = 0.30

# Grace period (seconds) before dropping a track when detection flickers/disappears
TRACK_MAX_MISSING_SECONDS = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Grouped Crops & Object Clustering Deduplication (VLM Bypass)
# ─────────────────────────────────────────────────────────────────────────────
# Cosine distance threshold (DBSCAN eps) for crop similarity (<= 0.35 = same object)
CROP_CLUSTER_EPS = 0.35

# Interval (minutes) for cleaning up object folders (keep max 2 crops per folder)
CROP_CLEANUP_INTERVAL_MINUTES = 60

# Max crops retained per object folder during 60-minute cleanup
MAX_CROPS_PER_CLUSTER = 2

# Reset interval (hours): delete all object folders after 2 days (48 hrs)
CROP_RESET_HOURS = 48

# Cooldown (seconds) before sending duplicate user alerts for the SAME ongoing incident
INCIDENT_ALERT_COOLDOWN_SECONDS = 300.0



# ─────────────────────────────────────────────────────────────────────────────
# Rolling Frame Buffer
# ─────────────────────────────────────────────────────────────────────────────
# Total seconds of frames kept in the rolling buffer
BUFFER_SECONDS = 20

# ─────────────────────────────────────────────────────────────────────────────
# Known-Normal Zone (SQLite + IoU)
# ─────────────────────────────────────────────────────────────────────────────
# If IoU between a new detection and a saved known-normal bbox is >= this
# threshold, the detection is considered the same object → skip VLM routing.
IOU_OVERLAP_THRESHOLD = 0.30

# After this many days without re-detection, a known-normal zone expires
# and will be re-evaluated if seen again.
KNOWN_ZONE_EXPIRY_DAYS = 7

# ─────────────────────────────────────────────────────────────────────────────
# Hardware / Routing
# ─────────────────────────────────────────────────────────────────────────────
# Minimum FREE VRAM (GB) required to load the local Qwen model
VLLM_MIN_VRAM_GB = 8.0

# ─────────────────────────────────────────────────────────────────────────────
# Local VLLM (Qwen2.5-VL-7B-Instruct)
# ─────────────────────────────────────────────────────────────────────────────
VLLM_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Frames-per-second sent to the VLM (1 frame/sec for a 20s clip = 20 frames)
VLLM_FPS = 1.0

# Min/max pixel area for video frames sent to VLM
VLLM_MIN_PIXELS = 28 * 28 * 4
VLLM_MAX_PIXELS = 28 * 28 * 400

# Max new tokens the model is allowed to generate
VLLM_MAX_NEW_TOKENS = 256

# If VLM confidence is below this value, escalate to Gemini API
VLLM_MIN_CONFIDENCE = 0.60

# ─────────────────────────────────────────────────────────────────────────────
# Gemini API
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-3.6-flash"

# Seconds to wait between Gemini API retry attempts on rate-limit errors
GEMINI_RETRY_DELAY = 5

# Maximum number of retries for Gemini API calls
GEMINI_MAX_RETRIES = 3

# ─────────────────────────────────────────────────────────────────────────────
# Alert Severity Filter
# ─────────────────────────────────────────────────────────────────────────────
# Only events at or above this severity level are packaged as real alerts.
# Options: "low", "medium", "high"
ALERT_MIN_SEVERITY = "medium"
