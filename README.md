# 🔥 Fire & Smoke Abnormal Detection System

A modular, production-grade pipeline that detects fire and smoke from camera streams or video files using **YOLO v26** for real-time detection and a **Vision-Language Model (VLM)** for intelligent analysis and false-alarm filtering.

---

## 📁 Project Structure

```
abnormal_detection/
│
├── main.py                        # ← START HERE: entry point
├── config.py                      # All tunable constants (thresholds, paths, etc.)
├── requirements.txt               # Python dependencies
├── .env                           # GEMINI_API_KEY (already set)
│
├── weights/
│   └── best.pt                    # YOLO v26 nano trained weights
│
├── core/                          # Core pipeline modules
│   ├── stream_reader.py           # Video/stream reader + rolling 20s buffer
│   ├── detector.py                # YOLO v26 + ByteTrack inference wrapper
│   ├── logical_tracker.py         # Kalman Filter position-based Re-ID tracker (FireSmokeTrackManager)
│   ├── feature_extractor.py       # MobileNetV3-Small crop appearance feature extractor
│   ├── crop_cluster.py            # Grouped crops object clustering & VLM bypass deduplication
│   ├── vlm_worker.py              # Async non-blocking background VLM queue worker
│   ├── crop_saver.py              # Crop archive saving utility
│   └── clip_exporter.py           # Export last-20s buffer to .mp4


├── db/                            # Database layer
│   ├── schema.sql                 # SQLite table definitions
│   └── zone_db.py                 # IoU-based known-zone matching & alert logging
│
├── hardware/
│   └── gpu_checker.py             # Detect GPU/VRAM → decide routing
│
├── vllm/                          # Local Qwen2.5-VL-7B model
│   ├── prompts.py                 # JUDGE_PROMPT + context builder
│   ├── loader.py                  # Singleton model loader (4-bit quantized)
│   ├── analyzer.py                # analyze_clip() + parse_judge_output()
│   └── confidence_check.py        # Escalation decision (low conf → Gemini)
│
├── api/
│   └── gemini_analyzer.py         # Gemini 2.0 Flash API (cloud fallback)
│
├── alerts/
│   └── alert_handler.py           # Package + store alert (clip + JSON + message)
│
├── data/
│   ├── detection.db               # SQLite database (auto-created)
│   ├── temp/                      # Temporary clip files (auto-cleaned)
│   └── videos/                    # Test videos
│
├── alerts_output/                 # Saved alert packages (auto-created)
│   └── cam0_20260919_020000/
│       ├── clip.mp4
│       ├── result.json
│       └── alert_message.txt
│
└── Data/                          # Training datasets (existing)
```

---

## 🔄 Full Workflow

```
Stream/File
    │
    ▼
[StreamReader] ─── rolling 20s buffer ──────────────────────┐
    │                                                         │
    ▼                                                         │
[YOLO Detector] → detections                                 │
    │                                                         │
    ▼                                                         │
[EventTracker] → stable for 3s?                              │
    │ YES                                                     │
    ▼                                                         │
[ZoneDB] → known-normal zone (IoU ≥ 0.3)?                   │
    │ NO (new thing!)              │ YES (skip)               │
    ▼                              └──────────────────────────┘
[ClipExporter] → grabs buffer ──────────────────────────────┘
    │
    ▼
[GPU Checker] → local_vllm or gemini_api?
    │
    ├── local_vllm → [Qwen2.5-VL-7B] → confidence OK?
    │                                      │ NO → escalate
    │                                      └───→ [Gemini API]
    │
    └── gemini_api → [Gemini 2.0 Flash]
    │
    ▼
Result: is_abnormal?
    │
    ├── TRUE  → [AlertHandler] → saves clip + JSON + message → logs to DB
    │
    └── FALSE → [ZoneDB] → save as known-normal zone (skip next time)
```

---

## 🚀 How to Run

### Option 1: Uploaded Video File (test mode)

```bash
# Test with a 12-minute video file
python main.py --source Data/videos/bucket11.mp4

# With live display window (press Q to quit)
python main.py --source Data/videos/bucket11.mp4 --show

# With custom camera ID
python main.py --source Data/videos/printer31.mp4 --camera-id factory_cam --show
```

### Option 2: Live Webcam Stream

```bash
# Default webcam (camera index 0)
python main.py --source 0

# Specific webcam index
python main.py --source 1 --camera-id cam1 --show
```

### Option 3: RTSP Network Camera

```bash
python main.py --source rtsp://192.168.1.100:554/stream --camera-id outdoor_cam
```

### Option 4: Force a specific VLM backend

```bash
# Always use Gemini API (even if GPU is available)
python main.py --source 0 --route gemini_api

# Always use local Qwen (requires GPU with ≥8 GB VRAM)
python main.py --source 0 --route local_vllm
```

---

## ⚙️ Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `YOLO_CONF_THRESHOLD` | `0.45` | Min confidence for YOLO detections |
| `DETECTION_STABILITY_SECONDS` | `3` | Seconds of continuous detection before triggering |
| `EVENT_COOLDOWN_SECONDS` | `30` | Cooldown after event fires (prevents re-triggering) |
| `BUFFER_SECONDS` | `20` | Rolling buffer length in seconds |
| `IOU_OVERLAP_THRESHOLD` | `0.30` | IoU to match a detection to a known-normal zone |
| `KNOWN_ZONE_EXPIRY_DAYS` | `7` | Days before a known zone expires |
| `VLLM_MIN_VRAM_GB` | `8.0` | Min free VRAM to use local Qwen model |
| `VLLM_MIN_CONFIDENCE` | `0.60` | Below this → escalate to Gemini API |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model to use |
| `ALERT_MIN_SEVERITY` | `medium` | Only alert for medium/high severity events |

---

## 🧠 VLM Routing Logic

```
At startup:
  Has CUDA GPU?  →  No  → Use Gemini API
                    Yes → Free VRAM ≥ 8 GB?  →  No  → Use Gemini API
                                                Yes → Use Local Qwen2.5-VL-7B

During event:
  Local Qwen result confidence < 0.6?  →  Yes → Escalate to Gemini API
```

---

## 🗄️ Database (`data/detection.db`)

### `known_normal_zones`
Stores bounding boxes of things YOLO detects that VLM confirmed are **normal** (e.g. factory chimney, pilot flame, steam). Next time YOLO detects something with IoU ≥ 0.3 overlap → skip VLM entirely.

| Column | Description |
|--------|-------------|
| `camera_id` | Which camera |
| `class_name` | "fire" or "smoke" |
| `x1, y1, x2, y2` | Bounding box |
| `hit_count` | How many times re-detected |
| `last_seen` | Auto-expires after 7 days |

### `alert_events`
Every confirmed abnormal event is logged here.

| Column | Description |
|--------|-------------|
| `event_type` | fire / smoke / false_alarm |
| `severity` | low / medium / high |
| `confidence` | VLM confidence (0–1) |
| `clip_path` | Path to saved 20s clip |
| `alert_dir` | Full alert package folder |
| `router` | local_vllm or gemini_api |

---

## 🚨 Alert Packages

When an abnormal event is detected, a package is saved:

```
alerts_output/
  cam0_20260919_020000/
    ├── clip.mp4              ← 20-second video evidence
    ├── result.json           ← Full VLM analysis output
    └── alert_message.txt     ← Human-readable summary
```

**Example `alert_message.txt`:**
```
🚨 FIRE & SMOKE DETECTION ALERT
─────────────────────────────────────────────
Camera    : factory_cam
Timestamp : 20260919_020000
Event     : FIRE
Severity  : 🔴 HIGH
Confidence: 85%
Analyzed by: gemini_api
─────────────────────────────────────────────
Description:
  A bright flame is visible in the lower-left area...

Reasoning:
  Consistent flame movement confirms an active fire.
```

---

## 📦 Installation

### Minimal (Gemini API only — no GPU required)
```bash
pip install ultralytics opencv-python python-dotenv google-generativeai
```

### Full (with local Qwen VLLM on GPU)
```bash
# 1. Install PyTorch with CUDA (from pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install all requirements
pip install -r requirements.txt
```

> ⚠️ **Windows Note**: `bitsandbytes` (required for 4-bit local Qwen) does **not** support native Windows.  
> Use WSL2 or run on Linux. On Windows without WSL, the system **automatically routes all events to Gemini API**.

---

## 📊 What Works Now

| Feature | Status |
|---------|--------|
| YOLO v26 fire/smoke detection | ✅ |
| Rolling 20-second frame buffer | ✅ |
| 3-second stability check | ✅ |
| Known-normal zone DB with IoU | ✅ |
| Clip export from buffer | ✅ |
| GPU/VRAM auto-routing | ✅ |
| Local Qwen2.5-VL-7B analysis | ✅ |
| Gemini API analysis (fallback) | ✅ |
| Low-confidence escalation | ✅ |
| Alert package storage | ✅ |
| SQLite alert event logging | ✅ |
| Webcam stream support | ✅ |
| Video file support | ✅ |
| RTSP stream support | ✅ |
| Live display with bboxes (`--show`) | ✅ |

## 🔮 Coming Next (v2)

| Feature | Status |
|---------|--------|
| Multiple simultaneous cameras | 🔜 |
| Telegram alert integration | 🔜 |
| Streamlit web dashboard | 🔜 |
| Parallel Gemini for multi-camera | 🔜 |

---

## 🗒️ Notes & Improvements Made

1. **Spaces in `best (2).pt`** — Copied to `weights/best.pt` to avoid path issues on Windows.
2. **Known-zone IoU** — Factory smoke that runs all day is learned after first VLM confirmation and never re-analyzed.
3. **Gemini file cleanup** — Uploaded files are deleted immediately after use to avoid hitting quota.
4. **Rolling buffer deque** — O(1) append/pop, automatically drops frames older than 20s.
5. **Singleton VLLM model** — Loaded once, reused for all events. No reload cost per detection.
6. **4-bit quantization** — Qwen runs in ~6 GB VRAM (notebook confirmed), minimum set to 8 GB for safety headroom.
