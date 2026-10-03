# Fire & Smoke Abnormal Detection System — Full Architecture Plan

## Background

The project currently has two Kaggle notebooks:
- **`clip-model.ipynb`** — Loads `Qwen2.5-VL-7B-Instruct` (VLLM), trims last 20s from a video, sends frames + judge prompt, parses JSON response, and decides whether to alert.
- **`Yolo26.ipynb`** / **`data_prep.ipynb`** — YOLO v26 training, evaluation, and inference on fire/smoke dataset.

The goal is to refactor this into a **production-grade modular system** that runs locally.

---

## Open Questions / Design Decisions

> [!IMPORTANT]
> **Notification channel**: You mentioned Telegram or Streamlit web alert as future work. For now the system will **store alert packages** (video clip + JSON result + message) in a folder ready to be consumed. You can plug in Telegram/Streamlit later.

> [!IMPORTANT]
> **VLLM model**: The notebook uses `Qwen2.5-VL-7B-Instruct` from HuggingFace (local GPU). This is what we call "vllm" throughout this plan. It is **not** a vllm-server; it is a local Transformers inference pipeline.

> [!WARNING]
> **`best (2).pt`** is your production YOLO v26 nano weights. We rename it to `weights/best.pt` internally to avoid spaces in paths.

> [!NOTE]
> **Coordinate overlap logic**: Two bounding boxes are considered the "same object" if their IoU (Intersection over Union) ≥ 0.3. This avoids storing duplicate known-normal zones across frames.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                      main.py (entry point)                 │
│  --source 0 (webcam) | --source video.mp4 | --source rtsp  │
└────────────────────┬───────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │  core/stream_reader.py │  ← rolling 20-sec buffer
         └───────────┬───────────┘
                     │  frames
         ┌───────────▼───────────┐
         │  core/detector.py      │  ← YOLO v26 nano inference
         └───────────┬───────────┘
                     │  detections
         ┌───────────▼───────────┐
         │  core/event_tracker.py │  ← 3-sec stability check
         └───────────┬───────────┘
                     │  stable_event
         ┌───────────▼───────────┐
         │  db/zone_db.py         │  ← SQLite: known-normal zones
         └───────────┬───────────┘
                     │  is_new?
         ┌───────────▼───────────┐
         │  core/clip_exporter.py │  ← save last-20s clip to disk
         └───────────┬───────────┘
                     │  clip_path
┌────────────────────▼──────────────────────────────────────┐
│              hardware/gpu_checker.py                       │
│  Has GPU? → enough VRAM? → route to local or Gemini API    │
└────┬──────────────────────────────────┬───────────────────┘
     │ GPU OK                           │ No GPU / low VRAM
┌────▼──────────────┐       ┌───────────▼──────────────────┐
│ vllm/analyzer.py  │       │  api/gemini_analyzer.py       │
│ Qwen2.5-VL-7B     │       │  Gemini 1.5/2.0 Flash API     │
└────┬──────────────┘       └───────────┬──────────────────┘
     └──────────────┬───────────────────┘
                    │  JSON result
         ┌──────────▼──────────┐
         │  core/alert_handler │  ← store alert package
         └─────────────────────┘
```

---

## Proposed File Structure

```
abnormal_detection/
│
├── main.py                        # Entry point
├── config.py                      # All tunable constants
├── .env                           # GEMINI_API_KEY
├── requirements.txt
│
├── weights/
│   └── best.pt                    # Copy of "best (2).pt"
│
├── core/
│   ├── __init__.py
│   ├── stream_reader.py           # VideoCapture + rolling frame buffer
│   ├── detector.py                # YOLO v26 wrapper
│   ├── event_tracker.py           # 3-sec stability logic
│   └── clip_exporter.py           # Save last-20s buffer as mp4
│
├── db/
│   ├── __init__.py
│   ├── zone_db.py                 # SQLite CRUD + IoU overlap check
│   └── schema.sql                 # DB schema (auto-applied on init)
│
├── vllm/
│   ├── __init__.py
│   ├── loader.py                  # Load Qwen model (lazy, singleton)
│   ├── prompts.py                 # JUDGE_PROMPT + any future prompts
│   ├── analyzer.py                # analyze_clip(), parse_judge_output()
│   └── confidence_check.py        # Is confidence < threshold? → escalate
│
├── api/
│   ├── __init__.py
│   └── gemini_analyzer.py         # Gemini 1.5/2.0 Flash via google-genai
│
├── hardware/
│   ├── __init__.py
│   └── gpu_checker.py             # Check GPU, VRAM, decide routing
│
├── alerts/
│   ├── __init__.py
│   └── alert_handler.py           # Store alert package (clip + JSON + txt)
│
├── data/                          # (existing datasets, keep as-is)
├── runs/                          # (YOLO training runs, keep as-is)
└── README.md                      # Full documentation
```

---

## Module-by-Module Plan

### `config.py`
- `YOLO_WEIGHTS` = `"weights/best.pt"`
- `YOLO_CONF_THRESHOLD` = `0.45`
- `DETECTION_STABILITY_SECONDS` = `3`
- `BUFFER_SECONDS` = `20`
- `IOU_OVERLAP_THRESHOLD` = `0.3` (for known-zone matching)
- `VLLM_MIN_CONFIDENCE` = `0.6` (below this → escalate to Gemini)
- `VLLM_MIN_VRAM_GB` = `8` (minimum free VRAM to use local model)
- `GEMINI_MODEL` = `"gemini-2.0-flash"`
- `ALERTS_DIR` = `"alerts_output/"`

---

### `core/stream_reader.py`
**Rolling 20-second frame buffer** using `collections.deque`.

- `StreamReader(source)` — source can be `int` (webcam), path to `.mp4`, or RTSP URL.
- `read()` → yields `(frame, timestamp)` one at a time.
- Internally keeps a `deque` of `(frame, timestamp)` for the last `BUFFER_SECONDS` seconds.
- `get_buffer_clip()` → writes buffer frames to a temp `.mp4` and returns the path.

**How the rolling buffer works:**
Every second a new frame arrives → oldest frame (beyond 20s window) is popped from left → buffer always contains exactly last 20 seconds.

---

### `core/detector.py`
- `Detector(weights_path)` — loads YOLO v26 nano model once.
- `detect(frame)` → returns list of `Detection(class_name, confidence, bbox_xyxy)`.
- Filters by `YOLO_CONF_THRESHOLD`.

---

### `core/event_tracker.py`
- Tracks consecutive detection state.
- `update(detections)` → returns `True` if a **stable event** is confirmed (detection present for ≥ 3 seconds continuously).
- If detection disappears before 3s, timer resets.
- Once triggered, a **cooldown** of 30s prevents re-triggering on the same ongoing event.

---

### `db/zone_db.py` — SQLite Schema

**Table: `known_normal_zones`**
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto |
| `camera_id` | TEXT | Camera source identifier |
| `class_name` | TEXT | "fire" or "smoke" |
| `x1, y1, x2, y2` | REAL | Bounding box |
| `first_seen` | DATETIME | When first recorded |
| `last_seen` | DATETIME | Updated on re-detection |
| `hit_count` | INTEGER | How many times detected |

**Table: `alert_events`**
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto |
| `camera_id` | TEXT | Camera source |
| `timestamp` | DATETIME | When triggered |
| `event_type` | TEXT | fire/smoke/false_alarm |
| `severity` | TEXT | low/medium/high |
| `confidence` | REAL | Model confidence |
| `description` | TEXT | Model description |
| `clip_path` | TEXT | Path to saved video clip |
| `is_abnormal` | BOOLEAN | Final judgment |

**Overlap check logic:**  
When a new detection bbox comes in → query all known zones for same `camera_id` + `class_name` → compute IoU against each → if IoU ≥ 0.3, it's a known zone → update `last_seen` and skip VLM routing.

---

### `core/clip_exporter.py`
- `export_clip(buffer_frames, fps, output_dir)` → saves frames as `.mp4`, returns file path.
- Filename: `clip_CAMERA_TIMESTAMP.mp4`.

---

### `hardware/gpu_checker.py`
Decision tree:
```
has_cuda? 
  └─ NO → use Gemini API
  └─ YES → free_vram >= VLLM_MIN_VRAM_GB?
              └─ NO  → use Gemini API
              └─ YES → use local Qwen VLLM
```
- `get_routing_decision()` → returns `"local"` or `"api"`
- Checks once at startup, result is cached.

---

### `vllm/loader.py`
- Singleton pattern: loads `Qwen2.5-VL-7B-Instruct` with 4-bit quantization **once**.
- `get_model_and_processor()` → returns `(model, processor)`.
- Uses same `BitsAndBytesConfig` as notebook.

### `vllm/prompts.py`
- `JUDGE_PROMPT` — exact prompt from notebook (refactored into module).
- Future: add camera-specific context injection.

### `vllm/analyzer.py`
- `analyze_clip(clip_path)` → calls model, returns raw text.
- `parse_judge_output(raw_text)` → extracts JSON dict.

### `vllm/confidence_check.py`
- `needs_escalation(result)` → `True` if `confidence < VLLM_MIN_CONFIDENCE`.

---

### `api/gemini_analyzer.py`
- Uses `google-generativeai` SDK with `GEMINI_API_KEY` from `.env`.
- `analyze_clip_gemini(clip_path)` → uploads video, sends same `JUDGE_PROMPT`, parses JSON.
- Used when: no GPU, low VRAM, or VLLM confidence too low.

---

### `alerts/alert_handler.py`
- `handle_alert(result, clip_path, camera_id)`:
  - Saves clip to `alerts_output/CAMERA_TIMESTAMP/`
  - Writes `result.json` with full VLM output
  - Writes `alert_message.txt` with human-readable summary
  - Logs to `alert_events` table in SQLite
  - **Returns** the alert package dict (ready for Telegram/Streamlit later)

---

### `main.py`
```
1. Parse args: --source, --camera-id
2. gpu_checker → decide routing
3. Load YOLO detector
4. Open StreamReader
5. Loop:
   a. Read frame → detector.detect()
   b. event_tracker.update()
   c. If stable event:
      - get detections bboxes
      - zone_db.is_known_normal(bboxes) ?
          YES → skip
          NO  → export clip → route to VLLM or Gemini
      - If result.is_abnormal:
          alert_handler.handle_alert()
          zone_db.log_alert()
      - If result.is_normal:
          zone_db.save_known_zone(bboxes)
```

---

## Verification Plan

### Automated
- `python main.py --source Data/videos/bucket11.mp4` — should trigger alert (fire)
- `python main.py --source Data/videos/printer31.mp4` — should trigger alert (smoke)
- `python main.py --source 0` — webcam stream

### Manual
- Inspect `alerts_output/` folder for saved clip + JSON + message
- Check SQLite DB with any SQLite browser
- Verify GPU routing logic prints correct decision at startup

---

## What Is NOT in Scope (this version)
- Multiple simultaneous cameras (next version)
- Telegram bot integration
- Streamlit dashboard
- RTSP authentication
