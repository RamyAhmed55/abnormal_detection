"""
dashboard/app.py
─────────────────
FireGuard AI — Flask Dashboard Backend

Provides:
  - Login / Session management (simple username/password)
  - REST API for alerts, saved videos, stats
  - WebSocket (SocketIO) for live detection streaming
  - Video upload + real-time YOLO processing pipeline
  - Alert video streaming / playback
"""

import sys
import os
from pathlib import Path


# Add project root and backend to path so imports work correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import json
import logging
import shutil
import threading
import time
import uuid
from functools import wraps
from typing import Optional

from flask import (
    Flask, render_template, request, jsonify, session,
    redirect, url_for, send_file, abort, send_from_directory,
)
from flask_socketio import SocketIO, emit, disconnect

import config

# ─────────────────────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = PROJECT_ROOT
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
SAVED_VIDEOS_DIR = BASE_DIR / "data" / "saved_videos"
ALERTS_DIR = Path(config.ALERTS_DIR)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAVED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
ALERTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("DASHBOARD_SECRET_KEY", "fireguard-secret-2024")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB upload limit

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    max_http_buffer_size=50 * 1024 * 1024,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Simple User Store (replace with DB for production)
# ─────────────────────────────────────────────────────────────────────────────

USERS = {
    "admin": {"password": "admin123", "name": "Administrator", "role": "admin"},
    "demo":  {"password": "demo",     "name": "Demo User",     "role": "viewer"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Active Processing Sessions
# ─────────────────────────────────────────────────────────────────────────────

_active_sessions: dict = {}   # sid -> {"stop": Event, "thread": Thread}


# ─────────────────────────────────────────────────────────────────────────────
# Auth Helpers
# ─────────────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            if request.is_json:
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────────────────────
# Page Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "username" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = USERS.get(username)
        if user and user["password"] == password:
            session["username"] = username
            session["name"] = user["name"]
            session["role"] = user["role"]
            return redirect(url_for("dashboard"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=session)


@app.route("/upload")
@login_required
def upload_page():
    return render_template("upload.html", user=session)


@app.route("/alerts")
@login_required
def alerts_page():
    return render_template("alerts.html", user=session)


@app.route("/saved")
@login_required
def saved_page():
    return render_template("saved.html", user=session)


# ─────────────────────────────────────────────────────────────────────────────
# API: Stats
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/stats")
@login_required
def api_stats():
    alerts = _load_all_alerts()
    total = len(alerts)
    abnormal = sum(1 for a in alerts if a.get("is_abnormal"))
    normal = total - abnormal
    saved = len(list(SAVED_VIDEOS_DIR.glob("*.mp4")))
    return jsonify({
        "total_alerts": total,
        "abnormal_events": abnormal,
        "normal_events": normal,
        "saved_videos": saved,
        "active_cameras": 0,  # Placeholder for future camera support
    })


# ─────────────────────────────────────────────────────────────────────────────
# API: Alerts
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/alerts")
@login_required
def api_alerts():
    alerts = _load_all_alerts()
    alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify(alerts)


@app.route("/api/alerts/<alert_id>/video")
@login_required
def alert_video(alert_id):
    """Stream the alert clip video."""
    alert_dir = ALERTS_DIR / alert_id
    clip_path = alert_dir / "clip.mp4"
    if not clip_path.exists():
        abort(404)
    return send_file(str(clip_path), mimetype="video/mp4", conditional=True)


@app.route("/api/alerts/<alert_id>/save", methods=["POST"])
@login_required
def save_alert_video(alert_id):
    """Copy alert clip to saved videos collection."""
    alert_dir = ALERTS_DIR / alert_id
    clip_path = alert_dir / "clip.mp4"
    result_path = alert_dir / "result.json"

    if not clip_path.exists():
        return jsonify({"error": "Alert clip not found"}), 404

    # Load result for metadata
    result = {}
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Save to saved_videos with descriptive filename
    event_type = result.get("event_type", "event").replace(" ", "_")
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest_name = f"{event_type}_{alert_id[:12]}_{ts}.mp4"
    dest_path = SAVED_VIDEOS_DIR / dest_name

    shutil.copy2(clip_path, dest_path)

    # Save metadata JSON alongside
    meta = {
        "original_alert_id": alert_id,
        "saved_at": ts,
        "event_type": result.get("event_type", "unknown"),
        "severity": result.get("severity", "unknown"),
        "description": result.get("description", ""),
        "is_abnormal": result.get("is_abnormal", False),
        "filename": dest_name,
    }
    (SAVED_VIDEOS_DIR / dest_name.replace(".mp4", ".json")).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Saved alert video: %s -> %s", alert_id, dest_name)
    return jsonify({"success": True, "filename": dest_name})


# ─────────────────────────────────────────────────────────────────────────────
# API: Saved Videos
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/saved")
@login_required
def api_saved():
    videos = []
    for mp4 in sorted(SAVED_VIDEOS_DIR.glob("*.mp4"), reverse=True):
        meta_path = mp4.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        videos.append({
            "filename": mp4.name,
            "size_mb": round(mp4.stat().st_size / (1024 * 1024), 1),
            "event_type": meta.get("event_type", "unknown"),
            "severity": meta.get("severity", "unknown"),
            "description": meta.get("description", ""),
            "saved_at": meta.get("saved_at", ""),
            "is_abnormal": meta.get("is_abnormal", False),
        })
    return jsonify(videos)


@app.route("/api/saved/<filename>/video")
@login_required
def saved_video_stream(filename):
    """Stream a saved video."""
    safe = Path(filename).name  # Prevent path traversal
    path = SAVED_VIDEOS_DIR / safe
    if not path.exists() or path.suffix != ".mp4":
        abort(404)
    return send_file(str(path), mimetype="video/mp4", conditional=True)


@app.route("/api/saved/<filename>/delete", methods=["DELETE"])
@login_required
def delete_saved_video(filename):
    safe = Path(filename).name
    for ext in [".mp4", ".json"]:
        p = SAVED_VIDEOS_DIR / (safe.replace(".mp4", ext))
        if p.exists():
            p.unlink()
    return jsonify({"success": True})


# ─────────────────────────────────────────────────────────────────────────────
# API: Video Upload
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
@login_required
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    session_id = str(uuid.uuid4())[:8]
    upload_path = UPLOAD_DIR / f"{session_id}{ext}"
    file.save(str(upload_path))

    logger.info("Uploaded video: %s (session=%s)", file.filename, session_id)
    return jsonify({
        "success": True,
        "session_id": session_id,
        "filename": file.filename,
        "path": str(upload_path),
    })


# ─────────────────────────────────────────────────────────────────────────────
# SocketIO: Live Detection Processing
# ─────────────────────────────────────────────────────────────────────────────

@socketio.on("start_detection")
def handle_start_detection(data):
    """
    Start the detection pipeline on an uploaded video.
    Streams annotated frames + events back via SocketIO.
    """
    if "username" not in session:
        emit("error", {"message": "Not authenticated"})
        return

    session_id = data.get("session_id")
    if not session_id:
        emit("error", {"message": "No session_id provided"})
        return

    # Find the uploaded file
    upload_path = None
    for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        candidate = UPLOAD_DIR / f"{session_id}{ext}"
        if candidate.exists():
            upload_path = candidate
            break

    if upload_path is None:
        emit("error", {"message": f"Video file not found for session {session_id}"})
        return

    sid = request.sid
    stop_event = threading.Event()
    _active_sessions[sid] = {"stop": stop_event}

    thread = threading.Thread(
        target=_run_detection_pipeline,
        args=(str(upload_path), sid, stop_event, session_id),
        daemon=True,
    )
    _active_sessions[sid]["thread"] = thread
    thread.start()

    emit("detection_started", {"session_id": session_id})
    logger.info("Detection started | session=%s | sid=%s", session_id, sid)


@socketio.on("stop_detection")
def handle_stop_detection(data):
    sid = request.sid
    if sid in _active_sessions:
        _active_sessions[sid]["stop"].set()
        logger.info("Detection stopped by user | sid=%s", sid)


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    if sid in _active_sessions:
        _active_sessions[sid]["stop"].set()
        _active_sessions.pop(sid, None)


# ─────────────────────────────────────────────────────────────────────────────
# Detection Pipeline (runs in background thread, emits events via SocketIO)
# ─────────────────────────────────────────────────────────────────────────────

def _run_detection_pipeline(video_path: str, sid: str, stop_event: threading.Event, session_id: str):
    """
    Runs YOLO + tracking + VLM pipeline and emits events back to the client.
    This runs completely in a background thread (non-blocking for UI).
    """
    import base64
    import cv2

    try:
        from core.detector import Detector
        from core.logical_tracker import FireSmokeTrackManager
        from core.stream_reader import StreamReader
        from core.feature_extractor import crop_with_padding
        from core.crop_saver import save_crop
        from core.clip_exporter import cleanup_temp_clips
        from db.zone_db import ZoneDB
        from core.vlm_worker import VLMWorker
        from hardware.gpu_checker import get_routing_decision

        from core.crop_cluster import get_crop_cluster_manager
        get_crop_cluster_manager().reset_clusters()

        cleanup_temp_clips()

        # Initialize components
        db = ZoneDB()
        db.initialize()

        route = get_routing_decision(force=None)
        vlm_worker = VLMWorker(db=db, route=route)
        vlm_worker.start()

        detector = Detector()
        manager = FireSmokeTrackManager(
            confirmation_seconds=config.DETECTION_STABILITY_SECONDS,
            max_missed_seconds=config.MAX_MISSED_SECONDS,
            reid_max_distance_px=config.REID_MAX_DISTANCE_PX,
            assumed_fps=config.ASSUMED_FPS,
        )
        reader = StreamReader(source=video_path, camera_id=f"upload_{session_id}")

        COLOR_MAP = {"fire": (0, 140, 255), "smoke": (180, 180, 180)}
        frame_count = 0
        prev_time = time.time()
        EMIT_EVERY_N_FRAMES = 3  # Emit every 3rd frame to reduce bandwidth

        camera_id = f"upload_{session_id}"

        for frame, timestamp in reader.read():
            if stop_event.is_set():
                break

            frame_count += 1
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Run YOLO tracking
            tracked_results = detector.track(frame)
            seen_raw_ids = set()
            annotated = frame.copy()

            detections_info = []

            for raw_track_id, det in tracked_results:
                seen_raw_ids.add(raw_track_id)
                bbox = det.bbox_xyxy
                class_name = det.class_name
                conf = det.confidence

                crop = crop_with_padding(frame, bbox, padding_ratio=config.CROP_PADDING_RATIO)

                logical_track, status = manager.update(
                    raw_track_id=raw_track_id,
                    camera_id=camera_id,
                    class_name=class_name,
                    bbox=bbox,
                    now=timestamp,
                )

                if status == "confirmed_new":
                    saved_path = save_crop(
                        crop_bgr=crop,
                        logical_id=logical_track.logical_id,
                        camera_id=camera_id,
                        class_name=class_name,
                        timestamp=timestamp,
                    )
                    logical_track.saved_crop_path = saved_path
                    vlm_worker.enqueue(
                        buffer_frames=reader.get_buffer_frames(),
                        fps=reader.fps,
                        camera_id=camera_id,
                        class_name=class_name,
                        logical_id=logical_track.logical_id,
                        bbox=bbox,
                        crop_bgr=crop,
                    )
                    # Notify UI immediately that analysis started
                    socketio.emit("vlm_analyzing", {
                        "logical_id": logical_track.logical_id,
                        "class_name": class_name,
                        "message": f"Analyzing L{logical_track.logical_id} ({class_name})..."
                    }, to=sid)

                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                color = COLOR_MAP.get(class_name, (0, 255, 0))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                duration = logical_track.active_duration
                label = f"L{logical_track.logical_id} {class_name} {conf:.0%} ({duration:.1f}s)"
                cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                detections_info.append({
                    "logical_id": logical_track.logical_id,
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "duration": round(duration, 1),
                    "status": status,
                })

            # Mark missed tracks
            for raw_id in list(manager.active_tracks.keys()):
                if raw_id not in seen_raw_ids:
                    manager.mark_missed(raw_id, now=timestamp)

            # Draw HUD (clean FPS and Tracks count, hidden frame count)
            cv2.rectangle(annotated, (5, 5), (250, 40), (0, 0, 0), -1)
            cv2.putText(annotated, f"FPS: {fps:.0f} | Tracks: {len(manager.get_all_active_logical_tracks())}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

            # Emit annotated frame (every N frames to limit bandwidth)
            if frame_count % EMIT_EVERY_N_FRAMES == 0:
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_b64 = base64.b64encode(buf).decode("ascii")
                socketio.emit("frame", {
                    "frame": frame_b64,
                    "fps": round(fps, 1),
                    "frame_count": frame_count,
                    "detections": detections_info,
                    "vlm_status": vlm_worker.latest_status_message,
                }, to=sid)

            # Check if VLM produced a new result (FIXED DICT LOOKUP)
            sess_dict = _active_sessions.get(sid, {})
            last_emitted = sess_dict.get("_last_emitted_result") if isinstance(sess_dict, dict) else None
            if vlm_worker.last_result and vlm_worker.last_result != last_emitted:
                result = vlm_worker.last_result.copy()
                if sid in _active_sessions:
                    _active_sessions[sid]["_last_emitted_result"] = vlm_worker.last_result

                socketio.emit("vlm_result", result, to=sid)

        # Pipeline done
        vlm_worker.stop()
        reader.release()

        # Wait a moment for any final VLM results, then emit completion
        time.sleep(2)
        final_result = vlm_worker.last_result if vlm_worker.last_result else None
        socketio.emit("detection_complete", {
            "total_frames": frame_count,
            "final_result": final_result,
            "alerts": _get_recent_alerts(camera_id),
        }, to=sid)

        logger.info("Detection complete | session=%s | frames=%d", session_id, frame_count)

    except Exception as e:
        logger.exception("Detection pipeline error: %s", e)
        socketio.emit("error", {"message": str(e)}, to=sid)
    finally:
        _active_sessions.pop(sid, None)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_all_alerts() -> list:
    """Load all alert packages from the alerts_output directory."""
    alerts = []
    if not ALERTS_DIR.exists():
        return alerts

    for alert_dir in ALERTS_DIR.iterdir():
        if not alert_dir.is_dir():
            continue
        result_path = alert_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            msg_path = alert_dir / "alert_message.txt"
            message = msg_path.read_text(encoding="utf-8") if msg_path.exists() else ""
            has_clip = (alert_dir / "clip.mp4").exists()

            alerts.append({
                "id": alert_dir.name,
                "timestamp": result.get("timestamp", alert_dir.name.split("_")[-2] if "_" in alert_dir.name else ""),
                "camera_id": result.get("camera_id", alert_dir.name.split("_")[0]),
                "is_abnormal": result.get("is_abnormal", False),
                "event_type": result.get("event_type", "unknown"),
                "severity": result.get("severity", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "description": result.get("description", ""),
                "reasoning": result.get("reasoning", ""),
                "router": result.get("_router", "unknown"),
                "suppressed": result.get("suppressed", False),
                "object_id": result.get("object_id", ""),
                "has_clip": has_clip,
                "message": message,
            })
        except Exception as e:
            logger.warning("Failed to load alert %s: %s", alert_dir.name, e)

    return alerts


def _get_recent_alerts(camera_id: str, limit: int = 5) -> list:
    """Get recent alerts for a specific camera."""
    all_alerts = _load_all_alerts()
    cam_alerts = [a for a in all_alerts if camera_id in a.get("camera_id", "")]
    cam_alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return cam_alerts[:limit]


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🔥 FireGuard AI Dashboard")
    print("  http://127.0.0.1:5000")
    print("  Login: admin / admin123")
    print("=" * 55 + "\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
