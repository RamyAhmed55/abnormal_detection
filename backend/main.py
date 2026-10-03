"""
main.py — Fire & Smoke Detection System
═══════════════════════════════════════
Entry point for the production fire & smoke detection pipeline.

Features
────────
1. Two-Tier Re-ID Tracker (FireSmokeTrackManager): Merges raw ByteTrack ID switches
   into stable Logical Tracks (L1, L2) using MobileNetV3-Small appearance embeddings + spatial proximity.
2. Non-Blocking Async VLM Analysis (VLMWorker): Exports 20s clips and routes to
   Local Qwen2.5-VL / Gemini API in a background thread, preventing video playback freezing!
3. Known-Normal Memory (ZoneDB): Skips VLM analysis for pre-registered normal zones.
4. Clean Live HUD Overlay: Shows real-time FPS, stable Logical IDs, and VLM status banner.

Usage
─────
  # Webcam
  python main.py --source 0 --show

  # Test Video File
  python main.py --source Data/videos/printer31.mp4 --show

  # Force Gemini API
  python main.py --source Data/videos/bucket11.mp4 --route gemini_api --show
"""

import argparse
import logging
import os
import sys
import time

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

import cv2
import config
from core.stream_reader import StreamReader
from core.detector import Detector
from core.logical_tracker import FireSmokeTrackManager
from core.feature_extractor import crop_with_padding
from core.crop_saver import save_crop
from core.vlm_worker import VLMWorker
from core.clip_exporter import cleanup_temp_clips
from db.zone_db import ZoneDB
from hardware.gpu_checker import get_routing_decision, print_hardware_summary

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("detection.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────

def run(source, camera_id: str, route: str, show: bool):
    print_hardware_summary()
    cleanup_temp_clips()

    # 1. Initialize SQLite Database
    db = ZoneDB()
    db.initialize()

    # 2. Initialize VLM Async Worker (non-blocking background thread)
    vlm_worker = VLMWorker(db=db, route=route)
    vlm_worker.start()

    # 3. Initialize Detector, Tracker, and StreamReader
    detector = Detector()
    manager = FireSmokeTrackManager(
        confirmation_seconds=config.DETECTION_STABILITY_SECONDS,
        max_missed_seconds=config.MAX_MISSED_SECONDS,
        reid_max_distance_px=config.REID_MAX_DISTANCE_PX,
        assumed_fps=config.ASSUMED_FPS,
    )
    reader = StreamReader(source=source, camera_id=camera_id)

    logger.info("Pipeline started | source=%r | camera_id=%s | route=%s", source, camera_id, route)

    prev_frame_time = time.time()
    frame_count = 0

    try:
        for frame, timestamp in reader.read():
            frame_count += 1
            now = time.time()

            # Calculate FPS
            fps = 1.0 / (now - prev_frame_time) if (now - prev_frame_time) > 0 else 0.0
            prev_frame_time = now

            # ── 1. YOLO + ByteTrack Detection ──────────────────────────────────
            tracked_results = detector.track(frame)
            seen_raw_ids_this_frame = set()

            annotated_frame = frame.copy() if show else None

            # ── 2. Update Logical Tracker with Detections ──────────────────────
            for raw_track_id, det in tracked_results:
                seen_raw_ids_this_frame.add(raw_track_id)
                bbox = det.bbox_xyxy
                class_name = det.class_name

                # Extract crop with context padding
                crop = crop_with_padding(frame, bbox, padding_ratio=config.CROP_PADDING_RATIO)

                # Update Kalman Position Re-ID Logical Tracker
                logical_track, status = manager.update(
                    raw_track_id=raw_track_id,
                    camera_id=camera_id,
                    class_name=class_name,
                    bbox=bbox,
                    now=timestamp,
                )

                # ── On Confirmed Stable Event -> Enqueue to VLMWorker (Async!) ──
                if status == "confirmed_new":
                    saved_path = save_crop(
                        crop_bgr=crop,
                        logical_id=logical_track.logical_id,
                        camera_id=camera_id,
                        class_name=class_name,
                        timestamp=timestamp,
                    )
                    logical_track.saved_crop_path = saved_path

                    logger.info(
                        "⚡ Confirmed Stable Event for LogicalTrack L%d (%s) -> Enqueuing VLM task",
                        logical_track.logical_id, class_name,
                    )

                    # NON-BLOCKING: Enqueue payload to background worker thread!
                    vlm_worker.enqueue(
                        buffer_frames=reader.get_buffer_frames(),
                        fps=reader.fps,
                        camera_id=camera_id,
                        class_name=class_name,
                        logical_id=logical_track.logical_id,
                        bbox=bbox,
                        crop_bgr=crop,
                    )


            # ── 3. Mark Missed Tracks ──────────────────────────────────────────
            for raw_id in list(manager.active_tracks.keys()):
                if raw_id not in seen_raw_ids_this_frame:
                    manager.mark_missed(raw_id, now=timestamp)

            # ── 4. Live Display Overlay (HUD) ──────────────────────────────────
            if show:
                _draw_hud(
                    frame=annotated_frame,
                    active_logical_tracks=manager.get_all_active_logical_tracks(),
                    fps=fps,
                    vlm_status=vlm_worker.latest_status_message,
                    timestamp=timestamp,
                )
                cv2.imshow("Fire & Smoke Detection — Stable Re-ID", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit key pressed — stopping pipeline.")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        vlm_worker.stop()
        reader.release()
        cv2.destroyAllWindows()
        logger.info("Pipeline stopped. Total frames processed: %d", frame_count)


# ─────────────────────────────────────────────────────────────────────────────
# HUD Display Helper
# ─────────────────────────────────────────────────────────────────────────────

def _draw_hud(frame, active_logical_tracks, fps: float, vlm_status: str, timestamp: float):
    """Draw clean bounding boxes, stable Logical IDs, FPS, and VLM status banner."""
    COLOR_MAP = {"fire": (0, 140, 255), "smoke": (180, 180, 180)}

    for track in active_logical_tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        color = COLOR_MAP.get(track.class_name, (0, 255, 0))

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label: e.g. L1 fire (2.4s)
        duration = track.active_duration
        label = f"L{track.logical_id} {track.class_name} ({duration:.1f}s)"
        cv2.putText(
            frame, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )

    # ── Top HUD Banner ────────────────────────────────────────────────────────
    cv2.rectangle(frame, (10, 10), (450, 75), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (450, 75), (50, 50, 50), 1)

    fps_text = f"FPS: {int(fps)} | Tracks: {len(active_logical_tracks)}"
    cv2.putText(frame, fps_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    status_text = f"VLM Status: {vlm_status}"
    status_color = (0, 215, 255) if "Analyzing" in vlm_status else (200, 200, 200)
    cv2.putText(frame, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Fire & Smoke Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: 0 (webcam), path to .mp4 file, or RTSP URL. Default: 0",
    )
    parser.add_argument(
        "--camera-id",
        default="cam0",
        help="Unique identifier for camera. Default: cam0",
    )
    parser.add_argument(
        "--route",
        choices=["auto", "local_vllm", "gemini_api"],
        default="auto",
        help="Force VLM routing strategy. Default: auto",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live detection window with bounding boxes and HUD.",
    )
    return parser.parse_args()


def _resolve_source(raw: str):
    try:
        return int(raw)
    except ValueError:
        return raw


if __name__ == "__main__":
    args = _parse_args()
    source = _resolve_source(args.source)
    route = get_routing_decision(
        force=None if args.route == "auto" else args.route
    )
    run(
        source=source,
        camera_id=args.camera_id,
        route=route,
        show=args.show,
    )
