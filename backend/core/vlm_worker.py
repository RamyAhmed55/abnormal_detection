"""
core/vlm_worker.py
──────────────────
Async Background Queue Worker for VLM Clip Export, Clustering & Analysis.

Problem Solved
──────────────
1. Running VLM inference or cloud API calls synchronously blocks the main video
   reading loop, freezing the OpenCV playback window for 2 to 10 seconds.
2. Multiple video clips were being exported and sent to VLM for the SAME physical
   object when tracking was re-established.

Solution
────────
VLMWorker runs in a background daemon thread:
1. Non-blockingly receives confirmed event payloads from main video loop.
2. Checks CropClusterManager (Notebook Cell 7 logic):
   - If crop matches an EXISTING object cluster folder (old object) -> CANCEL VLM process & delete video!
   - If crop is a NEW object cluster -> Export 20s clip, route to VLM, and check Incident Anti-Spam Guard!
"""

import queue
import threading
import logging
import time
import os
from typing import Dict, Any, List

import config
from core.clip_exporter import export_clip
from core.crop_cluster import get_crop_cluster_manager
from db.zone_db import ZoneDB
from alerts.alert_handler import handle_alert

logger = logging.getLogger(__name__)


def _run_vlm(clip_path: str, route: str, camera_id: str) -> dict:
    """Route clip to VLM (Local Qwen or Gemini API)."""
    if route == "local_vllm":
        from vllm.analyzer import analyze_clip
        from vllm.confidence_check import needs_escalation

        logger.info("Sending clip to local VLLM (Qwen2.5-VL-7B)…")
        result = analyze_clip(clip_path, camera_id=camera_id)

        if needs_escalation(result):
            logger.info("Low confidence — escalating to Gemini API…")
            from api.gemini_analyzer import analyze_clip_gemini
            result = analyze_clip_gemini(clip_path, camera_id=camera_id)

        return result
    else:  # gemini_api / auto
        from api.gemini_analyzer import analyze_clip_gemini
        logger.info("Sending clip to Gemini API…")
        return analyze_clip_gemini(clip_path, camera_id=camera_id)


class VLMWorker:
    """
    Background worker thread that processes VLM jobs asynchronously.
    """

    def __init__(self, db: ZoneDB, route: str = "auto"):
        self.db = db
        self.route = route
        self._job_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._process_queue,
            name="VLMWorkerThread",
            daemon=True,
        )
        self.is_busy = False
        self.last_result: Dict[str, Any] = {}
        self.latest_status_message: str = "Idle"

    def start(self):
        """Start the background worker thread."""
        self._stop_event.clear()
        self._worker_thread.start()
        logger.info("VLM Worker background thread started.")

    def stop(self):
        """Signal worker thread to shut down."""
        self._stop_event.set()
        self._job_queue.put(None)  # Sentinel to unblock queue.get()

    def enqueue(
        self,
        buffer_frames: List[Any],
        fps: float,
        camera_id: str,
        class_name: str,
        logical_id: int,
        bbox: tuple,
        crop_bgr: Any,
    ):
        """
        Non-blockingly enqueue a VLM task payload.
        """
        payload = {
            "buffer_frames": buffer_frames,
            "fps": fps,
            "camera_id": camera_id,
            "class_name": class_name,
            "logical_id": logical_id,
            "bbox": bbox,
            "crop_bgr": crop_bgr,
            "timestamp": time.time(),
        }
        self._job_queue.put(payload)
        logger.info("Enqueued VLM task for LogicalTrack L%d (queue size=%d)", logical_id, self._job_queue.qsize())

    def _process_queue(self):
        cluster_manager = get_crop_cluster_manager()

        while not self._stop_event.is_set():
            try:
                job = self._job_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if job is None:
                break

            self.is_busy = True
            camera_id = job["camera_id"]
            logical_id = job["logical_id"]
            class_name = job["class_name"]
            crop_bgr = job["crop_bgr"]

            logger.info("Worker started processing event L%d (%s)...", logical_id, class_name)

            try:
                # ── 1. Check Crop Cluster Deduplication (Notebook Cell 7 Logic) ──
                is_new_object, object_id, saved_crop = cluster_manager.process_new_crop(
                    crop_bgr=crop_bgr,
                    class_name=class_name,
                    camera_id=camera_id,
                    timestamp=job["timestamp"],
                )

                if not is_new_object:
                    # ── OLD OBJECT -> BYPASS VLM & CANCEL VIDEO EXPORT! ────────
                    logger.info(
                        "🛑 Track L%d matched existing object cluster '%s' -> VLM analysis BYPASSED!",
                        logical_id, object_id,
                    )
                    self.latest_status_message = f"L{logical_id} matched {object_id} (VLM Bypassed)"
                    continue

                # ── 2. NEW OBJECT -> Export 20s clip & Route to VLM! ─────────
                self.latest_status_message = f"Analyzing L{logical_id} ({object_id})..."
                logger.info("⚡ NEW OBJECT '%s' (Track L%d) -> Exporting clip & Routing to VLM...", object_id, logical_id)

                clip_path = export_clip(
                    buffer_frames=job["buffer_frames"],
                    fps=job["fps"],
                    camera_id=camera_id,
                )

                # Check if detection matches known-normal zone in SQLite DB
                from core.detector import Detection
                det = Detection(class_name=class_name, confidence=1.0, bbox_xyxy=job["bbox"])

                if self.db.is_known_normal(camera_id, [det]):
                    logger.info("Track L%d matches known-normal zone in SQLite DB — skipping VLM.", logical_id)
                    self.db.save_known_zone(camera_id, [det])
                    self.latest_status_message = f"L{logical_id} matched Known Normal Zone"
                    if os.path.exists(clip_path):
                        os.remove(clip_path)
                    continue

                # 3. Route clip to VLM
                result = _run_vlm(clip_path, self.route, camera_id)
                self.last_result = result

                if "error" in result:
                    logger.error("VLM analysis error: %s", result.get("error"))
                    self.latest_status_message = f"Error: {result.get('error')}"
                else:
                    is_abnormal = result.get("is_abnormal", False)
                    event_type = result.get("event_type", class_name)

                    if is_abnormal:
                        package = handle_alert(
                            result=result,
                            clip_path=clip_path,
                            camera_id=camera_id,
                            object_id=object_id,
                            db=self.db,
                        )
                        if package and package.get("suppressed"):
                            self.latest_status_message = f"⚠️ Ongoing Incident: L{logical_id} ({object_id})"
                        else:
                            self.latest_status_message = f"🚨 ALERT! L{logical_id} {event_type.upper()}"
                            logger.warning("🚨 ABNORMAL EVENT DETECTED for Track L%d: %s", logical_id, result.get("description"))
                    else:
                        logger.info("✅ Normal event confirmed for Track L%d — saving zone to DB.", logical_id)
                        self.db.save_known_zone(camera_id, [det])
                        self.latest_status_message = f"✅ Normal: L{logical_id} saved to DB"

                # Clean up temporary clip file if not saved in an alert package
                if os.path.exists(clip_path):
                    try:
                        os.remove(clip_path)
                    except OSError:
                        pass

            except Exception as err:
                logger.exception("Unexpected error in VLM worker processing: %s", err)
                self.latest_status_message = f"Worker Error: {err}"
            finally:
                self.is_busy = False
                self._job_queue.task_done()
