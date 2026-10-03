"""
db/zone_db.py
─────────────
SQLite interface for:
  1. Checking whether a new detection overlaps a known-normal zone (IoU).
  2. Saving a new known-normal zone when VLM says the event is normal.
  3. Logging alert events when VLM says the event is abnormal.

IoU Overlap Logic
─────────────────
Two bounding boxes refer to the "same object" when their
Intersection-over-Union (IoU) is >= config.IOU_OVERLAP_THRESHOLD (0.3).

IoU = Area(Intersection) / Area(Union)

A value of 0.3 is deliberately lenient because:
  - Camera angle may shift slightly over time.
  - Object (e.g. chimney plume) grows/shrinks between frames.
  - We want to avoid re-alerting the same source repeatedly.
"""

import sqlite3
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import config
from core.detector import Detection

logger = logging.getLogger(__name__)

# Path to the SQL schema file (same directory as this file)
_SCHEMA_FILE = Path(__file__).parent / "schema.sql"


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """
    Compute Intersection-over-Union between two bounding boxes.

    Parameters
    ----------
    box_a, box_b : (x1, y1, x2, y2)

    Returns
    -------
    float in [0, 1]
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


# ─────────────────────────────────────────────────────────────────────────────
# Database class
# ─────────────────────────────────────────────────────────────────────────────

class ZoneDB:
    """
    Manages the SQLite database for known-normal zones and alert events.

    Usage
    -----
    db = ZoneDB()
    db.initialize()   # call once at startup
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def initialize(self):
        """Apply schema.sql to create tables if they don't exist."""
        schema = _SCHEMA_FILE.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(schema)
        logger.info("Database initialized at: %s", self.db_path)

    # ── Known-Normal Zone checks ──────────────────────────────────────────────

    def is_known_normal(
        self,
        camera_id: str,
        detections: List[Detection],
    ) -> bool:
        """
        Return True if ALL of the given detections match a known-normal zone
        for this camera (IoU >= threshold).

        If even one detection is NEW (no matching zone), returns False so the
        event is routed to the VLM.
        """
        # Expire stale zones first
        self._expire_old_zones()

        for det in detections:
            if not self._detection_matches_zone(camera_id, det):
                return False
        return True

    def save_known_zone(
        self,
        camera_id: str,
        detections: List[Detection],
        note: str = "",
    ):
        """
        Save detections as known-normal zones (after VLM says 'normal').
        If a zone already exists with high IoU, update last_seen + hit_count.
        """
        for det in detections:
            existing_id = self._find_matching_zone_id(camera_id, det)
            if existing_id is not None:
                self._update_zone(existing_id)
                logger.debug(
                    "Updated known zone id=%d for camera=%s class=%s",
                    existing_id, camera_id, det.class_name,
                )
            else:
                self._insert_zone(camera_id, det, note)
                logger.info(
                    "New known-normal zone saved: camera=%s class=%s bbox=%s",
                    camera_id, det.class_name, det.bbox_xyxy,
                )

    # ── Alert event logging ───────────────────────────────────────────────────

    def log_alert(
        self,
        camera_id: str,
        result: dict,
        clip_path: str,
        alert_dir: str,
        router: str,
    ) -> int:
        """
        Insert a row into alert_events.

        Parameters
        ----------
        result : dict
            Parsed VLM output (keys: is_abnormal, event_type, severity,
            confidence, description, reasoning).
        router : str
            "local_vllm" or "gemini_api"

        Returns
        -------
        int : Row ID of the inserted alert.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO alert_events
                    (camera_id, event_type, severity, confidence,
                     is_abnormal, description, reasoning, clip_path,
                     alert_dir, router)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id,
                    result.get("event_type", "unknown"),
                    result.get("severity", "low"),
                    result.get("confidence", 0.0),
                    1 if result.get("is_abnormal") else 0,
                    result.get("description", ""),
                    result.get("reasoning", ""),
                    clip_path,
                    alert_dir,
                    router,
                ),
            )
            row_id = cursor.lastrowid
        logger.info("Alert logged to DB: id=%d camera=%s", row_id, camera_id)
        return row_id

    def get_recent_alerts(self, limit: int = 20) -> List[dict]:
        """Return the most recent alert events."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM alert_events ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Internals ─────────────────────────────────────────────────────────────

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _detection_matches_zone(self, camera_id: str, det: Detection) -> bool:
        """Return True if the detection overlaps any known zone by IoU >= threshold."""
        return self._find_matching_zone_id(camera_id, det) is not None

    def _find_matching_zone_id(
        self, camera_id: str, det: Detection
    ) -> Optional[int]:
        """Return the ID of the first matching zone, or None."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, x1, y1, x2, y2 FROM known_normal_zones
                WHERE camera_id = ? AND class_name = ?
                """,
                (camera_id, det.class_name),
            ).fetchall()

        for row in rows:
            iou = _compute_iou(
                det.bbox_xyxy,
                (row["x1"], row["y1"], row["x2"], row["y2"]),
            )
            if iou >= config.IOU_OVERLAP_THRESHOLD:
                return row["id"]

        return None

    def _insert_zone(self, camera_id: str, det: Detection, note: str):
        x1, y1, x2, y2 = det.bbox_xyxy
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO known_normal_zones
                    (camera_id, class_name, x1, y1, x2, y2, note)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (camera_id, det.class_name, x1, y1, x2, y2, note),
            )

    def _update_zone(self, zone_id: int):
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE known_normal_zones
                SET last_seen = datetime('now'),
                    hit_count = hit_count + 1
                WHERE id = ?
                """,
                (zone_id,),
            )

    def _expire_old_zones(self):
        """Remove zones not seen for KNOWN_ZONE_EXPIRY_DAYS days."""
        cutoff = (
            datetime.utcnow() - timedelta(days=config.KNOWN_ZONE_EXPIRY_DAYS)
        ).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM known_normal_zones WHERE last_seen < ?",
                (cutoff,),
            )
