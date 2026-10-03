"""
core/event_tracker.py
─────────────────────
Implements per-object IoU tracking with flicker tolerance (grace period)
and 3-second stability verification.

How it works
────────────
1. Assigns a unique track_id to each newly detected fire/smoke object.
2. Tracks objects across frames using bounding-box IoU (Intersection-over-Union).
3. Grace Period (Flicker Tolerance): If YOLO misses an object for a brief moment
   (up to config.TRACK_MAX_MISSING_SECONDS, e.g. 1.0s), the track timer is NOT reset!
4. Continuous Duration: Once a track's active duration (last_seen - first_seen)
   reaches config.DETECTION_STABILITY_SECONDS (3.0s), a stable event is fired.
5. Cooldown: Prevents re-firing on the exact same tracked object once triggered.
"""

import time
import logging
from typing import List, Optional, Tuple, Dict

import config
from core.detector import Detection

logger = logging.getLogger(__name__)


def compute_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union between two bounding boxes (x1, y1, x2, y2)."""
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


class TrackedObject:
    """
    Represents an active tracked object over consecutive frames.
    """

    def __init__(self, track_id: int, detection: Detection, timestamp: float):
        self.track_id = track_id
        self.class_name = detection.class_name
        self.bbox_xyxy = detection.bbox_xyxy
        self.confidence = detection.confidence
        self.first_seen_ts = timestamp
        self.last_seen_ts = timestamp
        self.peak_detection = detection
        self.is_triggered = False

    def update(self, detection: Detection, timestamp: float):
        self.bbox_xyxy = detection.bbox_xyxy
        self.confidence = detection.confidence
        self.last_seen_ts = timestamp
        if detection.confidence > self.peak_detection.confidence:
            self.peak_detection = detection

    def elapsed_seconds(self, current_ts: float) -> float:
        """Total time elapsed since first seen up to last seen or current timestamp."""
        return self.last_seen_ts - self.first_seen_ts


class EventTracker:
    """
    Per-object tracker that tracks detections across frames, tolerates flickering,
    and fires events when an object remains stable for >= 3 seconds.
    """

    def __init__(self):
        self._next_track_id: int = 1
        self._active_tracks: Dict[int, TrackedObject] = {}
        self._cooldown_until: float = 0.0

    def update(
        self,
        detections: List[Detection],
        timestamp: float,
    ) -> Tuple[bool, List[Detection]]:
        """
        Update tracker state with current frame detections.

        Parameters
        ----------
        detections : List[Detection]
            Detections from the current frame.
        timestamp : float
            Current stream timestamp in seconds.

        Returns
        -------
        (is_stable_event, peak_detections) : Tuple[bool, List[Detection]]
            is_stable_event — True if a track reached 3 seconds of stability.
            peak_detections — Peak detection bbox list for the triggered track.
        """
        # ── 1. Expire stale tracks beyond grace period ─────────────────────────
        expired_ids = [
            tid
            for tid, track in self._active_tracks.items()
            if (timestamp - track.last_seen_ts) > config.TRACK_MAX_MISSING_SECONDS
        ]
        for tid in expired_ids:
            logger.debug(
                "Track ID #%d (%s) lost after grace period — removing.",
                tid,
                self._active_tracks[tid].class_name,
            )
            del self._active_tracks[tid]

        # ── 2. Match current detections with active tracks ────────────────────
        unmatched_detections = list(detections)

        for track in self._active_tracks.values():
            best_iou = 0.0
            best_det_idx = -1

            for idx, det in enumerate(unmatched_detections):
                if det.class_name != track.class_name:
                    continue
                iou = compute_iou(track.bbox_xyxy, det.bbox_xyxy)
                if iou >= config.TRACKER_IOU_THRESHOLD and iou > best_iou:
                    best_iou = iou
                    best_det_idx = idx

            if best_det_idx >= 0:
                matched_det = unmatched_detections.pop(best_det_idx)
                track.update(matched_det, timestamp)

        # ── 3. Create new tracks for unmatched detections ──────────────────────
        for det in unmatched_detections:
            new_track = TrackedObject(self._next_track_id, det, timestamp)
            self._active_tracks[self._next_track_id] = new_track
            logger.debug(
                "New track created: ID #%d | class=%s | bbox=%s",
                self._next_track_id,
                det.class_name,
                det.bbox_xyxy,
            )
            self._next_track_id += 1

        # ── 4. Cooldown guard ──────────────────────────────────────────────────
        if timestamp < self._cooldown_until:
            return False, []

        # ── 5. Check if any active track reached stability threshold ───────────
        for track in self._active_tracks.values():
            if track.is_triggered:
                continue

            duration = track.elapsed_seconds(timestamp)
            if duration >= config.DETECTION_STABILITY_SECONDS:
                logger.info(
                    "⚡ Track #%d (%s) stable for %.1fs >= %.1fs threshold!",
                    track.track_id,
                    track.class_name,
                    duration,
                    config.DETECTION_STABILITY_SECONDS,
                )
                track.is_triggered = True
                self._cooldown_until = timestamp + config.EVENT_COOLDOWN_SECONDS
                return True, [track.peak_detection]

        return False, []

    @property
    def active_tracks(self) -> List[TrackedObject]:
        return list(self._active_tracks.values())

    def seconds_since_first_detection(self, current_ts: float) -> float:
        """Return maximum duration (seconds) among currently active tracks."""
        if not self._active_tracks:
            return 0.0
        return max(t.elapsed_seconds(current_ts) for t in self._active_tracks.values())

