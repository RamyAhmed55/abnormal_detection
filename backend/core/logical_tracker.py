"""
core/logical_tracker.py
────────────────────────
FireSmokeTrackManager implements Kalman Filter-based Persistent Tracking.

Why Kalman Filter Position Tracking:
Fire & smoke sources are stationary or semi-stationary. Visual appearance
changes chaotically frame to frame, making appearance matching unreliable.
The Kalman Filter predicts expected object position [cx, cy, w, h] and velocity,
allowing stable re-identification across multi-second gaps (e.g. 7-10s) based
on predicted spatial position alone.
"""

import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    # Custom lightweight KalmanFilter fallback if filterpy is missing
    class KalmanFilter:
        def __init__(self, dim_x: int, dim_z: int):
            self.dim_x = dim_x
            self.dim_z = dim_z
            self.x = np.zeros((dim_x,), dtype=np.float64)
            self.F = np.eye(dim_x, dtype=np.float64)
            self.H = np.zeros((dim_z, dim_x), dtype=np.float64)
            self.P = np.eye(dim_x, dtype=np.float64)
            self.R = np.eye(dim_z, dtype=np.float64)
            self.Q = np.eye(dim_x, dtype=np.float64)

        def predict(self):
            self.x = np.dot(self.F, self.x)
            self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        def update(self, z: np.ndarray):
            y = z - np.dot(self.H, self.x)
            S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.pinv(S))
            self.x = self.x + np.dot(K, y)
            I = np.eye(self.dim_x)
            self.P = np.dot(I - np.dot(K, self.H), self.P)

import config

logger = logging.getLogger(__name__)


def create_kalman_filter(bbox: Tuple[float, float, float, float]) -> KalmanFilter:
    """
    State vector: [cx, cy, w, h, vcx, vcy, vw, vh]
    Measurement: [cx, cy, w, h]
    """
    kf = KalmanFilter(dim_x=8, dim_z=4)

    x1, y1, x2, y2 = bbox
    cx, cy, w, h = (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1), max(1.0, y2 - y1)

    kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)

    kf.F = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float64)

    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.float64)

    kf.P *= 10.0
    kf.R *= 5.0
    kf.Q = np.eye(8) * 0.5

    return kf


def kalman_predicted_bbox(kf: KalmanFilter) -> Tuple[float, float, float, float]:
    """Extract (x1, y1, x2, y2) from predicted Kalman state."""
    cx, cy, w, h = float(kf.x[0]), float(kf.x[1]), float(kf.x[2]), float(kf.x[3])
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


@dataclass
class LogicalTrack:
    """
    Represents a stable logical track (e.g. L1, L2) that survives raw ID switches.
    """
    logical_id: int
    raw_track_ids: Set[int]
    class_name: str
    camera_id: str
    bbox: Tuple[float, float, float, float]
    kalman: KalmanFilter
    first_seen: float
    last_seen: float
    consecutive_missed: int = 0
    is_lost: bool = False
    lost_at: Optional[float] = None
    confirmed_sent_to_vlm: bool = False
    saved_crop_path: Optional[str] = None
    frames_since_update: int = 0

    @property
    def active_duration(self) -> float:
        return self.last_seen - self.first_seen


class FireSmokeTrackManager:
    """
    Manages active tracks using Kalman Filter predictions for position-based Re-ID.
    """

    def __init__(
        self,
        confirmation_seconds: float = config.DETECTION_STABILITY_SECONDS,
        max_missed_seconds: float = config.MAX_MISSED_SECONDS,
        reid_max_distance_px: float = config.REID_MAX_DISTANCE_PX,
        assumed_fps: float = config.ASSUMED_FPS,
    ):
        self.confirmation_seconds = confirmation_seconds
        self.max_missed_seconds = max_missed_seconds
        self.reid_max_distance_px = reid_max_distance_px
        self.assumed_fps = assumed_fps

        self.active_tracks: Dict[int, LogicalTrack] = {}  # raw_track_id -> LogicalTrack
        self.recently_lost: List[LogicalTrack] = []      # Lost tracks candidates
        self._next_logical_id = 1

    @staticmethod
    def _box_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _try_reidentify(
        self,
        camera_id: str,
        class_name: str,
        bbox: Tuple[float, float, float, float],
        now: float,
    ) -> Optional[LogicalTrack]:
        """
        Match against Kalman-predicted position of recently lost tracks.
        """
        best_match = None
        best_dist = float("inf")

        for lost_track in self.recently_lost:
            if lost_track.camera_id != camera_id or lost_track.class_name != class_name:
                continue

            if lost_track.lost_at is not None and (now - lost_track.lost_at) > self.max_missed_seconds:
                continue

            predicted_bbox = kalman_predicted_bbox(lost_track.kalman)
            dist = self._distance(self._box_center(bbox), self._box_center(predicted_bbox))

            if dist <= self.reid_max_distance_px and dist < best_dist:
                best_dist = dist
                best_match = lost_track

        if best_match is not None:
            logger.info(
                "🔄 Kalman Position Re-ID Match! Re-attaching to LogicalTrack L%d (dist=%.1fpx)",
                best_match.logical_id, best_dist,
            )

        return best_match

    def update(
        self,
        raw_track_id: int,
        camera_id: str,
        class_name: str,
        bbox: Tuple[float, float, float, float],
        now: float = None,
    ) -> Tuple[LogicalTrack, str]:
        """
        Update tracker with frame detection.

        Returns
        -------
        (logical_track, status)
        status: "pending" | "confirmed_new" | "continuing"
        """
        if now is None:
            now = time.time()

        x1, y1, x2, y2 = bbox
        cx, cy, w, h = (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1), max(1.0, y2 - y1)
        z_measurement = np.array([cx, cy, w, h], dtype=np.float64)

        if raw_track_id in self.active_tracks:
            track = self.active_tracks[raw_track_id]
            track.kalman.predict()
            track.kalman.update(z_measurement)
            track.bbox = bbox
            track.last_seen = now
            track.consecutive_missed = 0
            track.frames_since_update = 0
            status = "continuing"

        else:
            match = self._try_reidentify(camera_id, class_name, bbox, now)

            if match is not None:
                match.kalman.predict()
                match.kalman.update(z_measurement)
                match.raw_track_ids.add(raw_track_id)
                match.bbox = bbox
                match.last_seen = now
                match.consecutive_missed = 0
                match.frames_since_update = 0
                match.is_lost = False
                match.lost_at = None

                if match in self.recently_lost:
                    self.recently_lost.remove(match)

                self.active_tracks[raw_track_id] = match
                track = match
                status = "continuing"
            else:
                track = LogicalTrack(
                    logical_id=self._next_logical_id,
                    raw_track_ids={raw_track_id},
                    class_name=class_name,
                    camera_id=camera_id,
                    bbox=bbox,
                    kalman=create_kalman_filter(bbox),
                    first_seen=now,
                    last_seen=now,
                )
                self._next_logical_id += 1
                self.active_tracks[raw_track_id] = track
                status = "pending"

        if track.confirmed_sent_to_vlm:
            return track, "continuing"

        elapsed = track.last_seen - track.first_seen
        if elapsed >= self.confirmation_seconds and status != "pending":
            track.confirmed_sent_to_vlm = True
            logger.info("⚡ Track L%d confirmed stable after %.1fs!", track.logical_id, elapsed)
            return track, "confirmed_new"

        return track, "pending" if not track.confirmed_sent_to_vlm else "continuing"

    def mark_missed(self, raw_track_id: int, now: float = None):
        """Called for active raw tracks not detected in the current frame."""
        if now is None:
            now = time.time()

        if raw_track_id not in self.active_tracks:
            return

        track = self.active_tracks[raw_track_id]
        track.consecutive_missed += 1
        track.frames_since_update += 1
        track.kalman.predict()  # Extrapolate position through the gap

        if track.consecutive_missed >= 5 and not track.is_lost:
            track.is_lost = True
            track.lost_at = now
            del self.active_tracks[raw_track_id]
            self.recently_lost.append(track)
            logger.debug("LogicalTrack L%d moved to recently_lost.", track.logical_id)

        # Remove tracks lost longer than max_missed_seconds
        self.recently_lost = [
            t for t in self.recently_lost
            if t.lost_at is not None and (now - t.lost_at) <= self.max_missed_seconds
        ]

    def get_all_active_logical_tracks(self) -> List[LogicalTrack]:
        seen = set()
        unique_tracks = []
        for track in self.active_tracks.values():
            if track.logical_id not in seen:
                seen.add(track.logical_id)
                unique_tracks.append(track)
        return unique_tracks
