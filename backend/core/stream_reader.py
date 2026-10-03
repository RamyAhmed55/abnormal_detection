"""
core/stream_reader.py
─────────────────────
Handles reading frames from any video source (webcam, file, RTSP) and
maintains a rolling 20-second frame buffer.

How the buffer works
────────────────────
Every time a frame is read, it is timestamped and appended to a deque.
Frames older than BUFFER_SECONDS are automatically dropped from the left.
When an event fires, call get_buffer_clip() to export those frames to disk.
"""

import cv2
import time
import logging
import os
from collections import deque
from pathlib import Path
from typing import Generator, Tuple, List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


class StreamReader:
    """
    Reads frames from a video source and keeps a rolling frame buffer.

    Parameters
    ----------
    source : int | str
        Camera index (0 for default webcam), path to a video file, or
        RTSP/HTTP stream URL.
    camera_id : str
        Human-readable identifier for this camera, used in filenames and DB.
    """

    def __init__(self, source, camera_id: str = "cam0"):
        self.source = source
        self.camera_id = camera_id

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source: {source!r}. "
                "For webcam use 0, for file pass the full path."
            )

        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # For video files: total duration in seconds (0 = live stream / unknown)
        total_frames = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration: float = total_frames / self.fps if total_frames > 0 else 0.0

        # Rolling buffer: each entry is (frame_ndarray, timestamp_float)
        max_frames = int(self.fps * config.BUFFER_SECONDS) + int(self.fps)
        self._buffer: deque = deque(maxlen=max_frames)

        self._is_file = isinstance(source, str) and os.path.isfile(source)

        logger.info(
            "StreamReader opened | source=%r | fps=%.1f | size=%dx%d | camera_id=%s",
            source, self.fps, self.width, self.height, camera_id,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def read(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generator that yields (frame, timestamp) continuously.

        For live streams: timestamp = wall-clock time (time.time()).
        For video files:  timestamp = presentation time in seconds.

        The rolling buffer is updated on every yield.
        """
        while True:
            ret, frame = self._cap.read()
            if not ret:
                if self._is_file:
                    logger.info("End of video file reached: %s", self.source)
                    break
                else:
                    # Live stream blip — try to recover
                    logger.warning("Frame grab failed, retrying in 0.1s …")
                    time.sleep(0.1)
                    continue

            # Timestamp
            if self._is_file:
                ts = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:
                ts = time.time()

            # Maintain rolling buffer
            self._buffer.append((frame, ts))
            self._drop_old_frames(ts)

            yield frame, ts

    def get_buffer_frames(self) -> List[Tuple[np.ndarray, float]]:
        """
        Returns a snapshot of the current rolling buffer contents as a list
        of (frame, timestamp) tuples ordered oldest→newest.
        """
        return list(self._buffer)

    def release(self):
        """Release the underlying VideoCapture."""
        self._cap.release()
        logger.info("StreamReader released: %s", self.source)

    # ─────────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────────

    def _drop_old_frames(self, current_ts: float):
        """Remove frames from the left of the buffer that are older than BUFFER_SECONDS."""
        cutoff = current_ts - config.BUFFER_SECONDS
        while self._buffer and self._buffer[0][1] < cutoff:
            self._buffer.popleft()
