"""
core/clip_exporter.py
─────────────────────
Exports the rolling frame buffer to a .mp4 file on disk.

The exported clip always represents the last BUFFER_SECONDS (20s) of
footage — the moment the event was confirmed. This clip is what gets
sent to the VLM for analysis.
"""

import cv2
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


def export_clip(
    buffer_frames: List[Tuple[np.ndarray, float]],
    fps: float,
    camera_id: str,
    output_dir: str = None,
) -> str:
    """
    Write a list of (frame, timestamp) pairs to an .mp4 file.

    Parameters
    ----------
    buffer_frames : List[Tuple[np.ndarray, float]]
        Frames from StreamReader.get_buffer_frames().
    fps : float
        Frame rate used when writing the output file.
    camera_id : str
        Camera identifier, used in the output filename.
    output_dir : str | None
        Directory to save the clip in. Defaults to config.TEMP_DIR.

    Returns
    -------
    str
        Absolute path to the saved .mp4 clip.

    Raises
    ------
    ValueError
        If buffer_frames is empty.
    RuntimeError
        If the VideoWriter fails to open.
    """
    if not buffer_frames:
        raise ValueError("Cannot export clip: buffer is empty.")

    out_dir = Path(output_dir or config.TEMP_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    filename = f"clip_{camera_id}_{timestamp_str}.mp4"
    out_path = str(out_dir / filename)

    # Determine frame dimensions from the first frame
    first_frame, _ = buffer_frames[0]
    height, width = first_frame.shape[:2]

    # Use mp4v codec (widely supported; H.264 requires extra libs on Windows)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {out_path}")

    for frame, _ in buffer_frames:
        writer.write(frame)

    writer.release()
    size_mb = os.path.getsize(out_path) / 1_048_576
    logger.info(
        "Clip exported | path=%s | frames=%d | size=%.2f MB",
        out_path,
        len(buffer_frames),
        size_mb,
    )
    return out_path


def cleanup_temp_clips(max_age_seconds: int = 3600):
    """
    Delete temp clips older than max_age_seconds to avoid disk bloat.
    Called automatically at the start of each main.py run.
    """
    temp_dir = Path(config.TEMP_DIR)
    if not temp_dir.exists():
        return

    now = time.time()
    removed = 0
    for f in temp_dir.glob("clip_*.mp4"):
        if now - f.stat().st_mtime > max_age_seconds:
            f.unlink()
            removed += 1

    if removed:
        logger.info("Cleaned up %d old temp clip(s).", removed)
