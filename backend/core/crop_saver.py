"""
core/crop_saver.py
──────────────────
Utility for saving cropped object images to disk under `saved_crops/`.
Helps audit tracker decisions and build offline visual similarity archives.
"""

import os
import time
import cv2
import numpy as np
import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def save_crop(
    crop_bgr: np.ndarray,
    logical_id: int,
    camera_id: str,
    class_name: str,
    timestamp: float = None,
) -> str:
    """
    Saves a padded crop around a confirmed detection to CROPS_OUTPUT_DIR.

    Parameters
    ----------
    crop_bgr : np.ndarray
        Cropped image array (BGR).
    logical_id : int
        Stable logical track ID.
    camera_id : str
        Camera source identifier.
    class_name : str
        "fire" or "smoke".
    timestamp : float, optional
        Event timestamp. Defaults to current unix epoch.

    Returns
    -------
    str
        Absolute file path to the saved crop image.
    """
    os.makedirs(config.CROPS_OUTPUT_DIR, exist_ok=True)

    if timestamp is None:
        timestamp = time.time()

    filename = f"cam_{camera_id}_id{logical_id}_{class_name}_{int(timestamp)}.jpg"
    filepath = os.path.join(config.CROPS_OUTPUT_DIR, filename)

    if crop_bgr is not None and crop_bgr.size > 0:
        cv2.imwrite(filepath, crop_bgr)
        logger.debug("Saved crop image: %s", filepath)

    return filepath
