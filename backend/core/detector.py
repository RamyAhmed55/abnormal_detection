"""
core/detector.py
─────────────────
YOLO v26 nano wrapper for fire & smoke detection.

Returns a list of Detection objects per frame. The model is loaded once
at startup and reused across all frames.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """
    Represents a single YOLO detection result.

    Attributes
    ----------
    class_name : str
        e.g. "fire" or "smoke"
    confidence : float
        Detection confidence score in [0, 1]
    bbox_xyxy : Tuple[float, float, float, float]
        Bounding box as (x1, y1, x2, y2) in pixel coordinates
    """
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox_xyxy
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }


class Detector:
    """
    Loads a YOLO v26 model and runs inference on individual frames.

    Parameters
    ----------
    weights_path : str
        Path to the .pt weights file (default: config.YOLO_WEIGHTS).
    conf_threshold : float
        Minimum confidence to keep a detection (default: config.YOLO_CONF_THRESHOLD).
    img_size : int
        Inference image size (default: config.YOLO_IMG_SIZE).
    """

    def __init__(
        self,
        weights_path: str = None,
        conf_threshold: float = None,
        img_size: int = None,
    ):
        from ultralytics import YOLO  # import here so errors are clear

        self.weights_path = weights_path or config.YOLO_WEIGHTS
        self.conf_threshold = conf_threshold or config.YOLO_CONF_THRESHOLD
        self.img_size = img_size or config.YOLO_IMG_SIZE

        logger.info("Loading YOLO model from: %s", self.weights_path)
        self._model = YOLO(self.weights_path)
        logger.info("YOLO model loaded successfully.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from OpenCV.

        Returns
        -------
        List[Detection]
            All detections above the confidence threshold.
        """
        results = self._model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=self.img_size,
            verbose=False,
        )

        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                cls_idx = int(boxes.cls[i].item())
                class_name = (
                    self._model.names.get(cls_idx, str(cls_idx))
                    if hasattr(self._model, "names")
                    else str(cls_idx)
                )

                # Filter to known classes only
                if class_name.lower() not in config.YOLO_CLASS_NAMES:
                    continue

                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                detections.append(
                    Detection(
                        class_name=class_name.lower(),
                        confidence=conf,
                        bbox_xyxy=(x1, y1, x2, y2),
                    )
                )

        return detections

    def track(self, frame: np.ndarray) -> List[Tuple[int, Detection]]:
        """
        Run YOLO + ByteTrack tracking on a frame.

        Returns
        -------
        List[Tuple[int, Detection]]
            List of (raw_track_id, Detection) pairs.
        """
        results = self._model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            imgsz=self.img_size,
            verbose=False,
        )

        tracked_detections: List[Tuple[int, Detection]] = []

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                raw_id = int(boxes.id[i].item())
                cls_idx = int(boxes.cls[i].item())
                class_name = (
                    self._model.names.get(cls_idx, str(cls_idx))
                    if hasattr(self._model, "names")
                    else str(cls_idx)
                )

                if class_name.lower() not in config.YOLO_CLASS_NAMES:
                    continue

                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                det = Detection(
                    class_name=class_name.lower(),
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                )
                tracked_detections.append((raw_id, det))

        return tracked_detections

