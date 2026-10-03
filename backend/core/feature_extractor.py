"""
core/feature_extractor.py
─────────────────────────
Lightweight visual feature extractor using pretrained MobileNetV3-Small.
Extracts 576-dimensional appearance embeddings from cropped detection regions
to enable robust Re-Identification (Re-ID) across flickering / ID switches.
"""

import logging
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T

import config

logger = logging.getLogger(__name__)

# Global singleton feature extractor instance
_FEATURE_EXTRACTOR = None


class FeatureExtractor:
    """
    Pretrained MobileNetV3-Small feature extractor (frozen classifier).
    Converts image crops (BGR) into normalized 1D embedding vectors.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Initializing FeatureExtractor (MobileNetV3-Small) on device: %s", self.device)

        # Load MobileNetV3-Small backbone with default pretrained weights
        weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT
        backbone = tv_models.mobilenet_v3_small(weights=weights)
        backbone.classifier = nn.Identity()  # Strip classification head, retain feature pool
        backbone.eval()
        backbone.to(self.device)

        self._backbone = backbone

        # Image preprocessing pipeline
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def get_crop_embedding(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a normalized 576-dim feature vector from a BGR crop.

        Parameters
        ----------
        crop_bgr : np.ndarray
            OpenCV BGR image crop.

        Returns
        -------
        np.ndarray (shape: (576,)) or None if crop is empty.
        """
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
            return None

        try:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            tensor = self._transform(crop_rgb).unsqueeze(0).to(self.device)
            embedding = self._backbone(tensor).cpu().numpy().flatten()

            # L2 normalize for fast cosine distance via dot product
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding
        except Exception as err:
            logger.warning("Error computing crop embedding: %s", err)
            return None


def get_feature_extractor() -> FeatureExtractor:
    """Singleton getter for FeatureExtractor."""
    global _FEATURE_EXTRACTOR
    if _FEATURE_EXTRACTOR is None:
        _FEATURE_EXTRACTOR = FeatureExtractor()
    return _FEATURE_EXTRACTOR


def crop_with_padding(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    padding_ratio: float = config.CROP_PADDING_RATIO,
) -> np.ndarray:
    """
    Crop the detection region from frame with margin padding around it.

    Parameters
    ----------
    frame : np.ndarray
        Source full video frame (BGR).
    bbox : (x1, y1, x2, y2)
        Bounding box coordinates.
    padding_ratio : float
        Percentage of box dimensions to extend as spatial margin context.

    Returns
    -------
    np.ndarray
        Cropped BGR image array.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1

    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    px1 = max(0, int(x1 - pad_x))
    py1 = max(0, int(y1 - pad_y))
    px2 = min(w, int(x2 + pad_x))
    py2 = min(h, int(y2 + pad_y))

    crop = frame[py1:py2, px1:px2]
    if crop.size == 0:
        return frame[int(y1):int(y2), int(x1):int(x2)]
    return crop


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """
    Compute cosine similarity between two 1D embedding vectors.
    Since embeddings are L2 normalized, cosine sim is simply dot product.
    """
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
