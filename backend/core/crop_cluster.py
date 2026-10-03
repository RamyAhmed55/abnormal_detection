"""
core/crop_cluster.py
────────────────────
Grouped Crops & Object Clustering Manager.

Problem Solved
──────────────
When tracking loses an object and assigns a new ID (or when multiple crops arrive
for the same physical object), the pipeline previously cut a 20s clip and sent it
to VLM every time. This created duplicate video clips and unnecessary VLM calls.

Solution (Notebook Cell 7 Logic)
────────────────────────────────
1. Maintains visual clusters under `grouped_crops/object_id_N/`.
2. When a confirmed detection crop arrives, extracts feature embeddings (ResNet50 / MobileNetV3)
   and computes Cosine Distance against existing object folders.
3. If Cosine Distance <= CROP_CLUSTER_EPS (0.35):
   - Classified as an OLD / EXISTING object.
   - Crop is saved to the existing object folder.
   - VLM video export & API call is CANCELLED! (Video clip deleted).
4. If Cosine Distance > CROP_CLUSTER_EPS (0.35):
   - Classified as a NEW object.
   - New folder `object_id_N` is created.
   - Video clip IS sent to VLM for analysis.
5. Automatic Maintenance:
   - 60-Minute Cleanup: Retains at most 2 crop images per object folder.
   - 48-Hour Reset: Clears all object folders and resets object ID counter after 2 days.
"""

import os
import shutil
import time
import glob
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import numpy as np

import config
from core.feature_extractor import get_feature_extractor, cosine_similarity

logger = logging.getLogger(__name__)


class CropClusterManager:
    """
    Manages object crop clustering under grouped_crops/ and decides whether
    a crop represents a NEW object (send to VLM) or OLD object (bypass VLM).
    """

    def __init__(
        self,
        grouped_dir: str = config.GROUPED_CROPS_DIR,
        eps: float = config.CROP_CLUSTER_EPS,
        cleanup_minutes: int = config.CROP_CLEANUP_INTERVAL_MINUTES,
        reset_hours: int = config.CROP_RESET_HOURS,
        max_crops_per_cluster: int = config.MAX_CROPS_PER_CLUSTER,
    ):
        self.grouped_dir = grouped_dir
        self.eps = eps
        self.cleanup_seconds = cleanup_minutes * 60
        self.reset_seconds = reset_hours * 3600
        self.max_crops_per_cluster = max_crops_per_cluster

        os.makedirs(self.grouped_dir, exist_ok=True)

        self._next_object_id = self._scan_next_object_id()
        self._cluster_embeddings: Dict[str, List[np.ndarray]] = {}
        self._cluster_metadata: Dict[str, Dict[str, str]] = {}
        self._last_cleanup_time = time.time()
        self._system_start_time = time.time()

        # Load existing cluster embeddings from disk on startup
        self._reload_clusters_from_disk()


    def _scan_next_object_id(self) -> int:
        """Scan grouped_crops directory to find the next available object ID index."""
        existing_dirs = glob.glob(os.path.join(self.grouped_dir, "object_id_*"))
        max_id = -1
        for d in existing_dirs:
            dirname = os.path.basename(d)
            try:
                idx = int(dirname.replace("object_id_", ""))
                if idx > max_id:
                    max_id = idx
            except ValueError:
                pass
        return max_id + 1

    def _reload_clusters_from_disk(self):
        """Read crop images from existing object_id_* folders and cache embeddings."""
        feature_extractor = get_feature_extractor()
        object_folders = glob.glob(os.path.join(self.grouped_dir, "object_id_*"))

        for folder in object_folders:
            object_id = os.path.basename(folder)
            image_files = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))

            embeddings = []
            for img_path in image_files[:5]:  # Read top 5 images per folder
                try:
                    import cv2
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is not None:
                        emb = feature_extractor.get_crop_embedding(img_bgr)
                        if emb is not None:
                            embeddings.append(emb)
                except Exception as err:
                    logger.warning("Error reading cluster crop %s: %s", img_path, err)

            if embeddings:
                self._cluster_embeddings[object_id] = embeddings

        logger.info("Loaded %d object clusters from %s", len(self._cluster_embeddings), self.grouped_dir)

    def reset_clusters(self):
        """Force clear all cluster embeddings in RAM and rescan disk."""
        self._cluster_embeddings.clear()
        self._cluster_metadata.clear()
        self._next_object_id = self._scan_next_object_id()
        self._reload_clusters_from_disk()

    def process_new_crop(
        self,
        crop_bgr: np.ndarray,
        class_name: str,
        camera_id: str,
        timestamp: float = None,
    ) -> Tuple[bool, str, str]:
        """
        Process a new crop image and decide if it belongs to a NEW or OLD object.

        Parameters
        ----------
        crop_bgr : np.ndarray
            Crop image array.
        class_name : str
            "fire" or "smoke".
        camera_id : str
            Camera ID identifier.
        timestamp : float, optional
            Timestamp of detection.

        Returns
        -------
        (is_new_object, object_id, crop_file_path)
            is_new_object: True if crop is NEW -> Send to VLM!
                           False if crop is OLD -> Cancel/Bypass VLM!
        """
        if timestamp is None:
            timestamp = time.time()

        # Run periodic maintenance cleanups
        self.check_and_run_cleanups(timestamp)

        # Sync memory cache with disk: purge any cluster from RAM if its folder was deleted by user
        for obj_id in list(self._cluster_embeddings.keys()):
            folder_path = os.path.join(self.grouped_dir, obj_id)
            if not os.path.exists(folder_path):
                self._cluster_embeddings.pop(obj_id, None)
                self._cluster_metadata.pop(obj_id, None)

        feature_extractor = get_feature_extractor()
        crop_emb = feature_extractor.get_crop_embedding(crop_bgr)

        best_object_id = None
        best_dist = 1.0  # Cosine distance = 1 - cosine_similarity

        if crop_emb is not None:
            for obj_id, emb_list in self._cluster_embeddings.items():
                meta = self._cluster_metadata.get(obj_id, {})
                if meta.get("class_name") and meta.get("class_name") != class_name:
                    continue
                if meta.get("camera_id") and meta.get("camera_id") != camera_id:
                    continue

                for ref_emb in emb_list:
                    sim = cosine_similarity(crop_emb, ref_emb)
                    dist = 1.0 - sim
                    if dist < best_dist:
                        best_dist = dist
                        best_object_id = obj_id

        # Distance threshold check: <= 0.35 means SAME object (old object)
        if best_object_id is not None and best_dist <= self.eps:

            # ── OLD OBJECT DETECTED ──────────────────────────────────────────
            logger.info(
                "MATCHED OLD OBJECT: crop matches cluster '%s' (dist=%.3f <= eps=%.2f) -> BYPASS VLM!",
                best_object_id, best_dist, self.eps
            )
            target_dir = os.path.join(self.grouped_dir, best_object_id)
            os.makedirs(target_dir, exist_ok=True)

            filename = f"cam_{camera_id}_{class_name}_{int(timestamp)}.jpg"
            saved_path = os.path.join(target_dir, filename)
            if crop_bgr is not None and crop_bgr.size > 0:
                import cv2
                cv2.imwrite(saved_path, crop_bgr)

            # Add crop embedding to cluster cache
            if crop_emb is not None:
                self._cluster_embeddings[best_object_id].append(crop_emb)

            return False, best_object_id, saved_path

        else:
            # ── GENUINELY NEW OBJECT DETECTED ────────────────────────────────
            new_object_id = f"object_id_{self._next_object_id}"
            self._next_object_id += 1

            logger.info(
                "NEW OBJECT DETECTED: created cluster '%s' (min_dist=%.3f > eps=%.2f) -> ROUTE TO VLM!",
                new_object_id, best_dist, self.eps
            )
            target_dir = os.path.join(self.grouped_dir, new_object_id)
            os.makedirs(target_dir, exist_ok=True)

            filename = f"cam_{camera_id}_{class_name}_{int(timestamp)}.jpg"
            saved_path = os.path.join(target_dir, filename)
            if crop_bgr is not None and crop_bgr.size > 0:
                import cv2
                cv2.imwrite(saved_path, crop_bgr)

            if crop_emb is not None:
                self._cluster_embeddings[new_object_id] = [crop_emb]

            self._cluster_metadata[new_object_id] = {
                "class_name": class_name,
                "camera_id": camera_id,
            }

            return True, new_object_id, saved_path


    def check_and_run_cleanups(self, current_time: float = None):
        """Perform 60-minute crop pruning and 48-hour full reset cleanups."""
        if current_time is None:
            current_time = time.time()

        if self._last_cleanup_time > current_time:
            self._last_cleanup_time = current_time
        if self._system_start_time > current_time:
            self._system_start_time = current_time

        # ── 1. 48-Hour Full Reset (Every 2 days) ──────────────────────────────
        if (current_time - self._system_start_time) >= self.reset_seconds:
            logger.warning("🧹 48-Hour Reset Triggered: Clearing all object cluster folders and resetting object IDs.")
            try:
                if os.path.exists(self.grouped_dir):
                    shutil.rmtree(self.grouped_dir)
                os.makedirs(self.grouped_dir, exist_ok=True)
                self._cluster_embeddings.clear()
                self._cluster_metadata.clear()
                self._next_object_id = 0
                self._system_start_time = current_time
                self._last_cleanup_time = current_time
            except Exception as err:
                logger.error("Error during 48-hour reset cleanup: %s", err)
            return

        # ── 2. 60-Minute Pruning (Keep max 2 crops per folder) ───────────────
        if (current_time - self._last_cleanup_time) >= self.cleanup_seconds:
            logger.info("🧹 60-Minute Cleanup Triggered: Pruning crop files in object folders to max %d.", self.max_crops_per_cluster)
            try:
                object_folders = glob.glob(os.path.join(self.grouped_dir, "object_id_*"))
                for folder in object_folders:
                    files = sorted(
                        glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")),
                        key=os.path.getmtime,
                    )
                    if len(files) > self.max_crops_per_cluster:
                        files_to_delete = files[:-self.max_crops_per_cluster]
                        for fpath in files_to_delete:
                            try:
                                os.remove(fpath)
                            except OSError:
                                pass
                        logger.debug("Pruned %d old crops in %s", len(files_to_delete), folder)

                self._last_cleanup_time = current_time
            except Exception as err:
                logger.error("Error during 60-minute pruning cleanup: %s", err)



# Global singleton instance
_CROP_CLUSTER_MANAGER = None


def get_crop_cluster_manager() -> CropClusterManager:
    """Singleton getter for CropClusterManager."""
    global _CROP_CLUSTER_MANAGER
    if _CROP_CLUSTER_MANAGER is None:
        _CROP_CLUSTER_MANAGER = CropClusterManager()
    return _CROP_CLUSTER_MANAGER
