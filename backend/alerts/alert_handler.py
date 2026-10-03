"""
alerts/alert_handler.py
────────────────────────
Packages and stores an alert event when the VLM judges a detection as abnormal.

Incident Deduplication & Anti-Spam Guard
─────────────────────────────────────────
When a fire or smoke incident is ongoing, the fire will continue burning over time.
To prevent spamming the user with duplicate alerts for the exact same ongoing accident:
- Active incidents are tracked per `(camera_id, object_id)`.
- If an alert occurs for an ongoing incident within INCIDENT_ALERT_COOLDOWN_SECONDS (5 minutes),
  the alert is logged to the DB with `suppressed=True`, but duplicate notifications to the user are blocked.
- If a NEW incident occurs on a different camera, object, or after cooldown expires, a new alert is delivered.
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Tuple

import config

logger = logging.getLogger(__name__)

# Severity numeric map for comparison
_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}
_MIN_SEVERITY_RANK = _SEVERITY_RANK.get(config.ALERT_MIN_SEVERITY, 1)

# Cache for active ongoing incidents: (camera_id, object_id) -> last_alert_timestamp
_ACTIVE_INCIDENTS: Dict[Tuple[str, str], float] = {}


def handle_alert(
    result: dict,
    clip_path: str,
    camera_id: str,
    object_id: str = "unknown_object",
    db=None,
) -> Optional[dict]:
    """
    Save an alert package to disk and log it to the database.

    Parameters
    ----------
    result : dict
        Parsed VLM output dict.
    clip_path : str
        Path to the exported .mp4 clip.
    camera_id : str
        Camera source identifier.
    object_id : str
        Grouped object cluster ID (e.g. object_id_0).
    db : ZoneDB | None
        If provided, the alert is logged to the database.

    Returns
    -------
    dict | None
        Alert package dict if severity passes filter and alert is not suppressed.
    """
    severity = result.get("severity", "low")
    if _SEVERITY_RANK.get(severity, 0) < _MIN_SEVERITY_RANK:
        logger.info(
            "Alert filtered (severity=%s < min=%s)", severity, config.ALERT_MIN_SEVERITY
        )
        return None

    now = time.time()
    incident_key = (camera_id, object_id)
    last_alert_ts = _ACTIVE_INCIDENTS.get(incident_key, 0.0)
    is_suppressed = False

    # ── Incident Deduplication Check ──────────────────────────────────────────
    if (now - last_alert_ts) < config.INCIDENT_ALERT_COOLDOWN_SECONDS:
        is_suppressed = True
        logger.info(
            "🚨 ONGOING INCIDENT: Fire/smoke active for '%s' on %s. Suppressing duplicate user alert.",
            object_id, camera_id
        )
    else:
        _ACTIVE_INCIDENTS[incident_key] = now

    # ── Create alert directory ─────────────────────────────────────────────────
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    alert_dir = Path(config.ALERTS_DIR) / f"{camera_id}_{object_id}_{timestamp_str}"
    alert_dir.mkdir(parents=True, exist_ok=True)

    # ── Copy clip ──────────────────────────────────────────────────────────────
    dest_clip = str(alert_dir / "clip.mp4")
    if os.path.exists(clip_path):
        shutil.copy2(clip_path, dest_clip)

    # ── Write result JSON ──────────────────────────────────────────────────────
    result_copy = dict(result)
    result_copy["object_id"] = object_id
    result_copy["suppressed"] = is_suppressed

    result_path = alert_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_copy, f, indent=2, ensure_ascii=False)

    # ── Write human-readable message ───────────────────────────────────────────
    message = _build_message(result_copy, camera_id, object_id, timestamp_str, is_suppressed)
    msg_path = alert_dir / "alert_message.txt"
    msg_path.write_text(message, encoding="utf-8")

    # ── Log to database ────────────────────────────────────────────────────────
    if db is not None:
        db.log_alert(
            camera_id=camera_id,
            result=result_copy,
            clip_path=dest_clip,
            alert_dir=str(alert_dir),
            router=result.get("_router", "unknown"),
        )

    # ── Build and return alert package ────────────────────────────────────────
    package = {
        "camera_id": camera_id,
        "object_id": object_id,
        "timestamp": timestamp_str,
        "alert_dir": str(alert_dir),
        "clip_path": dest_clip,
        "result_json_path": str(result_path),
        "message_path": str(msg_path),
        "message": message,
        "is_abnormal": result.get("is_abnormal", False),
        "event_type": result.get("event_type", "unknown"),
        "severity": severity,
        "confidence": result.get("confidence", 0.0),
        "description": result.get("description", ""),
        "router": result.get("_router", "unknown"),
        "suppressed": is_suppressed,
    }

    if is_suppressed:
        logger.info("Alert package logged to DB but suppressed from user delivery (ongoing incident).")
        return package

    logger.warning(
        "🚨 ALERT DELIVERED | camera=%s | object=%s | event=%s | severity=%s",
        camera_id, object_id, result.get("event_type"), severity
    )
    return package


def _build_message(
    result: dict, camera_id: str, object_id: str, timestamp_str: str, suppressed: bool
) -> str:
    """Build human-readable alert message."""
    is_abnormal = result.get("is_abnormal", False)
    event_type = result.get("event_type", "unknown").upper()
    severity = result.get("severity", "unknown").upper()
    confidence = result.get("confidence", 0.0)
    description = result.get("description", "No description provided.")
    reasoning = result.get("reasoning", "")
    router = result.get("_router", "unknown")

    status_icon = "🚨" if is_abnormal else "✅"
    severity_icon = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴"}.get(severity, "⚪")

    suppressed_banner = "\n[NOTE: ONGOING INCIDENT — USER NOTIFICATION SUPPRESSED]\n" if suppressed else ""

    lines = [
        f"{status_icon} FIRE & SMOKE DETECTION ALERT{suppressed_banner}",
        f"{'─' * 45}",
        f"Camera     : {camera_id}",
        f"Object ID  : {object_id}",
        f"Timestamp  : {timestamp_str}",
        f"Event      : {event_type}",
        f"Severity   : {severity_icon} {severity}",
        f"Confidence : {confidence:.0%}",
        f"Analyzed by: {router}",
        f"{'─' * 45}",
        f"Description:",
        f"  {description}",
        "",
        f"Reasoning:",
        f"  {reasoning}",
        f"{'─' * 45}",
        "Action: Check the video clip in the alert directory.",
    ]

    return "\n".join(lines)
