-- db/schema.sql
-- SQLite schema for the fire/smoke detection system.
-- Applied automatically by zone_db.py on first run.

-- ─────────────────────────────────────────────────────────────────────────────
-- known_normal_zones
-- Stores bounding boxes of objects that have been verified as NORMAL
-- (e.g. factory chimney, pilot light). When a new detection's IoU with
-- any row here is >= IOU_OVERLAP_THRESHOLD, we skip the VLM pipeline.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS known_normal_zones (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT    NOT NULL,
    class_name  TEXT    NOT NULL,   -- "fire" or "smoke"
    x1          REAL    NOT NULL,
    y1          REAL    NOT NULL,
    x2          REAL    NOT NULL,
    y2          REAL    NOT NULL,
    first_seen  DATETIME NOT NULL DEFAULT (datetime('now')),
    last_seen   DATETIME NOT NULL DEFAULT (datetime('now')),
    hit_count   INTEGER NOT NULL DEFAULT 1,
    note        TEXT                -- optional human label
);

CREATE INDEX IF NOT EXISTS idx_known_zones_camera
    ON known_normal_zones (camera_id, class_name);

-- ─────────────────────────────────────────────────────────────────────────────
-- alert_events
-- Logs every event that was judged ABNORMAL by the VLM pipeline.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alert_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT    NOT NULL,
    timestamp   DATETIME NOT NULL DEFAULT (datetime('now')),
    event_type  TEXT    NOT NULL,   -- "fire" | "smoke" | "false_alarm" | "other"
    severity    TEXT    NOT NULL,   -- "low" | "medium" | "high"
    confidence  REAL    NOT NULL,
    is_abnormal INTEGER NOT NULL,   -- 1 = abnormal, 0 = normal
    description TEXT,
    reasoning   TEXT,
    clip_path   TEXT,               -- path to the saved 20-second clip
    alert_dir   TEXT,               -- path to the full alert package folder
    router      TEXT                -- "local_vllm" or "gemini_api"
);

CREATE INDEX IF NOT EXISTS idx_alert_events_camera
    ON alert_events (camera_id, timestamp);
