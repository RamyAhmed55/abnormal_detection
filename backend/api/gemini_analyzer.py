"""
api/gemini_analyzer.py
───────────────────────
Sends a video clip to the Gemini API for analysis.

Used in three cases:
  1. No CUDA GPU detected on startup.
  2. Free VRAM is below the required threshold.
  3. Local VLLM confidence was too low (escalation).

The same JUDGE_PROMPT is used as with the local model, ensuring consistent
output format regardless of which backend is used.

Gemini Video API notes
──────────────────────
- Files are uploaded first via the Files API (google.genai).
- The upload returns a file object that is then referenced in the message.
- Uploaded files are automatically deleted by Google after 48 hours.
- We also delete them immediately after use to avoid quota buildup.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import config
from vllm.prompts import build_prompt

logger = logging.getLogger(__name__)


def analyze_clip_gemini(
    clip_path: str,
    camera_id: str = "",
    location_hint: str = "",
) -> dict:
    """
    Upload a video clip to Gemini and return the analysis result.

    Parameters
    ----------
    clip_path : str
        Path to the local .mp4 clip file.
    camera_id : str
        Camera identifier for context injection.
    location_hint : str
        Optional location description.

    Returns
    -------
    dict
        Parsed result with keys: description, is_abnormal, event_type,
        severity, confidence, reasoning.
        On error: {"error": ..., "raw_output": ...}.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai is not installed. Run: pip install google-genai"
        )

    # Load API key from environment (dotenv should be loaded by main.py)
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment. Check your .env file."
        )

    # Initialize the new Client object
    client = genai.Client(api_key=api_key)

    prompt = build_prompt(camera_id=camera_id, location_hint=location_hint)
    uploaded_file = None

    for attempt in range(1, config.GEMINI_MAX_RETRIES + 1):
        try:
            # ── Upload video ──────────────────────────────────────────────────
            logger.info(
                "Uploading clip to Gemini Files API (attempt %d): %s",
                attempt, clip_path,
            )
            uploaded_file = client.files.upload(
                file=clip_path,
                config=types.UploadFileConfig(mime_type="video/mp4"),
            )

            # Wait for processing (Gemini needs to transcode the video)
            _wait_for_file_active(client, uploaded_file)

            # ── Run inference ─────────────────────────────────────────────────
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )

            raw_text = response.text
            logger.debug("Gemini raw output:\n%s", raw_text)

            result = _parse_output(raw_text)
            result["_router"] = "gemini_api"
            return result

        except Exception as exc:
            logger.warning("Gemini API attempt %d failed: %s", attempt, exc)
            if attempt < config.GEMINI_MAX_RETRIES:
                time.sleep(config.GEMINI_RETRY_DELAY)
            else:
                logger.error("All Gemini retries exhausted.")
                return {"error": str(exc), "raw_output": ""}

        finally:
            # Always clean up the uploaded file to free quota
            if uploaded_file is not None:
                try:
                    client.files.delete(name=uploaded_file.name)
                    logger.debug("Deleted Gemini file: %s", uploaded_file.name)
                except Exception:
                    pass  # Non-critical


def _wait_for_file_active(client, uploaded_file, timeout: int = 120):
    """
    Poll until the uploaded file reaches ACTIVE state.
    Gemini needs time to transcode video files before they can be used.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        file_info = client.files.get(name=uploaded_file.name)
        state = file_info.state.name
        if state == "ACTIVE":
            logger.debug("Gemini file is ACTIVE: %s", uploaded_file.name)
            return
        elif state == "FAILED":
            raise RuntimeError(
                f"Gemini file processing failed: {uploaded_file.name}"
            )
        logger.debug("Waiting for Gemini file to become ACTIVE (state=%s)…", state)
        time.sleep(3)

    raise TimeoutError(
        f"Gemini file did not become ACTIVE within {timeout}s: {uploaded_file.name}"
    )


def _parse_output(raw_text: str) -> dict:
    """Extract JSON from Gemini response (same logic as vllm/analyzer.py)."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)

    if not match:
        logger.error("No JSON found in Gemini output: %s", raw_text[:300])
        return {"error": "Failed to extract JSON", "raw_output": raw_text}

    try:
        parsed = json.loads(match.group(0))
        logger.info(
            "Gemini result | is_abnormal=%s | event=%s | severity=%s | confidence=%.2f",
            parsed.get("is_abnormal"),
            parsed.get("event_type"),
            parsed.get("severity"),
            parsed.get("confidence", 0),
        )
        return parsed

    except json.JSONDecodeError as exc:
        logger.error("Gemini JSON parse error: %s | raw: %s", exc, raw_text[:300])
        return {"error": "Invalid JSON", "raw_output": raw_text}