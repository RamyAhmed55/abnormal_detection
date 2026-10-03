"""
vllm/prompts.py
───────────────
All prompts used by the local VLM (Qwen2.5-VL-7B) and the Gemini API.

Keeping prompts here (not inside analyzer functions) makes it easy to
iterate on prompt engineering without touching logic code.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Main judge prompt — sent with every video clip analysis request.
# This is the same prompt refined in the Kaggle notebook, kept identical
# so the model's behaviour is predictable.
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_PROMPT = """\
You are a video analysis system for security surveillance cameras.
You have been alerted by an initial detection system (YOLO) that this clip
may contain fire or smoke. Your task is to carefully analyze the video
and provide an accurate judgment.

Analyze the video and return JSON only, with no additional text before
or after it, using exactly this structure:

{
  "description": "A brief and clear description of what is happening in the video (2-3 sentences)",
  "is_abnormal": true or false,
  "event_type": "fire" or "smoke" or "false_alarm" or "other",
  "severity": "low" or "medium" or "high",
  "confidence": number between 0 and 1,
  "reasoning": "A brief reason for your judgment (one sentence)"
}

Important guidelines:
- If what you see is only red lighting, light reflection, sunset,
  or another non-fire phenomenon, set is_abnormal=false and
  event_type="false_alarm".
- If you are not completely certain but there are genuine indicators
  of smoke or fire, prefer caution (is_abnormal=true with low severity)
  rather than completely ignoring the potential hazard.
- Focus on movement and changes over time in the video, not only
  colors or visual patterns in a single frame.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: inject camera context into the prompt (optional enrichment)
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(camera_id: str = "", location_hint: str = "") -> str:
    """
    Build a context-enriched judge prompt.

    Parameters
    ----------
    camera_id : str
        Camera identifier, e.g. "cam0" or "entrance_cam".
    location_hint : str
        Optional description of where this camera is installed,
        e.g. "factory floor near boiler room".
        Providing context helps the model reduce false alarms.

    Returns
    -------
    str
        Complete prompt string ready to send to the model.
    """
    context = ""
    if camera_id or location_hint:
        parts = []
        if camera_id:
            parts.append(f"Camera ID: {camera_id}")
        if location_hint:
            parts.append(f"Location: {location_hint}")
        context = "\n\nCamera context:\n" + "\n".join(parts) + "\n"

    return JUDGE_PROMPT + context
