"""
vllm/analyzer.py
─────────────────
Core analysis functions for the local Qwen2.5-VL-7B model.

Public API
──────────
  analyze_clip(clip_path, camera_id, location_hint) -> dict
      Full pipeline: load model → build messages → generate → parse JSON.

  parse_judge_output(raw_text) -> dict
      Extracts JSON from raw model output (handles markdown code blocks too).
"""

import gc
import json
import logging
import re
from typing import Optional

import config
from vllm.loader import get_model_and_processor
from vllm.prompts import build_prompt

logger = logging.getLogger(__name__)


def analyze_clip(
    clip_path: str,
    camera_id: str = "",
    location_hint: str = "",
) -> dict:
    """
    Run the local Qwen2.5-VL-7B model on a video clip.

    Parameters
    ----------
    clip_path : str
        Path to the .mp4 clip (should be ~20 seconds).
    camera_id : str
        Camera identifier for context injection into prompt.
    location_hint : str
        Optional location description for context enrichment.

    Returns
    -------
    dict
        Parsed VLM output with keys:
          description, is_abnormal, event_type, severity, confidence, reasoning.
        On error, returns {"error": ..., "raw_output": ...}.
    """
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = get_model_and_processor()

    # Clean VRAM before allocation
    gc.collect()
    torch.cuda.empty_cache()

    prompt = build_prompt(camera_id=camera_id, location_hint=location_hint)

    # Build multi-modal message with video + text prompt
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": clip_path,
                    "min_pixels": config.VLLM_MIN_PIXELS,
                    "max_pixels": config.VLLM_MAX_PIXELS,
                    "fps": config.VLLM_FPS,  # 1 frame/sec for 20s clip = 20 frames
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    logger.info("Running VLLM inference on clip: %s", clip_path)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=config.VLLM_MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    raw_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    logger.debug("VLLM raw output:\n%s", raw_text)

    # Free VRAM tensors
    del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
    gc.collect()
    torch.cuda.empty_cache()

    result = parse_judge_output(raw_text)
    result["_router"] = "local_vllm"
    return result


def parse_judge_output(raw_text: str) -> dict:
    """
    Extract and parse the JSON object from raw model output.

    The model is instructed to return only JSON, but may sometimes wrap it
    in a markdown code block (```json ... ```) — this function handles both.

    Parameters
    ----------
    raw_text : str
        Raw text output from the model.

    Returns
    -------
    dict
        Parsed result dict, or {"error": ..., "raw_output": ...} on failure.
    """
    # Try to find JSON object in the text (handles markdown fences too)
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)

    if not match:
        logger.error("No JSON found in VLLM output: %s", raw_text[:300])
        return {"error": "Failed to extract JSON", "raw_output": raw_text}

    try:
        parsed = json.loads(match.group(0))
        logger.info(
            "VLLM result | is_abnormal=%s | event=%s | severity=%s | confidence=%.2f",
            parsed.get("is_abnormal"),
            parsed.get("event_type"),
            parsed.get("severity"),
            parsed.get("confidence", 0),
        )
        return parsed

    except json.JSONDecodeError as exc:
        logger.error("JSON parse error: %s | raw: %s", exc, raw_text[:300])
        return {"error": "Invalid JSON", "raw_output": raw_text}
