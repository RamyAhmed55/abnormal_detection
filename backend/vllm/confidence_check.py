"""
vllm/confidence_check.py
─────────────────────────
Determines whether a VLLM result should be escalated to the Gemini API.

Escalation happens when:
  1. The model's confidence score is below VLLM_MIN_CONFIDENCE.
  2. The result contains a parse error (model didn't return valid JSON).

In both cases, we escalate to Gemini API to get a second opinion.
"""

import logging

import config

logger = logging.getLogger(__name__)


def needs_escalation(result: dict) -> bool:
    """
    Return True if the VLLM result is uncertain enough to escalate.

    Parameters
    ----------
    result : dict
        Parsed output from vllm.analyzer.parse_judge_output().

    Returns
    -------
    bool
    """
    # Error case — model output was unparseable
    if "error" in result:
        logger.warning(
            "VLLM returned a parse error — escalating to Gemini API. Error: %s",
            result.get("error"),
        )
        return True

    confidence = result.get("confidence", 0.0)

    if confidence < config.VLLM_MIN_CONFIDENCE:
        logger.info(
            "VLLM confidence %.2f < threshold %.2f — escalating to Gemini API.",
            confidence,
            config.VLLM_MIN_CONFIDENCE,
        )
        return True

    logger.debug("VLLM confidence %.2f OK — no escalation needed.", confidence)
    return False
