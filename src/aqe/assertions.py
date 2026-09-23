"""Structured assertion checks against driver evidence."""

import json
from typing import Any


def evaluate_assertion(assertion: str, evidence: dict[str, Any]) -> bool:
    """Judge substring, JSON field, or HTTP status assertions."""
    text = assertion.strip()
    blob = " ".join(
        str(evidence.get(key) or "")
        for key in ("summary", "stdout", "stderr", "body")
    )
    if text.startswith("contains:"):
        return text.split(":", 1)[1] in blob
    if text.startswith("json:"):
        expression = text.split(":", 1)[1]
        field, separator, expected = expression.partition("=")
        if not separator:
            return False
        payload = evidence.get("json")
        if payload is None and evidence.get("stdout"):
            try:
                payload = json.loads(str(evidence["stdout"]))
            except json.JSONDecodeError:
                return False
        if not isinstance(payload, dict):
            return False
        return str(payload.get(field)) == expected
    if text.startswith("status:"):
        return str(evidence.get("status")) == text.split(":", 1)[1]
    return text in blob
