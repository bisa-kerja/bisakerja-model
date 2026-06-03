"""Safe observability payload builders for Model API responses/log hooks.

Only operational metadata belongs here. Raw CV text, tokens, DB URLs, and other
PII must never be copied into these payloads.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SENSITIVE_OBSERVABILITY_KEYS = frozenset(
    {
        "authorization",
        "token",
        "serviceToken",
        "databaseUrl",
        "dbUrl",
        "rawCv",
        "cvText",
        "cvContent",
        "rawPayload",
    }
)

SAFE_OBSERVABILITY_KEYS = frozenset(
    {
        "requestId",
        "modelVersion",
        "artifactHash",
        "candidateCount",
        "parseQuality",
        "parseLatencyMs",
        "embeddingLatencyMs",
        "tensorflowLatencyMs",
        "wrapperLatencyMs",
        "totalLatencyMs",
        "errorCode",
        "fallbackReason",
        "warmupCompleted",
        "warmupLatencyMs",
    }
)


def build_safe_observability_event(**values: object) -> dict[str, object]:
    """Return allow-listed telemetry metadata with no sensitive key names."""

    event: dict[str, object] = {}
    for key, value in values.items():
        if key in SENSITIVE_OBSERVABILITY_KEYS or key not in SAFE_OBSERVABILITY_KEYS:
            continue
        if value is None:
            continue
        event[key] = value
    return event


def assert_safe_observability_event(event: Mapping[str, Any]) -> None:
    """Raise when observability payload contains unsafe keys or raw text-like data."""

    unsafe = sorted(key for key in event if key in SENSITIVE_OBSERVABILITY_KEYS or key not in SAFE_OBSERVABILITY_KEYS)
    if unsafe:
        raise ValueError(f"unsafe observability keys: {unsafe}")
    for key, value in event.items():
        if isinstance(value, str) and ("%PDF" in value or "postgresql://" in value or "Bearer " in value):
            raise ValueError(f"unsafe observability value at {key}")
