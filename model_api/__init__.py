"""Bisakerja Phase 26 Model API serving package.

This package is intentionally separate from ``legacy/api``. It contains the
production serving layout that consumes Phase 25 TensorFlow artifacts without
retraining, backend DB access, persistence, auth, or GenAI calls.
"""

from __future__ import annotations

PACKAGE_NAME = "bisakerja-model-api"
PACKAGE_VERSION = "0.1.0-phase26.4"

__all__ = ["PACKAGE_NAME", "PACKAGE_VERSION"]
