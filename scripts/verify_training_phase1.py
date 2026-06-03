#!/usr/bin/env python3
"""Verify the extracted training package without relying on notebook state."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "training" / "configs" / "jobfit_v2.yaml"
DEFAULT_REPORT = ROOT / "reports" / "training_phase1_verification.json"
REQUIRED_PACKAGE_FILES = [
    "training/__init__.py",
    "training/configs/jobfit_v2.yaml",
    "training/configs/cv_quality_v1.yaml",
    "training/data/build_pairs.py",
    "training/data/normalize_experience.py",
    "training/data/normalize_skills.py",
    "training/features/text_builder.py",
    "training/features/embedding.py",
    "training/features/skill_alias.py",
    "training/models/model_def.py",
    "training/model_def.py",
    "training/models/jobfit_ranker.py",
    "training/models/ats_quality.py",
    "training/evaluation/baselines.py",
    "training/evaluation/metrics.py",
    "training/evaluation/calibration.py",
    "training/evaluation/report.py",
    "training/export/export_model.py",
    "training/export/export_model_card.py",
    "training/train.py",
    "training/evaluate.py",
]


def _safe_check(name: str, check_fn, *args) -> dict[str, Any]:
    """Run a verification check and record import/runtime failures as report data."""
    try:
        return check_fn(*args)
    except Exception as exc:
        return {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "check": name,
        }


def _check_environment() -> dict[str, Any]:
    """Record active Python runtime and dependency availability for manual verification."""
    modules = ["numpy", "pandas", "pyarrow", "sklearn", "tensorflow"]
    availability: dict[str, dict[str, Any]] = {}
    for module_name in modules:
        spec = importlib.util.find_spec(module_name)
        availability[module_name] = {"available": spec is not None}
    return {
        "passed": True,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "modules": availability,
    }


def _check_file_inventory() -> dict[str, Any]:
    """Check that all files required by the extraction scope exist."""
    missing = [path for path in REQUIRED_PACKAGE_FILES if not (ROOT / path).exists()]
    return {"passed": not missing, "missing": missing, "checked": REQUIRED_PACKAGE_FILES}


def _check_config(config_path: Path) -> dict[str, Any]:
    """Load the versioned config and return the key values required by the scope."""
    from training.config import load_training_config

    config = load_training_config(config_path)
    return {
        "passed": True,
        "model_name": config["model"]["name"],
        "model_version": config["model"]["version"],
        "label_version": config["label"]["version"],
        "split_seed": config["split"]["seed"],
        "embedding_model": config["features"]["embedding_model"],
        "paths": config["paths"],
    }


def _check_dataset(config_path: Path) -> dict[str, Any]:
    """Validate pair artifact loading and deterministic split generation."""
    from training.config import load_training_config, resolve_repo_path
    from training.data.build_pairs import load_pairs, split_pair_indices

    config = load_training_config(config_path)
    pairs = load_pairs(resolve_repo_path(config["paths"]["pairs"]))
    train_idx, val_idx = split_pair_indices(
        pairs,
        seed=int(config["split"]["seed"]),
        validation_size=float(config["split"]["validation_size"]),
    )
    return {
        "passed": True,
        "pairs_total": int(len(pairs)),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "columns": list(pairs.columns),
    }


def _check_entrypoints() -> dict[str, Any]:
    """Confirm train and evaluate modules expose CLI parser builders."""
    from training.evaluate import build_parser as build_evaluate_parser
    from training.train import build_parser as build_train_parser

    train_parser = build_train_parser()
    evaluate_parser = build_evaluate_parser()
    return {
        "passed": True,
        "train_prog": train_parser.prog,
        "evaluate_prog": evaluate_parser.prog,
    }


def _check_tensorflow_model_load(config_path: Path) -> dict[str, Any]:
    """Load the configured .keras artifact when TensorFlow is installed locally."""
    if importlib.util.find_spec("tensorflow") is None:
        return {
            "passed": False,
            "skipped": True,
            "reason": "TensorFlow is not installed in the active Python environment.",
        }

    from training.config import load_training_config, resolve_repo_path
    from training.models.model_def import load_jobfit_model

    config = load_training_config(config_path)
    model_path = resolve_repo_path(config["paths"]["model"])
    model = load_jobfit_model(model_path, compile=False)
    return {
        "passed": True,
        "skipped": False,
        "model_path": str(model_path.relative_to(ROOT)),
        "model_name": getattr(model, "name", None),
        "input_names": [getattr(item, "name", str(item)) for item in getattr(model, "inputs", [])],
    }


def run_verification(config_path: Path) -> dict[str, Any]:
    """Run all Phase 1 checks and keep TensorFlow load as an explicit optional gate."""
    checks = {
        "environment": _safe_check("environment", _check_environment),
        "file_inventory": _safe_check("file_inventory", _check_file_inventory),
        "config": _safe_check("config", _check_config, config_path),
        "dataset": _safe_check("dataset", _check_dataset, config_path),
        "entrypoints": _safe_check("entrypoints", _check_entrypoints),
        "tensorflow_model_load": _safe_check(
            "tensorflow_model_load", _check_tensorflow_model_load, config_path
        ),
    }
    required_passed = all(
        result.get("passed", False)
        for name, result in checks.items()
        if name != "tensorflow_model_load"
    )
    return {
        "schema_version": "training-phase1-verification-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(ROOT)),
        "required_checks_passed": required_passed,
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the verification script."""
    parser = argparse.ArgumentParser(description="Verify training package extraction.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Training config path.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Output JSON report path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write verification report and return non-zero only for required check failures."""
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path

    payload = run_verification(config_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
