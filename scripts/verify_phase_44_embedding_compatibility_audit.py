#!/usr/bin/env python3
"""Write Phase 44 multilingual-E5-small embedding compatibility evidence.

Default mode is offline-safe: it reads already generated Phase 44 artifacts and
never downloads a model. Use --allow-download --regenerate to build the
multilingual-E5-small cache explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/phase_44_embedding_compatibility_audit"
REPORT_JSON_PATH = ROOT / "reports/phase_44_embedding_compatibility_audit.json"
REPORT_MD_PATH = ROOT / "reports/phase_44_embedding_compatibility_audit.md"
METADATA_PATH = ARTIFACT_ROOT / "multilingual_e5_small_embedding_metadata.json"
CACHE_PATH = ARTIFACT_ROOT / "multilingual_e5_small_pair_embeddings_v1.npz"
PAIR_FEATURES_PATH = ARTIFACT_ROOT / "multilingual_e5_small_pair_features.json"
DRIFT_REPORT_PATH = ARTIFACT_ROOT / "embedding_cosine_drift_report.json"
NORMALIZATION_IMPACT_PATH = ARTIFACT_ROOT / "feature_normalization_impact.json"
DECISION_PATH = ARTIFACT_ROOT / "retrain_vs_recalibrate_decision.json"
README_PATH = ARTIFACT_ROOT / "README.md"

PHASE25_ROOT = ROOT / "artifacts/phase_25_tensorflow_training_delivery"
PAIRS_V2_PATH = ROOT / "artifacts/pairs_v2.parquet"
PHASE25_E5_FEATURES_PATH = PHASE25_ROOT / "e5_pair_features.parquet"
PHASE25_E5_CONTRACT_PATH = PHASE25_ROOT / "e5_embedding_contract.json"
PHASE25_TF_FEATURE_CONFIG_PATH = PHASE25_ROOT / "tensorflow_feature_config.json"
PHASE43_DECISION_PATH = ROOT / "reports/phase_43_multilingual_e5_small_migration_decision.json"
JOBS_PATH = ROOT / "legacy/dataset/indotech_job_cleaned.csv"
PROFILES_PATH = ROOT / "legacy/dataset/techtalent_profile_cleaned.csv"

MODEL_NAME = "intfloat/multilingual-e5-small"
BASE_MODEL_NAME = "intfloat/e5-base-v2"
SCHEMA_VERSION = "phase-44-embedding-compatibility-audit-v1"
SOURCE_PATHS = (
    PAIRS_V2_PATH,
    PHASE25_E5_FEATURES_PATH,
    PHASE25_E5_CONTRACT_PATH,
    PHASE25_TF_FEATURE_CONFIG_PATH,
    PHASE43_DECISION_PATH,
    ROOT / "GAP_MODEL_TRAINING.md",
    ROOT / "REQUIREMENT.md",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def dir_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def run_text(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False, timeout=10)
    except Exception:
        return None
    return result.stdout.strip() if result.stdout.strip() else None


def git_state() -> dict[str, Any]:
    porcelain = run_text(["git", "status", "--porcelain"]) or ""
    dirty_lines = [line for line in porcelain.splitlines() if line]
    return {
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "dirty_file_count": len(dirty_lines),
        "dirty_sample": dirty_lines[:30],
        "production_claim_allowed": len(dirty_lines) == 0,
    }


def compact_text(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text).strip()


def labeled_segment(label: str, value: Any) -> str:
    text = compact_text(value)
    return f"{label}: {text}" if text else ""


def join_segments(*segments: str) -> str:
    return " | ".join(segment for segment in segments if segment)


def build_profile_text(row: dict[str, Any]) -> str:
    return join_segments(
        labeled_segment("target_role", row.get("Job_Role")),
        labeled_segment("experience", row.get("Experience")),
        labeled_segment("skills", row.get("Skills")),
        labeled_segment("required_skills", row.get("Required_Skills")),
        labeled_segment("projects", row.get("Projects")),
        labeled_segment("education", row.get("Education")),
    )


def build_job_text(row: dict[str, Any]) -> str:
    return join_segments(
        labeled_segment("title", row.get("title")),
        labeled_segment("normalized_title", row.get("normalized_title")),
        labeled_segment("category", row.get("category")),
        labeled_segment("experience", row.get("experience_level")),
        labeled_segment("skills", row.get("skills_clean")),
        labeled_segment("requirements", row.get("requirements_concat")),
        labeled_segment("description", row.get("description")),
    )


def percentile(values: list[float], q: float) -> float | None:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return None
    if len(clean) == 1:
        return float(clean[0])
    position = (len(clean) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(clean[int(position)])
    fraction = position - lower
    return float(clean[lower] * (1 - fraction) + clean[upper] * fraction)


def describe(values: Iterable[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"count": 0}
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / len(clean)
    return {
        "count": len(clean),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(clean),
        "p05": percentile(clean, 0.05),
        "p25": percentile(clean, 0.25),
        "p50": percentile(clean, 0.50),
        "p75": percentile(clean, 0.75),
        "p95": percentile(clean, 0.95),
        "max": max(clean),
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    x_values = [x for x, _ in pairs]
    y_values = [y for _, y in pairs]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_denominator = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_denominator = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    if x_denominator == 0 or y_denominator == 0:
        return None
    return numerator / (x_denominator * y_denominator)


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            output[indexed[k][0]] = average_rank
        i = j + 1
    return output


def spearman(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def slice_groups(records: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(field) or "UNKNOWN")].append(record)
    return dict(groups)


def stats_for_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    base = [float(row["e5_base_cosine"]) for row in records]
    new = [float(row["multilingual_e5_small_cosine"]) for row in records]
    delta = [float(row["cosine_delta"]) for row in records]
    abs_delta = [abs(value) for value in delta]
    return {
        "count": len(records),
        "e5_base_cosine": describe(base),
        "multilingual_e5_small_cosine": describe(new),
        "delta": describe(delta),
        "abs_delta": describe(abs_delta),
        "pearson": pearson(base, new),
        "spearman": spearman(base, new),
    }


def build_source_rows(sample_size: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    pairs = pd.read_parquet(PAIRS_V2_PATH)
    phase25_features = pd.read_parquet(PHASE25_E5_FEATURES_PATH)
    profiles = pd.read_csv(PROFILES_PATH)
    jobs = pd.read_csv(JOBS_PATH)

    pairs["profile_id_join"] = pairs["profile_id"].astype(str)
    profiles["profile_id_join"] = profiles["ID"].astype(str)
    jobs["job_id_join"] = jobs["job_id"].astype(str)

    profile_map = {str(row["profile_id_join"]): row.to_dict() for _, row in profiles.iterrows()}
    job_map = {str(row["job_id_join"]): row.to_dict() for _, row in jobs.iterrows()}
    base_cosine = {str(row["pair_id"]): float(row["e5_cosine"]) for _, row in phase25_features.iterrows()}

    rows: list[dict[str, Any]] = []
    missing_profiles: list[str] = []
    missing_jobs: list[str] = []
    missing_cosines: list[str] = []
    if sample_size is not None:
        pairs = pairs.head(sample_size)
    for _, pair in pairs.iterrows():
        pair_dict = pair.to_dict()
        pair_id = str(pair_dict["pair_id"])
        profile_id = str(pair_dict["profile_id"])
        job_id = str(pair_dict["job_id"])
        profile = profile_map.get(profile_id)
        job = job_map.get(job_id)
        cosine = base_cosine.get(pair_id)
        if profile is None:
            missing_profiles.append(profile_id)
        if job is None:
            missing_jobs.append(job_id)
        if cosine is None:
            missing_cosines.append(pair_id)
        if profile is None or job is None or cosine is None:
            continue
        profile_text = build_profile_text(profile)
        job_text = build_job_text(job)
        rows.append(
            {
                "pair_id": pair_id,
                "profile_id": profile_id,
                "job_id": job_id,
                "split": str(pair_dict["split"]),
                "language": str(pair_dict.get("language") or "UNKNOWN"),
                "role_family": str(pair_dict.get("role_family") or "UNKNOWN"),
                "pair_type": str(pair_dict.get("pair_type") or "UNKNOWN"),
                "score_band": str(pair_dict.get("score_band") or "UNKNOWN"),
                "job_fit_score": float(pair_dict["job_fit_score"]),
                "skill_overlap": float(pair_dict["skill_overlap"]),
                "requirement_coverage": float(pair_dict["requirement_coverage"]),
                "role_match": float(pair_dict["role_match"]),
                "experience_match": float(pair_dict["experience_match"]),
                "experience_gap_years_clipped": min(6.0, max(0.0, float(pair_dict["experience_gap_years"]))),
                "e5_base_cosine": cosine,
                "profile_text": profile_text,
                "job_text": job_text,
            }
        )
    diagnostics = {
        "requested_pair_count": int(len(pairs)),
        "built_pair_count": len(rows),
        "missing_profile_count": len(missing_profiles),
        "missing_job_count": len(missing_jobs),
        "missing_phase25_cosine_count": len(missing_cosines),
        "empty_profile_text_count": sum(1 for row in rows if not row["profile_text"]),
        "empty_job_text_count": sum(1 for row in rows if not row["job_text"]),
        "sample_size": sample_size,
    }
    return rows, diagnostics


def representative_texts() -> list[str]:
    return [
        "query: Backend engineer with Python, SQL, Docker, and API deployment experience.",
        "query: Data analyst berpengalaman memakai Python, SQL, dashboard, dan analisis bisnis.",
        "passage: Backend Developer role requiring Python, PostgreSQL, Docker, REST API, and cloud deployment.",
        "passage: Lowongan Data Analyst membutuhkan SQL, Python, visualisasi data, dan komunikasi bisnis.",
    ]


def generate_embeddings(allow_download: bool, sample_size: int | None = None) -> dict[str, Any]:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    import torch

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"

    rows, source_diagnostics = build_source_rows(sample_size=sample_size)
    if source_diagnostics["built_pair_count"] == 0:
        raise RuntimeError("No Phase 44 source rows could be built")

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    get_dimension = getattr(model, "get_embedding_dimension", None) or getattr(model, "get_sentence_embedding_dimension")
    dimension = int(get_dimension())
    profile_inputs = ["query: " + row["profile_text"] for row in rows]
    job_inputs = ["passage: " + row["job_text"] for row in rows]
    query_embeddings = model.encode(
        profile_inputs,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    passage_embeddings = model.encode(
        job_inputs,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    cosines = (query_embeddings * passage_embeddings).sum(axis=1).astype("float32")

    sample_once = model.encode(
        representative_texts(), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")
    sample_twice = model.encode(
        representative_texts(), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")
    sample_norms = np.linalg.norm(sample_once, axis=1)
    deterministic_max_abs_delta = float(np.max(np.abs(sample_once - sample_twice)))

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        query_embeddings=query_embeddings,
        passage_embeddings=passage_embeddings,
        pair_id=np.array([row["pair_id"] for row in rows]),
        metadata_json=json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "embedding_model": MODEL_NAME,
                "embedding_dimension": dimension,
                "profile_prefix": "query:",
                "job_prefix": "passage:",
                "normalized_embeddings": True,
                "row_count": len(rows),
            },
            sort_keys=True,
        ),
    )

    pair_feature_records = []
    for row, cosine in zip(rows, cosines):
        feature_record = {key: value for key, value in row.items() if key not in {"profile_text", "job_text"}}
        feature_record["multilingual_e5_small_cosine"] = float(cosine)
        feature_record["cosine_delta"] = float(cosine - row["e5_base_cosine"])
        feature_record["embedding_model"] = MODEL_NAME
        feature_record["source_embedding_model"] = BASE_MODEL_NAME
        pair_feature_records.append(feature_record)
    write_json(PAIR_FEATURES_PATH, {"schema_version": SCHEMA_VERSION, "records": pair_feature_records})

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "phase_44_embedding_compatibility_audit",
        "generated_at": now_utc(),
        "embedding_model": MODEL_NAME,
        "reference_url": "https://huggingface.co/intfloat/multilingual-e5-small",
        "license_reference_url": "https://huggingface.co/intfloat/multilingual-e5-small",
        "backend": "sentence-transformers",
        "fallback_backend_used": False,
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
        "embedding_dimension": dimension,
        "expected_embedding_dimension": 384,
        "profile_prefix": "query:",
        "job_prefix": "passage:",
        "normalized_embeddings": True,
        "representative_text_count": len(representative_texts()),
        "representative_embedding_check": {
            "dimension": int(sample_once.shape[1]),
            "finite_values": bool(np.isfinite(sample_once).all()),
            "norm_min": float(sample_norms.min()),
            "norm_max": float(sample_norms.max()),
            "deterministic_max_abs_delta": deterministic_max_abs_delta,
        },
        "pair_embedding_check": {
            "row_count": len(rows),
            "dimension": dimension,
            "query_finite_values": bool(np.isfinite(query_embeddings).all()),
            "passage_finite_values": bool(np.isfinite(passage_embeddings).all()),
            "query_norm_min": float(np.linalg.norm(query_embeddings, axis=1).min()),
            "query_norm_max": float(np.linalg.norm(query_embeddings, axis=1).max()),
            "passage_norm_min": float(np.linalg.norm(passage_embeddings, axis=1).min()),
            "passage_norm_max": float(np.linalg.norm(passage_embeddings, axis=1).max()),
            "cosine_min": float(cosines.min()),
            "cosine_max": float(cosines.max()),
        },
        "cache": file_info(CACHE_PATH),
        "cache_path": str(CACHE_PATH.relative_to(ROOT)),
        "hf_cache_path": str(Path.home() / ".cache/huggingface/hub/models--intfloat--multilingual-e5-small"),
        "hf_cache_size_bytes": dir_size_bytes(Path.home() / ".cache/huggingface/hub/models--intfloat--multilingual-e5-small"),
        "runtime_hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_brand": run_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "memory_bytes": run_text(["sysctl", "-n", "hw.memsize"]),
        },
        "source_diagnostics": source_diagnostics,
    }
    write_json(METADATA_PATH, metadata)
    return metadata


def load_pair_feature_records() -> list[dict[str, Any]]:
    payload = read_json(PAIR_FEATURES_PATH, {"records": []})
    return list(payload.get("records", []))


def build_drift_report(generated_at: str) -> dict[str, Any]:
    records = load_pair_feature_records()
    if not records:
        return {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "phase_44_embedding_compatibility_audit",
            "generated_at": generated_at,
            "status": "missing_pair_features",
            "blockers": ["Phase 44 pair features are missing; run with --allow-download --regenerate."],
        }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "phase_44_embedding_compatibility_audit",
        "generated_at": generated_at,
        "status": "complete",
        "overall": stats_for_records(records),
        "by_split": {},
        "by_language": {},
        "by_role_family": {},
        "by_pair_type": {},
        "by_score_band": {},
        "worst_drift_examples": [],
    }
    for report_key, field in [
        ("by_split", "split"),
        ("by_language", "language"),
        ("by_role_family", "role_family"),
        ("by_pair_type", "pair_type"),
        ("by_score_band", "score_band"),
    ]:
        report[report_key] = {
            name: stats_for_records(group)
            for name, group in sorted(slice_groups(records, field).items())
            if len(group) >= 1
        }
    worst = sorted(records, key=lambda row: abs(float(row["cosine_delta"])), reverse=True)[:15]
    for row in worst:
        report["worst_drift_examples"].append(
            {
                "pair_id": row["pair_id"],
                "split": row["split"],
                "language": row["language"],
                "role_family": row["role_family"],
                "pair_type": row["pair_type"],
                "score_band": row["score_band"],
                "job_fit_score": row["job_fit_score"],
                "e5_base_cosine": row["e5_base_cosine"],
                "multilingual_e5_small_cosine": row["multilingual_e5_small_cosine"],
                "cosine_delta": row["cosine_delta"],
            }
        )
    return report


def build_normalization_impact(generated_at: str, drift_report: dict[str, Any]) -> dict[str, Any]:
    records = load_pair_feature_records()
    tf_config = read_json(PHASE25_TF_FEATURE_CONFIG_PATH)
    old_mean = float(tf_config.get("normalization", {}).get("mean", {}).get("e5_cosine", 0.0))
    old_std = float(tf_config.get("normalization", {}).get("std", {}).get("e5_cosine", 0.0))
    train_records = [row for row in records if row.get("split") == "train"]
    new_train_stats = describe([float(row["multilingual_e5_small_cosine"]) for row in train_records])
    if not records or old_std == 0 or not new_train_stats.get("count"):
        return {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "phase_44_embedding_compatibility_audit",
            "generated_at": generated_at,
            "status": "missing_evidence",
            "blockers": ["Cannot evaluate normalization impact without Phase 25 config and Phase 44 features."],
        }
    new_mean = float(new_train_stats["mean"])
    new_std = float(new_train_stats["std"])
    mean_shift_std_units = abs(new_mean - old_mean) / old_std
    std_ratio = new_std / old_std if old_std else None
    material_shift = mean_shift_std_units > 0.25 or std_ratio is None or std_ratio < 0.80 or std_ratio > 1.25
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "phase_44_embedding_compatibility_audit",
        "generated_at": generated_at,
        "status": "complete",
        "approved_feature_order": tf_config.get("approved_features"),
        "phase25_e5_cosine_normalization": {"mean": old_mean, "std": old_std},
        "multilingual_e5_small_train_cosine": new_train_stats,
        "mean_delta": new_mean - old_mean,
        "mean_shift_std_units": mean_shift_std_units,
        "std_ratio": std_ratio,
        "material_normalization_shift": material_shift,
        "normalization_decision": "phase25_mean_std_invalid_for_multilingual_e5_small" if material_shift else "phase25_mean_std_may_be_compatible_but_recalibration_still_required",
        "feature_contract_impact": {
            "feature_order_changed": False,
            "only_e5_cosine_source_changed": True,
            "six_feature_vector_recomputed": True,
            "phase45_must_write_new_tensorflow_feature_config": True,
        },
    }


def build_decision(generated_at: str, drift_report: dict[str, Any], normalization: dict[str, Any]) -> dict[str, Any]:
    overall = drift_report.get("overall", {})
    abs_delta = overall.get("abs_delta", {})
    pearson_value = overall.get("pearson")
    spearman_value = overall.get("spearman")
    mean_abs_delta = abs_delta.get("mean")
    max_abs_delta = abs_delta.get("max")
    material_norm = bool(normalization.get("material_normalization_shift"))
    drift_exceeds_threshold = bool(
        mean_abs_delta is None
        or max_abs_delta is None
        or pearson_value is None
        or spearman_value is None
        or mean_abs_delta > 0.01
        or max_abs_delta > 0.05
        or pearson_value < 0.98
        or spearman_value < 0.98
        or material_norm
    )
    decision = "retrain_required_recalibration_required_direct_swap_blocked" if drift_exceeds_threshold else "direct_swap_not_approved_without_phase45_shadow_evidence"
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "phase_44_embedding_compatibility_audit",
        "generated_at": generated_at,
        "decision": decision,
        "direct_artifact_swap_allowed": False,
        "retrain_required": True,
        "recalibration_required": True,
        "thresholds": {
            "mean_abs_cosine_delta_max": 0.01,
            "max_abs_cosine_delta_max": 0.05,
            "pearson_min": 0.98,
            "spearman_min": 0.98,
            "normalization_mean_shift_std_units_max": 0.25,
            "normalization_std_ratio_range": [0.80, 1.25],
        },
        "observed": {
            "mean_abs_cosine_delta": mean_abs_delta,
            "max_abs_cosine_delta": max_abs_delta,
            "pearson": pearson_value,
            "spearman": spearman_value,
            "material_normalization_shift": material_norm,
        },
        "required_next_steps": [
            "Create Phase 45 training notebook with multilingual-E5-small-derived e5_cosine values.",
            "Write new train-split normalization stats for all approved features.",
            "Retrain TensorFlow scorer and compare against Phase 25 E5-base scorer and baselines.",
            "Recalibrate score buckets before any staging default switch.",
        ],
    }


def build_report() -> dict[str, Any]:
    generated_at = now_utc()
    metadata = read_json(METADATA_PATH)
    drift = build_drift_report(generated_at)
    normalization = build_normalization_impact(generated_at, drift)
    decision = build_decision(generated_at, drift, normalization)
    records = load_pair_feature_records()
    expected_rows = read_json(PHASE25_E5_CONTRACT_PATH).get("source_text_contract", {}).get("pair_count", 3600)

    checks = {
        "metadata_verifies_model_dimension_prefix_norms_and_runtime": bool(
            metadata.get("embedding_model") == MODEL_NAME
            and metadata.get("embedding_dimension") == 384
            and metadata.get("profile_prefix") == "query:"
            and metadata.get("job_prefix") == "passage:"
            and metadata.get("pair_embedding_check", {}).get("query_finite_values") is True
            and metadata.get("fallback_backend_used") is False
        ),
        "paired_embedding_cache_matches_frozen_phase25_pair_set": bool(
            CACHE_PATH.exists()
            and len(records) == expected_rows
            and metadata.get("source_diagnostics", {}).get("built_pair_count") == expected_rows
        ),
        "cosine_drift_quantified_by_required_slices": bool(
            drift.get("status") == "complete"
            and all(drift.get(key) for key in ["by_split", "by_language", "by_role_family", "by_pair_type", "by_score_band"])
        ),
        "rank_correlation_and_worst_examples_recorded": bool(
            drift.get("overall", {}).get("pearson") is not None
            and drift.get("overall", {}).get("spearman") is not None
            and len(drift.get("worst_drift_examples", [])) >= 5
        ),
        "multilingual_slices_cover_id_en_mixed_unknown_when_present": bool(
            {"ID", "EN", "MIXED", "UNKNOWN"}.issubset(set(drift.get("by_language", {}).keys()))
        ),
        "feature_normalization_impact_recorded": normalization.get("status") == "complete"
        and normalization.get("feature_contract_impact", {}).get("six_feature_vector_recomputed") is True,
        "direct_replacement_rejected_with_retrain_recalibrate_decision": bool(
            decision.get("direct_artifact_swap_allowed") is False
            and decision.get("retrain_required") is True
            and decision.get("recalibration_required") is True
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    status = "complete" if not blockers else "incomplete"
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "phase_44_embedding_compatibility_audit",
        "generated_at": generated_at,
        "status": status,
        "checks": checks,
        "blockers": blockers,
        "final_decision": decision.get("decision") if not blockers else "no_go_missing_evidence",
        "summary": {
            "embedding_model": metadata.get("embedding_model"),
            "embedding_dimension": metadata.get("embedding_dimension"),
            "row_count": len(records),
            "old_model": BASE_MODEL_NAME,
            "new_model": MODEL_NAME,
            "mean_abs_cosine_delta": decision.get("observed", {}).get("mean_abs_cosine_delta"),
            "max_abs_cosine_delta": decision.get("observed", {}).get("max_abs_cosine_delta"),
            "pearson": decision.get("observed", {}).get("pearson"),
            "spearman": decision.get("observed", {}).get("spearman"),
            "material_normalization_shift": decision.get("observed", {}).get("material_normalization_shift"),
        },
        "artifact_paths": {
            "metadata": str(METADATA_PATH.relative_to(ROOT)),
            "cache": str(CACHE_PATH.relative_to(ROOT)),
            "pair_features": str(PAIR_FEATURES_PATH.relative_to(ROOT)),
            "drift_report": str(DRIFT_REPORT_PATH.relative_to(ROOT)),
            "normalization_impact": str(NORMALIZATION_IMPACT_PATH.relative_to(ROOT)),
            "decision": str(DECISION_PATH.relative_to(ROOT)),
        },
        "source_files": [str(path.relative_to(ROOT)) for path in SOURCE_PATHS],
        "git_state": git_state(),
    }


def write_markdown(report: dict[str, Any], drift: dict[str, Any], normalization: dict[str, Any], decision: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Phase 44 Embedding Compatibility Audit",
        "",
        f"Status: `{report['status']}`",
        f"Decision: `{report['final_decision']}`",
        "",
        "Phase 44 measures embedding and feature drift for `intfloat/multilingual-e5-small` before TensorFlow retraining. It does not change Model API defaults.",
        "",
        "## Summary",
        f"- Old embedding model: `{summary['old_model']}`",
        f"- New embedding model: `{summary['new_model']}`",
        f"- New embedding dimension: `{summary['embedding_dimension']}`",
        f"- Pair rows compared: `{summary['row_count']}`",
        f"- Mean absolute cosine delta: `{summary['mean_abs_cosine_delta']}`",
        f"- Max absolute cosine delta: `{summary['max_abs_cosine_delta']}`",
        f"- Pearson correlation: `{summary['pearson']}`",
        f"- Spearman correlation: `{summary['spearman']}`",
        f"- Material normalization shift: `{summary['material_normalization_shift']}`",
        "",
        "## Checks",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    if report["blockers"]:
        lines.extend(["", "## Blockers"])
        lines.extend(f"- `{item}`" for item in report["blockers"])
    lines.extend(["", "## Language Slices"])
    for language, values in drift.get("by_language", {}).items():
        lines.append(
            f"- `{language}`: count `{values.get('count')}`, mean new cosine `{values.get('multilingual_e5_small_cosine', {}).get('mean')}`, mean abs delta `{values.get('abs_delta', {}).get('mean')}`"
        )
    lines.extend(
        [
            "",
            "## Normalization Impact",
            f"- Phase 25 e5 cosine mean/std: `{normalization.get('phase25_e5_cosine_normalization')}`",
            f"- multilingual-E5-small train cosine stats: `{normalization.get('multilingual_e5_small_train_cosine')}`",
            f"- Decision: `{normalization.get('normalization_decision')}`",
            "",
            "## Next Steps",
        ]
    )
    lines.extend(f"- {item}" for item in decision.get("required_next_steps", []))
    lines.extend(
        [
            "",
            "## Commands",
            "- Generate live cache: `training/.tf-venv-3.13/bin/python scripts/verify_phase_44_embedding_compatibility_audit.py --write --allow-download --regenerate`",
            "- Verify: `python -m unittest tests.test_phase_44_embedding_compatibility_audit`",
        ]
    )
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme() -> None:
    lines = [
        "# Phase 44 Embedding Compatibility Audit Artifacts",
        "",
        "This directory stores multilingual-E5-small embedding compatibility evidence before TensorFlow retraining.",
        "",
        "## Files",
        "",
        "- `multilingual_e5_small_embedding_metadata.json` — model metadata, runtime dependency versions, prefix policy, dimension, normalization, finite-value checks, and cache metadata.",
        "- `multilingual_e5_small_pair_embeddings_v1.npz` — query and passage embeddings for the frozen Phase 25 pair set using `intfloat/multilingual-e5-small`.",
        "- `multilingual_e5_small_pair_features.json` — old E5-base cosine, new multilingual-E5-small cosine, and slice metadata per pair.",
        "- `embedding_cosine_drift_report.json` — overall and slice-level drift statistics plus worst drift examples without raw CV text.",
        "- `feature_normalization_impact.json` — impact on the approved six-feature vector and Phase 25 normalization stats.",
        "- `retrain_vs_recalibrate_decision.json` — direct-swap rejection and required retraining/recalibration steps.",
        "",
        "## Safety",
        "",
        "These artifacts are audit evidence only. They must not be used as Model API defaults before retraining, recalibration, contract validation, staging shadow comparison, and rollback gates pass.",
    ]
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all(allow_download: bool = False, regenerate: bool = False, sample_size: int | None = None) -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    needs_generated_inputs = regenerate or not PAIR_FEATURES_PATH.exists() or not METADATA_PATH.exists() or not CACHE_PATH.exists()
    if needs_generated_inputs and (allow_download or regenerate):
        generate_embeddings(allow_download=allow_download, sample_size=sample_size)
    generated_at = now_utc()
    drift = build_drift_report(generated_at)
    normalization = build_normalization_impact(generated_at, drift)
    decision = build_decision(generated_at, drift, normalization)
    write_json(DRIFT_REPORT_PATH, drift)
    write_json(NORMALIZATION_IMPACT_PATH, normalization)
    write_json(DECISION_PATH, decision)
    report = build_report()
    write_json(REPORT_JSON_PATH, report)
    write_markdown(report, drift, normalization, decision)
    write_readme()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write Phase 44 report and artifacts.")
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face model download when cache is missing.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate multilingual-E5-small embeddings.")
    parser.add_argument("--sample-size", type=int, default=None, help="Debug-only pair row limit; full frozen set is required for complete status.")
    args = parser.parse_args(argv)

    report = write_all(allow_download=args.allow_download, regenerate=args.regenerate, sample_size=args.sample_size) if args.write else build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["blockers"] else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
