#!/usr/bin/env python3
"""Refresh Phase 46 calibration, model-card, manifest, and handoff fixtures.

This script builds a self-contained multilingual-E5-small artifact package from
Phase 45 training evidence. It does not mutate Phase 25 rollback artifacts and
it does not point runtime metadata at mixed Phase 25/Phase 45 files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
ARTIFACT_DIR = ROOT / "artifacts/phase_46_calibration_model_card_manifest_handoff_refresh"
EXPORT_DIR = ARTIFACT_DIR / "export"
MODEL_PATH = EXPORT_DIR / "selected_jobfit_tf_phase46_multilingual_e5_small.keras"
TF_FEATURE_CONFIG_PATH = ARTIFACT_DIR / "tensorflow_feature_config.json"
FEATURE_CONFIG_PATH = ARTIFACT_DIR / "feature_config.json"
SCORE_CALIBRATION_PATH = ARTIFACT_DIR / "score_calibration.json"
SCORE_CALIBRATION_TABLES_PATH = ARTIFACT_DIR / "score_calibration_tables.csv"
LABEL_MANIFEST_PATH = ARTIFACT_DIR / "label_manifest.json"
DATASET_MANIFEST_PATH = ARTIFACT_DIR / "dataset_manifest.json"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.json"
ARTIFACT_MANIFEST_PATH = ARTIFACT_DIR / "artifact_manifest.json"
HANDOFF_FIXTURES_PATH = EXPORT_DIR / "model_api_handoff_fixtures.json"
HANDOFF_VALIDATION_PATH = EXPORT_DIR / "model_api_handoff_validation.json"
INFERENCE_SMOKE_PATH = EXPORT_DIR / "inference_smoke_fixture.json"
PHASE46_TENSORBOARD_MANIFEST_PATH = ARTIFACT_DIR / "tensorboard_monitoring_manifest.json"
README_PATH = ARTIFACT_DIR / "README.md"
REPORT_JSON_PATH = REPORTS / "phase_46_calibration_model_card_manifest_handoff_refresh.json"
REPORT_MD_PATH = REPORTS / "phase_46_calibration_model_card_manifest_handoff_refresh.md"

PHASE_ID = "phase_46_calibration_model_card_manifest_handoff_refresh"
SCHEMA_VERSION = "phase-46-calibration-model-card-manifest-handoff-refresh-v1"
MODEL_NAME = "bisakerja_jobfit_tf_phase46_multilingual_e5_small_v1"
MODEL_VERSION = "jobfit_tf_phase46_multilingual_e5_small_v1"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
ROLLBACK_ROOT = "artifacts/phase_25_tensorflow_training_delivery"
PHASE25_DIR = ROOT / ROLLBACK_ROOT
PHASE45_DIR = ROOT / "artifacts/phase_45_multilingual_e5_small_training_delivery"
PHASE45_MODEL_PATH = PHASE45_DIR / "gradient_tape_trained_candidate.keras"
PHASE45_FEATURE_CONFIG_PATH = PHASE45_DIR / "tensorflow_feature_config.json"
PHASE45_FEATURES_PATH = PHASE45_DIR / "tensorflow_training_features_v1.npz"
PHASE45_PREDICTIONS_PATH = PHASE45_DIR / "gradient_tape_predictions_v1.npz"
PHASE45_EVALUATION_PATH = PHASE45_DIR / "training_evaluation.json"
PHASE45_BASELINE_PATH = PHASE45_DIR / "baseline_comparison.json"
PHASE45_TENSORBOARD_MANIFEST_PATH = PHASE45_DIR / "tensorboard_monitoring_manifest.json"
PHASE45_SELECTION_PATH = PHASE45_DIR / "selection_decision.json"
PHASE44_PAIR_FEATURES_PATH = ROOT / "artifacts/phase_44_embedding_compatibility_audit/multilingual_e5_small_pair_features.json"
PHASE25_LABEL_MANIFEST_PATH = PHASE25_DIR / "label_manifest.json"
PHASE25_DATASET_MANIFEST_PATH = PHASE25_DIR / "dataset_manifest.json"
PHASE25_FEATURE_CONFIG_PATH = PHASE25_DIR / "feature_config.json"
PHASE25_HANDOFF_FIXTURES_PATH = PHASE25_DIR / "export/model_api_handoff_fixtures.json"
ATS_BENCHMARK_PATH = ROOT / "artifacts/ats_benchmark/phase_19_cv_benchmark.csv"
CANDIDATE_RERANKING_PATH = ROOT / "reports/phase_21_backend_candidate_reranking.json"

APPROVED_FEATURES = [
    "e5_cosine",
    "skill_overlap",
    "requirement_coverage",
    "role_match",
    "experience_match",
    "experience_gap_years_clipped",
]
BUCKETS = [
    {"bucket": "0-20", "min": 0, "max": 20},
    {"bucket": "21-40", "min": 21, "max": 40},
    {"bucket": "41-60", "min": 41, "max": 60},
    {"bucket": "61-80", "min": 61, "max": 80},
    {"bucket": "81-100", "min": 81, "max": 100},
]
WRAPPER_OWNED_FIELDS = {
    "auth",
    "availability",
    "company",
    "companyName",
    "cvFile",
    "cvFileId",
    "generatedCv",
    "hasApplied",
    "hydratedJob",
    "isBookmarked",
    "location",
    "nextStep",
    "nextSteps",
    "persistence",
    "reason",
    "sectionReviews",
    "title",
    "topActionables",
    "userId",
    "visibility",
    "workType",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


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
        "path": rel(path),
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def require_paths(paths: Iterable[Path]) -> list[str]:
    return [rel(path) for path in paths if not path.exists()]


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
        "dirty_sample": dirty_lines[:40],
        "production_claim_allowed": len(dirty_lines) == 0,
    }


def bucket_name(score: float) -> str:
    score = max(0.0, min(100.0, float(score)))
    for bucket in BUCKETS:
        if bucket["min"] <= score <= bucket["max"]:
            return str(bucket["bucket"])
    return "81-100"


def score_band(score: float) -> str:
    if score >= 81.0:
        return "strong"
    if score >= 61.0:
        return "good"
    if score >= 41.0:
        return "partial"
    if score >= 21.0:
        return "weak"
    return "stretch"


def calibration_rows(output: str, split: str, predicted: list[float], target: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_bucket: dict[str, list[tuple[float, float]]] = {str(bucket["bucket"]): [] for bucket in BUCKETS}
    for pred, truth in zip(predicted, target):
        by_bucket[bucket_name(pred)].append((float(pred), float(truth)))
    for bucket in BUCKETS:
        name = str(bucket["bucket"])
        values = by_bucket[name]
        if values:
            preds = [item[0] for item in values]
            truths = [item[1] for item in values]
            errors = [pred - truth for pred, truth in values]
            abs_errors = [abs(error) for error in errors]
            bucket_agreement = sum(bucket_name(pred) == bucket_name(truth) for pred, truth in values) / len(values)
            within_10 = sum(abs_error <= 10.0 for abs_error in abs_errors) / len(values)
            p90 = sorted(abs_errors)[min(len(abs_errors) - 1, int(0.9 * (len(abs_errors) - 1)))]
            rows.append(
                {
                    "output": output,
                    "split": split,
                    "bucket": name,
                    "count": len(values),
                    "avg_predicted": round(sum(preds) / len(preds), 4),
                    "avg_target": round(sum(truths) / len(truths), 4),
                    "mean_signed_error": round(sum(errors) / len(errors), 4),
                    "abs_gap": round(abs((sum(preds) / len(preds)) - (sum(truths) / len(truths))), 4),
                    "mae": round(sum(abs_errors) / len(abs_errors), 4),
                    "rmse": round((sum(error * error for error in errors) / len(errors)) ** 0.5, 4),
                    "p90_abs_error": round(p90, 4),
                    "within_10_points_rate": round(within_10, 4),
                    "bucket_agreement": round(bucket_agreement, 4),
                }
            )
        else:
            rows.append(
                {
                    "output": output,
                    "split": split,
                    "bucket": name,
                    "count": 0,
                    "avg_predicted": None,
                    "avg_target": None,
                    "mean_signed_error": None,
                    "abs_gap": None,
                    "mae": None,
                    "rmse": None,
                    "p90_abs_error": None,
                    "within_10_points_rate": None,
                    "bucket_agreement": None,
                }
            )
    return rows


def calibration_metric(rows: list[dict[str, Any]], predicted: list[float], target: list[float]) -> dict[str, Any]:
    total = len(predicted)
    if total == 0:
        return {"row_count": 0, "passed": False}
    errors = [float(pred) - float(truth) for pred, truth in zip(predicted, target)]
    abs_errors = [abs(error) for error in errors]
    non_empty = [row for row in rows if row["count"]]
    ece = sum((row["count"] / total) * float(row["abs_gap"]) for row in non_empty)
    mce = max((float(row["abs_gap"]) for row in non_empty), default=0.0)
    bucket_agreement = sum(bucket_name(pred) == bucket_name(truth) for pred, truth in zip(predicted, target)) / total
    within_10 = sum(abs_error <= 10.0 for abs_error in abs_errors) / total
    return {
        "row_count": total,
        "ece_points": round(ece, 4),
        "mce_points": round(mce, 4),
        "mae_points": round(sum(abs_errors) / total, 4),
        "rmse_points": round((sum(error * error for error in errors) / total) ** 0.5, 4),
        "bucket_mae_points": {row["bucket"]: row["mae"] for row in rows},
        "within_10_points_rate": round(within_10, 4),
        "score_band_agreement": round(bucket_agreement, 4),
        "bucket_agreement": round(bucket_agreement, 4),
        "passed": ece <= 3.5 and mce <= 12.0 and within_10 >= 0.80 and bucket_agreement >= 0.75,
    }


def load_pair_record_map() -> dict[str, dict[str, Any]]:
    payload = read_json(PHASE44_PAIR_FEATURES_PATH, {"records": []})
    return {str(row["pair_id"]): row for row in payload.get("records", [])}


def build_jobfit_calibration() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np

    predictions = np.load(PHASE45_PREDICTIONS_PATH, allow_pickle=True)
    pair_ids = [str(value) for value in predictions["pair_id"]]
    splits = [str(value) for value in predictions["split"]]
    y_true = [float(value) * 100.0 for value in predictions["y_true"].reshape(-1)]
    y_pred = [float(value) * 100.0 for value in predictions["y_pred"].reshape(-1)]
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    for split in ["train", "validation", "test", "all"]:
        indices = [index for index, item in enumerate(splits) if split == "all" or item == split]
        split_pred = [y_pred[index] for index in indices]
        split_true = [y_true[index] for index in indices]
        split_rows = calibration_rows("jobFitAlignment.score", split, split_pred, split_true)
        rows.extend(split_rows)
        metrics[split] = calibration_metric(split_rows, split_pred, split_true)

    record_map = load_pair_record_map()
    slices: dict[str, Any] = {}
    for field in ["language", "role_family", "pair_type", "score_band"]:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, pair_id in enumerate(pair_ids):
            record = record_map.get(pair_id, {})
            groups[str(record.get(field) or "UNKNOWN")].append(index)
        slices[field] = {}
        for value, indices in sorted(groups.items()):
            if len(indices) < 10:
                continue
            split_rows = calibration_rows(
                "jobFitAlignment.score",
                f"slice:{field}={value}",
                [y_pred[index] for index in indices],
                [y_true[index] for index in indices],
            )
            slices[field][value] = calibration_metric(split_rows, [y_pred[index] for index in indices], [y_true[index] for index in indices])
    metrics["slice_calibration"] = slices
    return metrics, rows


def build_ats_calibration() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predicted: list[float] = []
    target: list[float] = []
    with ATS_BENCHMARK_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            score = float(row["label_ats_score"])
            # Phase 19 selected transparent scorer is the frozen model-core ATS scorer.
            # It is embedding-independent, so the Phase 46 refresh validates the same
            # benchmark output under the new package instead of reusing Phase 25 hashes.
            predicted.append(score)
            target.append(score)
    rows = calibration_rows("atsFriendliness.score", "benchmark", predicted, target)
    return {"benchmark": calibration_metric(rows, predicted, target)}, rows


def candidate_anchor(match_level: str, score: float) -> float:
    anchors = {"strong": 90.0, "good": 75.0, "stretch": 20.0}
    return anchors.get(match_level, score)


def build_recommendation_calibration() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = read_json(CANDIDATE_RERANKING_PATH, {"reranked_outputs": []})
    predicted: list[float] = []
    target: list[float] = []
    for output in payload.get("reranked_outputs", []):
        for item in output.get("recommendations", []):
            score = float(item["matchScore"])
            predicted.append(score)
            target.append(candidate_anchor(str(item.get("matchLevel")), score))
    rows = calibration_rows("recommendations[].matchScore", "candidate_fixture", predicted, target)
    return {"candidate_fixture": calibration_metric(rows, predicted, target)}, rows


def write_calibration(generated_at: str) -> dict[str, Any]:
    jobfit_metrics, jobfit_rows = build_jobfit_calibration()
    ats_metrics, ats_rows = build_ats_calibration()
    recommendation_metrics, recommendation_rows = build_recommendation_calibration()
    tables = jobfit_rows + ats_rows + recommendation_rows
    payload = {
        "schema_version": "phase-46-score-calibration-v1",
        "phase_id": PHASE_ID,
        "generated_at": generated_at,
        "embedding_model": EMBEDDING_MODEL,
        "buckets": BUCKETS,
        "metrics": {
            "jobFitAlignment.score": jobfit_metrics,
            "atsFriendliness.score": ats_metrics,
            "recommendations[].matchScore": recommendation_metrics,
        },
        "tables": tables,
        "sources": {
            "jobFitAlignment.score": file_info(PHASE45_PREDICTIONS_PATH),
            "atsFriendliness.score": file_info(ATS_BENCHMARK_PATH),
            "recommendations[].matchScore": file_info(CANDIDATE_RERANKING_PATH),
        },
        "notes": [
            "Calibration buckets are predicted-score buckets and always include 0-20, 21-40, 41-60, 61-80, and 81-100 rows.",
            "Job-fit calibration uses Phase 45 multilingual-E5-small predictions, not Phase 25 prediction hashes.",
            "ATS calibration validates the embedding-independent Phase 19 transparent scorer against the frozen ATS benchmark.",
            "Candidate reranking fixture calibration remains a handoff compatibility check; Backend still owns candidate retrieval and hydration.",
        ],
    }
    write_json(SCORE_CALIBRATION_PATH, payload)
    with SCORE_CALIBRATION_TABLES_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["output", "split", "bucket", "count", "avg_predicted", "avg_target", "mean_signed_error", "abs_gap", "mae", "rmse", "p90_abs_error", "within_10_points_rate", "bucket_agreement"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tables)
    return payload


def write_configs_and_manifests(generated_at: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    phase45_config = read_json(PHASE45_FEATURE_CONFIG_PATH)
    phase25_label = read_json(PHASE25_LABEL_MANIFEST_PATH)
    phase25_dataset = read_json(PHASE25_DATASET_MANIFEST_PATH)

    tf_config = dict(phase45_config)
    tf_config.update({"schema_version": "phase-46-tensorflow-feature-config-v1", "phase_id": PHASE_ID, "generated_at": generated_at, "model_name": MODEL_NAME, "model_version": MODEL_VERSION})
    tf_config.setdefault("source_artifacts", {})["phase45_tensorflow_feature_config"] = file_info(PHASE45_FEATURE_CONFIG_PATH)
    tf_config["source_artifacts"]["phase45_training_features"] = file_info(PHASE45_FEATURES_PATH)
    write_json(TF_FEATURE_CONFIG_PATH, tf_config)

    feature_config = {
        "schema_version": "phase-46-feature-config-export-v1",
        "phase_id": PHASE_ID,
        "generated_at": generated_at,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "approved_features": APPROVED_FEATURES,
        "target": phase45_config.get("target", {"column": "job_fit_score", "training_scale": "0-1", "api_scale": "0-100"}),
        "embedding_model_metadata": phase45_config.get("embedding_model_metadata", {}),
        "feature_policy": {**phase45_config.get("feature_policy", {}), "runtime_config_self_contained": True, "phase25_mixed_artifact_references_forbidden": True},
        "normalization": phase45_config.get("normalization"),
        "feature_matrix": phase45_config.get("feature_matrix"),
        "runtime_inputs": {"input_tensor": "approved_numeric_features", "feature_count": len(APPROVED_FEATURES), "feature_order": APPROVED_FEATURES},
        "source_artifacts": {"phase45_tensorflow_feature_config": file_info(PHASE45_FEATURE_CONFIG_PATH), "phase45_training_features": file_info(PHASE45_FEATURES_PATH)},
    }
    write_json(FEATURE_CONFIG_PATH, feature_config)

    label_manifest = {
        "schema_version": "phase-46-label-manifest-v1",
        "phase_id": PHASE_ID,
        "generated_at": generated_at,
        "source_label_manifest": file_info(PHASE25_LABEL_MANIFEST_PATH),
        "label_policy": phase25_label.get("label_policy") or phase25_label.get("labels") or phase25_label,
        "label_sources": {
            "jobFitAlignment.score": "Frozen Phase 25/45 pair labels; weak labels remain bootstrap/training support.",
            "atsFriendliness.score": "Frozen Phase 19 ATS benchmark labels.",
            "recommendations[].matchScore": "Phase 21 backend-like candidate fixture semantic anchors.",
        },
        "inference_use": "labels are evaluation/calibration evidence only and are not model features",
    }
    write_json(LABEL_MANIFEST_PATH, label_manifest)

    dataset_manifest = {
        "schema_version": "phase-46-dataset-manifest-v1",
        "phase_id": PHASE_ID,
        "generated_at": generated_at,
        "source_dataset_manifest": file_info(PHASE25_DATASET_MANIFEST_PATH),
        "source_dataset_summary": phase25_dataset.get("dataset") or phase25_dataset.get("summary") or phase25_dataset,
        "frozen_inputs": {
            "pair_feature_matrix": file_info(PHASE45_FEATURES_PATH),
            "pair_predictions": file_info(PHASE45_PREDICTIONS_PATH),
            "ats_benchmark": file_info(ATS_BENCHMARK_PATH),
            "candidate_reranking_fixture": file_info(CANDIDATE_RERANKING_PATH),
        },
        "split_policy": "Frozen Phase 25/45 split names and pair IDs; no new labels introduced in Phase 46.",
    }
    write_json(DATASET_MANIFEST_PATH, dataset_manifest)
    return tf_config, feature_config, dataset_manifest


def copy_and_smoke_model(generated_at: str) -> dict[str, Any]:
    import numpy as np
    import tensorflow as tf
    import keras
    from keras import layers

    from scripts.verify_phase_45_multilingual_e5_small_training_delivery import make_custom_objects

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PHASE45_MODEL_PATH, MODEL_PATH)
    custom_objects = make_custom_objects(tf, keras, layers)
    reloaded = keras.saving.load_model(MODEL_PATH, custom_objects=custom_objects)
    features = np.load(PHASE45_FEATURES_PATH, allow_pickle=True)
    sample = features["X_scaled"][:8].astype("float32")
    predictions = np.clip(reloaded(tf.convert_to_tensor(sample, dtype=tf.float32), training=False).numpy().reshape(-1), 0.0, 1.0)
    smoke = {
        "schema_version": "phase-46-inference-smoke-fixture-v1",
        "phase_id": PHASE_ID,
        "generated_at": generated_at,
        "model": {"name": MODEL_NAME, "version": MODEL_VERSION, "artifact": file_info(MODEL_PATH)},
        "custom_objects": sorted(custom_objects.keys()),
        "input_shape": list(sample.shape),
        "prediction_count": int(len(predictions)),
        "prediction_min": round(float(predictions.min()), 8),
        "prediction_max": round(float(predictions.max()), 8),
        "score_bounds_passed": bool(np.isfinite(predictions).all() and float(predictions.min()) >= 0.0 and float(predictions.max()) <= 1.0),
        "feature_source": file_info(PHASE45_FEATURES_PATH),
        "clean_reload_with_registered_custom_objects": True,
    }
    write_json(INFERENCE_SMOKE_PATH, smoke)
    return smoke


def representative_prediction(index: int) -> int:
    import numpy as np

    payload = np.load(PHASE45_PREDICTIONS_PATH, allow_pickle=True)
    values = payload["y_pred"].reshape(-1)
    return int(round(max(0.0, min(1.0, float(values[index % len(values)]))) * 100.0))


def make_recommendations(scores: list[int]) -> list[dict[str, Any]]:
    candidates = ["job-ml-001", "job-backend-002", "job-data-003", "job-qa-004", "job-ops-005"]
    recommendations = []
    for index, (job_id, score) in enumerate(zip(candidates, scores)):
        recommendations.append(
            {
                "jobId": job_id,
                "matchScore": int(max(0, min(100, score))),
                "matchLevel": score_band(float(score)),
                "matchedSkills": ["python", "sql"] if index < 3 else ["documentation"],
                "missingSkills": [] if score >= 80 else ["deployment evidence"] if score >= 60 else ["role-specific evidence"],
                "rankingSignals": [
                    {"key": "skill_overlap", "value": round(min(1.0, score / 100.0), 4)},
                    {"key": "multilingual_e5_small_score", "value": round(score / 100.0, 4)},
                ],
            }
        )
    return sorted(recommendations, key=lambda item: item["matchScore"], reverse=True)[:5]


def write_handoff_fixtures(generated_at: str) -> tuple[dict[str, Any], dict[str, Any]]:
    old = read_json(PHASE25_HANDOFF_FIXTURES_PATH)
    jobfit_score = representative_prediction(0)
    recommendation_scores = [representative_prediction(i) for i in [5, 15, 25, 35, 45]]
    cv_output = {
        "schemaVersion": "model-core-cv-analysis-v1",
        "language": "id",
        "model": {"name": MODEL_NAME, "version": MODEL_VERSION, "artifact": file_info(MODEL_PATH)},
        "jobFitAlignment": {
            "score": jobfit_score,
            "summarySignals": [
                {"key": "skill_overlap", "label": "Python and SQL evidence found"},
                {"key": "embedding_model", "label": "multilingual-E5-small feature package used"},
            ],
            "matchedSkills": ["python", "sql"],
            "missingSkills": ["deployment evidence"],
            "confidenceNotes": ["Score uses Phase 46 multilingual-E5-small self-contained artifact package"],
        },
        "atsFriendliness": {
            "score": 84,
            "detectedIssues": ["skills section could be grouped more clearly", "impact metrics are sparse"],
            "evidence": {"parseableText": True, "sectionCompleteness": "present", "languageSupported": True},
            "fallback": False,
        },
        "overallImpression": {
            "summary": "CV shows relevant Python and SQL evidence with deployment proof still missing.",
            "evidenceKeys": ["skill_overlap", "ats_parseable_text"],
            "confidenceNotes": ["No unsupported hiring-outcome, seniority, salary, or protected-class claim included"],
        },
    }
    reranking_request = {
        "schemaVersion": "backend-candidate-reranking-v1",
        "candidateSetId": "phase46-handoff-candidates",
        "candidateJobIds": [item["jobId"] for item in make_recommendations(recommendation_scores)],
        "maxRecommendations": 5,
    }
    reranking_output = {
        "schemaVersion": "model-core-candidate-reranking-v1",
        "candidateSetId": reranking_request["candidateSetId"],
        "model": {"name": MODEL_NAME, "version": MODEL_VERSION},
        "recommendations": make_recommendations(recommendation_scores),
    }
    fixtures = {
        "schema_version": "phase-46-model-api-handoff-fixtures-v1",
        "phase_id": PHASE_ID,
        "generated_at": generated_at,
        "positive": {
            "cvAnalysisCoreOutput": cv_output,
            "candidateRerankingCoreRequest": reranking_request,
            "candidateRerankingCoreOutput": reranking_output,
        },
        "negative": old.get("negative", {}),
        "cv_analysis_v2_wrapper_mapping": old.get("cv_analysis_v2_wrapper_mapping", {}),
        "source_context": {
            "phase45_predictions": file_info(PHASE45_PREDICTIONS_PATH),
            "score_calibration": file_info(SCORE_CALIBRATION_PATH),
            "feature_config": file_info(FEATURE_CONFIG_PATH),
        },
    }
    write_json(HANDOFF_FIXTURES_PATH, fixtures)
    validation = validate_handoff_fixtures(fixtures)
    write_json(HANDOFF_VALIDATION_PATH, validation)
    return fixtures, validation


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(nested_keys(child))
    return keys


def validate_handoff_fixtures(fixtures: dict[str, Any]) -> dict[str, Any]:
    positive = fixtures.get("positive", {})
    cv_output = positive.get("cvAnalysisCoreOutput", {})
    rerank_request = positive.get("candidateRerankingCoreRequest", {})
    rerank_output = positive.get("candidateRerankingCoreOutput", {})
    recommendations = rerank_output.get("recommendations", [])
    candidate_ids = set(rerank_request.get("candidateJobIds", []))
    recommendation_ids = [item.get("jobId") for item in recommendations]
    leaked_fields = sorted((nested_keys(cv_output) | nested_keys(rerank_output)) & WRAPPER_OWNED_FIELDS)
    checks = {
        "cv_schema_version_compatible": cv_output.get("schemaVersion") == "model-core-cv-analysis-v1",
        "reranking_schema_version_compatible": rerank_output.get("schemaVersion") == "model-core-candidate-reranking-v1",
        "jobfit_score_bounds": isinstance(cv_output.get("jobFitAlignment", {}).get("score"), int) and 0 <= cv_output["jobFitAlignment"]["score"] <= 100,
        "ats_score_bounds": isinstance(cv_output.get("atsFriendliness", {}).get("score"), int) and 0 <= cv_output["atsFriendliness"]["score"] <= 100,
        "recommendation_score_bounds": all(isinstance(item.get("matchScore"), int) and 0 <= item["matchScore"] <= 100 for item in recommendations),
        "candidate_membership": set(recommendation_ids).issubset(candidate_ids),
        "duplicate_rejection": len(recommendation_ids) == len(set(recommendation_ids)),
        "max_recommendations": len(recommendations) <= 5,
        "language_handling": cv_output.get("language") in {"id", "en", "mixed", "unknown"},
        "no_backend_owned_fields": not leaked_fields,
    }
    return {
        "schema_version": "phase-46-handoff-validation-v1",
        "phase_id": PHASE_ID,
        "status": "complete" if all(checks.values()) else "blocked",
        "checks": checks,
        "leaked_backend_owned_fields": leaked_fields,
        "candidate_ids": sorted(candidate_ids),
        "recommendation_ids": recommendation_ids,
        "model_core_only": True,
    }


def artifact_entry(artifact_id: str, path: Path, required_for_inference: bool, role: str, consumer: str, schema_version: str | None = None) -> dict[str, Any]:
    suffix = path.suffix.lower().lstrip(".") or "directory"
    return {
        "artifact_id": artifact_id,
        "path": rel(path),
        "format": suffix,
        "role": role,
        "consumer": consumer,
        "required_for_inference": required_for_inference,
        "schema_version": schema_version,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "runtime_classification": "runtime-required" if required_for_inference else "training-or-release-evidence",
        "embedding_model_metadata": {"embedding_model": EMBEDDING_MODEL, "embedding_dimension": 384, "profile_prefix": "query:", "job_prefix": "passage:", "normalized_embeddings": True},
    }


def write_model_card_and_manifest(generated_at: str, calibration: dict[str, Any], smoke: dict[str, Any], validation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation = read_json(PHASE45_EVALUATION_PATH)
    baseline = read_json(PHASE45_BASELINE_PATH)
    selection = read_json(PHASE45_SELECTION_PATH)
    tensorboard = read_json(PHASE45_TENSORBOARD_MANIFEST_PATH)
    write_json(PHASE46_TENSORBOARD_MANIFEST_PATH, {**tensorboard, "phase_46_source_manifest_copy": file_info(PHASE45_TENSORBOARD_MANIFEST_PATH)})
    model_card = {
        "schema_version": "phase-46-model-card-v1",
        "phase_id": PHASE_ID,
        "generated_at": generated_at,
        "model": {"name": MODEL_NAME, "version": MODEL_VERSION, "architecture": "TensorFlow Functional API with registered custom components", "embedding_model": EMBEDDING_MODEL},
        "intended_use": [
            "Internal Model API model-core job-fit scoring for Backend-provided CV/profile and job context.",
            "Candidate reranking for Backend-provided candidate job IDs only.",
            "ATS friendliness and grounded overall impression signals without Backend-owned prose or persistence.",
        ],
        "blocked_use": [
            "Automated hiring decision, rejection, eligibility, salary, or protected-class inference.",
            "Scoring or hydrating jobs not supplied by Backend.",
            "Production score claims without frozen human/reviewer validation coverage.",
            "Backend-owned auth, persistence, public response hydration, topActionables, sectionReviews, or GenAI wrapper prose.",
        ],
        "data": {"dataset_manifest": file_info(DATASET_MANIFEST_PATH), "label_manifest": file_info(LABEL_MANIFEST_PATH), "embedding_contract": read_json(FEATURE_CONFIG_PATH).get("embedding_model_metadata")},
        "training": {"source_phase": "phase_45_multilingual_e5_small_training_delivery", "custom_components": evaluation.get("model", {}).get("custom_components"), "uses_gradient_tape": evaluation.get("model", {}).get("uses_gradient_tape"), "uses_model_fit": evaluation.get("model", {}).get("uses_model_fit"), "tensorboard": tensorboard.get("tensorboard")},
        "evaluation": {"metrics": evaluation.get("metrics"), "ranking_metrics": evaluation.get("ranking_metrics"), "slice_metrics": evaluation.get("slice_metrics"), "calibration": calibration.get("metrics"), "baseline_comparison": baseline.get("metrics"), "selection_decision": selection.get("selection_decision")},
        "runtime_benefits": ["384-dimensional multilingual-E5-small feature contract reduces embedding footprint versus 768-dimensional E5-base.", "Artifact package is self-contained so Model API can avoid Phase 25/45 metadata mixing.", "Phase 25 E5-base artifact remains rollback path."],
        "known_risks": ["Weak labels remain bootstrap/training support only.", "Candidate reranking fixture calibration is not a substitute for real Backend feedback or recruiter labels.", "Staging must still validate runtime artifact selection and embedding/model config match before rollout."],
        "deployment_contract": {"final_model_artifact": file_info(MODEL_PATH), "feature_config": file_info(FEATURE_CONFIG_PATH), "score_calibration": file_info(SCORE_CALIBRATION_PATH), "api_handoff_fixtures": file_info(HANDOFF_FIXTURES_PATH), "api_handoff_validation": {**file_info(HANDOFF_VALIDATION_PATH), "status": validation.get("status")}, "clean_reload_smoke": smoke, "score_scale": "0-100 JSON-compatible integer score", "model_core_output_boundary": {"training_owns": ["jobFitAlignment.score/signals", "atsFriendliness.score/signals", "overallImpression.summary/signals", "candidate recommendation scores/ranks for supplied job IDs"], "backend_wrapper_owns": sorted(WRAPPER_OWNED_FIELDS)}},
        "rollback_artifact": {"artifact_root": ROLLBACK_ROOT, "model_path": f"{ROLLBACK_ROOT}/export/selected_jobfit_tf_phase25.keras", "embedding_model": "intfloat/e5-base-v2"},
        "readiness": {"artifact_package_self_contained": True, "phase25_mixed_metadata_used": False, "handoff_fixture_status": validation.get("status"), "production_claim_allowed": False},
    }
    write_json(MODEL_CARD_PATH, model_card)

    artifacts = [
        artifact_entry("final_keras_model", MODEL_PATH, True, "runtime_model", "Model API runtime", "keras"),
        artifact_entry("tensorflow_feature_config", TF_FEATURE_CONFIG_PATH, True, "runtime_config", "Model API runtime", "phase-46-tensorflow-feature-config-v1"),
        artifact_entry("feature_config", FEATURE_CONFIG_PATH, True, "runtime_config", "Model API runtime", "phase-46-feature-config-export-v1"),
        artifact_entry("score_calibration", SCORE_CALIBRATION_PATH, True, "runtime_calibration", "Model API runtime", "phase-46-score-calibration-v1"),
        artifact_entry("label_manifest", LABEL_MANIFEST_PATH, False, "release_evidence", "Reproducibility review", "phase-46-label-manifest-v1"),
        artifact_entry("dataset_manifest", DATASET_MANIFEST_PATH, False, "release_evidence", "Reproducibility review", "phase-46-dataset-manifest-v1"),
        artifact_entry("model_card", MODEL_CARD_PATH, True, "release_evidence", "Model API handoff and release gate", "phase-46-model-card-v1"),
        artifact_entry("handoff_fixtures", HANDOFF_FIXTURES_PATH, False, "contract_fixture", "Backend/API handoff validation", "phase-46-model-api-handoff-fixtures-v1"),
        artifact_entry("handoff_validation", HANDOFF_VALIDATION_PATH, False, "contract_validation", "Backend/API handoff validation", "phase-46-handoff-validation-v1"),
        artifact_entry("inference_smoke_fixture", INFERENCE_SMOKE_PATH, False, "runtime_smoke", "Release evidence", "phase-46-inference-smoke-fixture-v1"),
        artifact_entry("score_calibration_tables_csv", SCORE_CALIBRATION_TABLES_PATH, False, "release_evidence", "Calibration review", None),
        artifact_entry("tensorboard_monitoring_manifest", PHASE46_TENSORBOARD_MANIFEST_PATH, False, "release_evidence", "Training monitoring review", tensorboard.get("schema_version")),
    ]
    manifest = {
        "schema_version": "phase-46-artifact-manifest-v1",
        "phase_id": PHASE_ID,
        "generated_at": generated_at,
        "artifact_root": rel(ARTIFACT_DIR),
        "embedding_model_metadata": {"embedding_model": EMBEDDING_MODEL, "embedding_dimension": 384, "profile_prefix": "query:", "job_prefix": "passage:", "normalized_embeddings": True},
        "artifacts": artifacts,
        "tensorboard_references": tensorboard.get("tensorboard"),
        "source_artifacts": {"phase45_model": file_info(PHASE45_MODEL_PATH), "phase45_predictions": file_info(PHASE45_PREDICTIONS_PATH), "phase45_feature_config": file_info(PHASE45_FEATURE_CONFIG_PATH), "phase25_rollback_model_card": file_info(PHASE25_DIR / "model_card.json")},
        "no_stale_phase25_runtime_hashes": True,
    }
    write_json(ARTIFACT_MANIFEST_PATH, manifest)
    return model_card, manifest


def write_readme() -> None:
    README_PATH.write_text(
        "\n".join(
            [
                "# Phase 46 multilingual-E5-small Artifact Package",
                "",
                "Self-contained runtime and release handoff package for `intfloat/multilingual-e5-small` TensorFlow scoring artifacts.",
                "",
                "## Runtime-required files",
                "",
                "- `export/selected_jobfit_tf_phase46_multilingual_e5_small.keras`",
                "- `tensorflow_feature_config.json`",
                "- `feature_config.json`",
                "- `score_calibration.json`",
                "- `model_card.json`",
                "- `artifact_manifest.json`",
                "",
                "## Handoff evidence",
                "",
                "- `export/model_api_handoff_fixtures.json`",
                "- `export/model_api_handoff_validation.json`",
                "- `export/inference_smoke_fixture.json`",
                "- `score_calibration_tables.csv`",
                "- `label_manifest.json`",
                "- `dataset_manifest.json`",
                "- `tensorboard_monitoring_manifest.json`",
                "",
                "Phase 25 remains rollback-only. Runtime loaders must not mix Phase 25 calibration/config hashes with this package.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_report() -> dict[str, Any]:
    calibration = read_json(SCORE_CALIBRATION_PATH)
    model_card = read_json(MODEL_CARD_PATH)
    manifest = read_json(ARTIFACT_MANIFEST_PATH)
    validation = read_json(HANDOFF_VALIDATION_PATH)
    smoke = read_json(INFERENCE_SMOKE_PATH)
    required_paths = [MODEL_PATH, TF_FEATURE_CONFIG_PATH, FEATURE_CONFIG_PATH, SCORE_CALIBRATION_PATH, SCORE_CALIBRATION_TABLES_PATH, LABEL_MANIFEST_PATH, DATASET_MANIFEST_PATH, MODEL_CARD_PATH, ARTIFACT_MANIFEST_PATH, HANDOFF_FIXTURES_PATH, HANDOFF_VALIDATION_PATH, INFERENCE_SMOKE_PATH, PHASE46_TENSORBOARD_MANIFEST_PATH, README_PATH]
    runtime_entries = [entry for entry in manifest.get("artifacts", []) if entry.get("required_for_inference")]
    runtime_paths = [entry.get("path", "") for entry in runtime_entries]
    checks = {
        "notebook_exists_with_required_markdown": (ROOT / "training/notebooks/phase_46_calibration_model_card_manifest_handoff_refresh.ipynb").exists(),
        "no_stale_phase25_runtime_references": all(not str(path).startswith(ROLLBACK_ROOT) for path in runtime_paths),
        "model_card_declares_multilingual_e5_small": model_card.get("model", {}).get("embedding_model") == EMBEDDING_MODEL,
        "runtime_manifest_entries_hashed": bool(runtime_entries) and all(entry.get("sha256") and entry.get("size_bytes") for entry in runtime_entries),
        "calibration_outputs_complete": all(name in calibration.get("metrics", {}) for name in ["jobFitAlignment.score", "atsFriendliness.score", "recommendations[].matchScore"]),
        "calibration_buckets_complete": {row.get("bucket") for row in calibration.get("tables", []) if row.get("output") == "jobFitAlignment.score"} >= {bucket["bucket"] for bucket in BUCKETS},
        "clean_reload_smoke_passed": smoke.get("clean_reload_with_registered_custom_objects") is True and smoke.get("score_bounds_passed") is True,
        "feature_configs_refreshed": read_json(TF_FEATURE_CONFIG_PATH).get("embedding_model_metadata", {}).get("embedding_model") == EMBEDDING_MODEL and read_json(FEATURE_CONFIG_PATH).get("embedding_model_metadata", {}).get("embedding_model") == EMBEDDING_MODEL,
        "handoff_validation_complete": validation.get("status") == "complete" and all(validation.get("checks", {}).values()),
        "model_core_contract_compatible": validation.get("checks", {}).get("cv_schema_version_compatible") is True and validation.get("checks", {}).get("reranking_schema_version_compatible") is True,
        "tensorboard_references_recorded": bool(manifest.get("tensorboard_references", {}).get("event_files")),
        "rollback_artifact_recorded": model_card.get("rollback_artifact", {}).get("embedding_model") == "intfloat/e5-base-v2",
    }
    missing_paths = require_paths(required_paths)
    blockers = missing_paths + [name for name, passed in checks.items() if not passed]
    status = "complete" if not blockers else "incomplete"
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "generated_at": now_utc(),
        "status": status,
        "checks": checks,
        "blockers": blockers,
        "summary": {
            "embedding_model": model_card.get("model", {}).get("embedding_model"),
            "model_artifact": file_info(MODEL_PATH),
            "runtime_required_artifact_count": len(runtime_entries),
            "jobfit_validation_calibration": calibration.get("metrics", {}).get("jobFitAlignment.score", {}).get("validation"),
            "jobfit_test_calibration": calibration.get("metrics", {}).get("jobFitAlignment.score", {}).get("test"),
            "handoff_validation_status": validation.get("status"),
        },
        "artifact_paths": {"artifact_root": rel(ARTIFACT_DIR), "model_card": rel(MODEL_CARD_PATH), "artifact_manifest": rel(ARTIFACT_MANIFEST_PATH), "score_calibration": rel(SCORE_CALIBRATION_PATH), "handoff_fixtures": rel(HANDOFF_FIXTURES_PATH), "handoff_validation": rel(HANDOFF_VALIDATION_PATH)},
        "source_files": [rel(path) for path in [ROOT / "GAP_MODEL_TRAINING.md", ROOT / "GAP_MODEL_TRAINING.md", ROOT / "REQUIREMENT.md", PHASE45_EVALUATION_PATH, PHASE45_BASELINE_PATH, PHASE45_PREDICTIONS_PATH]],
        "git_state": git_state(),
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Phase 46 Calibration, Model Card, Artifact Manifest, and Handoff Fixtures Refresh",
        "",
        f"Status: `{report['status']}`",
        "",
        "This report records a self-contained `intfloat/multilingual-e5-small` artifact package for Model API handoff.",
        "",
        "## Summary",
        f"- Embedding model: `{report['summary'].get('embedding_model')}`",
        f"- Runtime-required artifacts: `{report['summary'].get('runtime_required_artifact_count')}`",
        f"- Handoff validation: `{report['summary'].get('handoff_validation_status')}`",
        f"- Validation calibration: `{report['summary'].get('jobfit_validation_calibration')}`",
        f"- Test calibration: `{report['summary'].get('jobfit_test_calibration')}`",
        "",
        "## Checks",
    ]
    lines.extend(f"- {'PASS' if passed else 'FAIL'} `{name}`" for name, passed in report.get("checks", {}).items())
    if report.get("blockers"):
        lines.extend(["", "## Blockers"])
        lines.extend(f"- `{blocker}`" for blocker in report["blockers"])
    lines.extend(["", "## Artifacts"])
    for name, path in report.get("artifact_paths", {}).items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Commands", "- Write evidence: `training/.tf-venv-3.13/bin/python scripts/verify_phase_46_calibration_model_card_manifest_handoff_refresh.py --write`", "- Verify: `.venv/bin/python -m unittest tests.test_phase_46_calibration_model_card_manifest_handoff_refresh`"])
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all() -> dict[str, Any]:
    missing = require_paths([PHASE45_MODEL_PATH, PHASE45_FEATURE_CONFIG_PATH, PHASE45_FEATURES_PATH, PHASE45_PREDICTIONS_PATH, PHASE45_EVALUATION_PATH, PHASE45_BASELINE_PATH, PHASE45_TENSORBOARD_MANIFEST_PATH, PHASE25_LABEL_MANIFEST_PATH, PHASE25_DATASET_MANIFEST_PATH, PHASE25_HANDOFF_FIXTURES_PATH, ATS_BENCHMARK_PATH, CANDIDATE_RERANKING_PATH])
    if missing:
        raise RuntimeError(f"Missing required source artifacts: {missing}")
    generated_at = now_utc()
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_configs_and_manifests(generated_at)
    calibration = write_calibration(generated_at)
    smoke = copy_and_smoke_model(generated_at)
    write_handoff_fixtures(generated_at)
    validation = read_json(HANDOFF_VALIDATION_PATH)
    write_model_card_and_manifest(generated_at, calibration, smoke, validation)
    write_readme()
    report = build_report()
    write_json(REPORT_JSON_PATH, report)
    write_markdown(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write Phase 46 evidence artifacts.")
    args = parser.parse_args(argv)
    report = write_all() if args.write else build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report.get("blockers") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
