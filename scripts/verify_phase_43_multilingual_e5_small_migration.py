#!/usr/bin/env python3
"""Write Phase 43 multilingual-E5-small migration decision evidence.

This phase freezes the current E5-base baseline before any embedding runtime or
artifact-default changes. It intentionally does not download or run
intfloat/multilingual-e5-small.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/phase_43_multilingual_e5_small_migration"
REPORT_JSON_PATH = ROOT / "reports/phase_43_multilingual_e5_small_migration_decision.json"
REPORT_MD_PATH = ROOT / "reports/phase_43_multilingual_e5_small_migration_decision.md"
RUNTIME_BASELINE_PATH = ARTIFACT_ROOT / "e5_base_runtime_baseline.json"
QUALITY_BASELINE_PATH = ARTIFACT_ROOT / "e5_base_quality_baseline.json"
DECISION_PATH = ARTIFACT_ROOT / "migration_decision_risk_register.json"

PHASE25_REPORT = ROOT / "reports/phase_25_tensorflow_training_delivery.json"
PHASE25_MODEL_CARD = ROOT / "artifacts/phase_25_tensorflow_training_delivery/model_card.json"
PHASE25_MANIFEST = ROOT / "artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json"
PHASE25_TF_FEATURE_CONFIG = ROOT / "artifacts/phase_25_tensorflow_training_delivery/tensorflow_feature_config.json"
PHASE25_CALIBRATION = ROOT / "artifacts/phase_25_tensorflow_training_delivery/score_calibration.json"
PHASE25_DATA_REUSE = ROOT / "reports/phase_25_data_feature_reuse.json"
PHASE25_SELECTION = ROOT / "reports/phase_25_baseline_selection_gate.json"
PHASE25_HANDOFF_FIXTURES = ROOT / "artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_fixtures.json"
PHASE27_PRODUCTION_SMOKE = ROOT / "reports/phase_27_9_model_api_production_smoke.json"
WARMUP_REPORT = ROOT / "reports/ai_cv_analyzer_warmup_staging.json"
PHASE39_COPY_REPORT = ROOT / "reports/phase_39_product_copy_localization.json"

SOURCE_PATHS = (
    PHASE25_REPORT,
    PHASE25_MODEL_CARD,
    PHASE25_MANIFEST,
    PHASE25_TF_FEATURE_CONFIG,
    PHASE25_CALIBRATION,
    PHASE25_DATA_REUSE,
    PHASE25_SELECTION,
    PHASE25_HANDOFF_FIXTURES,
    PHASE27_PRODUCTION_SMOKE,
    WARMUP_REPORT,
    PHASE39_COPY_REPORT,
    ROOT / "GAP_MODEL_TRAINING.md",
    ROOT / "REQUIREMENT.md",
    ROOT / "model_api/README.md",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


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
    head = run_text(["git", "rev-parse", "HEAD"])
    porcelain = run_text(["git", "status", "--porcelain"]) or ""
    dirty_lines = [line for line in porcelain.splitlines() if line]
    return {
        "commit": head,
        "dirty_file_count": len(dirty_lines),
        "dirty_sample": dirty_lines[:20],
        "production_claim_allowed": len(dirty_lines) == 0,
    }


def system_snapshot() -> dict[str, Any]:
    hf_cache = Path.home() / ".cache/huggingface/hub/models--intfloat--e5-base-v2"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_brand": run_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": run_text(["sysctl", "-n", "hw.memsize"]),
        "disk_usage_repo": shutil.disk_usage(ROOT)._asdict(),
        "phase25_artifact_root_size_bytes": dir_size_bytes(ROOT / "artifacts/phase_25_tensorflow_training_delivery"),
        "e5_base_hf_cache_path": str(hf_cache),
        "e5_base_hf_cache_size_bytes": dir_size_bytes(hf_cache),
        "docker_image_size": {
            "status": "not_found_in_local_docker_images",
            "command": "docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E 'bisakerja|model'",
            "note": "No matching local Docker image was present during Phase 43 local capture; staging/VPS capture must fill this before rollout.",
        },
    }


def build_runtime_baseline(generated_at: str) -> dict[str, Any]:
    phase25_report = read_json(PHASE25_REPORT)
    model_card = read_json(PHASE25_MODEL_CARD)
    tf_config = read_json(PHASE25_TF_FEATURE_CONFIG)
    smoke = read_json(PHASE27_PRODUCTION_SMOKE)
    warmup = read_json(WARMUP_REPORT)

    return {
        "schema_version": "phase-43-e5-base-runtime-baseline-v1",
        "phase_id": "phase_43_multilingual_e5_small_migration",
        "generated_at": generated_at,
        "baseline_model": {
            "embedding_model": model_card.get("data", {}).get("embedding_contract", {}).get("embedding_model", "intfloat/e5-base-v2"),
            "embedding_dimension": model_card.get("data", {}).get("embedding_contract", {}).get("embedding_dimension"),
            "embedding_backend": model_card.get("data", {}).get("embedding_contract", {}).get("backend"),
            "profile_prefix": model_card.get("data", {}).get("embedding_contract", {}).get("profile_prefix"),
            "job_prefix": model_card.get("data", {}).get("embedding_contract", {}).get("job_prefix"),
            "normalized_embeddings": model_card.get("data", {}).get("embedding_contract", {}).get("normalized_embeddings"),
            "tensorflow_model_version": phase25_report.get("requirement_targets", {}).get("tensorflow_export", {}).get("path"),
            "model_card_version": model_card.get("model", {}).get("version"),
            "artifact_hash": phase25_report.get("requirement_targets", {}).get("tensorflow_export", {}).get("sha256")
            or model_card.get("deployment_contract", {}).get("final_model_artifact", {}).get("sha256"),
            "feature_order": tf_config.get("approved_features"),
        },
        "endpoint_latency_baseline": {
            "local_live_probe_2026_06_04": {
                "command": "MODEL_API_ENV=local MODEL_API_SERVICE_TOKEN=<local-service-token> .venv/bin/python -m uvicorn model_api.app:create_app --factory --host 127.0.0.1 --port 8011; curl /live,/health,/ready,/model-info",
                "live": {"status_code": 200, "time_total_seconds": 0.003328, "body_key": "live=true"},
                "health_loading": {"status_code": 200, "time_total_seconds": 0.003029, "body_key": "message=loading"},
                "ready_loading": {"status_code": 200, "time_total_seconds": 0.002189, "body_key": "ready=false while TensorFlow loads"},
                "model_info_loading": {"status_code": 200, "time_total_seconds": 0.006660, "body_key": "ready=false, artifact paths exposed to internal caller only"},
            },
            "phase27_live_fastapi_first_cv_analysis": smoke.get("latency"),
            "phase27_endpoint_latency_ms": smoke.get("gates", [{}])[-1].get("evidence", {}).get("endpoint_latency_ms") if smoke.get("gates") else {},
            "warmup_ready_before": warmup.get("ready_before"),
            "warmup_ready_after": warmup.get("ready_after"),
            "warm_internal_cv_analysis": warmup.get("warmup"),
        },
        "runtime_commands": {
            "model_api_local_start": "MODEL_API_ENV=local MODEL_API_SERVICE_TOKEN=<token> uvicorn model_api.app:create_app --factory --host 127.0.0.1 --port 8000",
            "live": "curl http://127.0.0.1:8000/live",
            "health": "curl http://127.0.0.1:8000/health",
            "ready": "curl http://127.0.0.1:8000/ready",
            "model_info": "curl -H 'authorization: Bearer ${MODEL_API_SERVICE_TOKEN}' http://127.0.0.1:8000/model-info",
            "first_internal_cv_analysis": "python scripts/verify_phase_27_9_model_api_production_smoke.py --write --run-live",
            "warm_internal_cv_analysis": "python scripts/warmup_ai_cv_analyzer_runtime.py --model-api-url http://127.0.0.1:8000 --token ${MODEL_API_SERVICE_TOKEN} --latency-budget-ms 30000 --output reports/ai_cv_analyzer_warmup_staging.json",
            "resource_inventory": "du -sh artifacts/phase_25_tensorflow_training_delivery ~/.cache/huggingface/hub/models--intfloat--e5-base-v2; df -h .; docker images",
        },
        "resource_baseline": system_snapshot(),
        "git_state": git_state(),
        "source_reports": {
            "phase25_final_gate": file_info(PHASE25_REPORT),
            "phase27_model_api_smoke": file_info(PHASE27_PRODUCTION_SMOKE),
            "warmup_report": file_info(WARMUP_REPORT),
            "model_card": file_info(PHASE25_MODEL_CARD),
            "tensorflow_feature_config": file_info(PHASE25_TF_FEATURE_CONFIG),
        },
        "limitations": [
            "Local Phase 43 probe captured process liveness/readiness behavior, not VPS/container production load.",
            "Docker image size was not available in local Docker images and must be captured in Phase 48 staging deployment evidence.",
            "No multilingual-E5-small runtime was executed in this phase; Phase 44 owns embedding compatibility measurement.",
        ],
    }


def representative_examples(fixtures: dict[str, Any]) -> dict[str, Any]:
    positive = fixtures.get("positive", {})
    cv_core = positive.get("cvAnalysisCoreOutput") or positive.get("cv_analysis_core_output") or {}
    rerank = positive.get("candidateRerankingCoreOutput") or {}
    return {
        "cv_analysis_core_output_excerpt": {
            "schemaVersion": cv_core.get("schemaVersion"),
            "language": cv_core.get("language"),
            "jobFitAlignment": cv_core.get("jobFitAlignment"),
            "atsFriendliness": cv_core.get("atsFriendliness"),
            "overallImpression": cv_core.get("overallImpression"),
        },
        "candidate_reranking_top_examples": rerank.get("recommendations", [])[:3],
        "public_response_mapping": fixtures.get("cv_analysis_v2_wrapper_mapping", {}).get("model_core_to_wrapper", []),
    }


def build_quality_baseline(generated_at: str) -> dict[str, Any]:
    phase25_report = read_json(PHASE25_REPORT)
    model_card = read_json(PHASE25_MODEL_CARD)
    calibration = read_json(PHASE25_CALIBRATION)
    data_reuse = read_json(PHASE25_DATA_REUSE)
    selection = read_json(PHASE25_SELECTION)
    fixtures = read_json(PHASE25_HANDOFF_FIXTURES)
    copy_report = read_json(PHASE39_COPY_REPORT)

    return {
        "schema_version": "phase-43-e5-base-quality-baseline-v1",
        "phase_id": "phase_43_multilingual_e5_small_migration",
        "generated_at": generated_at,
        "model_version": model_card.get("model", {}).get("version"),
        "artifact_hash": model_card.get("deployment_contract", {}).get("final_model_artifact", {}).get("sha256"),
        "embedding_contract": model_card.get("data", {}).get("embedding_contract"),
        "jobfit_metrics": phase25_report.get("metrics"),
        "baseline_comparison": selection.get("comparison_rows"),
        "strict_gate_checks": selection.get("strict_gate_checks"),
        "calibration_metrics": calibration.get("metrics"),
        "calibration_tables": calibration.get("tables"),
        "score_distribution": {
            "pair_score_band_counts": data_reuse.get("pair_dataset", {}).get("score_band_counts"),
            "split_counts": data_reuse.get("pair_dataset", {}).get("split_counts"),
            "language_counts": data_reuse.get("pair_dataset", {}).get("language_counts"),
            "pair_type_counts": data_reuse.get("pair_dataset", {}).get("pair_type_counts"),
            "score_range": data_reuse.get("pair_dataset", {}).get("score_range"),
        },
        "candidate_reranking_examples": representative_examples(fixtures).get("candidate_reranking_top_examples"),
        "representative_cv_analyzer_outputs": representative_examples(fixtures),
        "public_copy_baseline": {
            "language_policy": copy_report.get("language_policy"),
            "sample_before_after": copy_report.get("sample_before_after"),
            "decision": copy_report.get("review_result"),
        },
        "source_reports": {
            "phase25_final_gate": file_info(PHASE25_REPORT),
            "model_card": file_info(PHASE25_MODEL_CARD),
            "calibration": file_info(PHASE25_CALIBRATION),
            "data_feature_reuse": file_info(PHASE25_DATA_REUSE),
            "baseline_selection": file_info(PHASE25_SELECTION),
            "handoff_fixtures": file_info(PHASE25_HANDOFF_FIXTURES),
            "phase39_copy_report": file_info(PHASE39_COPY_REPORT),
        },
    }


def build_decision(generated_at: str) -> dict[str, Any]:
    return {
        "schema_version": "phase-43-multilingual-e5-small-migration-decision-v1",
        "phase_id": "phase_43_multilingual_e5_small_migration",
        "generated_at": generated_at,
        "decision": "approved_for_staging_experiment_only",
        "current_baseline": {
            "embedding_model": "intfloat/e5-base-v2",
            "artifact_root": "artifacts/phase_25_tensorflow_training_delivery/",
            "rollback_policy": "Phase 25 artifacts remain immutable rollback baseline; migration artifacts must live under Phase 43+ namespaces.",
        },
        "migration_hypothesis": [
            "intfloat/multilingual-e5-small should reduce startup latency and first embedding latency versus intfloat/e5-base-v2.",
            "Smaller model/cache should reduce RAM pressure, disk use, image pull/cache warmup time, and timeout risk on VPS/container staging.",
            "Multilingual training may improve Indonesian/English mixed CV-job semantic coverage versus English-focused E5-base behavior.",
            "Lower resource use may make warmup more reliable before demo/staging traffic.",
        ],
        "migration_risks": [
            "Cosine distribution drift can invalidate Phase 25 e5_cosine normalization mean/std.",
            "Score calibration drift can make 0-100 score semantics inaccurate without recalibration.",
            "Ranking order can change, including top recommendation swaps that affect user-facing hydrated jobs.",
            "English-only semantic quality may regress even if Indonesian slices improve.",
            "Multilingual tokenization can change skill/requirement similarity and false-positive overlap behavior.",
            "User-facing recommendation prose may change because Backend wrapper uses model-core evidence and rank order.",
            "Direct runtime swap can silently mix E5-base artifacts with multilingual-E5-small embeddings unless artifact/runtime checks are added.",
        ],
        "go_no_go_thresholds": {
            "runtime": {
                "live_health_ready": "All /live, /health, and /ready probes must return 200; /ready.ready=true after warmup before traffic.",
                "first_internal_cv_analysis_ms": "Must be <= current E5-base first inference baseline and <= 30000 ms staging timeout budget.",
                "warm_internal_cv_analysis_ms": "Must be <= 5000 ms and preferably improve E5-base warm baseline by >= 30%.",
                "embedding_latency_ms": "Must be <= E5-base warm embedding baseline and preferably improve by >= 30%.",
                "ram": "Peak RSS must not exceed E5-base baseline; staging target is >= 25% reduction or explicit no-go review.",
                "disk_cache": "HF/E5 cache size must be smaller than E5-base cache; any larger cache requires explicit rejection or infra approval.",
                "docker_image": "Image size must not increase versus E5-base staging image; missing image-size evidence blocks rollout beyond experiment.",
            },
            "quality": {
                "mae_normalized": "Validation and test MAE must remain <= 0.02 and no worse than Phase 25 by more than 0.002 normalized.",
                "mae_points": "Validation/test MAE must remain <= 2.0 points and no worse than Phase 25 by more than 0.25 points.",
                "r2": "Validation/test R² must remain >= 0.98 or trigger model rejection review.",
                "spearman": "Validation/test Spearman must remain >= 0.99 and no lower than Phase 25 by more than 0.01.",
                "score_band_agreement": "Validation/test score-band agreement must remain >= 0.98 and no lower than Phase 25 by more than 0.01.",
                "high_fit_recall": "Validation/test high-fit recall must remain >= 0.98 and no lower than Phase 25 by more than 0.02.",
                "calibration": "ECE must not increase by more than 0.5 points and max bucket gap must remain <= 3.5 points after recalibration.",
                "ranking": "NDCG/MAP must not regress by more than 0.02; top recommendation swaps require review when score delta > 5 points.",
                "language_slices": "ID/EN/MIXED/UNKNOWN slices must be reported; any ID improvement cannot mask EN regression beyond allowed deltas.",
            },
            "contract": {
                "schema": "Model-core response must stay model-core-cv-analysis-v1 compatible unless a later phase explicitly versions it.",
                "candidate_membership": "All recommendations must remain subset of backend-provided candidates, max 5, unique job IDs.",
                "owned_fields": "No backend/wrapper-owned fields may appear in model-core output.",
            },
            "rollback": {
                "artifact_switch": "Phase 25 E5-base artifact paths and env must remain documented and selectable.",
                "no_overwrite": "Do not overwrite artifacts/phase_25_tensorflow_training_delivery or selected_jobfit_tf_phase25.keras.",
                "rollback_drill": "Staging rollback command must be tested before default promotion.",
            },
        },
        "artifact_namespace": {
            "reserved_root": "artifacts/phase_43_multilingual_e5_small_migration/",
            "later_training_root": "artifacts/phase_45_multilingual_e5_small_training_delivery/",
            "forbidden_root": "artifacts/phase_25_tensorflow_training_delivery/",
            "report_paths": [
                "reports/phase_43_multilingual_e5_small_migration_decision.json",
                "reports/phase_43_multilingual_e5_small_migration_decision.md",
            ],
        },
        "model_version_naming": {
            "reserved_training_model_version": "jobfit_tf_phase43_multilingual_e5_small_v1",
            "later_selected_model_version": "jobfit_tf_phase45_multilingual_e5_small_v1",
            "compatible_core_schema": "model-core-cv-analysis-v1",
            "compatibility_note": "Schema compatibility does not imply score/ranking equivalence; quality, calibration, and shadow comparison must pass first.",
        },
        "stakeholder_decision_record": {
            "status": "approved_for_staging_experiment_only_until_phase44_49_gates_pass",
            "approved_changes_now": [
                "Capture E5-base runtime/quality baseline.",
                "Reserve artifact/model namespaces.",
                "Proceed to Phase 44 embedding compatibility audit.",
            ],
            "blocked_changes_now": [
                "No production/staging env default switch to intfloat/multilingual-e5-small.",
                "No direct artifact swap without retraining/recalibration evidence.",
                "No quality-equivalence claim before shadow comparison and gate evidence.",
            ],
        },
    }


def build_report() -> dict[str, Any]:
    generated_at = now_utc()
    runtime = build_runtime_baseline(generated_at)
    quality = build_quality_baseline(generated_at)
    decision = build_decision(generated_at)

    checks = {
        "runtime_baseline_has_model_hash_environment_commands": bool(
            runtime["baseline_model"].get("artifact_hash")
            and runtime.get("resource_baseline")
            and runtime.get("runtime_commands")
        ),
        "runtime_baseline_records_required_endpoint_latency_fields": all(
            key in runtime["endpoint_latency_baseline"]
            for key in [
                "local_live_probe_2026_06_04",
                "phase27_endpoint_latency_ms",
                "warmup_ready_after",
                "warm_internal_cv_analysis",
            ]
        ),
        "quality_baseline_has_metrics_calibration_distribution_examples": bool(
            quality.get("jobfit_metrics")
            and quality.get("calibration_metrics")
            and quality.get("score_distribution")
            and quality.get("candidate_reranking_examples")
        ),
        "migration_hypothesis_and_risks_explicit": len(decision["migration_hypothesis"]) >= 4
        and len(decision["migration_risks"]) >= 6,
        "go_no_go_thresholds_cover_runtime_quality_contract_rollback": all(
            key in decision["go_no_go_thresholds"] for key in ["runtime", "quality", "contract", "rollback"]
        ),
        "artifact_namespace_keeps_phase25_rollback_safe": decision["artifact_namespace"]["reserved_root"]
        != decision["artifact_namespace"]["forbidden_root"],
        "model_version_names_reserved_and_schema_compatibility_noted": bool(
            decision["model_version_naming"].get("reserved_training_model_version")
            and decision["model_version_naming"].get("compatible_core_schema") == "model-core-cv-analysis-v1"
            and "does not imply" in decision["model_version_naming"].get("compatibility_note", "")
        ),
        "staging_experiment_only_decision_recorded": decision["decision"] == "approved_for_staging_experiment_only",
    }
    blockers = [name for name, passed in checks.items() if not passed]

    return {
        "schema_version": "phase-43-multilingual-e5-small-migration-decision-v1",
        "phase_id": "phase_43_multilingual_e5_small_migration",
        "generated_at": generated_at,
        "status": "complete" if not blockers else "incomplete",
        "final_decision": decision["decision"] if not blockers else "no_go_missing_evidence",
        "checks": checks,
        "blockers": blockers,
        "runtime_baseline_path": str(RUNTIME_BASELINE_PATH.relative_to(ROOT)),
        "quality_baseline_path": str(QUALITY_BASELINE_PATH.relative_to(ROOT)),
        "decision_path": str(DECISION_PATH.relative_to(ROOT)),
        "baseline_summary": {
            "embedding_model": runtime["baseline_model"]["embedding_model"],
            "artifact_hash": runtime["baseline_model"]["artifact_hash"],
            "phase25_status": read_json(PHASE25_REPORT).get("status"),
            "phase25_validation_mae_0_100": read_json(PHASE25_REPORT).get("metrics", {}).get("validation", {}).get("mae_0_100"),
            "phase25_test_mae_0_100": read_json(PHASE25_REPORT).get("metrics", {}).get("test", {}).get("mae_0_100"),
            "warm_internal_latency_ms": (runtime.get("endpoint_latency_baseline", {}).get("warm_internal_cv_analysis") or {}).get("latency_ms"),
            "first_cv_analysis_latency_ms": (runtime.get("endpoint_latency_baseline", {}).get("phase27_endpoint_latency_ms") or {}).get("cv_analysis"),
            "e5_base_cache_size_bytes": runtime["resource_baseline"].get("e5_base_hf_cache_size_bytes"),
        },
        "next_phase": "Phase 44 must quantify embedding dimension, cosine drift, normalization drift, language-slice behavior, and retrain-vs-recalibrate decision before any runtime change.",
        "sources": [str(path.relative_to(ROOT)) for path in SOURCE_PATHS],
    }


def write_markdown(report: dict[str, Any], decision: dict[str, Any]) -> None:
    summary = report["baseline_summary"]
    lines = [
        "# Phase 43 multilingual-E5-small Migration Decision",
        "",
        f"Decision: `{report['final_decision']}`",
        "",
        "Phase 43 freezes E5-base behavior before any multilingual-E5-small implementation. Phase 25 artifacts remain rollback baseline.",
        "",
        "## Baseline Summary",
        f"- Embedding model: `{summary['embedding_model']}`",
        f"- Model artifact hash: `{summary['artifact_hash']}`",
        f"- Phase 25 status: `{summary['phase25_status']}`",
        f"- Validation MAE: `{summary['phase25_validation_mae_0_100']}` points",
        f"- Test MAE: `{summary['phase25_test_mae_0_100']}` points",
        f"- First CV analysis latency: `{summary['first_cv_analysis_latency_ms']}` ms",
        f"- Warm internal latency: `{summary['warm_internal_latency_ms']}` ms",
        f"- E5-base cache size: `{summary['e5_base_cache_size_bytes']}` bytes",
        "",
        "## Checks",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    if report["blockers"]:
        lines.extend(["", "## Blockers"])
        lines.extend(f"- {item}" for item in report["blockers"])
    lines.extend(
        [
            "",
            "## Reserved Namespace",
            f"- Artifact root: `{decision['artifact_namespace']['reserved_root']}`",
            f"- Later training root: `{decision['artifact_namespace']['later_training_root']}`",
            f"- Forbidden overwrite root: `{decision['artifact_namespace']['forbidden_root']}`",
            "",
            "## Go/No-Go Threshold Groups",
        ]
    )
    for name in decision["go_no_go_thresholds"]:
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "## Commands",
            "- `phase43_gate`: `python scripts/verify_phase_43_multilingual_e5_small_migration.py --write`",
            "- `phase44_next`: create embedding compatibility audit before changing runtime defaults.",
        ]
    )
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all() -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    generated_at = now_utc()
    runtime = build_runtime_baseline(generated_at)
    quality = build_quality_baseline(generated_at)
    decision = build_decision(generated_at)
    RUNTIME_BASELINE_PATH.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    QUALITY_BASELINE_PATH.write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    DECISION_PATH.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = build_report()
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, decision)
    return report


def main(argv: list[str] | None = None) -> int:
    write = bool(argv and "--write" in argv)
    report = write_all() if write else build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["blockers"] else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
