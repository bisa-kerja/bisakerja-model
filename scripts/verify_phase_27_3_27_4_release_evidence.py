from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "training/notebooks"
HUMAN_LABELS_PATH = ROOT / "artifacts/manual_validation/phase_16_human_labels_frozen.csv"
PHASE16_REPORT_PATH = ROOT / "reports/phase_16_human_validation_label_governance.json"
PHASE25_LABEL_MANIFEST_PATH = ROOT / "artifacts/phase_25_tensorflow_training_delivery/label_manifest.json"
REPORT_JSON_PATH = ROOT / "reports/phase_27_3_27_4_notebook_label_gate.json"
REPORT_MD_PATH = ROOT / "reports/phase_27_3_27_4_notebook_label_gate.md"
LABEL_POLICY_PATH = ROOT / "artifacts/manual_validation/phase_27_production_label_policy.json"

ACTIVE_NOTEBOOK_GLOB = "phase_*.ipynb"
RETIRED_NOTEBOOKS = {
    "phase_13_data_snapshot_contract_freezing.ipynb": "Phase 13 execution evidence is frozen in reports/phase_13_*.json; executable cells are retired for production notebook hygiene.",
}

PRODUCTION_LABEL_POLICY = {
    "schema_version": "phase-27-production-label-policy-v1",
    "policy_id": "jobfit-production-human-validation-policy-v1",
    "label_sources": {
        "allowed_for_training": [
            "weak-label-balanced-v2 bootstrap labels",
            "feature-derived pair labels used only to fit preliminary scorer",
        ],
        "required_for_production_score_claims": [
            "frozen human or recruiter-reviewed validation labels",
            "at least two independent reviewers per validation item",
            "reviewer agreement metrics recorded before model-card release",
            "slice coverage across score band, role family, language, experience band, and pair type",
        ],
        "forbidden": [
            "human validation labels as model input features",
            "single-reviewer-only validation for production score claims",
            "weak-label-only evidence for production score accuracy claims",
            "validation labels exposed to inference payloads or generated product copy",
        ],
    },
    "minimum_release_validation_set": {
        "unique_items": 600,
        "reviewers_per_item": 2,
        "score_bands": ["low", "medium", "high"],
        "minimum_items_per_score_band": 150,
        "required_slice_dimensions": ["score_band", "role_family", "language", "experience_band", "pair_type"],
        "minimum_items_per_required_slice_bucket": 20,
    },
    "production_claim_gate": {
        "allow_production_score_claims_only_when": [
            "minimum_release_validation_set is satisfied",
            "all required slice dimensions are present and covered",
            "mean weighted kappa is recorded and >= 0.60",
            "human-label evaluation beats baseline and reports MAE/RMSE/R2/Spearman/slice stability",
        ],
        "block_when": [
            "unique frozen human validation items are below threshold",
            "any item has fewer than two independent reviewers",
            "labels are single-reviewer, unlabeled, or not frozen",
            "required slice dimensions are missing or under-covered",
            "model card still frames weak-label metrics as production score evidence",
        ],
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def notebook_summary(path: Path) -> dict[str, Any]:
    notebook = load_json(path)
    error_outputs: list[dict[str, Any]] = []
    unexecuted_code_cells: list[int] = []
    code_cell_count = 0
    markdown_cell_count = 0
    retired = path.name in RETIRED_NOTEBOOKS

    for index, cell in enumerate(notebook.get("cells", []), start=1):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        if cell_type == "markdown":
            markdown_cell_count += 1
        if cell_type == "code":
            code_cell_count += 1
            if source.strip() and cell.get("execution_count") is None:
                unexecuted_code_cells.append(index)
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    error_outputs.append(
                        {
                            "cell": index,
                            "ename": output.get("ename"),
                            "evalue": output.get("evalue"),
                        }
                    )

    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "retired": retired,
        "retirement_reason": RETIRED_NOTEBOOKS.get(path.name),
        "cell_counts": {"code": code_cell_count, "markdown": markdown_cell_count},
        "error_output_count": len(error_outputs),
        "error_outputs": error_outputs,
        "unexecuted_code_cell_count": len(unexecuted_code_cells),
        "unexecuted_code_cells": unexecuted_code_cells,
    }


def scan_notebooks() -> dict[str, Any]:
    notebooks = [p for p in sorted(NOTEBOOK_DIR.glob(ACTIVE_NOTEBOOK_GLOB)) if ".ipynb_checkpoints" not in p.parts]
    summaries = [notebook_summary(path) for path in notebooks]
    error_notebooks = [item for item in summaries if item["error_output_count"] > 0]
    active_unexecuted = [
        item
        for item in summaries
        if item["unexecuted_code_cell_count"] > 0 and not item["retired"]
    ]
    return {
        "schema_version": "phase-27-notebook-hygiene-scan-v1",
        "active_notebook_count": len(summaries),
        "retired_notebooks": [item for item in summaries if item["retired"]],
        "error_notebook_count": len(error_notebooks),
        "active_unexecuted_notebook_count": len(active_unexecuted),
        "notebooks": summaries,
        "status": "PASS" if not error_notebooks and not active_unexecuted else "FAIL",
    }


def parse_evidence_note(note: str, key: str) -> str | None:
    match = re.search(rf"{re.escape(key)}=([^;\.]+)", note or "")
    if not match:
        return None
    return match.group(1).strip().strip('"')


def load_human_labels() -> list[dict[str, str]]:
    if not HUMAN_LABELS_PATH.exists():
        return []
    with HUMAN_LABELS_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_label_evidence() -> dict[str, Any]:
    rows = load_human_labels()
    policy = PRODUCTION_LABEL_POLICY["minimum_release_validation_set"]
    by_item: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_item[row.get("review_item_id", "")].append(row)

    reviewer_counts_per_item = {item_id: len({r.get("reviewer_id", "") for r in item_rows if r.get("reviewer_id")}) for item_id, item_rows in by_item.items()}
    score_band_counts = Counter(row.get("reviewer_job_fit_band", "UNKNOWN") for row in rows)
    item_band_counts = Counter()
    pair_type_counts = Counter()
    role_family_counts = Counter()
    recommendation_relevance_counts = Counter(row.get("recommendation_relevance", "UNKNOWN") for row in rows)
    unsupported_claim_flags = Counter(row.get("unsupported_claim_flag", "UNKNOWN") for row in rows)
    disagreement_flags = Counter(row.get("disagreement_flag", "UNKNOWN") for row in rows)

    for item_rows in by_item.values():
        first = item_rows[0]
        item_band_counts[first.get("reviewer_job_fit_band", "UNKNOWN")] += 1
        note = first.get("evidence_notes", "")
        role_family_counts[parse_evidence_note(note, "role_family") or "UNKNOWN"] += 1
        pair_type_counts[parse_evidence_note(note, "pair_type") or "UNKNOWN"] += 1

    minimum_reviewers_per_item = min(reviewer_counts_per_item.values()) if reviewer_counts_per_item else 0
    max_reviewers_per_item = max(reviewer_counts_per_item.values()) if reviewer_counts_per_item else 0
    unique_items = len(by_item)
    reviewer_count = len({row.get("reviewer_id", "") for row in rows if row.get("reviewer_id")})

    covered_bands = sorted(band for band in policy["score_bands"] if item_band_counts.get(band, 0) > 0)
    slice_dimensions = {
        "score_band": dict(item_band_counts),
        "role_family": dict(role_family_counts),
        "pair_type": dict(pair_type_counts),
        "language": {},
        "experience_band": {},
    }

    blockers: list[str] = []
    if unique_items < policy["unique_items"]:
        blockers.append(f"frozen human validation item count {unique_items} < required {policy['unique_items']}")
    if minimum_reviewers_per_item < policy["reviewers_per_item"]:
        blockers.append(f"minimum reviewers per item {minimum_reviewers_per_item} < required {policy['reviewers_per_item']}")
    for band in policy["score_bands"]:
        if item_band_counts.get(band, 0) < policy["minimum_items_per_score_band"]:
            blockers.append(
                f"score band {band} has {item_band_counts.get(band, 0)} items < required {policy['minimum_items_per_score_band']}"
            )
    for dimension in policy["required_slice_dimensions"]:
        buckets = slice_dimensions.get(dimension, {})
        if not buckets:
            blockers.append(f"required slice dimension {dimension} missing from frozen labels")
        elif any(count < policy["minimum_items_per_required_slice_bucket"] for count in buckets.values()):
            blockers.append(f"required slice dimension {dimension} has under-covered buckets: {buckets}")

    phase16_report = load_json(PHASE16_REPORT_PATH) if PHASE16_REPORT_PATH.exists() else None
    phase25_manifest = load_json(PHASE25_LABEL_MANIFEST_PATH) if PHASE25_LABEL_MANIFEST_PATH.exists() else None

    weak_label_only_training = True
    if phase25_manifest:
        source = phase25_manifest.get("labels", {}).get("jobFitAlignment.score", {}).get("source", "")
        human_validation = phase25_manifest.get("labels", {}).get("jobFitAlignment.score", {}).get("human_validation")
        weak_label_only_training = "pairs_v2" in source and human_validation == "evaluation_only"

    return {
        "schema_version": "phase-27-label-evidence-gate-v1",
        "policy_path": str(LABEL_POLICY_PATH.relative_to(ROOT)),
        "label_path": str(HUMAN_LABELS_PATH.relative_to(ROOT)),
        "label_sha256": sha256_file(HUMAN_LABELS_PATH) if HUMAN_LABELS_PATH.exists() else None,
        "row_count": len(rows),
        "unique_review_items": unique_items,
        "reviewer_count": reviewer_count,
        "minimum_reviewers_per_item": minimum_reviewers_per_item,
        "maximum_reviewers_per_item": max_reviewers_per_item,
        "score_band_counts_by_row": dict(score_band_counts),
        "score_band_counts_by_item": dict(item_band_counts),
        "recommendation_relevance_counts_by_row": dict(recommendation_relevance_counts),
        "unsupported_claim_flags_by_row": dict(unsupported_claim_flags),
        "disagreement_flags_by_row": dict(disagreement_flags),
        "slice_dimensions": slice_dimensions,
        "covered_score_bands": covered_bands,
        "weak_labels_allowed_only_as_bootstrap_training_support": True,
        "weak_label_only_training_currently_used": weak_label_only_training,
        "human_labels_are_model_inputs": False,
        "production_score_claims_allowed": not blockers,
        "blockers": blockers,
        "source_reports": {
            "phase16_human_validation_label_governance": phase16_report,
            "phase25_label_manifest": phase25_manifest,
        },
        "status": "PASS" if not blockers else "FAIL",
    }


def build_report() -> dict[str, Any]:
    notebook_gate = scan_notebooks()
    label_gate = summarize_label_evidence()
    final_decision = "production-ready" if notebook_gate["status"] == "PASS" and label_gate["status"] == "PASS" else "blocked"
    return {
        "schema_version": "phase-27-3-27-4-notebook-label-gate-v1",
        "phase_id": "phase_27_3_27_4_notebook_hygiene_label_evidence",
        "generated_at": now_iso(),
        "references": ["GAP_MODEL_TRAINING.md#phase-27", "GAP_MODEL_TRAINING.md", "REQUIREMENT.md"],
        "steps": {
            "27.3": {
                "name": "Repair notebook hygiene",
                "status": notebook_gate["status"],
                "notebook_gate": notebook_gate,
            },
            "27.4": {
                "name": "Upgrade label evidence beyond weak-label-only training",
                "status": label_gate["status"],
                "production_label_policy": PRODUCTION_LABEL_POLICY,
                "label_gate": label_gate,
            },
        },
        "final_decision": final_decision,
    }


def write_markdown(report: dict[str, Any]) -> None:
    notebook_gate = report["steps"]["27.3"]["notebook_gate"]
    label_gate = report["steps"]["27.4"]["label_gate"]
    lines = [
        "# Phase 27.3-27.4 Notebook and Label Evidence Gate",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Final decision: **{report['final_decision']}**",
        "",
        "## Notebook hygiene",
        "",
        f"Status: **{notebook_gate['status']}**",
        f"Active notebooks scanned: `{notebook_gate['active_notebook_count']}`",
        f"Notebooks with saved error outputs: `{notebook_gate['error_notebook_count']}`",
        f"Active notebooks with unexecuted code cells: `{notebook_gate['active_unexecuted_notebook_count']}`",
        "",
        "Retired notebooks:",
    ]
    for item in notebook_gate["retired_notebooks"]:
        lines.append(f"- `{item['path']}` — {item['retirement_reason']}")
    if not notebook_gate["retired_notebooks"]:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Production label policy",
            "",
            f"Policy artifact: `{LABEL_POLICY_PATH.relative_to(ROOT)}`",
            f"Frozen label file: `{HUMAN_LABELS_PATH.relative_to(ROOT)}`",
            f"Status: **{label_gate['status']}**",
            f"Production score claims allowed: `{label_gate['production_score_claims_allowed']}`",
            f"Unique review items: `{label_gate['unique_review_items']}`",
            f"Reviewer count: `{label_gate['reviewer_count']}`",
            f"Minimum reviewers per item: `{label_gate['minimum_reviewers_per_item']}`",
            "",
            "Current blockers:",
        ]
    )
    for blocker in label_gate["blockers"]:
        lines.append(f"- {blocker}")
    if not label_gate["blockers"]:
        lines.append("- None")
    lines.extend(
        [
            "",
            "Weak labels remain allowed only as bootstrap/training support. Human/recruiter-reviewed frozen labels are mandatory for production score claims.",
            "",
        ]
    )
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 27.3 notebook hygiene and Phase 27.4 label evidence gates.")
    parser.add_argument("--write", action="store_true", help="Write JSON/Markdown reports and label policy artifact.")
    args = parser.parse_args()

    if args.write:
        write_json(LABEL_POLICY_PATH, PRODUCTION_LABEL_POLICY)
    report = build_report()
    if args.write:
        write_json(REPORT_JSON_PATH, report)
        write_markdown(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["final_decision"] == "production-ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
