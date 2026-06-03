#!/usr/bin/env python3
"""Generate Phase 0 baseline audit for legacy Bisakerja training artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

# Repository paths stay explicit so the audit can be rerun from any working directory.
ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
OUTPUT = ROOT / "reports" / "training_audit_v1.json"

# Baseline settings mirror the original notebook validation logic.
TOLERANCE = 0.15
SPLIT_SEED = 42
VAL_SIZE = 0.15

# Score bands make the narrow weak-label distribution visible in the audit report.
LABEL_BANDS = {
    "low": (0.0, 0.34),
    "medium": (0.35, 0.64),
    "high": (0.65, 1.0),
}
# Artifact inventory uses logical paths from the old model card and actual paths in this repo.
REQUIRED_ARTIFACTS = {
    "models/model_jobfit_v1.keras": LEGACY / "models" / "model_jobfit_v1.keras",
    "artifacts/pairs.parquet": LEGACY / "artifacts" / "pairs.parquet",
    "artifacts/job_index.json": LEGACY / "artifacts" / "job_index.json",
    "cache/all_job_embeddings.npy": LEGACY / "cache" / "all_job_embeddings.npy",
    "cache/job_embeddings.npy": LEGACY / "cache" / "job_embeddings.npy",
    "cache/profile_embeddings.npy": LEGACY / "cache" / "profile_embeddings.npy",
    "reports/model_card.json": LEGACY / "reports" / "model_card.json",
}
# Source CSVs are included to make row counts and file hashes reproducible.
DATASETS = {
    "jobs": LEGACY / "dataset" / "indotech_job_cleaned.csv",
    "profiles": LEGACY / "dataset" / "techtalent_profile_cleaned.csv",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 hash without loading large artifacts into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_base(path: Path) -> dict[str, Any]:
    """Return common file metadata used by artifacts and dataset entries."""
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def artifact_inventory() -> dict[str, Any]:
    """Collect current artifact presence, hashes, sizes, and lightweight shape metadata."""
    inventory: dict[str, Any] = {}
    for logical_path, path in REQUIRED_ARTIFACTS.items():
        item: dict[str, Any] = {
            "logical_path": logical_path,
            "actual_path": str(path.relative_to(ROOT)),
            **file_base(path),
        }
        if item["exists"] and path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
            item["shape"] = list(arr.shape)
            item["dtype"] = str(arr.dtype)
        elif item["exists"] and path.suffix == ".parquet":
            df = pd.read_parquet(path)
            item["rows"] = int(len(df))
            item["columns"] = list(df.columns)
        elif item["exists"] and path.suffix == ".json":
            with path.open() as handle:
                data = json.load(handle)
            item["json_type"] = type(data).__name__
            if isinstance(data, dict):
                keys = list(data.keys())
                item["entries"] = len(data)
                if len(keys) <= 50:
                    item["top_level_keys"] = keys
                else:
                    item["sample_keys"] = keys[:10]
            elif isinstance(data, list):
                item["entries"] = len(data)
        inventory[logical_path] = item
    return inventory


def read_model_card() -> dict[str, Any]:
    """Load the legacy model card that contains current model metrics and gate status."""
    with (LEGACY / "reports" / "model_card.json").open() as handle:
        return json.load(handle)


def skill_set(value: Any) -> set[str]:
    """Normalize comma-separated skills into a lowercase set for overlap baselines."""
    return {part.strip().lower() for part in str(value).split(",") if part.strip()}


def jaccard(a: Any, b: Any) -> float:
    """Compute skill overlap as a simple non-model baseline."""
    left = skill_set(a)
    right = skill_set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def cosine_similarity_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute row-wise cosine similarity for paired profile and job embeddings."""
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = left_norm * right_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.divide(
            np.sum(left * right, axis=1),
            denom,
            out=np.zeros(len(left), dtype=np.float32),
            where=denom != 0,
        )
    return np.clip(scores, 0.0, 1.0)


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Return shared regression metrics for model and baseline comparisons."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "accuracy_at_tolerance_0_15": round(float(np.mean(np.abs(y_true - y_pred) <= TOLERANCE)), 6),
        "r2": round(float(r2_score(y_true, y_pred)), 6),
        "prediction_mean": round(float(np.mean(y_pred)), 6),
        "prediction_median": round(float(np.median(y_pred)), 6),
    }


def split_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Recreate the notebook train/validation split grouped by profile_id."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SPLIT_SEED)
    groups = df["profile_id"].values
    return next(splitter.split(df, groups=groups))


def label_distribution(values: pd.Series | np.ndarray) -> dict[str, Any]:
    """Summarize weak-label range, central tendency, and low/medium/high bands."""
    series = pd.Series(values)
    bands: dict[str, int] = {}
    banded_mask = pd.Series(False, index=series.index)
    for name, (lower, upper) in LABEL_BANDS.items():
        mask = series.between(lower, upper, inclusive="both")
        bands[name] = int(mask.sum())
        banded_mask = banded_mask | mask
    return {
        "count": int(series.count()),
        "min": round(float(series.min()), 6),
        "max": round(float(series.max()), 6),
        "mean": round(float(series.mean()), 6),
        "median": round(float(series.median()), 6),
        "std": round(float(series.std()), 6),
        "p25": round(float(series.quantile(0.25)), 6),
        "p75": round(float(series.quantile(0.75)), 6),
        "bands": bands,
        "unbanded_count": int((~banded_mask).sum()),
    }


def dataset_counts(df_pairs: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> dict[str, Any]:
    """Report source dataset sizes and validation split counts."""
    counts: dict[str, Any] = {
        "pairs_total": int(len(df_pairs)),
        "unique_profiles_in_pairs": int(df_pairs["profile_id"].nunique()),
        "unique_jobs_in_pairs": int(df_pairs["job_id"].nunique()),
        "train_pairs": int(len(train_idx)),
        "val_pairs": int(len(val_idx)),
        "split_strategy": {
            "type": "GroupShuffleSplit",
            "group_column": "profile_id",
            "test_size": VAL_SIZE,
            "random_state": SPLIT_SEED,
        },
    }
    for name, path in DATASETS.items():
        item: dict[str, Any] = {"path": str(path.relative_to(ROOT)), **file_base(path)}
        if item["exists"]:
            item["rows"] = int(len(pd.read_csv(path)))
        counts[f"{name}_dataset"] = item
    return counts


def baseline_metrics(df_pairs: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> dict[str, Any]:
    """Evaluate required Phase 0 baselines on the recreated validation split."""
    y_train = df_pairs["fit_score"].to_numpy(dtype=np.float32)[train_idx]
    y_val = df_pairs["fit_score"].to_numpy(dtype=np.float32)[val_idx]

    # Constant baselines use train-only statistics to avoid validation leakage.
    train_mean = float(np.mean(y_train))
    train_median = float(np.median(y_train))

    # Skill Jaccard baseline checks whether simple overlap already explains the weak label.
    skill_pred = df_pairs.iloc[val_idx].apply(
        lambda row: jaccard(row["profile_skills"], row["job_skills"]), axis=1
    ).to_numpy(dtype=np.float32)

    # Memory mapping keeps large embedding arrays cheap to inspect during audit generation.
    profile_embs = np.load(LEGACY / "cache" / "profile_embeddings.npy", mmap_mode="r")
    job_embs = np.load(LEGACY / "cache" / "job_embeddings.npy", mmap_mode="r")
    cosine_pred = cosine_similarity_rows(profile_embs[val_idx], job_embs[val_idx])

    return {
        "label_target": "fit_score",
        "evaluation_split": "validation",
        "tolerance": TOLERANCE,
        "constant_mean": {
            "train_label_mean": round(train_mean, 6),
            **metric_block(y_val, np.full_like(y_val, train_mean)),
        },
        "constant_median": {
            "train_label_median": round(train_median, 6),
            **metric_block(y_val, np.full_like(y_val, train_median)),
        },
        "skill_jaccard_only": metric_block(y_val, skill_pred),
        "cosine_embedding_only": {
            "note": "Row-wise cosine(profile_embedding, job_embedding), clipped to [0, 1].",
            **metric_block(y_val, cosine_pred),
        },
    }


def main() -> None:
    """Build the audit JSON and mark the old gate as prototype-only."""
    df_pairs = pd.read_parquet(LEGACY / "artifacts" / "pairs.parquet")
    train_idx, val_idx = split_indices(df_pairs)
    model_card = read_model_card()
    current_r2 = float(model_card.get("metrics", {}).get("r2", 0.0))

    report = {
        "schema_version": "training-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_snapshot": "legacy",
        "artifact_inventory": artifact_inventory(),
        "dataset_row_counts": dataset_counts(df_pairs, train_idx, val_idx),
        "label_distribution": {
            "overall": label_distribution(df_pairs["fit_score"]),
            "train": label_distribution(df_pairs["fit_score"].to_numpy()[train_idx]),
            "validation": label_distribution(df_pairs["fit_score"].to_numpy()[val_idx]),
        },
        "current_metrics": model_card.get("metrics", {}),
        "baseline_metrics": baseline_metrics(df_pairs, train_idx, val_idx),
        "gate_assessment": {
            "model_card_gate_passed": bool(model_card.get("gate_passed")),
            "model_card_gate_type": "prototype",
            "production_gate_passed": False,
            "blockers": [
                "Current gate only checks MAE <= 0.10 and accuracy >= 0.75 at tolerance ±0.15.",
                f"Current R² is {current_r2:.4f}; negative R² means model does not beat mean baseline on validation.",
                "No high-fit labels above 0.65 in current pair dataset.",
            ],
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
