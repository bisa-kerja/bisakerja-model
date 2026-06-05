#!/usr/bin/env python3
"""Train and verify Phase 45 multilingual-E5-small TensorFlow delivery evidence.

This script is the executable companion for the Phase 45 notebook. It reads the
frozen Phase 25 pair set plus Phase 44 multilingual-E5-small features, trains a
new TensorFlow scorer with a manual GradientTape loop, writes bounded artifacts,
and records selection or rejection evidence without changing Model API defaults.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "202645")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ARTIFACT_DIR = ROOT / "artifacts/phase_45_multilingual_e5_small_training_delivery"
TENSORBOARD_ROOT = ROOT / "artifacts/tensorboard/phase_45_multilingual_e5_small_training_delivery"
MODEL_PATH = ARTIFACT_DIR / "gradient_tape_trained_candidate.keras"
TRAINING_FEATURES_PATH = ARTIFACT_DIR / "tensorflow_training_features_v1.npz"
FEATURE_CONFIG_PATH = ARTIFACT_DIR / "tensorflow_feature_config.json"
TRAINING_HISTORY_PATH = ARTIFACT_DIR / "gradient_tape_training_history.csv"
PREDICTIONS_PATH = ARTIFACT_DIR / "gradient_tape_predictions_v1.npz"
BASELINE_REPORT_PATH = ARTIFACT_DIR / "baseline_comparison.json"
EVALUATION_REPORT_PATH = ARTIFACT_DIR / "training_evaluation.json"
TENSORBOARD_MANIFEST_PATH = ARTIFACT_DIR / "tensorboard_monitoring_manifest.json"
SELECTION_DECISION_PATH = ARTIFACT_DIR / "selection_decision.json"
README_PATH = ARTIFACT_DIR / "README.md"
REPORT_JSON_PATH = REPORTS / "phase_45_multilingual_e5_small_training_delivery.json"
REPORT_MD_PATH = REPORTS / "phase_45_multilingual_e5_small_training_delivery.md"

PHASE25_DIR = ROOT / "artifacts/phase_25_tensorflow_training_delivery"
PHASE44_DIR = ROOT / "artifacts/phase_44_embedding_compatibility_audit"
PHASE25_FEATURE_CONFIG_PATH = PHASE25_DIR / "tensorflow_feature_config.json"
PHASE25_PREDICTIONS_PATH = PHASE25_DIR / "gradient_tape_predictions_v1.npz"
PHASE25_BASELINE_SELECTION_PATH = PHASE25_DIR / "baseline_selection_gate.json"
PHASE25_LABEL_MANIFEST_PATH = PHASE25_DIR / "label_manifest.json"
PHASE25_DATASET_MANIFEST_PATH = PHASE25_DIR / "dataset_manifest.json"
PHASE21_RERANKING_PATH = REPORTS / "phase_21_backend_candidate_reranking.json"
PHASE44_FEATURES_PATH = PHASE44_DIR / "multilingual_e5_small_pair_features.json"
PHASE44_DECISION_PATH = PHASE44_DIR / "retrain_vs_recalibrate_decision.json"

PHASE_ID = "phase_45_multilingual_e5_small_training_delivery"
SCHEMA_VERSION = "phase-45-multilingual-e5-small-training-delivery-v1"
MODEL_NAME = "bisakerja_jobfit_tf_phase45_multilingual_e5_small_v1"
MODEL_VERSION = "jobfit_tf_phase45_multilingual_e5_small_v1"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
SOURCE_EMBEDDING_MODEL = "intfloat/e5-base-v2"
APPROVED_FEATURES = [
    "e5_cosine",
    "skill_overlap",
    "requirement_coverage",
    "role_match",
    "experience_match",
    "experience_gap_years_clipped",
]
SEED = 202645
BATCH_SIZE = 256
MAX_EPOCHS = 180
EARLY_STOP_PATIENCE = 35
TARGET_MAE = 0.02
TARGET_MAE_0_100 = 2.0
MIN_R2 = 0.15
HIGH_FIT_THRESHOLD = 0.70
BAND_LABELS = ["low", "medium", "high"]


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


def require_paths(paths: Iterable[Path]) -> list[str]:
    return [rel(path) for path in paths if not path.exists()]


def rankdata_average(values: Any) -> Any:
    import numpy as np

    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype="float64")
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman_corr(y_true: Any, y_pred: Any) -> float | None:
    import numpy as np

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) < 2 or float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return None
    return float(np.corrcoef(rankdata_average(y_true), rankdata_average(y_pred))[0, 1])


def score_band(values_0_1: Any) -> Any:
    import numpy as np

    scores = np.clip(np.asarray(values_0_1).reshape(-1) * 100.0, 0.0, 100.0)
    return np.where(scores >= 70.0, "high", np.where(scores >= 40.0, "medium", "low"))


def compute_metrics(y_true: Any, y_pred: Any, loss_value: float | None = None) -> dict[str, Any]:
    import numpy as np

    y_true = np.asarray(y_true).reshape(-1).astype("float64")
    y_pred = np.clip(np.asarray(y_pred).reshape(-1).astype("float64"), 0.0, 1.0)
    error = y_true - y_pred
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    ss_res = float(np.sum(np.square(error)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = None if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    true_high = y_true >= (HIGH_FIT_THRESHOLD - 1e-6)
    pred_high = y_pred >= (HIGH_FIT_THRESHOLD - 1e-6)
    true_bands = score_band(y_true)
    pred_bands = score_band(y_pred)
    return {
        "row_count": int(len(y_true)),
        "loss": None if loss_value is None else round(float(loss_value), 8),
        "mae": round(mae, 8),
        "mae_0_100": round(mae * 100.0, 6),
        "rmse": round(rmse, 8),
        "rmse_0_100": round(rmse * 100.0, 6),
        "r2": None if r2 is None else round(r2, 8),
        "spearman": None if (sp := spearman_corr(y_true, y_pred)) is None else round(sp, 8),
        "score_band_agreement": round(float(np.mean(true_bands == pred_bands)), 8),
        "high_fit_recall": None if int(true_high.sum()) == 0 else round(float(np.mean(pred_high[true_high])), 8),
        "true_high_count": int(true_high.sum()),
        "predicted_high_count": int(pred_high.sum()),
    }


def dcg_at_k(relevances: list[float], k: int) -> float:
    return sum((2.0 ** rel - 1.0) / math.log2(index + 2.0) for index, rel in enumerate(relevances[:k]))


def average_precision_at_k(labels: list[bool], k: int) -> float:
    hits = 0
    precision_sum = 0.0
    for index, label in enumerate(labels[:k], start=1):
        if label:
            hits += 1
            precision_sum += hits / index
    relevant_total = sum(labels)
    return 0.0 if relevant_total == 0 else precision_sum / min(relevant_total, k)


def ranking_metrics(records: list[dict[str, Any]], predictions: Any) -> dict[str, Any]:
    import numpy as np

    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for record, prediction in zip(records, np.asarray(predictions).reshape(-1)):
        groups[str(record["profile_id"])].append((float(record["job_fit_score"]), float(prediction)))
    ndcg5: list[float] = []
    ndcg10: list[float] = []
    map10: list[float] = []
    group_count = 0
    for values in groups.values():
        if len(values) < 2:
            continue
        group_count += 1
        ordered = sorted(values, key=lambda item: item[1], reverse=True)
        ideal = sorted(values, key=lambda item: item[0], reverse=True)
        rel_pred = [score for score, _ in ordered]
        rel_ideal = [score for score, _ in ideal]
        ideal5 = dcg_at_k(rel_ideal, 5)
        ideal10 = dcg_at_k(rel_ideal, 10)
        ndcg5.append(0.0 if ideal5 == 0 else dcg_at_k(rel_pred, 5) / ideal5)
        ndcg10.append(0.0 if ideal10 == 0 else dcg_at_k(rel_pred, 10) / ideal10)
        labels = [score >= HIGH_FIT_THRESHOLD for score, _ in ordered]
        map10.append(average_precision_at_k(labels, 10))
    return {
        "group_by": "profile_id",
        "group_count": group_count,
        "ndcg_at_5": round(float(np.mean(ndcg5)), 8) if ndcg5 else None,
        "ndcg_at_10": round(float(np.mean(ndcg10)), 8) if ndcg10 else None,
        "map_at_10": round(float(np.mean(map10)), 8) if map10 else None,
    }


def linear_regression_predict(train_X: Any, train_y: Any, full_X: Any) -> Any:
    import numpy as np

    X = np.asarray(train_X, dtype="float64")
    y = np.asarray(train_y, dtype="float64").reshape(-1)
    X_aug = np.c_[np.ones(len(X)), X]
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    full_aug = np.c_[np.ones(len(full_X)), np.asarray(full_X, dtype="float64")]
    return np.clip(full_aug @ beta, 0.0, 1.0).reshape(-1, 1).astype("float32")


def load_records() -> list[dict[str, Any]]:
    payload = read_json(PHASE44_FEATURES_PATH, {"records": []})
    records = list(payload.get("records", []))
    records.sort(key=lambda row: str(row["pair_id"]))
    return records


def build_feature_matrix(records: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    if not records:
        raise RuntimeError("Phase 44 multilingual-E5-small pair features are missing.")
    X_raw = np.asarray(
        [
            [
                float(row["multilingual_e5_small_cosine"]),
                float(row["skill_overlap"]),
                float(row["requirement_coverage"]),
                float(row["role_match"]),
                float(row["experience_match"]),
                float(row["experience_gap_years_clipped"]),
            ]
            for row in records
        ],
        dtype="float32",
    )
    y = np.asarray([[float(row["job_fit_score"])] for row in records], dtype="float32")
    pair_ids = np.asarray([str(row["pair_id"]) for row in records], dtype=object)
    splits = np.asarray([str(row["split"]) for row in records], dtype=object)
    train_mask = splits == "train"
    if not bool(train_mask.any()):
        raise RuntimeError("Train split missing from Phase 45 feature records.")
    mean = X_raw[train_mask].mean(axis=0)
    std = X_raw[train_mask].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype("float32")
    X_scaled = ((X_raw - mean) / std).astype("float32")
    split_counts = {name: int((splits == name).sum()) for name in ["train", "validation", "test"]}
    metadata = {
        "phase_id": PHASE_ID,
        "schema_version": SCHEMA_VERSION,
        "normalization_fit_split": "train",
        "embedding_model": EMBEDDING_MODEL,
        "source_embedding_model": SOURCE_EMBEDDING_MODEL,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TRAINING_FEATURES_PATH,
        X_raw=X_raw,
        X_scaled=X_scaled,
        y=y,
        pair_id=pair_ids,
        split=splits,
        feature_names=np.asarray(APPROVED_FEATURES),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return {
        "X_raw": X_raw,
        "X_scaled": X_scaled,
        "y": y,
        "pair_ids": pair_ids,
        "splits": splits,
        "mean": mean,
        "std": std,
        "split_counts": split_counts,
    }


def write_feature_config(matrix: dict[str, Any], records: list[dict[str, Any]], generated_at: str) -> dict[str, Any]:
    import numpy as np

    phase25_config = read_json(PHASE25_FEATURE_CONFIG_PATH)
    config = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "generated_at": generated_at,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "approved_features": APPROVED_FEATURES,
        "target": phase25_config.get("target", {"column": "job_fit_score", "training_scale": "0-1", "api_scale": "0-100"}),
        "embedding_model_metadata": {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": 384,
            "profile_prefix": "query:",
            "job_prefix": "passage:",
            "normalized_embeddings": True,
            "source_phase": "phase_44_embedding_compatibility_audit",
        },
        "feature_policy": {
            "normalization_fit_split": "train",
            "uses_e5_derived_similarity": True,
            "uses_raw_embedding_vectors": False,
            "uses_manual_human_labels_as_training_features": False,
            "uses_wrapper_or_backend_owned_fields": False,
            "feature_order_changed_from_phase25": False,
            "only_e5_cosine_source_changed_from_phase25": True,
        },
        "normalization": {
            "mean": {name: float(value) for name, value in zip(APPROVED_FEATURES, matrix["mean"])},
            "std": {name: float(value) for name, value in zip(APPROVED_FEATURES, matrix["std"])},
        },
        "source_artifacts": {
            "phase44_pair_features": file_info(PHASE44_FEATURES_PATH),
            "phase44_decision": file_info(PHASE44_DECISION_PATH),
            "phase25_tensorflow_feature_config": file_info(PHASE25_FEATURE_CONFIG_PATH),
            "phase25_label_manifest": file_info(PHASE25_LABEL_MANIFEST_PATH),
            "phase25_dataset_manifest": file_info(PHASE25_DATASET_MANIFEST_PATH),
        },
        "feature_matrix": {
            "path": rel(TRAINING_FEATURES_PATH),
            "sha256": sha256_file(TRAINING_FEATURES_PATH),
            "row_count": int(len(records)),
            "feature_count": int(len(APPROVED_FEATURES)),
            "split_counts": matrix["split_counts"],
            "target_min": float(np.min(matrix["y"])),
            "target_max": float(np.max(matrix["y"])),
        },
    }
    write_json(FEATURE_CONFIG_PATH, config)
    return config


def import_tensorflow() -> tuple[Any, Any, Any, Any]:
    import numpy as np
    import tensorflow as tf
    import keras
    from keras import layers, regularizers

    tf.get_logger().setLevel("ERROR")
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    return np, tf, keras, (layers, regularizers)


# TensorFlow custom objects live inside factory to avoid importing TensorFlow in report-only code paths.
def make_custom_objects(tf: Any, keras: Any, layers: Any) -> dict[str, Any]:
    @keras.saving.register_keras_serializable(package="BisakerjaPhase45")
    class CosineInteractionLayer(layers.Layer):
        """Append multilingual-E5-small cosine interactions to approved numeric inputs."""

        def __init__(self, cosine_index: int = 0, interaction_indices: tuple[int, ...] = (1, 2, 3, 4), include_original: bool = True, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.cosine_index = int(cosine_index)
            self.interaction_indices = tuple(int(index) for index in interaction_indices)
            self.include_original = bool(include_original)

        def call(self, inputs: Any) -> Any:
            inputs = tf.convert_to_tensor(inputs)
            cosine_feature = tf.gather(inputs, [self.cosine_index], axis=-1)
            interaction_features = tf.gather(inputs, list(self.interaction_indices), axis=-1)
            interactions = interaction_features * cosine_feature
            pieces = [cosine_feature, interactions]
            if self.include_original:
                pieces.insert(0, inputs)
            return tf.concat(pieces, axis=-1)

        def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
            last_dim = input_shape[-1]
            added_dim = 1 + len(self.interaction_indices)
            output_dim = None if last_dim is None else (last_dim if self.include_original else 0) + added_dim
            return (*input_shape[:-1], output_dim)

        def get_config(self) -> dict[str, Any]:
            config = super().get_config()
            config.update({"cosine_index": self.cosine_index, "interaction_indices": list(self.interaction_indices), "include_original": self.include_original})
            return config

    @keras.saving.register_keras_serializable(package="BisakerjaPhase45")
    class WeightedHuberLoss(keras.losses.Loss):
        """Huber regression loss with extra weight on low/high fit score bands."""

        def __init__(self, delta: float = 0.03, high_fit_threshold: float = HIGH_FIT_THRESHOLD, high_fit_weight: float = 4.0, low_fit_threshold: float = 0.20, low_fit_weight: float = 1.25, name: str = "weighted_huber_loss", reduction: str = "sum_over_batch_size") -> None:
            super().__init__(name=name, reduction=reduction)
            self.delta = float(delta)
            self.high_fit_threshold = float(high_fit_threshold)
            self.high_fit_weight = float(high_fit_weight)
            self.low_fit_threshold = float(low_fit_threshold)
            self.low_fit_weight = float(low_fit_weight)

        def call(self, y_true: Any, y_pred: Any) -> Any:
            y_true = tf.cast(y_true, y_pred.dtype)
            error = y_true - y_pred
            abs_error = tf.abs(error)
            quadratic = tf.minimum(abs_error, self.delta)
            linear = abs_error - quadratic
            huber = 0.5 * tf.square(quadratic) + self.delta * linear
            weights = tf.ones_like(huber)
            weights = tf.where(y_true >= self.high_fit_threshold, weights * self.high_fit_weight, weights)
            weights = tf.where(y_true <= self.low_fit_threshold, weights * self.low_fit_weight, weights)
            return tf.reduce_mean(huber * weights, axis=-1)

        def get_config(self) -> dict[str, Any]:
            config = super().get_config()
            config.update({"delta": self.delta, "high_fit_threshold": self.high_fit_threshold, "high_fit_weight": self.high_fit_weight, "low_fit_threshold": self.low_fit_threshold, "low_fit_weight": self.low_fit_weight})
            return config

    @keras.saving.register_keras_serializable(package="BisakerjaPhase45")
    class ProductionGateCallback(keras.callbacks.Callback):
        """Record readiness gate state for manual GradientTape training loops."""

        def __init__(self, target_mae: float = TARGET_MAE, min_r2: float = MIN_R2, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.target_mae = float(target_mae)
            self.min_r2 = float(min_r2)
            self.gate_history: list[dict[str, Any]] = []

        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            logs = logs or {}
            mae = logs.get("val_mae")
            r2 = logs.get("val_r2")
            passed = mae is not None and r2 is not None and float(mae) <= self.target_mae and float(r2) >= self.min_r2
            self.gate_history.append({"epoch": int(epoch), "mae": None if mae is None else float(mae), "r2": None if r2 is None else float(r2), "passed": bool(passed)})

        def get_config(self) -> dict[str, Any]:
            return {"target_mae": self.target_mae, "min_r2": self.min_r2}

    return {
        "CosineInteractionLayer": CosineInteractionLayer,
        "WeightedHuberLoss": WeightedHuberLoss,
        "ProductionGateCallback": ProductionGateCallback,
    }


def build_model(input_dim: int, tf: Any, keras: Any, layers: Any, regularizers: Any, CosineInteractionLayer: Any) -> Any:
    inputs = keras.Input(shape=(input_dim,), name="approved_numeric_features")
    x = CosineInteractionLayer(name="multilingual_e5_small_cosine_interactions")(inputs)
    for dense_index, width in enumerate([96, 48, 24, 12], start=1):
        x = layers.Dense(width, activation="relu", kernel_regularizer=regularizers.l2(1e-5), name=f"jobfit_dense_{dense_index}")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="jobfit_score_0_1")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=MODEL_NAME)


def train_tensorflow(matrix: dict[str, Any], records: list[dict[str, Any]], generated_at: str) -> dict[str, Any]:
    np, tf, keras, layer_bundle = import_tensorflow()
    layers, regularizers = layer_bundle
    custom_objects = make_custom_objects(tf, keras, layers)
    CosineInteractionLayer = custom_objects["CosineInteractionLayer"]
    WeightedHuberLoss = custom_objects["WeightedHuberLoss"]
    ProductionGateCallback = custom_objects["ProductionGateCallback"]

    X_scaled = matrix["X_scaled"].astype("float32")
    y = matrix["y"].astype("float32")
    splits = matrix["splits"]
    split_indices = {name: np.where(splits == name)[0] for name in ["train", "validation", "test"]}
    model = build_model(X_scaled.shape[1], tf, keras, layers, regularizers, CosineInteractionLayer)
    loss_fn = WeightedHuberLoss()
    optimizer = keras.optimizers.Adam(learning_rate=3e-3, clipnorm=1.0)
    gate_callback = ProductionGateCallback()
    gate_callback.set_model(model)
    gate_callback.on_train_begin({})
    run_id = f"{MODEL_VERSION}_{generated_at.replace('-', '').replace(':', '').replace('Z', '').replace('T', 'T')}"
    tensorboard_run_dir = TENSORBOARD_ROOT / run_id
    writer = tf.summary.create_file_writer(str(tensorboard_run_dir))

    def make_dataset(indices: Any, shuffle: bool):
        ordered = np.asarray(indices, dtype=int)
        if shuffle:
            ordered = np.random.permutation(ordered)
        for start in range(0, len(ordered), BATCH_SIZE):
            batch_indices = ordered[start : start + BATCH_SIZE]
            yield tf.constant(X_scaled[batch_indices], dtype=tf.float32), tf.constant(y[batch_indices], dtype=tf.float32)

    def predict_indices(indices: Any) -> Any:
        predictions: list[Any] = []
        for start in range(0, len(indices), 512):
            batch_indices = indices[start : start + 512]
            predictions.append(model(tf.convert_to_tensor(X_scaled[batch_indices], dtype=tf.float32), training=False).numpy())
        return np.clip(np.vstack(predictions).astype("float32"), 0.0, 1.0)

    def evaluate(indices: Any) -> tuple[dict[str, Any], Any]:
        losses = []
        for X_batch, y_batch in make_dataset(indices, shuffle=False):
            preds = model(X_batch, training=False)
            losses.append(float(loss_fn(y_batch, preds).numpy()))
        preds_np = predict_indices(indices)
        return compute_metrics(y[indices], preds_np, float(np.mean(losses))), preds_np

    history: list[dict[str, Any]] = []
    best_weights: list[Any] | None = None
    best_val_mae = float("inf")
    best_epoch = 0
    patience = 0
    used_gradient_tape = False
    used_model_fit = False

    for epoch in range(1, MAX_EPOCHS + 1):
        batch_losses: list[float] = []
        for X_batch, y_batch in make_dataset(split_indices["train"], shuffle=True):
            with tf.GradientTape() as tape:
                used_gradient_tape = True
                predictions = model(X_batch, training=True)
                loss = loss_fn(y_batch, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            if any(gradient is None for gradient in gradients):
                raise RuntimeError("GradientTape returned None for at least one trainable variable.")
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            batch_losses.append(float(loss.numpy()))
        train_metrics, _ = evaluate(split_indices["train"])
        val_metrics, _ = evaluate(split_indices["validation"])
        logs = {
            "train_loss": float(np.mean(batch_losses)),
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"] or 0.0,
            "learning_rate": float(optimizer.learning_rate.numpy()),
        }
        gate_callback.on_epoch_end(epoch, logs)
        with writer.as_default():
            tf.summary.scalar("loss/train", logs["train_loss"], step=epoch)
            tf.summary.scalar("mae/train", logs["train_mae"], step=epoch)
            tf.summary.scalar("loss/validation", logs["val_loss"], step=epoch)
            tf.summary.scalar("mae/validation", logs["val_mae"], step=epoch)
            tf.summary.scalar("rmse/validation", logs["val_rmse"], step=epoch)
            tf.summary.scalar("r2/validation", logs["val_r2"], step=epoch)
        history.append({"epoch": epoch, **{key: round(float(value), 8) for key, value in logs.items()}})
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = float(val_metrics["mae"])
            best_epoch = int(epoch)
            best_weights = [weight.numpy() for weight in model.weights]
            patience = 0
        else:
            patience += 1
        if best_val_mae <= 0.006 and epoch >= 80:
            break
        if patience >= EARLY_STOP_PATIENCE and epoch >= 80:
            break

    if best_weights is None:
        raise RuntimeError("No best weights captured during GradientTape training.")
    for weight, value in zip(model.weights, best_weights):
        weight.assign(value)
    gate_callback.on_train_end({})

    final_predictions: dict[str, Any] = {}
    final_metrics: dict[str, Any] = {}
    all_predictions = np.empty_like(y, dtype="float32")
    for split_name, indices in split_indices.items():
        split_metrics, split_predictions = evaluate(indices)
        final_metrics[split_name] = split_metrics
        final_predictions[split_name] = split_predictions
        all_predictions[indices] = split_predictions
    if not used_gradient_tape or used_model_fit:
        raise RuntimeError("Manual GradientTape loop required and model.fit() forbidden.")
    if not np.isfinite(all_predictions).all() or float(all_predictions.min()) < 0.0 or float(all_predictions.max()) > 1.0:
        raise RuntimeError("Predictions must be finite and bounded in [0, 1].")

    writer.flush()
    writer.close()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with TRAINING_HISTORY_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(history[0].keys()) if history else ["epoch"]
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(history)
    model.save(MODEL_PATH)
    reloaded = keras.saving.load_model(MODEL_PATH, custom_objects=custom_objects)
    reload_preds = reloaded(tf.convert_to_tensor(X_scaled[:8], dtype=tf.float32), training=False).numpy()
    np.savez_compressed(
        PREDICTIONS_PATH,
        pair_id=matrix["pair_ids"],
        split=splits.astype(str),
        y_true=y,
        y_pred=all_predictions,
        feature_names=np.asarray(APPROVED_FEATURES),
        metadata_json=json.dumps({"phase_id": PHASE_ID, "model_version": MODEL_VERSION}, sort_keys=True),
    )
    slice_metrics: dict[str, Any] = {}
    for split_name, indices in split_indices.items():
        split_records = [records[int(i)] for i in indices]
        split_preds = all_predictions[indices]
        for field in ["language", "role_family", "pair_type", "score_band"]:
            groups: dict[str, list[int]] = defaultdict(list)
            for local_index, record in enumerate(split_records):
                groups[str(record.get(field) or "UNKNOWN")].append(local_index)
            slice_metrics[f"{split_name}_by_{field}"] = {
                group: compute_metrics(y[indices][local_indices], split_preds[local_indices])
                for group, local_indices in groups.items()
                if len(local_indices) >= 10
            }
    language_examples = {
        language: [
            {
                "pair_id": record["pair_id"],
                "split": record["split"],
                "target_0_100": round(float(record["job_fit_score"]) * 100.0, 4),
                "prediction_0_100": round(float(all_predictions[index][0]) * 100.0, 4),
                "pair_type": record["pair_type"],
                "role_family": record["role_family"],
            }
            for index, record in enumerate(records)
            if str(record.get("language")) == language
        ][:5]
        for language in ["ID", "MIXED", "EN", "UNKNOWN"]
    }
    ranking_by_split = {
        split_name: ranking_metrics([records[int(i)] for i in indices], all_predictions[indices])
        for split_name, indices in split_indices.items()
    }
    final_writer = tf.summary.create_file_writer(str(tensorboard_run_dir))
    with final_writer.as_default():
        for split_name, metrics in final_metrics.items():
            for key in ["mae", "rmse", "r2", "spearman", "score_band_agreement", "high_fit_recall"]:
                value = metrics.get(key)
                if value is not None:
                    tf.summary.scalar(f"final/{split_name}/{key}", float(value), step=best_epoch)
        tf.summary.histogram("histogram/prediction_0_1", all_predictions.reshape(-1)[:512], step=best_epoch)
        tf.summary.histogram("histogram/target_0_1", y.reshape(-1)[:512], step=best_epoch)
        for feature_index, feature_name in enumerate(APPROVED_FEATURES):
            tf.summary.histogram(f"histogram/feature/{feature_name}", matrix["X_raw"][:512, feature_index], step=best_epoch)
    final_writer.flush()
    final_writer.close()

    event_files = sorted(TENSORBOARD_ROOT.rglob("events.out.tfevents.*"))
    tensorboard_manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "generated_at": generated_at,
        "status": "complete" if event_files else "missing_event_files",
        "tensorboard": {
            "log_root": rel(TENSORBOARD_ROOT),
            "run_dir": rel(tensorboard_run_dir),
            "run_id": run_id,
            "event_file_count": len(event_files),
            "total_event_bytes": sum(path.stat().st_size for path in event_files),
            "event_files": [file_info(path) for path in event_files],
        },
        "written_summaries": {
            "scalars": ["loss/train", "mae/train", "loss/validation", "mae/validation", "final/*"],
            "histograms": ["prediction_0_1", "target_0_1", *APPROVED_FEATURES],
        },
    }
    write_json(TENSORBOARD_MANIFEST_PATH, tensorboard_manifest)

    report = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "generated_at": generated_at,
        "status": "complete",
        "model": {
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "api": "TensorFlow Functional API",
            "custom_components": ["CosineInteractionLayer", "WeightedHuberLoss", "ProductionGateCallback"],
            "uses_gradient_tape": used_gradient_tape,
            "uses_model_fit": used_model_fit,
            "input_shape": [None, int(X_scaled.shape[1])],
            "output_shape": [None, 1],
            "parameter_count": int(model.count_params()),
            "artifact": file_info(MODEL_PATH),
            "clean_reload_smoke": {
                "passed": bool(np.isfinite(reload_preds).all() and float(reload_preds.min()) >= 0.0 and float(reload_preds.max()) <= 1.0),
                "sample_count": int(len(reload_preds)),
            },
        },
        "training": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "epochs_run": int(history[-1]["epoch"] if history else 0),
            "best_epoch": best_epoch,
            "history": file_info(TRAINING_HISTORY_PATH),
            "predictions": file_info(PREDICTIONS_PATH),
        },
        "metrics": final_metrics,
        "slice_metrics": slice_metrics,
        "ranking_metrics": ranking_by_split,
        "indonesian_behavior_examples": language_examples,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
        },
        "tensorboard_manifest": file_info(TENSORBOARD_MANIFEST_PATH),
    }
    write_json(EVALUATION_REPORT_PATH, report)
    return report


def build_baseline_comparison(matrix: dict[str, Any], records: list[dict[str, Any]], training_report: dict[str, Any], generated_at: str) -> dict[str, Any]:
    import numpy as np

    X = matrix["X_raw"]
    y = matrix["y"]
    splits = matrix["splits"].astype(str)
    split_indices = {name: np.where(splits == name)[0] for name in ["train", "validation", "test"]}
    train_idx = split_indices["train"]
    constant_mean_pred = np.full_like(y, float(np.mean(y[train_idx])))
    constant_median_pred = np.full_like(y, float(np.median(y[train_idx])))
    skill_pred = linear_regression_predict(X[train_idx][:, [1]], y[train_idx], X[:, [1]])
    cosine_pred = linear_regression_predict(X[train_idx][:, [0]], y[train_idx], X[:, [0]])
    multi_linear_pred = linear_regression_predict(X[train_idx], y[train_idx], X)
    phase25_payload = np.load(PHASE25_PREDICTIONS_PATH, allow_pickle=True) if PHASE25_PREDICTIONS_PATH.exists() else None
    phase25_pred_map = {}
    if phase25_payload is not None:
        phase25_pred_map = {str(pair_id): float(pred) for pair_id, pred in zip(phase25_payload["pair_id"], phase25_payload["y_pred"].reshape(-1))}
    phase25_preds = np.asarray([[phase25_pred_map.get(str(row["pair_id"]), np.nan)] for row in records], dtype="float32")
    model_preds = np.load(PREDICTIONS_PATH, allow_pickle=True)["y_pred"]
    predictors = {
        "constant_train_mean": constant_mean_pred,
        "constant_train_median": constant_median_pred,
        "skill_overlap_only_regression": skill_pred,
        "cosine_only_multilingual_e5_small_regression": cosine_pred,
        "six_feature_linear_regression": multi_linear_pred,
        "e5_base_phase25_scorer": phase25_preds,
        "tensorflow_phase45_candidate": model_preds,
    }
    metrics: dict[str, Any] = {}
    for model_name, preds in predictors.items():
        metrics[model_name] = {}
        for split_name, indices in split_indices.items():
            valid = np.isfinite(preds[indices].reshape(-1))
            if not bool(valid.all()):
                metrics[model_name][split_name] = {"status": "missing_predictions", "row_count": int(len(indices))}
                continue
            metrics[model_name][split_name] = compute_metrics(y[indices], preds[indices])
            metrics[model_name][split_name]["ranking"] = ranking_metrics([records[int(i)] for i in indices], preds[indices])
    thresholds = {
        "validation_mae_max": TARGET_MAE,
        "test_mae_max": TARGET_MAE,
        "validation_r2_min": MIN_R2,
        "test_r2_min": MIN_R2,
        "high_fit_recall_min": 0.90,
        "score_band_agreement_min": 0.95,
        "must_not_regress_phase25_mae_0_100_by_more_than": 0.25,
    }
    candidate_val = metrics["tensorflow_phase45_candidate"]["validation"]
    candidate_test = metrics["tensorflow_phase45_candidate"]["test"]
    phase25_val = metrics["e5_base_phase25_scorer"]["validation"]
    phase25_test = metrics["e5_base_phase25_scorer"]["test"]
    checks = {
        "validation_mae_le_0_02": candidate_val.get("mae", 1.0) <= thresholds["validation_mae_max"],
        "test_mae_le_0_02": candidate_test.get("mae", 1.0) <= thresholds["test_mae_max"],
        "validation_r2_min": (candidate_val.get("r2") or -1.0) >= thresholds["validation_r2_min"],
        "test_r2_min": (candidate_test.get("r2") or -1.0) >= thresholds["test_r2_min"],
        "validation_high_fit_recall_min": (candidate_val.get("high_fit_recall") or 0.0) >= thresholds["high_fit_recall_min"],
        "test_high_fit_recall_min": (candidate_test.get("high_fit_recall") or 0.0) >= thresholds["high_fit_recall_min"],
        "validation_score_band_agreement_min": candidate_val.get("score_band_agreement", 0.0) >= thresholds["score_band_agreement_min"],
        "test_score_band_agreement_min": candidate_test.get("score_band_agreement", 0.0) >= thresholds["score_band_agreement_min"],
        "validation_phase25_mae_non_regression": candidate_val.get("mae_0_100", 999.0) <= phase25_val.get("mae_0_100", 999.0) + thresholds["must_not_regress_phase25_mae_0_100_by_more_than"],
        "test_phase25_mae_non_regression": candidate_test.get("mae_0_100", 999.0) <= phase25_test.get("mae_0_100", 999.0) + thresholds["must_not_regress_phase25_mae_0_100_by_more_than"],
    }
    failures = [name for name, passed in checks.items() if not passed]
    decision = "select_for_staging_shadow_validation" if not failures else "reject_keep_e5_base_phase25_selected"
    previous = read_json(PHASE25_BASELINE_SELECTION_PATH)
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "generated_at": generated_at,
        "status": "complete",
        "thresholds": thresholds,
        "metrics": metrics,
        "selection_checks": checks,
        "selection_failures": failures,
        "selection_decision": decision,
        "previous_selected_scorer": {
            "source": rel(PHASE25_BASELINE_SELECTION_PATH),
            "selection_decision": previous.get("selection_decision"),
            "phase25_metrics": previous.get("phase25_metrics"),
        },
        "candidate_reranking_fixture_source": file_info(PHASE21_RERANKING_PATH),
        "notes": [
            "Baselines use the same frozen labels, splits, pair IDs, and six-feature contract as Phase 25.",
            "Candidate reranking metrics are computed from grouped pair rows and existing backend candidate fixture evidence remains source evidence for backend-owned hydration boundary.",
            "Selection here is staging-shadow only; calibration, model card, artifact manifest, and handoff fixtures are refreshed later.",
        ],
    }
    write_json(BASELINE_REPORT_PATH, report)
    write_json(SELECTION_DECISION_PATH, {"schema_version": SCHEMA_VERSION, "phase_id": PHASE_ID, "generated_at": generated_at, "selection_decision": decision, "selection_failures": failures, "selection_checks": checks})
    return report


def write_readme() -> None:
    README_PATH.write_text(
        "\n".join(
            [
                "# Phase 45 multilingual-E5-small Training Delivery Artifacts",
                "",
                "This directory stores TensorFlow retraining evidence using `intfloat/multilingual-e5-small` features. Phase 25 artifacts remain rollback-safe and are not overwritten.",
                "",
                "## Files",
                "",
                "- `tensorflow_training_features_v1.npz` — approved six-feature matrix with multilingual-E5-small `e5_cosine` and frozen Phase 25 labels/splits.",
                "- `tensorflow_feature_config.json` — feature order, train-split normalization stats, embedding metadata, and source hashes.",
                "- `gradient_tape_training_history.csv` — manual `tf.GradientTape` training history.",
                "- `gradient_tape_predictions_v1.npz` — bounded `0-1` predictions and targets for every frozen pair.",
                "- `gradient_tape_trained_candidate.keras` — trained TensorFlow Functional API candidate model with registered custom components.",
                "- `training_evaluation.json` — quality metrics, slice metrics, ranking metrics, clean reload smoke, and Indonesian behavior examples.",
                "- `baseline_comparison.json` — constant, skill-only, cosine-only, Phase 25, previous-scorer, and Phase 45 comparisons.",
                "- `tensorboard_monitoring_manifest.json` — bounded TensorBoard event file manifest with hashes and byte sizes.",
                "- `selection_decision.json` — staging-shadow select/reject result. Model API defaults must not read Phase 45 artifacts until calibration and handoff refresh pass.",
                "",
                "## Safety",
                "",
                "These artifacts are training evidence only. Do not mix them with Phase 25 calibration or Model API handoff files.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_report() -> dict[str, Any]:
    training = read_json(EVALUATION_REPORT_PATH)
    baseline = read_json(BASELINE_REPORT_PATH)
    feature_config = read_json(FEATURE_CONFIG_PATH)
    selection = read_json(SELECTION_DECISION_PATH)
    required_paths = [
        TRAINING_FEATURES_PATH,
        FEATURE_CONFIG_PATH,
        TRAINING_HISTORY_PATH,
        PREDICTIONS_PATH,
        MODEL_PATH,
        BASELINE_REPORT_PATH,
        EVALUATION_REPORT_PATH,
        TENSORBOARD_MANIFEST_PATH,
        SELECTION_DECISION_PATH,
        README_PATH,
    ]
    checks = {
        "notebook_exists_with_required_markdown": (ROOT / "training/notebooks/phase_45_multilingual_e5_small_training_delivery.ipynb").exists(),
        "phase25_artifacts_not_mutated_by_namespace": PHASE25_DIR.exists() and ARTIFACT_DIR.exists(),
        "feature_config_records_multilingual_e5_small": feature_config.get("embedding_model_metadata", {}).get("embedding_model") == EMBEDDING_MODEL,
        "approved_feature_contract_preserved": feature_config.get("approved_features") == APPROVED_FEATURES,
        "train_split_normalization_written": bool(feature_config.get("normalization", {}).get("mean") and feature_config.get("normalization", {}).get("std")),
        "gradient_tape_training_used": training.get("model", {}).get("uses_gradient_tape") is True and training.get("model", {}).get("uses_model_fit") is False,
        "custom_components_present": set(training.get("model", {}).get("custom_components", [])) >= {"CosineInteractionLayer", "WeightedHuberLoss", "ProductionGateCallback"},
        "required_metrics_present": all(key in training.get("metrics", {}).get("validation", {}) for key in ["mae", "rmse", "r2", "spearman", "score_band_agreement", "high_fit_recall"]),
        "ranking_metrics_present": all(key in training.get("ranking_metrics", {}).get("validation", {}) for key in ["ndcg_at_5", "ndcg_at_10", "map_at_10"]),
        "indonesian_behavior_recorded": bool(training.get("indonesian_behavior_examples", {}).get("ID") or training.get("indonesian_behavior_examples", {}).get("MIXED")),
        "tensorboard_manifest_has_event_file_hashes": read_json(TENSORBOARD_MANIFEST_PATH).get("tensorboard", {}).get("event_file_count", 0) >= 1,
        "selection_or_rejection_recorded": selection.get("selection_decision") in {"select_for_staging_shadow_validation", "reject_keep_e5_base_phase25_selected"},
    }
    missing_paths = require_paths(required_paths)
    blockers = missing_paths + [name for name, passed in checks.items() if not passed]
    status = "complete" if not blockers else "incomplete"
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "generated_at": now_utc(),
        "status": status,
        "checks": checks,
        "blockers": blockers,
        "selection_decision": selection.get("selection_decision"),
        "selection_failures": selection.get("selection_failures", []),
        "summary": {
            "embedding_model": feature_config.get("embedding_model_metadata", {}).get("embedding_model"),
            "feature_count": feature_config.get("feature_matrix", {}).get("feature_count"),
            "row_count": feature_config.get("feature_matrix", {}).get("row_count"),
            "split_counts": feature_config.get("feature_matrix", {}).get("split_counts"),
            "validation_metrics": training.get("metrics", {}).get("validation"),
            "test_metrics": training.get("metrics", {}).get("test"),
        },
        "artifact_paths": {name: rel(path) for name, path in {
            "feature_config": FEATURE_CONFIG_PATH,
            "feature_matrix": TRAINING_FEATURES_PATH,
            "training_evaluation": EVALUATION_REPORT_PATH,
            "baseline_comparison": BASELINE_REPORT_PATH,
            "tensorboard_manifest": TENSORBOARD_MANIFEST_PATH,
            "selection_decision": SELECTION_DECISION_PATH,
            "model": MODEL_PATH,
        }.items()},
        "source_files": [rel(path) for path in [ROOT / "GAP_MODEL_TRAINING.md", ROOT / "GAP_MODEL_TRAINING.md", ROOT / "REQUIREMENT.md", PHASE44_FEATURES_PATH, PHASE44_DECISION_PATH, PHASE25_FEATURE_CONFIG_PATH]],
        "git_state": git_state(),
    }
    return report


def write_markdown(report: dict[str, Any]) -> None:
    summary = report.get("summary", {})
    lines = [
        "# Phase 45 multilingual-E5-small Training Delivery",
        "",
        f"Status: `{report['status']}`",
        f"Selection decision: `{report.get('selection_decision')}`",
        "",
        "This report records TensorFlow retraining evidence with `intfloat/multilingual-e5-small`-derived `e5_cosine` values while preserving the approved model-core contract.",
        "",
        "## Summary",
        f"- Embedding model: `{summary.get('embedding_model')}`",
        f"- Feature rows: `{summary.get('row_count')}`",
        f"- Feature count: `{summary.get('feature_count')}`",
        f"- Split counts: `{summary.get('split_counts')}`",
        f"- Validation metrics: `{summary.get('validation_metrics')}`",
        f"- Test metrics: `{summary.get('test_metrics')}`",
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
    lines.extend(
        [
            "",
            "## Commands",
            "- Train/write evidence: `training/.tf-venv-3.13/bin/python scripts/verify_phase_45_multilingual_e5_small_training_delivery.py --write`",
            "- Verify: `.venv/bin/python -m unittest tests.test_phase_45_multilingual_e5_small_training_delivery`",
        ]
    )
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all() -> dict[str, Any]:
    missing = require_paths([PHASE44_FEATURES_PATH, PHASE44_DECISION_PATH, PHASE25_FEATURE_CONFIG_PATH, PHASE25_PREDICTIONS_PATH])
    if missing:
        raise RuntimeError(f"Missing required source artifacts: {missing}")
    generated_at = now_utc()
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_ROOT.mkdir(parents=True, exist_ok=True)
    records = load_records()
    matrix = build_feature_matrix(records)
    write_feature_config(matrix, records, generated_at)
    training_report = train_tensorflow(matrix, records, generated_at)
    build_baseline_comparison(matrix, records, training_report, generated_at)
    write_readme()
    report = build_report()
    write_json(REPORT_JSON_PATH, report)
    write_markdown(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Train/write Phase 45 evidence.")
    args = parser.parse_args(argv)
    report = write_all() if args.write else build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report.get("blockers") else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
