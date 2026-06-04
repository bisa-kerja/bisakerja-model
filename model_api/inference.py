"""Inference service boundary for loaded TensorFlow model runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Protocol, Sequence

from .artifacts import ArtifactVerificationReport, VerifiedArtifact, load_json
from .config import ArtifactPaths
from .custom_objects import load_phase25_keras_model
from .errors import InferenceTimeoutError, ModelApiError, ModelLoadError, ModelNotReadyError
from .features import FeatureVector
from .schemas import MAX_RECOMMENDATIONS, MatchLevel, ModelArtifactIdentity, ModelIdentity, RecommendationScore


class PredictModel(Protocol):
    """Minimal protocol expected from loaded Keras model."""

    name: str

    def predict(self, values: Sequence[Sequence[float]], verbose: int = 0):  # pragma: no cover - runtime protocol
        ...


ModelLoader = Callable[[Path], PredictModel]

UNKNOWN_MODEL_NAME = "unknown_phase25_keras_model"
UNKNOWN_MODEL_VERSION = "unknown_phase25_model_version"


@dataclass(frozen=True)
class ScoreCalibrationBucket:
    min_score: int
    max_score: int
    mean_signed_error: float = 0.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ScoreCalibrationBucket":
        raw_min = payload.get("min")
        raw_max = payload.get("max")
        if not isinstance(raw_min, int) or isinstance(raw_min, bool):
            raw_min = _bucket_edge(payload.get("bucket"), 0)
        if not isinstance(raw_max, int) or isinstance(raw_max, bool):
            raw_max = _bucket_edge(payload.get("bucket"), 1)
        if raw_min is None or raw_max is None:
            raise ModelLoadError(f"Invalid score calibration bucket: {payload!r}")
        raw_error = payload.get("mean_signed_error", 0.0)
        mean_signed_error = float(raw_error) if isinstance(raw_error, (int, float)) and not isinstance(raw_error, bool) else 0.0
        return cls(min_score=raw_min, max_score=raw_max, mean_signed_error=mean_signed_error)

    def contains(self, score: int) -> bool:
        return self.min_score <= score <= self.max_score


@dataclass(frozen=True)
class ScoreCalibrationPolicy:
    """Bucket correction loaded from Phase 25 ``score_calibration.json``."""

    buckets: tuple[ScoreCalibrationBucket, ...] = ()
    output_name: str = "recommendations[].matchScore"

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        output_name: str = "recommendations[].matchScore",
    ) -> "ScoreCalibrationPolicy":
        table_buckets: list[ScoreCalibrationBucket] = []
        tables = payload.get("tables", ())
        if isinstance(tables, Sequence) and not isinstance(tables, (str, bytes, bytearray)):
            for row in tables:
                if not isinstance(row, Mapping) or row.get("output") != output_name:
                    continue
                table_buckets.append(ScoreCalibrationBucket.from_mapping(row))
        if table_buckets:
            return cls(buckets=tuple(table_buckets), output_name=output_name)

        fallback_buckets: list[ScoreCalibrationBucket] = []
        buckets = payload.get("buckets", ())
        if isinstance(buckets, Sequence) and not isinstance(buckets, (str, bytes, bytearray)):
            for row in buckets:
                if isinstance(row, Mapping):
                    fallback_buckets.append(ScoreCalibrationBucket.from_mapping(row))
        return cls(buckets=tuple(fallback_buckets), output_name=output_name)

    @classmethod
    def from_path(cls, path: Path, output_name: str = "recommendations[].matchScore") -> "ScoreCalibrationPolicy":
        return cls.from_mapping(load_json(path), output_name=output_name)

    def apply(self, score: int) -> int:
        for bucket in self.buckets:
            if bucket.contains(score):
                return clamp_score_0_100(score - bucket.mean_signed_error)
        return clamp_score_0_100(score)


def _bucket_edge(bucket: Any, index: int) -> int | None:
    if not isinstance(bucket, str) or "-" not in bucket:
        return None
    parts = bucket.split("-", 1)
    try:
        return int(parts[index])
    except (TypeError, ValueError):
        return None


def clamp_score_0_100(value: float) -> int:
    if not math.isfinite(value):
        raise ModelLoadError(f"Model prediction must be finite; actual={value!r}")
    return max(0, min(100, int(round(value))))


def score_0_1_to_0_100(value: float) -> int:
    return clamp_score_0_100(value * 100.0)


def match_level_for_score(score: int) -> MatchLevel:
    if score >= 81:
        return "strong"
    if score >= 61:
        return "good"
    return "stretch"


def _coerce_prediction_score(value: Any) -> float:
    """Coerce Keras/numpy/list scalar prediction into finite float score_0_1."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 1:
            raise ModelLoadError(f"Model prediction row must contain one score; actual={value!r}")
        value = value[0]
        if hasattr(value, "tolist"):
            value = value.tolist()
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ModelLoadError(f"Model prediction must be numeric; actual={value!r}")
    score = float(value)
    if not math.isfinite(score):
        raise ModelLoadError(f"Model prediction must be finite; actual={value!r}")
    return score


@dataclass(frozen=True)
class RuntimeState:
    ready: bool
    model_identity: ModelIdentity | None = None
    artifact_manifest_phase: str | None = None
    message: str = "not_loaded"
    loaded_at: str | None = None
    error_code: str | None = None


def utc_now_iso() -> str:
    """Return stable UTC timestamp for readiness metadata."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def find_verified_artifact(
    report: ArtifactVerificationReport,
    artifact_id: str,
) -> VerifiedArtifact | None:
    """Find verified artifact by manifest ID."""

    for artifact in report.verified_artifacts:
        if artifact.artifact_id == artifact_id:
            return artifact
    return None


def build_model_identity(
    paths: ArtifactPaths,
    artifact_report: ArtifactVerificationReport,
    loaded_model: PredictModel | None = None,
) -> ModelIdentity:
    """Build public model identity from Phase 25 model card and verified hash."""

    model_card = load_json(paths.model_card_path)
    model_section = model_card.get("model", {})
    if not isinstance(model_section, dict):
        model_section = {}

    model_name = str(model_section.get("name") or getattr(loaded_model, "name", "") or UNKNOWN_MODEL_NAME)
    model_version = str(model_section.get("version") or UNKNOWN_MODEL_VERSION)
    data_section = model_card.get("data", {})
    if not isinstance(data_section, Mapping):
        data_section = {}
    embedding_contract = data_section.get("embedding_contract")
    if not isinstance(embedding_contract, Mapping):
        embedding_contract = {}
    embedding_model = str(
        model_section.get("embedding_model")
        or embedding_contract.get("embedding_model")
        or (artifact_report.manifest.embedding_model_metadata or {}).get("embedding_model")
        or ""
    ) or None
    final_model = find_verified_artifact(artifact_report, "final_keras_model")
    artifact_path = paths.model_path if final_model is None else final_model.path
    artifact_sha256 = None if final_model is None else final_model.sha256

    return ModelIdentity(
        name=model_name,
        version=model_version,
        artifact=ModelArtifactIdentity(
            format=artifact_path.suffix or ".keras",
            path=str(artifact_path),
            sha256=artifact_sha256,
        ),
        artifact_phase=artifact_report.manifest.phase_id,
        embedding_model=embedding_model,
    )


class InferenceService:
    """Loaded model facade used by HTTP handlers.

    Startup calls ``load_once`` after artifact hash verification. Batch scoring,
    calibration, sorting, and max-item limiting are added in Step 26.7.
    """

    def __init__(
        self,
        model: PredictModel | None = None,
        state: RuntimeState | None = None,
        model_loader: ModelLoader = load_phase25_keras_model,
    ) -> None:
        self._model = model
        self._model_loader = model_loader
        self._state = state or RuntimeState(
            ready=model is not None,
            message="ready" if model is not None else "not_loaded",
        )
        self._load_count = 0

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def load_count(self) -> int:
        """Number of actual loader invocations in this process."""

        return self._load_count

    def load_once(self, paths: ArtifactPaths, artifact_report: ArtifactVerificationReport) -> RuntimeState:
        """Load Phase 25 Keras model at most once and capture readiness state.

        Loader failures are stored in state so health/model-info can expose not
        ready and inference handlers can return deterministic 503 responses.
        """

        if self._model is not None and self._state.ready and self._state.model_identity is not None:
            return self._state

        self._state = RuntimeState(
            ready=False,
            artifact_manifest_phase=artifact_report.manifest.phase_id,
            message="loading",
        )

        try:
            model = self._model
            if model is None:
                model = self._model_loader(paths.model_path)
                self._load_count += 1
            identity = build_model_identity(paths, artifact_report, model)
        except ModelApiError as exc:
            self._state = RuntimeState(
                ready=False,
                artifact_manifest_phase=artifact_report.manifest.phase_id,
                message=f"load_failed: {exc}",
                error_code=exc.code,
            )
            return self._state
        except Exception as exc:  # pragma: no cover - defensive runtime wrapper
            wrapped = ModelLoadError(f"TensorFlow model load failed: {exc}")
            self._state = RuntimeState(
                ready=False,
                artifact_manifest_phase=artifact_report.manifest.phase_id,
                message=str(wrapped),
                error_code=wrapped.code,
            )
            return self._state

        self._model = model
        self._state = RuntimeState(
            ready=True,
            model_identity=identity,
            artifact_manifest_phase=artifact_report.manifest.phase_id,
            message="ready",
            loaded_at=utc_now_iso(),
        )
        return self._state

    def require_ready(self) -> None:
        if self._model is None or not self._state.ready:
            detail = self._state.message or "TensorFlow model is not loaded yet"
            raise ModelNotReadyError(f"TensorFlow model is not ready: {detail}")

    def predict_recommendations(
        self,
        vectors: Sequence[FeatureVector],
        max_recommendations: int = MAX_RECOMMENDATIONS,
        calibration_policy: ScoreCalibrationPolicy | None = None,
        timeout_ms: int | None = None,
        allow_internal_full_ranking: bool = False,
    ) -> tuple[RecommendationScore, ...]:
        """Run batch model scoring, calibrate, sort, and cap recommendations."""

        self.require_ready()
        if not vectors:
            return ()
        if timeout_ms is not None and timeout_ms <= 0:
            raise InferenceTimeoutError(f"TensorFlow inference timeout before prediction; timeout_ms={timeout_ms}")

        model_rows = [vector.as_model_row() for vector in vectors]
        try:
            import numpy as np  # type: ignore[import-not-found]

            model_input = np.asarray(model_rows, dtype="float32")
        except ModuleNotFoundError:  # pragma: no cover - NumPy is a serving dependency
            model_input = model_rows
        started_at = monotonic()
        try:
            raw_predictions = self._model.predict(model_input, verbose=0)  # type: ignore[union-attr]
        except ModelApiError:
            raise
        except Exception as exc:  # pragma: no cover - defensive runtime wrapper
            raise ModelLoadError(f"TensorFlow model prediction failed: {exc}") from exc
        elapsed_ms = int((monotonic() - started_at) * 1000)
        if timeout_ms is not None and elapsed_ms > timeout_ms:
            raise InferenceTimeoutError(
                f"TensorFlow inference timeout; timeout_ms={timeout_ms} elapsed_ms={elapsed_ms}"
            )

        if hasattr(raw_predictions, "tolist"):
            raw_predictions = raw_predictions.tolist()
        if not isinstance(raw_predictions, Sequence) or isinstance(raw_predictions, (str, bytes, bytearray)):
            raise ModelLoadError(f"Model predictions must be a sequence; actual={raw_predictions!r}")
        if len(raw_predictions) != len(vectors):
            raise ModelLoadError(
                "Model prediction count mismatch; "
                f"expected={len(vectors)} actual={len(raw_predictions)}"
            )

        limit_cap = len(vectors) if allow_internal_full_ranking else MAX_RECOMMENDATIONS
        limit = max(0, min(limit_cap, int(max_recommendations)))
        recommendations: list[tuple[int, RecommendationScore]] = []
        for index, (vector, raw_prediction) in enumerate(zip(vectors, raw_predictions, strict=True)):
            score_0_1 = _coerce_prediction_score(raw_prediction)
            score = score_0_1_to_0_100(score_0_1)
            if calibration_policy is not None:
                score = calibration_policy.apply(score)
            recommendations.append(
                (
                    index,
                    RecommendationScore(
                        jobId=vector.candidate_id,
                        matchScore=score,
                        matchLevel=match_level_for_score(score),
                    ),
                )
            )

        recommendations.sort(key=lambda item: (-item[1].matchScore, item[0]))
        return tuple(recommendation for _, recommendation in recommendations[:limit])
