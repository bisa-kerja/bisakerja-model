"""Phase 25 feature construction for Model API inference.

The serving API must feed the TensorFlow model the exact six-feature vector
exported by Phase 25. This module owns deterministic text assembly, E5 cosine
similarity, rule-derived scalar features, finite validation, and train-split
normalization from ``tensorflow_feature_config.json``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re
from threading import Lock
from typing import Any, Protocol

from .artifacts import load_json
from .errors import FeatureBuildError
from .schemas import CandidateJobInput, CandidateScoringInput, SanitizedProfileInput

PHASE25_FEATURE_ORDER: tuple[str, ...] = (
    "e5_cosine",
    "skill_overlap",
    "requirement_coverage",
    "role_match",
    "experience_match",
    "experience_gap_years_clipped",
)

E5_MODEL_NAME = "intfloat/e5-base-v2"
E5_BACKEND_NAME = "sentence-transformers"
E5_PROFILE_PREFIX = "query:"
E5_JOB_PREFIX = "passage:"
E5_NORMALIZE_EMBEDDINGS = True

PRODUCTION_ENVIRONMENTS: frozenset[str] = frozenset({"staging", "production", "prod"})
FORBIDDEN_FALLBACK_BACKENDS: frozenset[str] = frozenset({"tf-idf", "tfidf", "local-hash", "local_hash", "hash"})

SKILL_ALIASES: Mapping[str, str] = {
    "js": "javascript",
    "javascript": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "reactjs": "react",
    "react.js": "react",
    "react js": "react",
    "react": "react",
    "nextjs": "next.js",
    "next.js": "next.js",
    "nodejs": "node.js",
    "node.js": "node.js",
    "node js": "node.js",
    "vuejs": "vue.js",
    "vue.js": "vue.js",
    "python": "python",
    "py": "python",
    "java": "java",
    "golang": "go",
    "go": "go",
    "laravel": "laravel",
    "php": "php",
    "sql": "sql",
    "mysql": "mysql",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "mongo": "mongodb",
    "mongodb": "mongodb",
    "aws": "aws",
    "gcp": "gcp",
    "google cloud": "gcp",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "k8s": "kubernetes",
    "machine learning": "machine learning",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "analisis data": "data analysis",
    "data analysis": "data analysis",
    "data analytics": "data analysis",
    "project management": "project management",
    "manajemen proyek": "project management",
    "html": "html",
    "css": "css",
    "tailwind": "tailwind",
    "jquery": "jquery",
    "c++": "c++",
    "c#": "c#",
    "linux": "linux",
    "ci/cd": "ci/cd",
    "ci cd": "ci/cd",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "rest api": "rest api",
    "api": "api",
}

EXPERIENCE_YEARS_BY_VALUE: Mapping[str, float] = {
    "fresher": 0.0,
    "entry": 0.0,
    "entry_level": 0.0,
    "entry level": 0.0,
    "junior": 1.5,
    "1-2 years": 1.5,
    "1 2 years": 1.5,
    "mid": 4.0,
    "mid_level": 4.0,
    "mid level": 4.0,
    "3-5 years": 4.0,
    "3 5 years": 4.0,
    "senior": 5.0,
    "5+ years": 5.0,
    "5 years": 5.0,
    "lead": 6.0,
    "manager": 7.0,
}

EXPERIENCE_NUMERIC_KEYS: tuple[str, ...] = (
    "experience_years",
    "experienceYears",
    "required_experience_years",
    "requiredExperienceYears",
    "job_experience_years",
    "jobExperienceYears",
)


@dataclass(frozen=True)
class EmbeddingModelMetadata:
    """Artifact-declared embedding model and prefix policy."""

    embedding_model: str = E5_MODEL_NAME
    profile_prefix: str = E5_PROFILE_PREFIX
    job_prefix: str = E5_JOB_PREFIX
    normalized_embeddings: bool = E5_NORMALIZE_EMBEDDINGS
    embedding_dimension: int | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "EmbeddingModelMetadata":
        if not isinstance(payload, Mapping):
            return cls()
        raw_model = payload.get("embedding_model") or payload.get("model_name")
        raw_profile_prefix = payload.get("profile_prefix")
        raw_job_prefix = payload.get("job_prefix")
        raw_normalized = payload.get("normalized_embeddings")
        raw_dimension = payload.get("embedding_dimension")
        return cls(
            embedding_model=str(raw_model or E5_MODEL_NAME),
            profile_prefix=str(raw_profile_prefix or E5_PROFILE_PREFIX),
            job_prefix=str(raw_job_prefix or E5_JOB_PREFIX),
            normalized_embeddings=bool(E5_NORMALIZE_EMBEDDINGS if raw_normalized is None else raw_normalized),
            embedding_dimension=int(raw_dimension) if isinstance(raw_dimension, int) and not isinstance(raw_dimension, bool) else None,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "embeddingModel": self.embedding_model,
            "profilePrefix": self.profile_prefix,
            "jobPrefix": self.job_prefix,
            "normalizedEmbeddings": self.normalized_embeddings,
            "embeddingDimension": self.embedding_dimension,
        }


class TextEmbeddingBackend(Protocol):
    """Minimal backend contract for E5 encoding."""

    model_name: str
    backend_name: str

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:  # pragma: no cover - protocol only
        ...


@dataclass(frozen=True)
class FeatureVector:
    """Single candidate pair vector in Phase 25 order."""

    candidate_id: str
    values: tuple[float, ...]

    def as_model_row(self) -> list[float]:
        return list(self.values)


@dataclass(frozen=True)
class CandidateFeatureVectors:
    """Raw and normalized vectors for one candidate pair."""

    candidate_id: str
    raw: FeatureVector
    normalized: FeatureVector

    def as_model_row(self) -> list[float]:
        return self.normalized.as_model_row()


@dataclass(frozen=True)
class TensorFlowFeatureConfig:
    """Approved feature order plus train-split normalization stats."""

    approved_features: tuple[str, ...]
    mean: Mapping[str, float]
    std: Mapping[str, float]
    model_name: str | None = None
    model_version: str | None = None
    schema_version: str | None = None
    embedding_metadata: EmbeddingModelMetadata = EmbeddingModelMetadata()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TensorFlowFeatureConfig":
        approved = tuple(str(value) for value in payload.get("approved_features", ()))
        assert_phase25_feature_order(approved)

        policy = payload.get("feature_policy", {})
        if isinstance(policy, Mapping) and policy.get("uses_e5_derived_similarity") is not True:
            raise FeatureBuildError("tensorflow_feature_config.json must require E5-derived similarity")

        normalization = payload.get("normalization")
        if not isinstance(normalization, Mapping):
            raise FeatureBuildError("tensorflow_feature_config.json missing normalization object")
        raw_mean = normalization.get("mean")
        raw_std = normalization.get("std")
        if not isinstance(raw_mean, Mapping) or not isinstance(raw_std, Mapping):
            raise FeatureBuildError("tensorflow_feature_config.json missing normalization mean/std")

        mean = _feature_stats(raw_mean, "normalization.mean")
        std = _feature_stats(raw_std, "normalization.std")
        for feature_name in PHASE25_FEATURE_ORDER:
            scale = std[feature_name]
            if scale <= 0.0:
                raise FeatureBuildError(f"Normalization std must be positive for {feature_name}")

        return cls(
            approved_features=approved,
            mean=mean,
            std=std,
            model_name=None if payload.get("model_name") is None else str(payload.get("model_name")),
            model_version=None if payload.get("model_version") is None else str(payload.get("model_version")),
            schema_version=None if payload.get("schema_version") is None else str(payload.get("schema_version")),
            embedding_metadata=EmbeddingModelMetadata.from_mapping(
                payload.get("embedding_model_metadata") if isinstance(payload.get("embedding_model_metadata"), Mapping) else None
            ),
        )

    @classmethod
    def from_path(cls, path: Path) -> "TensorFlowFeatureConfig":
        return cls.from_mapping(load_json(path))

    @property
    def embedding_model_name(self) -> str:
        return self.embedding_metadata.embedding_model

    @property
    def profile_prefix(self) -> str:
        return self.embedding_metadata.profile_prefix

    @property
    def job_prefix(self) -> str:
        return self.embedding_metadata.job_prefix

    def normalize(self, vector: FeatureVector) -> FeatureVector:
        return normalize_feature_vector(vector, self.mean, self.std)


class SentenceTransformerE5Embedder:
    """Lazy ``sentence-transformers`` E5 encoder for runtime inference."""

    model_name = E5_MODEL_NAME
    backend_name = E5_BACKEND_NAME

    def __init__(self, model_name: str = E5_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._model_lock = Lock()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
            except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency
                raise FeatureBuildError(f"sentence-transformers dependency missing for {self.model_name} embeddings: {exc}") from exc
            except Exception as exc:  # pragma: no cover - optional runtime dependency
                raise FeatureBuildError(f"sentence-transformers import failed for {self.model_name} embeddings: {exc!r}") from exc
            self._model = SentenceTransformer(self.model_name)
            return self._model

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        model = self._load_model()
        return model.encode(
            list(texts),
            normalize_embeddings=E5_NORMALIZE_EMBEDDINGS,
            convert_to_numpy=False,
        )


@dataclass(frozen=True)
class FeatureBuilder:
    """Build raw and normalized Phase 25 feature vectors for candidate pairs."""

    feature_config: TensorFlowFeatureConfig
    embedding_backend: TextEmbeddingBackend
    environment: str = "local"

    @classmethod
    def from_tensorflow_feature_config_path(
        cls,
        path: Path,
        embedding_backend: TextEmbeddingBackend | None = None,
        environment: str = "local",
    ) -> "FeatureBuilder":
        feature_config = TensorFlowFeatureConfig.from_path(path)
        return cls(
            feature_config=feature_config,
            embedding_backend=embedding_backend or SentenceTransformerE5Embedder(feature_config.embedding_model_name),
            environment=environment,
        )

    def build_candidate(self, profile: SanitizedProfileInput, candidate: CandidateJobInput) -> CandidateFeatureVectors:
        validate_e5_backend(self.embedding_backend, self.environment, expected_model_name=self.feature_config.embedding_model_name)
        raw = build_candidate_raw_feature_map(
            profile,
            candidate,
            self.embedding_backend,
            self.environment,
            expected_model_name=self.feature_config.embedding_model_name,
            profile_prefix=self.feature_config.profile_prefix,
            job_prefix=self.feature_config.job_prefix,
        )
        raw_vector = build_ordered_feature_vector(candidate.jobId, raw)
        normalized = self.feature_config.normalize(raw_vector)
        return CandidateFeatureVectors(candidate_id=candidate.jobId, raw=raw_vector, normalized=normalized)

    def build_batch(self, profile: SanitizedProfileInput, candidates: Sequence[CandidateJobInput]) -> tuple[CandidateFeatureVectors, ...]:
        validate_e5_backend(self.embedding_backend, self.environment, expected_model_name=self.feature_config.embedding_model_name)
        if not candidates:
            return ()

        profile_text = prefixed_e5_text(self.feature_config.profile_prefix, build_profile_e5_text(profile))
        job_texts = tuple(
            prefixed_e5_text(self.feature_config.job_prefix, build_job_e5_text(candidate.model_scoring_input))
            for candidate in candidates
        )
        try:
            encoded = self.embedding_backend.encode((profile_text, *job_texts))
        except FeatureBuildError:
            raise
        except Exception as exc:
            raise FeatureBuildError(f"E5 embedding failed: {exc}") from exc
        vectors = tuple(encoded)
        expected_count = len(candidates) + 1
        if len(vectors) != expected_count:
            raise FeatureBuildError(f"E5 backend must return {expected_count} vectors; actual={len(vectors)}")

        profile_vector = vectors[0]
        output: list[CandidateFeatureVectors] = []
        for candidate, job_vector in zip(candidates, vectors[1:], strict=True):
            e5_cosine = cosine_similarity(profile_vector, job_vector)
            raw = build_candidate_raw_feature_map_from_e5_cosine(profile, candidate, e5_cosine)
            raw_vector = build_ordered_feature_vector(candidate.jobId, raw)
            output.append(
                CandidateFeatureVectors(
                    candidate_id=candidate.jobId,
                    raw=raw_vector,
                    normalized=self.feature_config.normalize(raw_vector),
                )
            )
        return tuple(output)


def assert_phase25_feature_order(feature_names: Sequence[str]) -> None:
    actual = tuple(feature_names)
    if actual != PHASE25_FEATURE_ORDER:
        raise FeatureBuildError(f"Feature order mismatch. expected={PHASE25_FEATURE_ORDER}, actual={actual}")


def build_ordered_feature_vector(candidate_id: str, raw_features: Mapping[str, float]) -> FeatureVector:
    """Build finite unnormalized vector from already computed feature values."""

    values: list[float] = []
    missing: list[str] = []
    non_finite: list[str] = []
    for feature_name in PHASE25_FEATURE_ORDER:
        if feature_name not in raw_features:
            missing.append(feature_name)
            continue
        value = float(raw_features[feature_name])
        if not math.isfinite(value):
            non_finite.append(feature_name)
        values.append(value)
    if missing or non_finite:
        problems = []
        if missing:
            problems.append(f"missing={missing}")
        if non_finite:
            problems.append(f"non_finite={non_finite}")
        raise FeatureBuildError("Invalid Phase 25 feature vector: " + ", ".join(problems))
    return FeatureVector(candidate_id=str(candidate_id), values=tuple(values))


def normalize_feature_vector(vector: FeatureVector, mean: Mapping[str, float], std: Mapping[str, float]) -> FeatureVector:
    """Apply train-split normalization from tensorflow_feature_config.json."""

    normalized: list[float] = []
    for feature_name, value in zip(PHASE25_FEATURE_ORDER, vector.values, strict=True):
        scale = float(std[feature_name])
        if scale == 0.0:
            raise FeatureBuildError(f"Normalization std is zero for {feature_name}")
        normalized_value = (float(value) - float(mean[feature_name])) / scale
        if not math.isfinite(normalized_value):
            raise FeatureBuildError(f"Normalized feature is non-finite for {feature_name}")
        normalized.append(normalized_value)
    return FeatureVector(candidate_id=vector.candidate_id, values=tuple(normalized))


def build_feature_vectors_for_request(
    request: Any,
    feature_config: TensorFlowFeatureConfig,
    embedding_backend: TextEmbeddingBackend | None = None,
    environment: str = "local",
) -> tuple[CandidateFeatureVectors, ...]:
    """Build normalized vectors from CV-analysis or candidate-reranking request objects."""

    profile = getattr(request, "profile", None) or getattr(request, "profileFeatures", None)
    candidates = getattr(request, "jobCandidates", None)
    if not isinstance(profile, SanitizedProfileInput) or candidates is None:
        raise FeatureBuildError("Request must expose SanitizedProfileInput and jobCandidates")
    builder = FeatureBuilder(
        feature_config=feature_config,
        embedding_backend=embedding_backend or SentenceTransformerE5Embedder(feature_config.embedding_model_name),
        environment=environment,
    )
    return builder.build_batch(profile, candidates)


def build_candidate_raw_feature_map(
    profile: SanitizedProfileInput,
    candidate: CandidateJobInput,
    embedding_backend: TextEmbeddingBackend,
    environment: str = "local",
    expected_model_name: str = E5_MODEL_NAME,
    profile_prefix: str = E5_PROFILE_PREFIX,
    job_prefix: str = E5_JOB_PREFIX,
) -> dict[str, float]:
    """Compute raw Phase 25-compatible features before normalization."""

    validate_e5_backend(embedding_backend, environment, expected_model_name=expected_model_name)
    scoring = candidate.model_scoring_input
    profile_text = prefixed_e5_text(profile_prefix, build_profile_e5_text(profile))
    job_text = prefixed_e5_text(job_prefix, build_job_e5_text(scoring))
    e5_cosine = cosine_similarity_from_backend(embedding_backend, profile_text, job_text)
    return build_candidate_raw_feature_map_from_e5_cosine(profile, candidate, e5_cosine)


def build_candidate_raw_feature_map_from_e5_cosine(
    profile: SanitizedProfileInput,
    candidate: CandidateJobInput,
    e5_cosine: float,
) -> dict[str, float]:
    """Compute raw Phase 25-compatible features with precomputed E5 cosine."""

    scoring = candidate.model_scoring_input
    profile_skills = normalized_skill_set(profile.normalizedSkills)
    job_skills = normalized_skill_set((*scoring.requiredSkills, *scoring.requirements))
    skill_overlap = jaccard(profile_skills, job_skills)
    requirement_coverage = explicit_finite_feature(scoring, "requirement_coverage", aliases=("requirementCoverage",))
    if requirement_coverage is None:
        requirement_coverage = coverage(profile_skills, job_skills)

    role_match = compute_role_match(profile, scoring)
    profile_years = profile.experienceYears if profile.experienceYears is not None else experience_years_from_value(profile.experienceBand)
    job_years = candidate_experience_years(scoring)
    if profile_years is None or job_years is None:
        experience_gap = 6.0
        experience_match = 0.0
    else:
        experience_gap = clamp(abs(float(profile_years) - float(job_years)), 0.0, 6.0)
        experience_match = max(0.0, 1.0 - experience_gap / 6.0)

    raw = {
        "e5_cosine": clamp(e5_cosine, -1.0, 1.0),
        "skill_overlap": clamp(skill_overlap, 0.0, 1.0),
        "requirement_coverage": clamp(requirement_coverage, 0.0, 1.0),
        "role_match": clamp(role_match, 0.0, 1.0),
        "experience_match": clamp(experience_match, 0.0, 1.0),
        "experience_gap_years_clipped": clamp(experience_gap, 0.0, 6.0),
    }
    validate_feature_map(raw)
    return raw


def validate_e5_backend(
    backend: TextEmbeddingBackend,
    environment: str = "local",
    expected_model_name: str = E5_MODEL_NAME,
) -> None:
    backend_name = str(getattr(backend, "backend_name", "")).strip().lower()
    model_name = str(getattr(backend, "model_name", "")).strip()
    if model_name != expected_model_name:
        raise FeatureBuildError(f"E5 backend must use artifact-declared model {expected_model_name}; actual={model_name!r}")
    if not backend_name:
        raise FeatureBuildError("E5 backend must expose backend_name")
    is_fallback = any(token in backend_name for token in FORBIDDEN_FALLBACK_BACKENDS)
    if is_fallback:
        raise FeatureBuildError(f"Fallback embedding backend is forbidden in {environment}: {backend_name}")


def prefixed_e5_text(prefix: str, text: str) -> str:
    stripped = " ".join(text.split())
    if not stripped:
        raise FeatureBuildError(f"Cannot build E5 text for {prefix} empty content")
    return f"{prefix} {stripped}"


def build_profile_e5_text(profile: SanitizedProfileInput) -> str:
    parts = [
        profile.cvText,
        profile.profileText,
        "target roles: " + ", ".join(profile.targetRoles) if profile.targetRoles else "",
        "role family: " + profile.roleFamily if profile.roleFamily else "",
        "experience band: " + profile.experienceBand if profile.experienceBand else "",
        "skills: " + ", ".join(profile.normalizedSkills) if profile.normalizedSkills else "",
        "sections: " + ", ".join(profile.detectedCvSectionNames) if profile.detectedCvSectionNames else "",
    ]
    return join_text_parts(parts)


def build_job_e5_text(scoring: CandidateScoringInput) -> str:
    parts = [
        scoring.titleText,
        scoring.descriptionText,
        scoring.requirementSummary,
        "required skills: " + ", ".join(scoring.requiredSkills) if scoring.requiredSkills else "",
        "requirements: " + ", ".join(scoring.requirements) if scoring.requirements else "",
        "role family: " + scoring.roleFamily if scoring.roleFamily else "",
        "experience level: " + scoring.experienceLevel if scoring.experienceLevel else "",
        "experience band: " + scoring.experienceBand if scoring.experienceBand else "",
        "work type: " + scoring.workType if scoring.workType else "",
    ]
    return join_text_parts(parts)


def join_text_parts(parts: Sequence[str]) -> str:
    text = " | ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
    if not text:
        raise FeatureBuildError("Missing text/signals for E5 feature construction")
    return text


def cosine_similarity_from_backend(backend: TextEmbeddingBackend, profile_text: str, job_text: str) -> float:
    try:
        encoded = backend.encode((profile_text, job_text))
    except FeatureBuildError:
        raise
    except Exception as exc:
        raise FeatureBuildError(f"E5 embedding failed: {exc}") from exc
    vectors = list(encoded)
    if len(vectors) != 2:
        raise FeatureBuildError(f"E5 backend must return 2 vectors; actual={len(vectors)}")
    return cosine_similarity(vectors[0], vectors[1])


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    if not left_values or len(left_values) != len(right_values):
        raise FeatureBuildError("E5 vectors must be non-empty with matching dimensions")
    if not all(math.isfinite(value) for value in (*left_values, *right_values)):
        raise FeatureBuildError("E5 vectors must contain finite values")
    dot = sum(a * b for a, b in zip(left_values, right_values, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        raise FeatureBuildError("E5 vectors must not be zero vectors")
    value = dot / (left_norm * right_norm)
    if not math.isfinite(value):
        raise FeatureBuildError("E5 cosine is non-finite")
    return clamp(value, -1.0, 1.0)


def normalized_skill_set(values: Sequence[str]) -> set[str]:
    output: set[str] = set()
    for raw in values:
        for token in re.split(r"[,|;/]+|\s\|\s", str(raw)):
            normalized = normalize_skill_token(token)
            if normalized:
                output.add(normalized)
    return output


def normalize_skill_token(token: Any) -> str | None:
    text = "" if token is None else str(token).lower().strip()
    text = text.replace("react.js", "react js").replace("node.js", "node js").replace("next.js", "nextjs")
    text = re.sub(r"[^a-z0-9+#./ -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -_/.,")
    if not text:
        return None
    return SKILL_ALIASES.get(text, text)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def coverage(profile_skills: set[str], job_skills: set[str]) -> float:
    return len(profile_skills & job_skills) / len(job_skills) if job_skills else 0.0


def compute_role_match(profile: SanitizedProfileInput, scoring: CandidateScoringInput) -> float:
    profile_roles = {normalize_role(value) for value in (*profile.targetRoles, profile.roleFamily or "") if value}
    candidate_roles = {normalize_role(value) for value in (scoring.roleFamily or "", scoring.titleText) if value}
    profile_roles.discard("")
    candidate_roles.discard("")
    if not profile_roles or not candidate_roles:
        return 0.0
    return 1.0 if profile_roles & candidate_roles else 0.0


def normalize_role(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    aliases = {
        "front end": "frontend",
        "frontend": "frontend",
        "mobile": "mobile",
        "android": "mobile",
        "ios": "mobile",
        "back end": "backend",
        "backend": "backend",
        "api": "backend",
        "server": "backend",
        "full stack": "fullstack",
        "fullstack": "fullstack",
        "data": "data",
        "machine learning": "data",
        "ai": "data",
        "cloud": "cloud",
        "devops": "cloud",
        "security": "security",
        "qa": "quality_assurance",
        "quality": "quality_assurance",
        "web": "web",
        "developer": "software_engineering",
        "engineer": "software_engineering",
    }
    for marker, role in aliases.items():
        if marker in text:
            return role
    return text


def candidate_experience_years(scoring: CandidateScoringInput) -> float | None:
    for key in EXPERIENCE_NUMERIC_KEYS:
        if key in scoring.numericFeatures:
            value = float(scoring.numericFeatures[key])
            if math.isfinite(value):
                return value
    for value in (scoring.experienceLevel, scoring.experienceBand):
        years = experience_years_from_value(value)
        if years is not None:
            return years
    return None


def experience_years_from_value(value: str | None) -> float | None:
    if value is None:
        return None
    text = re.sub(r"[_\s]+", " ", str(value).strip().lower())
    text = text.replace("+", "+")
    if text in EXPERIENCE_YEARS_BY_VALUE:
        return EXPERIENCE_YEARS_BY_VALUE[text]
    underscored = text.replace(" ", "_")
    if underscored in EXPERIENCE_YEARS_BY_VALUE:
        return EXPERIENCE_YEARS_BY_VALUE[underscored]
    return None


def explicit_finite_feature(scoring: CandidateScoringInput, key: str, aliases: Sequence[str] = ()) -> float | None:
    values = [scoring.numericFeatures.get(key), *(scoring.numericFeatures.get(alias) for alias in aliases)]
    if key == "requirement_coverage":
        values.append(scoring.requirementCoverage)
    for value in values:
        if value is None:
            continue
        converted = float(value)
        if not math.isfinite(converted):
            raise FeatureBuildError(f"Explicit feature {key} must be finite")
        return converted
    return None


def validate_feature_map(raw: Mapping[str, float]) -> None:
    for feature_name in PHASE25_FEATURE_ORDER:
        value = raw.get(feature_name)
        if value is None or not math.isfinite(float(value)):
            raise FeatureBuildError(f"Feature {feature_name} must be finite")


def _feature_stats(raw: Mapping[str, Any], path: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for feature_name in PHASE25_FEATURE_ORDER:
        if feature_name not in raw:
            raise FeatureBuildError(f"{path}.{feature_name} is required")
        value = float(raw[feature_name])
        if not math.isfinite(value):
            raise FeatureBuildError(f"{path}.{feature_name} must be finite")
        output[feature_name] = value
    return output


def clamp(value: float, lower: float, upper: float) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise FeatureBuildError("Cannot clamp non-finite feature value")
    return min(upper, max(lower, converted))
