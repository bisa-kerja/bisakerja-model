"""Backend-aligned strict request/response schema names for Phase 26.

OpenAPI ``cv-analysis-v2`` is backend/frontend wrapper output. Model API returns
model-core payloads only; Backend API hydrates DB-owned job fields, persistence,
auth, OpenRouter prose, and final response formatting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Literal

from .errors import ContractValidationError

Language = Literal["id", "en"]
DbAnalysisLanguage = Literal["ID", "EN"]
InputMode = Literal["UPLOAD", "REFERENCE"]
CompareSource = Literal["BOOKMARK", "JOB_SEARCH", "DIRECT_JOB_DETAIL"]
MatchLevel = Literal["strong", "good", "stretch"]

ALLOWED_LANGUAGES: frozenset[str] = frozenset({"id", "en"})
DB_LANGUAGE_TO_API_LANGUAGE: Mapping[str, str] = {"ID": "id", "EN": "en"}
API_LANGUAGE_TO_DB_LANGUAGE: Mapping[str, str] = {"id": "ID", "en": "EN"}
ALLOWED_INPUT_MODES: frozenset[str] = frozenset({"UPLOAD", "REFERENCE"})
ALLOWED_COMPARE_SOURCES: frozenset[str] = frozenset({"BOOKMARK", "JOB_SEARCH", "DIRECT_JOB_DETAIL"})
ALLOWED_MATCH_LEVELS: frozenset[str] = frozenset({"strong", "good", "stretch"})

BACKEND_CV_ANALYSIS_SCHEMA_VERSION = "cv-analysis-v2"
MODEL_CORE_CV_ANALYSIS_SCHEMA_VERSION = "model-core-cv-analysis-v1"
MODEL_CORE_CANDIDATE_RERANKING_SCHEMA_VERSION = "model-core-candidate-reranking-v1"
MODEL_CORE_CANDIDATE_RERANKING_REQUEST_VERSION = "model-core-candidate-reranking-request-v1"
MODEL_CORE_CV_ANALYZER_INPUT_VERSION = "cv-analyzer-v1"

MAX_REQUEST_ID_LENGTH = 200
MAX_CANDIDATES = 50
MAX_RECOMMENDATIONS = 10
MAX_TEXT_CHARS = 20_000
MAX_SHORT_TEXT_CHARS = 500
MAX_LIST_ITEMS = 200
MAX_JOB_ROLES = 10
MAX_JOB_ROLE_CHARS = 120
MAX_REQUIREMENT_VALUE_CHARS = 500
_REQUIREMENT_OBJECT_FIELDS: frozenset[str] = frozenset({"type", "value", "priority"})
APPROVED_NUMERIC_SIGNAL_KEYS: frozenset[str] = frozenset(
    {
        "e5_cosine",
        "skill_overlap",
        "requirement_coverage",
        "role_match",
        "experience_match",
        "experience_gap_years_clipped",
        "requirementCoverage",
        "semanticSimilarity",
        "experience_years",
        "experienceYears",
        "required_experience_years",
        "requiredExperienceYears",
        "job_experience_years",
        "jobExperienceYears",
    }
)

CV_ANALYSIS_MODEL_CORE_REQUEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "requestId",
    "inputVersion",
    "language",
    "inputMode",
    "compareSource",
    "profile",
    "jobCandidates",
)
CANDIDATE_RERANKING_CORE_REQUEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "requestId",
    "schemaVersion",
    "candidateSetId",
    "language",
    "profileFeatures",
    "jobCandidates",
)

_PROFILE_FIELDS: frozenset[str] = frozenset(
    {
        "profileId",
        "cvFileId",
        "cvText",
        "profileText",
        "targetRoles",
        "normalizedSkills",
        "roleFamily",
        "experienceYears",
        "experienceBand",
        "embeddingTextHash",
        "detectedCvSectionNames",
    }
)
_SCORING_FIELDS: frozenset[str] = frozenset(
    {
        "titleText",
        "descriptionText",
        "requirementSummary",
        "requiredSkills",
        "requirements",
        "roleFamily",
        "experienceLevel",
        "workType",
        "experienceBand",
        "semanticSimilarity",
        "requirementCoverage",
        "numericFeatures",
        "numericSignals",
    }
)
_BACKEND_METADATA_FIELDS: frozenset[str] = frozenset(
    {
        "title",
        "companyName",
        "location",
        "locationDisplay",
        "workType",
        "experienceLevel",
        "postedAt",
        "sourceUpdatedAt",
        "source",
    }
)
_CANDIDATE_FIELDS: frozenset[str] = frozenset(
    {
        "jobId",
        "titleText",
        "descriptionText",
        "requirementSummary",
        "requiredSkills",
        "requirements",
        "roleFamily",
        "experienceLevel",
        "workType",
        "experienceBand",
        "semanticSimilarity",
        "requirementCoverage",
        "numericFeatures",
        "scoringInput",
        "backendMetadata",
    }
)
_RANKING_POLICY_FIELDS: frozenset[str] = frozenset(
    {"maxRecommendations", "requireCandidateJobIds", "deduplicateByJobId", "backendOwnsHydration"}
)


@dataclass(frozen=True)
class CandidateScoringInput:
    """Model-owned candidate evidence. Backend hydration metadata stays outside."""

    titleText: str = ""
    descriptionText: str = ""
    requirementSummary: str = ""
    requiredSkills: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    roleFamily: str | None = None
    experienceLevel: str | None = None
    workType: str | None = None
    experienceBand: str | None = None
    semanticSimilarity: float | None = None
    requirementCoverage: float | None = None
    numericFeatures: Mapping[str, float] = field(default_factory=dict)

    def has_scoring_evidence(self) -> bool:
        return any(
            (
                bool(self.titleText.strip()),
                bool(self.descriptionText.strip()),
                bool(self.requirementSummary.strip()),
                bool(self.requiredSkills),
                bool(self.requirements),
                self.roleFamily is not None,
                self.experienceLevel is not None,
                self.experienceBand is not None,
                self.semanticSimilarity is not None,
                self.requirementCoverage is not None,
                bool(self.numericFeatures),
            )
        )


@dataclass(frozen=True)
class CandidateBackendMetadata:
    """Backend-owned fields allowed in requests for trace/hydration only."""

    title: str | None = None
    companyName: str | None = None
    location: Mapping[str, str | None] = field(default_factory=dict)
    locationDisplay: str | None = None
    workType: str | None = None
    experienceLevel: str | None = None
    postedAt: str | None = None
    sourceUpdatedAt: str | None = None
    source: CompareSource | None = None


@dataclass(frozen=True)
class RankingPolicy:
    """Backend-supplied request limits. Public CV response cap remains five."""

    maxRecommendations: int = MAX_RECOMMENDATIONS
    requireCandidateJobIds: bool = True
    deduplicateByJobId: bool = True
    backendOwnsHydration: bool = True


@dataclass(frozen=True)
class CandidateJobInput:
    """Backend-provided candidate with scoring inputs separated from metadata."""

    jobId: str
    titleText: str = ""
    descriptionText: str = ""
    requirementSummary: str = ""
    requiredSkills: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    roleFamily: str | None = None
    experienceLevel: str | None = None
    workType: str | None = None
    numericFeatures: Mapping[str, float] = field(default_factory=dict)
    experienceBand: str | None = None
    semanticSimilarity: float | None = None
    requirementCoverage: float | None = None
    scoringInput: CandidateScoringInput | None = None
    backendMetadata: CandidateBackendMetadata | None = None

    @property
    def model_scoring_input(self) -> CandidateScoringInput:
        if self.scoringInput is not None:
            return self.scoringInput
        numeric_features = dict(self.numericFeatures)
        if self.semanticSimilarity is not None:
            numeric_features.setdefault("semanticSimilarity", self.semanticSimilarity)
        if self.requirementCoverage is not None:
            numeric_features.setdefault("requirementCoverage", self.requirementCoverage)
        return CandidateScoringInput(
            titleText=self.titleText,
            descriptionText=self.descriptionText,
            requirementSummary=self.requirementSummary,
            requiredSkills=self.requiredSkills,
            requirements=self.requirements,
            roleFamily=self.roleFamily,
            experienceLevel=self.experienceLevel,
            workType=self.workType,
            experienceBand=self.experienceBand,
            semanticSimilarity=self.semanticSimilarity,
            requirementCoverage=self.requirementCoverage,
            numericFeatures=numeric_features,
        )


@dataclass(frozen=True)
class SanitizedProfileInput:
    """Backend-prepared CV/profile signals. No raw secrets or unrelated PII."""

    profileId: str | None = None
    cvFileId: str | None = None
    cvText: str = ""
    profileText: str = ""
    targetRoles: tuple[str, ...] = ()
    normalizedSkills: tuple[str, ...] = ()
    roleFamily: str | None = None
    experienceYears: float | None = None
    experienceBand: str | None = None
    embeddingTextHash: str | None = None
    detectedCvSectionNames: tuple[str, ...] = ()

    def has_profile_evidence(self) -> bool:
        return any(
            (
                bool(self.cvText.strip()),
                bool(self.profileText.strip()),
                bool(self.targetRoles),
                bool(self.normalizedSkills),
                self.roleFamily is not None,
                self.experienceYears is not None,
                self.experienceBand is not None,
                self.embeddingTextHash is not None,
                bool(self.detectedCvSectionNames),
            )
        )


@dataclass(frozen=True)
class CvAnalysisModelCoreRequest:
    """Internal Model API request assembled by Backend API from OpenAPI/DB data."""

    requestId: str
    inputVersion: str
    language: Language
    inputMode: InputMode
    compareSource: CompareSource
    profile: SanitizedProfileInput
    jobCandidates: tuple[CandidateJobInput, ...]
    maxRecommendations: int = MAX_RECOMMENDATIONS
    rankingPolicy: RankingPolicy = field(default_factory=RankingPolicy)


@dataclass(frozen=True)
class CandidateRerankingCoreRequest:
    requestId: str
    schemaVersion: str
    candidateSetId: str
    language: Language
    profileFeatures: SanitizedProfileInput
    jobCandidates: tuple[CandidateJobInput, ...]
    maxRecommendations: int = MAX_RECOMMENDATIONS
    rankingPolicy: RankingPolicy = field(default_factory=RankingPolicy)


@dataclass(frozen=True)
class ModelArtifactIdentity:
    format: str = ".keras"
    path: str = "artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/export/selected_jobfit_tf_phase46_multilingual_e5_small.keras"
    sha256: str | None = None


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    version: str
    artifact: ModelArtifactIdentity | None = None
    artifact_phase: str | None = None
    embedding_model: str | None = None

    @property
    def artifact_sha256(self) -> str | None:
        return None if self.artifact is None else self.artifact.sha256


@dataclass(frozen=True)
class ScoreSignal:
    key: str
    label: str | None = None
    value: float | None = None


@dataclass(frozen=True)
class JobFitAlignmentCore:
    score: int
    matchedSkills: tuple[str, ...] = ()
    missingSkills: tuple[str, ...] = ()
    summarySignals: tuple[ScoreSignal, ...] = ()
    confidenceNotes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AtsFriendlinessCore:
    score: int
    detectedIssues: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    fallback: bool = False


@dataclass(frozen=True)
class OverallImpressionCore:
    score: int
    summary: str
    evidenceKeys: tuple[str, ...] = ()
    confidenceNotes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecommendationScore:
    jobId: str
    matchScore: int
    matchLevel: MatchLevel = "stretch"
    rankingSignals: tuple[ScoreSignal, ...] = ()
    matchedSkills: tuple[str, ...] = ()
    missingSkills: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateRerankingCoreOutput:
    requestId: str
    schemaVersion: str
    candidateSetId: str
    language: Language
    recommendations: tuple[RecommendationScore, ...]
    model: ModelIdentity
    rankedAt: str


@dataclass(frozen=True)
class CvAnalysisCoreOutput:
    """Model-owned CV output. Wrapper/backend-owned fields stay excluded."""

    requestId: str
    schemaVersion: str
    language: Language
    jobFitAlignment: JobFitAlignmentCore
    atsFriendliness: AtsFriendlinessCore
    overallImpression: OverallImpressionCore
    candidateReranking: CandidateRerankingCoreOutput | None
    model: ModelIdentity
    analyzedAt: str


@dataclass(frozen=True)
class BackendCvAnalysisWrapperExpectation:
    """OpenAPI final response fields Backend API must create after wrapping."""

    schemaVersion: str = BACKEND_CV_ANALYSIS_SCHEMA_VERSION
    requiredAnalysisResultFields: tuple[str, ...] = (
        "id",
        "schemaVersion",
        "jobFitAlignment",
        "atsFriendliness",
        "overallImpression",
        "topActionables",
        "sectionReviews",
        "jobRecommendations",
        "generatedCv",
        "model",
        "analyzedAt",
    )
    wrapperOwnedFields: tuple[str, ...] = (
        "id",
        "summary",
        "overallImpression",
        "topActionables",
        "sectionReviews",
        "jobRecommendations[].title",
        "jobRecommendations[].companyName",
        "jobRecommendations[].reason",
        "jobRecommendations[].nextStep",
        "generatedCv",
        "cvFile",
        "persistence",
        "auth",
    )


@dataclass(frozen=True)
class OpenRouterGenAiWrapperRequest:
    """Secondary wrapper prose request; disabled for core inference by default."""

    requestId: str
    language: Language
    modelCore: CvAnalysisCoreOutput
    sanitizedBackendContext: Mapping[str, Any] = field(default_factory=dict)
    model: str = "~openai/gpt-latest"


def _reject_unknown_fields(payload: Mapping[str, Any], allowed: frozenset[str], path: str, errors: list[str]) -> None:
    for key in payload:
        if key not in allowed:
            errors.append(f"{path}.{key} is not allowed")


def _require_mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be object")
        return {}
    return value


def _optional_string(value: Any, path: str, errors: list[str], max_length: int = MAX_SHORT_TEXT_CHARS) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        errors.append(f"{path} must be string or null")
        return None
    if len(value) > max_length:
        errors.append(f"{path} max length {max_length}; actual={len(value)}")
    return value


def _required_string(value: Any, path: str, errors: list[str], max_length: int = MAX_REQUEST_ID_LENGTH) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path} must be non-empty string")
        return ""
    if len(value) > max_length:
        errors.append(f"{path} max length {max_length}; actual={len(value)}")
    return value


def _string_tuple(
    value: Any,
    path: str,
    errors: list[str],
    max_items: int = MAX_LIST_ITEMS,
    max_item_chars: int = MAX_SHORT_TEXT_CHARS,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        errors.append(f"{path} must be array of strings")
        return ()
    if len(value) > max_items:
        errors.append(f"{path} max items {max_items}; actual={len(value)}")
    output: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{path}[{index}] must be non-empty string")
            continue
        if len(item) > max_item_chars:
            errors.append(f"{path}[{index}] max length {max_item_chars}; actual={len(item)}")
        output.append(item)
    return tuple(output)


def _finite_float(value: Any, path: str, errors: list[str]) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"{path} must be finite number")
        return None
    converted = float(value)
    if not math.isfinite(converted):
        errors.append(f"{path} must be finite number")
        return None
    return converted


def _requirements_tuple(value: Any, path: str, errors: list[str]) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        errors.append(f"{path} must be array of requirement strings or objects")
        return ()
    if len(value) > MAX_LIST_ITEMS:
        errors.append(f"{path} max items {MAX_LIST_ITEMS}; actual={len(value)}")
    output: list[str] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, Mapping):
            _reject_unknown_fields(item, _REQUIREMENT_OBJECT_FIELDS, item_path, errors)
            req_type = item.get("type")
            priority = item.get("priority")
            if not isinstance(req_type, str) or not req_type.strip():
                errors.append(f"{item_path}.type must be non-empty string")
            if not isinstance(priority, str) or not priority.strip():
                errors.append(f"{item_path}.priority must be non-empty string")
            value_text = item.get("value")
            if not isinstance(value_text, str):
                errors.append(f"{item_path}.value must be non-empty string")
                continue
            text = value_text.strip()
        else:
            errors.append(f"{item_path} must be string or requirement object")
            continue
        if not text:
            errors.append(f"{item_path}.value must be non-empty string" if isinstance(item, Mapping) else f"{item_path} must be non-empty string")
            continue
        if len(text) > MAX_REQUIREMENT_VALUE_CHARS:
            errors.append(f"{item_path} max length {MAX_REQUIREMENT_VALUE_CHARS}; actual={len(text)}")
        output.append(text)
    return tuple(output)


def _numeric_mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, float]:
    if value is None:
        return {}
    payload = _require_mapping(value, path, errors)
    output: dict[str, float] = {}
    for key, item in payload.items():
        if not isinstance(key, str) or not key:
            errors.append(f"{path} keys must be non-empty strings")
            continue
        if key not in APPROVED_NUMERIC_SIGNAL_KEYS:
            errors.append(f"{path}.{key} is not an approved Phase 25 numeric signal")
            continue
        converted = _finite_float(item, f"{path}.{key}", errors)
        if converted is not None:
            output[key] = converted
    return output


def _enum(value: Any, allowed: frozenset[str], path: str, errors: list[str]) -> str:
    if value not in allowed:
        errors.append(f"{path} must be one of {sorted(allowed)}")
        return ""
    return str(value)


def _request_limit(value: Any, path: str, errors: list[str]) -> int:
    if value is None:
        return MAX_RECOMMENDATIONS
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{path} must be integer 0-{MAX_RECOMMENDATIONS}")
        return MAX_RECOMMENDATIONS
    if value < 0 or value > MAX_RECOMMENDATIONS:
        errors.append(f"{path} must be integer 0-{MAX_RECOMMENDATIONS}; actual={value}")
    return value


def _bool_value(value: Any, path: str, errors: list[str], default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        errors.append(f"{path} must be boolean")
        return default
    return value


def _parse_profile(payload: Any, path: str, errors: list[str]) -> SanitizedProfileInput:
    data = _require_mapping(payload, path, errors)
    _reject_unknown_fields(data, _PROFILE_FIELDS, path, errors)
    profile = SanitizedProfileInput(
        profileId=_optional_string(data.get("profileId"), f"{path}.profileId", errors),
        cvFileId=_optional_string(data.get("cvFileId"), f"{path}.cvFileId", errors),
        cvText=_optional_string(data.get("cvText", ""), f"{path}.cvText", errors, MAX_TEXT_CHARS) or "",
        profileText=_optional_string(data.get("profileText", ""), f"{path}.profileText", errors, MAX_TEXT_CHARS) or "",
        targetRoles=_string_tuple(
            data.get("targetRoles"), f"{path}.targetRoles", errors, max_items=MAX_JOB_ROLES, max_item_chars=MAX_JOB_ROLE_CHARS
        ),
        normalizedSkills=_string_tuple(data.get("normalizedSkills"), f"{path}.normalizedSkills", errors),
        roleFamily=_optional_string(data.get("roleFamily"), f"{path}.roleFamily", errors),
        experienceYears=_finite_float(data.get("experienceYears"), f"{path}.experienceYears", errors),
        experienceBand=_optional_string(data.get("experienceBand"), f"{path}.experienceBand", errors),
        embeddingTextHash=_optional_string(data.get("embeddingTextHash"), f"{path}.embeddingTextHash", errors),
        detectedCvSectionNames=_string_tuple(
            data.get("detectedCvSectionNames"), f"{path}.detectedCvSectionNames", errors, max_items=50
        ),
    )
    if not profile.has_profile_evidence():
        errors.append(f"{path} must include sanitized cv/profile text or extracted signals")
    return profile


def _parse_scoring_input(payload: Any, path: str, errors: list[str]) -> CandidateScoringInput:
    data = _require_mapping(payload, path, errors)
    _reject_unknown_fields(data, _SCORING_FIELDS, path, errors)
    if "numericSignals" in data and "numericFeatures" in data:
        errors.append(f"{path} must provide numericSignals or numericFeatures, not both")
    scoring = CandidateScoringInput(
        titleText=_optional_string(data.get("titleText", ""), f"{path}.titleText", errors, MAX_SHORT_TEXT_CHARS) or "",
        descriptionText=_optional_string(data.get("descriptionText", ""), f"{path}.descriptionText", errors, MAX_TEXT_CHARS) or "",
        requirementSummary=_optional_string(
            data.get("requirementSummary", ""), f"{path}.requirementSummary", errors, MAX_TEXT_CHARS
        )
        or "",
        requiredSkills=_string_tuple(data.get("requiredSkills"), f"{path}.requiredSkills", errors),
        requirements=_requirements_tuple(data.get("requirements"), f"{path}.requirements", errors),
        roleFamily=_optional_string(data.get("roleFamily"), f"{path}.roleFamily", errors),
        experienceLevel=_optional_string(data.get("experienceLevel"), f"{path}.experienceLevel", errors),
        workType=_optional_string(data.get("workType"), f"{path}.workType", errors),
        experienceBand=_optional_string(data.get("experienceBand"), f"{path}.experienceBand", errors),
        semanticSimilarity=_finite_float(data.get("semanticSimilarity"), f"{path}.semanticSimilarity", errors),
        requirementCoverage=_finite_float(data.get("requirementCoverage"), f"{path}.requirementCoverage", errors),
        numericFeatures=_numeric_mapping(
            data.get("numericSignals", data.get("numericFeatures")),
            f"{path}.numericSignals" if "numericSignals" in data else f"{path}.numericFeatures",
            errors,
        ),
    )
    if not scoring.has_scoring_evidence():
        errors.append(f"{path} must include model-owned candidate scoring inputs")
    return scoring


def _parse_backend_metadata(payload: Any, path: str, errors: list[str]) -> CandidateBackendMetadata:
    if payload is None:
        return CandidateBackendMetadata()
    data = _require_mapping(payload, path, errors)
    _reject_unknown_fields(data, _BACKEND_METADATA_FIELDS, path, errors)
    location = data.get("location")
    if location is None:
        location_map: Mapping[str, str | None] = {}
    else:
        location_payload = _require_mapping(location, f"{path}.location", errors)
        location_map = {
            str(key): _optional_string(value, f"{path}.location.{key}", errors)
            for key, value in location_payload.items()
        }
    source: CompareSource | None = None
    if data.get("source") is not None:
        parsed_source = _enum(data.get("source"), ALLOWED_COMPARE_SOURCES, f"{path}.source", errors)
        source = parsed_source if parsed_source in ALLOWED_COMPARE_SOURCES else None  # type: ignore[assignment]
    return CandidateBackendMetadata(
        title=_optional_string(data.get("title"), f"{path}.title", errors),
        companyName=_optional_string(data.get("companyName"), f"{path}.companyName", errors),
        location=location_map,
        workType=_optional_string(data.get("workType"), f"{path}.workType", errors),
        experienceLevel=_optional_string(data.get("experienceLevel"), f"{path}.experienceLevel", errors),
        postedAt=_optional_string(data.get("postedAt"), f"{path}.postedAt", errors),
        locationDisplay=_optional_string(data.get("locationDisplay"), f"{path}.locationDisplay", errors),
        sourceUpdatedAt=_optional_string(data.get("sourceUpdatedAt"), f"{path}.sourceUpdatedAt", errors),
        source=source,
    )


def _parse_candidate(payload: Any, path: str, errors: list[str]) -> CandidateJobInput:
    data = _require_mapping(payload, path, errors)
    _reject_unknown_fields(data, _CANDIDATE_FIELDS, path, errors)
    job_id = _required_string(data.get("jobId"), f"{path}.jobId", errors)
    top_level_scoring = {field_name: data[field_name] for field_name in _SCORING_FIELDS if field_name in data}
    scoring_input = None
    if data.get("scoringInput") is not None:
        scoring_input = _parse_scoring_input(data.get("scoringInput"), f"{path}.scoringInput", errors)
    elif top_level_scoring:
        scoring_input = _parse_scoring_input(top_level_scoring, path, errors)
    else:
        errors.append(f"{path}.scoringInput is required")
    return CandidateJobInput(
        jobId=job_id,
        titleText=str(top_level_scoring.get("titleText", "")),
        descriptionText=str(top_level_scoring.get("descriptionText", "")),
        requirementSummary=str(top_level_scoring.get("requirementSummary", "")),
        requiredSkills=_string_tuple(top_level_scoring.get("requiredSkills"), f"{path}.requiredSkills", errors),
        requirements=_requirements_tuple(top_level_scoring.get("requirements"), f"{path}.requirements", errors),
        roleFamily=top_level_scoring.get("roleFamily") if isinstance(top_level_scoring.get("roleFamily"), str) else None,
        experienceLevel=(
            top_level_scoring.get("experienceLevel") if isinstance(top_level_scoring.get("experienceLevel"), str) else None
        ),
        workType=top_level_scoring.get("workType") if isinstance(top_level_scoring.get("workType"), str) else None,
        numericFeatures=_numeric_mapping(
            top_level_scoring.get("numericSignals", top_level_scoring.get("numericFeatures")),
            f"{path}.numericSignals" if "numericSignals" in top_level_scoring else f"{path}.numericFeatures",
            errors,
        ),
        experienceBand=(
            top_level_scoring.get("experienceBand") if isinstance(top_level_scoring.get("experienceBand"), str) else None
        ),
        semanticSimilarity=_finite_float(top_level_scoring.get("semanticSimilarity"), f"{path}.semanticSimilarity", errors),
        requirementCoverage=_finite_float(top_level_scoring.get("requirementCoverage"), f"{path}.requirementCoverage", errors),
        scoringInput=scoring_input,
        backendMetadata=_parse_backend_metadata(data.get("backendMetadata"), f"{path}.backendMetadata", errors),
    )


def _parse_candidates(payload: Any, path: str, errors: list[str]) -> tuple[CandidateJobInput, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        errors.append(f"{path} must be array")
        return ()
    if not payload:
        errors.append(f"{path} must contain at least 1 candidate")
    if len(payload) > MAX_CANDIDATES:
        errors.append(f"{path} max items {MAX_CANDIDATES}; actual={len(payload)}")
    candidates = tuple(_parse_candidate(item, f"{path}[{index}]", errors) for index, item in enumerate(payload))
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not candidate.jobId:
            continue
        if candidate.jobId in seen:
            errors.append(f"{path}[{index}].jobId duplicate: {candidate.jobId!r}")
        seen.add(candidate.jobId)
    return candidates


def _parse_ranking_policy(payload: Any, path: str, errors: list[str]) -> RankingPolicy:
    if payload is None:
        return RankingPolicy()
    data = _require_mapping(payload, path, errors)
    _reject_unknown_fields(data, _RANKING_POLICY_FIELDS, path, errors)
    if data.get("requireCandidateJobIds") is False:
        errors.append(f"{path}.requireCandidateJobIds must be true")
    if data.get("deduplicateByJobId") is False:
        errors.append(f"{path}.deduplicateByJobId must be true")
    if data.get("backendOwnsHydration") is False:
        errors.append(f"{path}.backendOwnsHydration must be true")
    return RankingPolicy(
        maxRecommendations=_request_limit(data.get("maxRecommendations"), f"{path}.maxRecommendations", errors),
        requireCandidateJobIds=_bool_value(data.get("requireCandidateJobIds"), f"{path}.requireCandidateJobIds", errors, True),
        deduplicateByJobId=_bool_value(data.get("deduplicateByJobId"), f"{path}.deduplicateByJobId", errors, True),
        backendOwnsHydration=_bool_value(data.get("backendOwnsHydration"), f"{path}.backendOwnsHydration", errors, True),
    )


def _root_mapping(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ContractValidationError(["$ must be object"])
    return payload


def parse_cv_analysis_model_core_request(payload: Any) -> CvAnalysisModelCoreRequest:
    """Parse and validate strict CV/profile scoring request payload."""

    data = _root_mapping(payload)
    errors: list[str] = []
    allowed = frozenset((*CV_ANALYSIS_MODEL_CORE_REQUEST_REQUIRED_FIELDS, "maxRecommendations", "rankingPolicy"))
    _reject_unknown_fields(data, allowed, "$", errors)
    for field_name in CV_ANALYSIS_MODEL_CORE_REQUEST_REQUIRED_FIELDS:
        if field_name not in data:
            errors.append(f"$.{field_name} is required")
    language = _enum(data.get("language"), ALLOWED_LANGUAGES, "$.language", errors)
    input_mode = _enum(data.get("inputMode"), ALLOWED_INPUT_MODES, "$.inputMode", errors)
    compare_source = _enum(data.get("compareSource"), ALLOWED_COMPARE_SOURCES, "$.compareSource", errors)
    input_version = _required_string(data.get("inputVersion"), "$.inputVersion", errors)
    if input_version and input_version != MODEL_CORE_CV_ANALYZER_INPUT_VERSION:
        errors.append(f"$.inputVersion must be {MODEL_CORE_CV_ANALYZER_INPUT_VERSION!r}")
    ranking_policy = _parse_ranking_policy(data.get("rankingPolicy"), "$.rankingPolicy", errors)
    max_recommendations = _request_limit(data.get("maxRecommendations", ranking_policy.maxRecommendations), "$.maxRecommendations", errors)
    request = CvAnalysisModelCoreRequest(
        requestId=_required_string(data.get("requestId"), "$.requestId", errors),
        inputVersion=input_version,
        language=language,  # type: ignore[arg-type]
        inputMode=input_mode,  # type: ignore[arg-type]
        compareSource=compare_source,  # type: ignore[arg-type]
        profile=_parse_profile(data.get("profile"), "$.profile", errors),
        jobCandidates=_parse_candidates(data.get("jobCandidates"), "$.jobCandidates", errors),
        maxRecommendations=max_recommendations,
        rankingPolicy=ranking_policy,
    )
    if errors:
        raise ContractValidationError(errors)
    return request


def parse_candidate_reranking_core_request(payload: Any) -> CandidateRerankingCoreRequest:
    """Parse and validate strict backend candidate reranking request payload."""

    data = _root_mapping(payload)
    errors: list[str] = []
    allowed = frozenset((*CANDIDATE_RERANKING_CORE_REQUEST_REQUIRED_FIELDS, "maxRecommendations", "rankingPolicy"))
    _reject_unknown_fields(data, allowed, "$", errors)
    for field_name in CANDIDATE_RERANKING_CORE_REQUEST_REQUIRED_FIELDS:
        if field_name not in data:
            errors.append(f"$.{field_name} is required")
    language = _enum(data.get("language"), ALLOWED_LANGUAGES, "$.language", errors)
    schema_version = _required_string(data.get("schemaVersion"), "$.schemaVersion", errors)
    if schema_version and schema_version != MODEL_CORE_CANDIDATE_RERANKING_REQUEST_VERSION:
        errors.append(f"$.schemaVersion must be {MODEL_CORE_CANDIDATE_RERANKING_REQUEST_VERSION!r}")
    ranking_policy = _parse_ranking_policy(data.get("rankingPolicy"), "$.rankingPolicy", errors)
    max_recommendations = _request_limit(data.get("maxRecommendations", ranking_policy.maxRecommendations), "$.maxRecommendations", errors)
    request = CandidateRerankingCoreRequest(
        requestId=_required_string(data.get("requestId"), "$.requestId", errors),
        schemaVersion=schema_version,
        candidateSetId=_required_string(data.get("candidateSetId"), "$.candidateSetId", errors),
        language=language,  # type: ignore[arg-type]
        profileFeatures=_parse_profile(data.get("profileFeatures"), "$.profileFeatures", errors),
        jobCandidates=_parse_candidates(data.get("jobCandidates"), "$.jobCandidates", errors),
        maxRecommendations=max_recommendations,
        rankingPolicy=ranking_policy,
    )
    if errors:
        raise ContractValidationError(errors)
    return request


# Backward-compatible aliases from initial 26.1 layout.
CandidateSignals = CandidateJobInput
CvAnalysisInferenceRequest = CvAnalysisModelCoreRequest
CvAnalysisCoreResponse = CvAnalysisCoreOutput
MODEL_CORE_SCHEMA_VERSION = MODEL_CORE_CV_ANALYSIS_SCHEMA_VERSION
