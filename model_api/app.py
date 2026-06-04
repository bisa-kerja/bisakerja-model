"""FastAPI application factory for Model API serving.

The factory keeps FastAPI optional until serving dependencies are pinned in the
Phase 26 packaging step. Importing this module must remain lightweight for tests.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
from dataclasses import replace
import json
import logging
import re
from secrets import compare_digest
from threading import BoundedSemaphore
from time import perf_counter
from typing import Any

from .artifacts import ArtifactVerificationReport, load_json, verify_runtime_artifacts
from .config import RuntimeConfig
from .errors import (
    ArtifactError,
    ContractValidationError,
    FeatureBuildError,
    InferenceTimeoutError,
    ModelApiError,
    ModelLoadError,
    ModelNotReadyError,
    UnsupportedArtifactVersionError,
)
from .features import (
    PHASE25_FEATURE_ORDER,
    SKILL_ALIASES,
    EmbeddingModelMetadata,
    SentenceTransformerE5Embedder,
    TensorFlowFeatureConfig,
    TextEmbeddingBackend,
    build_feature_vectors_for_request,
    normalize_role,
    normalized_skill_set,
    validate_e5_backend,
)
from .observability import build_safe_observability_event
from .pdf_parser import ats_score_from_pdf_evidence, normalized_skills_from_text, parse_pdf_bytes
from .inference import InferenceService, RuntimeState, ScoreCalibrationPolicy, match_level_for_score, utc_now_iso
from .schemas import (
    MAX_JOB_ROLE_CHARS,
    MAX_JOB_ROLES,
    MAX_RECOMMENDATIONS,
    MODEL_CORE_CANDIDATE_RERANKING_SCHEMA_VERSION,
    MODEL_CORE_CV_ANALYSIS_SCHEMA_VERSION,
    AtsFriendlinessCore,
    CandidateRerankingCoreRequest,
    CvAnalysisModelCoreRequest,
    JobFitAlignmentCore,
    OverallImpressionCore,
    SanitizedProfileInput,
    ScoreSignal,
    parse_candidate_reranking_core_request,
    parse_cv_analysis_model_core_request,
)
from .schemas import MODEL_CORE_CV_ANALYZER_INPUT_VERSION, ModelIdentity
from .validators import validate_model_core_payload


LOGGER = logging.getLogger(__name__)
SERVER_LOGGER = logging.getLogger("uvicorn.error")
MODEL_SIGNAL_MAX_ITEMS = 100
MODEL_SIGNAL_MAX_CHARS = 200
MODEL_SKILL_MAX_ITEMS = 100
MODEL_SKILL_MAX_CHARS = 120
CANONICAL_ENGLISH_TEXT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("drafting laporan keuangan", "financial reporting"),
    ("menyusun laporan keuangan", "financial reporting"),
    ("laporan keuangan", "financial reporting"),
    ("menganalisis data keuangan", "financial analysis"),
    ("analisis keuangan", "financial analysis"),
    ("analisa keuangan", "financial analysis"),
    ("data keuangan", "financial analysis"),
    ("dashboard keuangan", "dashboard reporting"),
    ("menganalisis data operasional", "operational analysis"),
    ("analisis operasional", "operational analysis"),
    ("data operasional", "operational analysis"),
    ("mendukung pengambilan keputusan bisnis", "business decision support"),
    ("pengambilan keputusan bisnis", "business decision support"),
    ("manajemen proyek", "project management"),
    ("analisis data", "data analysis"),
    ("kemampuan komunikasi", "communication skills"),
    ("komunikasi", "communication"),
    ("kerja sama tim", "teamwork"),
    ("pengalaman kerja", "work experience"),
    ("riwayat pekerjaan", "work history"),
    ("pendidikan", "education"),
    ("sertifikasi", "certification"),
    ("keahlian", "skills"),
    ("keterampilan", "skills"),
    ("kemampuan", "skills"),
    ("kontak tidak terdeteksi", "contact signal not detected"),
    ("sinyal kontak tidak terdeteksi", "contact signal not detected"),
    ("tanggal tidak terdeteksi", "date or timeline signal not detected"),
    ("sinyal dampak terukur tidak terdeteksi", "quantified impact signal not detected"),
    ("dampak terukur tidak terdeteksi", "quantified impact signal not detected"),
    ("judul peran tidak terdeteksi", "role title evidence not detected"),
    ("bukti perusahaan tidak terdeteksi", "company evidence not detected"),
    ("bahasa indonesia", "Indonesian language"),
    ("inggris", "English language"),
)


def _model_identity_payload(identity: ModelIdentity | None, *, include_artifact: bool = False) -> dict[str, object] | None:
    if identity is None:
        return None
    payload: dict[str, object] = {"name": identity.name, "version": identity.version}
    if not include_artifact:
        return payload
    if identity.artifact is not None:
        payload["artifact"] = {
            "format": identity.artifact.format,
            "path": identity.artifact.path,
            "sha256": identity.artifact.sha256,
        }
    payload["artifactPhase"] = identity.artifact_phase
    payload["embeddingModel"] = identity.embedding_model
    return payload


def _metadata_from_payload(payload: dict[str, Any]) -> EmbeddingModelMetadata | None:
    direct = payload.get("embedding_model_metadata")
    if isinstance(direct, dict):
        return EmbeddingModelMetadata.from_mapping(direct)
    embedding_contract = payload.get("embedding_contract")
    if isinstance(embedding_contract, dict):
        return EmbeddingModelMetadata.from_mapping(embedding_contract)
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("embedding_contract"), dict):
        return EmbeddingModelMetadata.from_mapping(data["embedding_contract"])
    model = payload.get("model")
    if isinstance(model, dict) and model.get("embedding_model"):
        return EmbeddingModelMetadata.from_mapping(model)
    return None


def _manifest_embedding_metadata(report: ArtifactVerificationReport) -> EmbeddingModelMetadata | None:
    if report.manifest.embedding_model_metadata:
        return EmbeddingModelMetadata.from_mapping(report.manifest.embedding_model_metadata)
    for entry in report.manifest.required_for_inference_entries():
        if entry.embedding_model_metadata:
            return EmbeddingModelMetadata.from_mapping(entry.embedding_model_metadata)
    return None


def _assert_same_embedding_metadata(name: str, actual: EmbeddingModelMetadata, expected: EmbeddingModelMetadata) -> None:
    mismatches: list[str] = []
    if actual.embedding_model != expected.embedding_model:
        mismatches.append(f"embedding_model expected={expected.embedding_model!r} actual={actual.embedding_model!r}")
    if actual.profile_prefix != expected.profile_prefix:
        mismatches.append(f"profile_prefix expected={expected.profile_prefix!r} actual={actual.profile_prefix!r}")
    if actual.job_prefix != expected.job_prefix:
        mismatches.append(f"job_prefix expected={expected.job_prefix!r} actual={actual.job_prefix!r}")
    if actual.normalized_embeddings != expected.normalized_embeddings:
        mismatches.append(
            "normalized_embeddings "
            f"expected={expected.normalized_embeddings!r} actual={actual.normalized_embeddings!r}"
        )
    if mismatches:
        raise ArtifactError(f"Embedding metadata mismatch in {name}: " + "; ".join(mismatches))


def validate_runtime_embedding_contract(
    *,
    paths: Any,
    artifact_report: ArtifactVerificationReport,
    feature_config: TensorFlowFeatureConfig,
    expected_embedding_model: str | None = None,
) -> EmbeddingModelMetadata:
    """Validate artifact-declared embedding model across runtime metadata."""

    sources: list[tuple[str, EmbeddingModelMetadata]] = []
    tf_payload = load_json(paths.tensorflow_feature_config_path)
    feature_payload = load_json(paths.feature_config_path)
    model_card_payload = load_json(paths.model_card_path)

    for name, metadata in (
        ("tensorflow_feature_config.json", _metadata_from_payload(tf_payload)),
        ("feature_config.json", _metadata_from_payload(feature_payload)),
        ("model_card.json", _metadata_from_payload(model_card_payload)),
        ("artifact_manifest.json", _manifest_embedding_metadata(artifact_report)),
    ):
        if metadata is not None:
            sources.append((name, metadata))

    if artifact_report.manifest.phase_id != "phase_25_tensorflow_training_delivery" and not any(
        source in {"tensorflow_feature_config.json", "feature_config.json"} for source, _metadata in sources
    ):
        raise ArtifactError("Embedding metadata missing from tensorflow_feature_config.json or feature_config.json")
    if not sources:
        sources.append(("tensorflow_feature_config.json", feature_config.embedding_metadata))

    declared = sources[0][1]
    for name, metadata in sources[1:]:
        _assert_same_embedding_metadata(name, metadata, declared)
    if expected_embedding_model and declared.embedding_model != expected_embedding_model:
        raise ArtifactError(
            "MODEL_API_EXPECTED_EMBEDDING_MODEL mismatch: "
            f"expected={expected_embedding_model!r} actual={declared.embedding_model!r}"
        )
    _assert_same_embedding_metadata("loaded TensorFlowFeatureConfig", feature_config.embedding_metadata, declared)
    return declared


def _runtime_state_payload(state: RuntimeState) -> dict[str, object]:
    return {
        "ready": state.ready,
        "message": state.message,
        "artifactManifestPhase": state.artifact_manifest_phase,
        "loadedAt": state.loaded_at,
        "errorCode": state.error_code,
        "model": _model_identity_payload(state.model_identity, include_artifact=True),
    }


def _artifact_verification_payload(report: ArtifactVerificationReport) -> dict[str, object]:
    return {
        "manifestPhase": report.manifest.phase_id,
        "runtimeArtifactIds": report.runtime_artifact_ids,
        "artifactHashes": report.artifact_hashes,
        "artifactSizes": report.artifact_sizes,
    }


def _score_signal_payload(signal: ScoreSignal) -> dict[str, object]:
    return {"key": signal.key, "label": signal.label, "value": signal.value}


def _canonical_english_text(value: Any) -> str:
    """Normalize model-core user-facing labels to English without touching proper nouns broadly."""

    text = str(value).strip()
    if not text:
        return ""
    alias_key = text.casefold()
    canonical_skill = SKILL_ALIASES.get(alias_key)
    if canonical_skill and canonical_skill != alias_key:
        return canonical_skill
    output = text
    for source, target in sorted(CANONICAL_ENGLISH_TEXT_REPLACEMENTS, key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])", re.I)
        output = pattern.sub(target, output)
    return re.sub(r"\s+", " ", output).strip()


def _bounded_text(value: Any, max_chars: int) -> str:
    text = _canonical_english_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _bounded_text_list(values: Any, *, max_items: int, max_chars: int) -> list[str]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        values = [values]
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = _bounded_text(value, max_chars)
        if not text or text in seen:
            continue
        output.append(text)
        seen.add(text)
        if len(output) >= max_items:
            break
    return output


def _skill_list(values: Any) -> list[str]:
    return _bounded_text_list(values, max_items=MODEL_SKILL_MAX_ITEMS, max_chars=MODEL_SKILL_MAX_CHARS)


def _signal_list(values: Any) -> list[str]:
    return _bounded_text_list(values, max_items=MODEL_SIGNAL_MAX_ITEMS, max_chars=MODEL_SIGNAL_MAX_CHARS)


def _score_signal_text(signal: ScoreSignal) -> str:
    if signal.label and signal.value is not None:
        return f"{signal.label}: {signal.value:.3f}"
    if signal.label:
        return signal.label
    if signal.value is not None:
        return f"{signal.key}: {signal.value:.3f}"
    return signal.key


def _recommendation_payload(recommendation) -> dict[str, object]:
    return {
        "jobId": recommendation.jobId,
        "matchScore": recommendation.matchScore,
        "matchLevel": recommendation.matchLevel,
        "rankingSignals": _signal_list([_score_signal_text(signal) for signal in recommendation.rankingSignals]),
        "matchedSkills": _skill_list(recommendation.matchedSkills),
        "missingSkills": _skill_list(recommendation.missingSkills),
    }


def _candidate_skill_evidence(profile: SanitizedProfileInput, candidate) -> tuple[tuple[str, ...], tuple[str, ...]]:
    profile_skills = normalized_skill_set(profile.normalizedSkills)
    candidate_skills = normalized_skill_set((*candidate.model_scoring_input.requiredSkills, *candidate.model_scoring_input.requirements))
    matched = tuple(_skill_list(sorted(profile_skills & candidate_skills)))
    missing = tuple(_skill_list(sorted(candidate_skills - profile_skills)))
    return matched, missing


def _recommendation_payloads_with_evidence(
    recommendations,
    profile: SanitizedProfileInput,
    candidates,
    vectors=(),
) -> list[dict[str, object]]:
    candidates_by_id = {candidate.jobId: candidate for candidate in candidates}
    vectors_by_id = {vector.candidate_id: vector for vector in vectors}
    payloads: list[dict[str, object]] = []
    for recommendation in recommendations:
        payload = _recommendation_payload(recommendation)
        candidate = candidates_by_id.get(recommendation.jobId)
        if candidate is not None:
            matched, missing = _candidate_skill_evidence(profile, candidate)
            vector = vectors_by_id.get(recommendation.jobId)
            payload["matchedSkills"] = _skill_list(matched)
            payload["missingSkills"] = _skill_list(missing)
            payload["rankingSignals"] = _signal_list(
                _ranking_signal_texts(
                    recommendation.rankingSignals,
                    profile,
                    candidate,
                    vector,
                    recommendation.matchScore,
                )
            )
        payloads.append(payload)
    return payloads


def _candidate_feature_scores(profile: SanitizedProfileInput, candidate, vector=None) -> dict[str, float]:
    matched, missing = _candidate_skill_evidence(profile, candidate)
    candidate_skill_count = len(matched) + len(missing)
    explicit_coverage = _feature_value_by_name(vector, "requirement_coverage") if vector is not None else 0.0
    coverage_value = (len(matched) / candidate_skill_count) if candidate_skill_count else explicit_coverage
    semantic = (_feature_value_by_name(vector, "e5_cosine") + 1.0) / 2.0 if vector is not None else 0.0
    return {
        "skillCoverage": max(0.0, min(1.0, coverage_value)),
        "semanticMatch": max(0.0, min(1.0, semantic)),
        "roleMatch": max(0.0, min(1.0, _feature_value_by_name(vector, "role_match") if vector is not None else 0.0)),
        "experienceMatch": max(0.0, min(1.0, _feature_value_by_name(vector, "experience_match") if vector is not None else 0.0)),
        "matchedSkillCount": float(len(matched)),
        "missingSkillCount": float(len(missing)),
    }


def _percent_signal(name: str, value: float) -> str:
    return f"{name}={round(max(0.0, min(1.0, value)) * 100)}%"


def _ranking_signal_texts(
    signals: tuple[ScoreSignal, ...],
    profile: SanitizedProfileInput,
    candidate,
    vector=None,
    match_score: int | None = None,
) -> list[str]:
    texts = [_score_signal_text(signal) for signal in signals]
    matched, missing = _candidate_skill_evidence(profile, candidate)
    scores = _candidate_feature_scores(profile, candidate, vector)
    texts.extend(
        (
            _percent_signal("skillCoverage", scores["skillCoverage"]),
            _percent_signal("semanticMatch", scores["semanticMatch"]),
            _percent_signal("roleMatch", scores["roleMatch"]),
            _percent_signal("experienceMatch", scores["experienceMatch"]),
        )
    )
    if match_score is not None:
        texts.append(f"calibratedMatchScore={match_score}")
    if matched:
        texts.append(f"matchedSkills={len(matched)}")
    if missing:
        texts.append(f"missingSkills={len(missing)}")
    scoring = candidate.model_scoring_input
    role_family = normalize_role(scoring.roleFamily or scoring.titleText)
    if role_family:
        texts.append(f"roleFamily={role_family}")
    if scoring.requiredSkills:
        texts.append("requiredSkills=" + ",".join(_skill_list(scoring.requiredSkills[:5])))
    if profile.experienceYears is not None and scoring.experienceBand:
        texts.append(f"experienceYears={profile.experienceYears:g}; candidateBand={scoring.experienceBand}")
    if profile.detectedCvSectionNames:
        texts.append("sections=" + ",".join(profile.detectedCvSectionNames[:5]))
    return _signal_list(texts)


def _feature_value_by_name(vector, feature_name: str) -> float:
    try:
        index = PHASE25_FEATURE_ORDER.index(feature_name)
    except ValueError:
        return 0.0
    try:
        return float(vector.raw.values[index])
    except (AttributeError, IndexError, TypeError, ValueError):
        return 0.0


def _feature_evidence_floor(profile: SanitizedProfileInput, candidate, vector) -> int:
    """Conservative floor to avoid tiny TensorFlow scores when deterministic evidence matches."""

    if candidate is None:
        return 0
    matched, _missing = _candidate_skill_evidence(profile, candidate)
    scores = _candidate_feature_scores(profile, candidate, vector)
    floor = 0.0
    if profile.cvText.strip() or profile.profileText.strip():
        floor += 12.0
    if profile.detectedCvSectionNames:
        floor += 5.0
    if profile.experienceYears is not None or profile.experienceBand:
        floor += 5.0
    if profile.normalizedSkills:
        floor += 3.0
    if matched:
        floor += min(8.0, len(matched) * 2.0)
    floor += 35.0 * scores["skillCoverage"]
    floor += 12.0 * scores["semanticMatch"]
    floor += 10.0 * scores["roleMatch"]
    floor += 8.0 * scores["experienceMatch"]
    cap = 55
    if scores["skillCoverage"] >= 0.4 or len(matched) >= 2:
        cap = 65
    if scores["skillCoverage"] >= 0.75 and scores["roleMatch"] >= 0.75:
        cap = 78
    return max(0, min(cap, round(floor)))


def _limit_recommendations(recommendations, max_recommendations: int):
    limit = max(0, min(MAX_RECOMMENDATIONS, int(max_recommendations)))
    return tuple(recommendations[:limit])


def _apply_deterministic_evidence_floor(recommendations, profile: SanitizedProfileInput, candidates, vectors):
    candidates_by_id = {candidate.jobId: candidate for candidate in candidates}
    vectors_by_id = {vector.candidate_id: vector for vector in vectors}
    adjusted = []
    for recommendation in recommendations:
        candidate = candidates_by_id.get(recommendation.jobId)
        vector = vectors_by_id.get(recommendation.jobId)
        floor = _feature_evidence_floor(profile, candidate, vector)
        if floor > recommendation.matchScore:
            recommendation = replace(
                recommendation,
                matchScore=floor,
                matchLevel=match_level_for_score(floor),
            )
        elif candidate is not None and vector is not None and recommendation.matchScore >= 70:
            scores = _candidate_feature_scores(profile, candidate, vector)
            unsupported_high_score = (
                scores["matchedSkillCount"] == 0.0
                and scores["roleMatch"] < 0.5
                and scores["semanticMatch"] < 0.75
            )
            if unsupported_high_score:
                capped_score = min(recommendation.matchScore, 60)
                recommendation = replace(
                    recommendation,
                    matchScore=capped_score,
                    matchLevel=match_level_for_score(capped_score),
                )
        adjusted.append(recommendation)
    adjusted.sort(key=lambda item: (-item.matchScore, item.jobId))
    return tuple(adjusted)


def _job_fit_evidence(profile: SanitizedProfileInput, top_candidate, matched_skills: tuple[str, ...], missing_skills: tuple[str, ...]) -> list[str]:
    evidence: list[str] = ["top candidate model-core score with deterministic evidence floor"]
    if top_candidate is not None:
        scoring = top_candidate.model_scoring_input
        if scoring.titleText:
            evidence.append(f"selected target role: {scoring.titleText}")
        role_family = normalize_role(scoring.roleFamily or scoring.titleText)
        if role_family:
            evidence.append(f"role family: {role_family}")
    if matched_skills:
        evidence.append(f"matched required skills: {', '.join(matched_skills[:8])}")
    if missing_skills:
        evidence.append(f"missing required skills: {', '.join(missing_skills[:8])}")
    if profile.experienceYears is not None:
        evidence.append(f"candidate experience years: {profile.experienceYears:g}")
    if top_candidate is not None and top_candidate.model_scoring_input.experienceBand:
        evidence.append(f"job experience band: {top_candidate.model_scoring_input.experienceBand}")
    if profile.targetRoles:
        evidence.append("target roles provided")
    if profile.detectedCvSectionNames:
        evidence.append("detected CV sections: " + ", ".join(profile.detectedCvSectionNames[:6]))
    return evidence


def _grounded_ats_issue_text(issue: str) -> str:
    issue_key = _canonical_english_text(issue).casefold()
    if "contact signal not detected" in issue_key:
        return "Contact evidence weak: email or phone signal was not detected by the parser. Fix: add selectable email and phone text near the top."
    if "date or timeline signal not detected" in issue_key:
        return "Timeline evidence weak: role or education dates were not detected. Fix: add month/year ranges for experience and education."
    if "quantified impact signal not detected" in issue_key:
        return "Impact evidence weak: measurable outcomes were not detected. Fix: add numbers such as %, users, revenue, latency, or project counts."
    if "standard cv sections not detected" in issue_key or "section evidence not provided" in issue_key:
        return "Section structure weak: standard CV headings were not detected. Fix: add clear Summary, Experience, Education, and Skills headings."
    if "image content present" in issue_key:
        return "Formatting/parser risk: image content can reduce ATS extraction. Fix: export a text-based PDF with simple one-column layout."
    if "role title evidence not detected" in issue_key:
        return "Role-title evidence weak: job titles were not detected. Fix: write explicit role titles above each work or project entry."
    if "company evidence not detected" in issue_key:
        return "Organization evidence weak: company or organization names were not detected. Fix: add employer or organization names for each role."
    if "education credential detail not detected" in issue_key:
        return "Education evidence weak: degree or institution detail was not detected. Fix: add degree, school, and graduation timeline."
    if "certification" in issue_key and "not detected" in issue_key:
        return "Certification evidence weak: certification signals were not detected. Fix: add relevant credential names only when they exist."
    if "empty" in issue_key or "no usable" in issue_key:
        return "Parser confidence low: no usable CV text was extracted. Fix: upload a text-based PDF instead of scanned or image-only content."
    return f"Parser evidence: {_bounded_text(issue, 120)}. Fix: make the relevant CV evidence explicit in selectable text."


def _grounded_ats_issues(raw_issues: Any, profile: SanitizedProfileInput) -> list[str]:
    source_issues = _signal_list(raw_issues)
    if not source_issues:
        if not profile.cvText.strip():
            source_issues = ["empty CV text after PDF parsing"]
        elif not profile.detectedCvSectionNames:
            source_issues = ["section evidence not provided"]
    grouped: list[str] = []
    seen_categories: set[str] = set()
    for issue in source_issues:
        text = _grounded_ats_issue_text(issue)
        category = text.split(":", 1)[0].casefold()
        if category in seen_categories:
            continue
        grouped.append(text)
        seen_categories.add(category)
    return _signal_list(grouped)


def _parse_confidence_label(parse_quality: Any) -> str:
    value = str(parse_quality or "").casefold()
    if value in {"high", "good"}:
        return "high"
    if value in {"medium", "partial"}:
        return "medium"
    if value in {"failed", "empty"}:
        return "failed"
    return "low"


def _target_role_label(profile: SanitizedProfileInput, top_candidate) -> str:
    if top_candidate is not None and top_candidate.model_scoring_input.titleText:
        return _bounded_text(top_candidate.model_scoring_input.titleText, 80)
    if profile.targetRoles:
        return _bounded_text(profile.targetRoles[0], 80)
    return "selected role"


def _first_sentence(value: str) -> str:
    text = value.split(". Fix:", 1)[0]
    return text.split(".", 1)[0].strip()


def _overall_impression_evidence(
    profile: SanitizedProfileInput,
    top_candidate,
    matched_skills: tuple[str, ...],
    missing_skills: tuple[str, ...],
    ats_issues: list[str],
    parse_quality: Any,
    parsed_pdf_evidence: dict[str, Any] | None = None,
) -> list[str]:
    confidence = _parse_confidence_label(parse_quality)
    role_label = _target_role_label(profile, top_candidate)
    sections = tuple(_skill_list(profile.detectedCvSectionNames[:4]))
    strengths: list[str] = []
    if matched_skills:
        strengths.append("matched skills " + ", ".join(matched_skills[:4]))
    if sections:
        strengths.append("CV sections " + ", ".join(sections))
    if profile.experienceYears is not None:
        strengths.append(f"experience evidence {profile.experienceYears:g} years")
    if parsed_pdf_evidence:
        if parsed_pdf_evidence.get("roleTitles"):
            strengths.append("role-title evidence detected")
        if parsed_pdf_evidence.get("companyNames"):
            strengths.append("organization evidence detected")
    if not strengths and profile.normalizedSkills:
        strengths.append("CV skill evidence " + ", ".join(_skill_list(profile.normalizedSkills[:4])))
    if not strengths:
        strengths.append("limited role-relevant evidence")

    if confidence in {"failed", "low"} or not profile.cvText.strip():
        summary = f"summary: Parser confidence {confidence}; {strengths[0]}. Treat job-fit and ATS scores cautiously until CV text extraction improves."
    else:
        second_signal = strengths[1] if len(strengths) > 1 else "parser confidence " + confidence
        summary = f"summary: For {role_label}, evidence shows {strengths[0]} plus {second_signal}."
    if ats_issues:
        summary = f"{summary} ATS risk: {_first_sentence(ats_issues[0])}."

    evidence = [summary, f"parser confidence: {confidence}"]
    if missing_skills:
        evidence.append("next improvement: add evidence for " + ", ".join(missing_skills[:4]))
    elif ats_issues:
        evidence.append("next improvement: " + _first_sentence(ats_issues[0]))
    if parsed_pdf_evidence and isinstance(parsed_pdf_evidence.get("wordCount"), int):
        evidence.append(f"parser evidence: {parsed_pdf_evidence.get('wordCount')} words; sections={len(profile.detectedCvSectionNames)}")
    return _signal_list(evidence)


def _latency_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _attach_observability(payload: dict[str, object], **values: object) -> None:
    event = build_safe_observability_event(**values)
    if event:
        payload["observability"] = event


def _warmup_cv_analysis_request() -> CvAnalysisModelCoreRequest:
    return parse_cv_analysis_model_core_request(
        {
            "requestId": "warmup_cv_analysis_staging_fixture",
            "inputVersion": MODEL_CORE_CV_ANALYZER_INPUT_VERSION,
            "language": "en",
            "inputMode": "UPLOAD",
            "compareSource": "JOB_SEARCH",
            "profile": {
                "cvText": "Summary Backend engineer. Skills Python SQL REST APIs. Experience 2020 to 2024.",
                "profileText": "Backend engineer building REST APIs with Python and SQL.",
                "targetRoles": ["Backend Engineer"],
                "normalizedSkills": ["python", "sql", "rest api"],
                "detectedCvSectionNames": ["summary", "skills", "experience"],
            },
            "jobCandidates": [
                {
                    "jobId": "warmup-job-backend-engineer",
                    "scoringInput": {
                        "titleText": "Backend Engineer",
                        "requirementSummary": "Build REST APIs using Python and SQL.",
                        "requiredSkills": ["python", "sql", "rest api"],
                        "roleFamily": "backend",
                    },
                }
            ],
            "rankingPolicy": {
                "maxRecommendations": 1,
                "requireCandidateJobIds": True,
                "deduplicateByJobId": True,
                "backendOwnsHydration": True,
            },
            "maxRecommendations": 1,
        }
    )


def _run_cv_analysis_warmup(
    *,
    service: InferenceService,
    feature_config: TensorFlowFeatureConfig,
    calibration_policy: ScoreCalibrationPolicy,
    embedding_backend: TextEmbeddingBackend,
    environment: str,
    timeout_ms: int | None,
) -> dict[str, object]:
    started_at = perf_counter()
    payload = build_cv_analysis_response_payload(
        _warmup_cv_analysis_request(),
        service=service,
        feature_config=feature_config,
        calibration_policy=calibration_policy,
        embedding_backend=embedding_backend,
        environment=environment,
        timeout_ms=timeout_ms,
        include_observability=True,
    )
    return {
        "completed": True,
        "latencyMs": _latency_ms(started_at),
        "modelVersion": (payload.get("model") or {}).get("version") if isinstance(payload.get("model"), dict) else None,
        "observability": payload.get("observability", {}),
    }


def build_cv_analysis_response_payload(
    request: CvAnalysisModelCoreRequest,
    service: InferenceService,
    feature_config: TensorFlowFeatureConfig,
    calibration_policy: ScoreCalibrationPolicy,
    embedding_backend: TextEmbeddingBackend | None = None,
    environment: str = "local",
    timeout_ms: int | None = None,
    include_observability: bool = False,
    parse_latency_ms: int = 0,
) -> dict[str, object]:
    """Build validated model-core CV-analysis response for HTTP route/tests."""

    state = service.state
    if state.model_identity is None:
        service.require_ready()
        state = service.state
    if state.model_identity is None:
        raise ModelNotReadyError("TensorFlow model identity is not available")

    total_started_at = perf_counter()
    embedding_started_at = perf_counter()
    candidate_vectors = build_feature_vectors_for_request(
        request,
        feature_config=feature_config,
        embedding_backend=embedding_backend,
        environment=environment,
    )
    embedding_latency_ms = _latency_ms(embedding_started_at)
    tensorflow_started_at = perf_counter()
    recommendations = service.predict_recommendations(
        tuple(vector.normalized for vector in candidate_vectors),
        max_recommendations=len(candidate_vectors),
        calibration_policy=calibration_policy,
        timeout_ms=timeout_ms,
        allow_internal_full_ranking=True,
    )
    recommendations = _limit_recommendations(
        _apply_deterministic_evidence_floor(recommendations, request.profile, request.jobCandidates, candidate_vectors),
        request.maxRecommendations,
    )
    tensorflow_latency_ms = _latency_ms(tensorflow_started_at)
    top_score = recommendations[0].matchScore if recommendations else 0
    top_candidate_by_id = {candidate.jobId: candidate for candidate in request.jobCandidates}
    top_candidate = top_candidate_by_id.get(recommendations[0].jobId) if recommendations else None
    matched_skills, missing_skills = _candidate_skill_evidence(request.profile, top_candidate) if top_candidate is not None else ((), ())
    profile_evidence_keys = [
        key
        for key, enabled in {
            "cvText": bool(request.profile.cvText.strip()),
            "profileText": bool(request.profile.profileText.strip()),
            "normalizedSkills": bool(request.profile.normalizedSkills),
            "targetRoles": bool(request.profile.targetRoles),
            "detectedCvSectionNames": bool(request.profile.detectedCvSectionNames),
        }.items()
        if enabled
    ]
    detected_sections = list(request.profile.detectedCvSectionNames)
    parse_quality = "high" if detected_sections else ("medium" if request.profile.cvText.strip() else "failed")
    job_fit = JobFitAlignmentCore(
        score=top_score,
        matchedSkills=matched_skills,
        missingSkills=missing_skills,
        summarySignals=tuple(ScoreSignal(key=feature_name) for feature_name in PHASE25_FEATURE_ORDER),
        confidenceNotes=("model-core job-fit score from Phase 25 TensorFlow candidate scorer",),
    )
    ats_score = 85 if request.profile.detectedCvSectionNames else 70
    ats = AtsFriendlinessCore(
        score=ats_score,
        detectedIssues=() if request.profile.detectedCvSectionNames else ("section evidence not provided",),
        evidence={"evidenceKeys": profile_evidence_keys, "placeholderPolicy": "deterministic_model_core_evidence"},
        fallback=True,
    )
    grounded_ats_issues = _grounded_ats_issues(ats.detectedIssues, request.profile)
    overall_score = round((job_fit.score + ats.score) / 2)
    overall = OverallImpressionCore(
        score=overall_score,
        summary="Grounded model-core impression prepared from deterministic CV and job-fit evidence.",
        evidenceKeys=("jobFitAlignment", "atsFriendliness", "candidateReranking"),
        confidenceNotes=("No external GenAI call is made by Model API core inference.",),
    )
    overall_evidence = _overall_impression_evidence(request.profile, top_candidate, matched_skills, missing_skills, grounded_ats_issues, parse_quality)
    payload: dict[str, object] = {
        "schemaVersion": MODEL_CORE_CV_ANALYSIS_SCHEMA_VERSION,
        "parsedCv": {
            "status": "parsed" if request.profile.cvText.strip() else "empty_text",
            "pageCount": 0,
            "textLength": len(request.profile.cvText),
            "detectedSections": detected_sections,
            "extractionEvidence": _signal_list(profile_evidence_keys),
        },
        "jobFitAlignment": {
            "score": job_fit.score,
            "matchedSignals": _signal_list([_score_signal_text(signal) for signal in job_fit.summarySignals]),
            "missingSignals": _signal_list(job_fit.confidenceNotes),
            "matchedSkills": _skill_list(job_fit.matchedSkills),
            "missingSkills": _skill_list(job_fit.missingSkills),
            "evidence": _signal_list([*_job_fit_evidence(request.profile, top_candidate, matched_skills, missing_skills), *profile_evidence_keys]),
        },
        "atsFriendliness": {
            "score": ats.score,
            "detectedIssues": grounded_ats_issues,
            "parseQuality": parse_quality,
            "evidence": _signal_list([str(value) for value in ats.evidence.get("evidenceKeys", profile_evidence_keys)] if isinstance(ats.evidence, dict) else profile_evidence_keys),
        },
        "overallImpression": {
            "score": overall.score,
            "evidence": overall_evidence,
        },
        "candidateReranking": {
            "recommendations": _recommendation_payloads_with_evidence(recommendations, request.profile, request.jobCandidates, candidate_vectors),
        },
        "model": _model_identity_payload(state.model_identity),
        "createdAt": utc_now_iso(),
    }
    if include_observability:
        _attach_observability(
            payload,
            requestId=request.requestId,
            modelVersion=state.model_identity.version,
            artifactHash=state.model_identity.artifact_sha256,
            candidateCount=len(request.jobCandidates),
            parseQuality=parse_quality,
            parseLatencyMs=parse_latency_ms,
            embeddingLatencyMs=embedding_latency_ms,
            tensorflowLatencyMs=tensorflow_latency_ms,
            wrapperLatencyMs=0,
            totalLatencyMs=_latency_ms(total_started_at),
        )
    validate_model_core_payload(payload, {candidate.jobId for candidate in request.jobCandidates}, request.maxRecommendations)
    return payload


def build_candidate_reranking_response_payload(
    request: CandidateRerankingCoreRequest,
    service: InferenceService,
    feature_config: TensorFlowFeatureConfig,
    calibration_policy: ScoreCalibrationPolicy,
    embedding_backend: TextEmbeddingBackend | None = None,
    environment: str = "local",
    timeout_ms: int | None = None,
) -> dict[str, object]:
    state = service.state
    if state.model_identity is None:
        service.require_ready()
        state = service.state
    if state.model_identity is None:
        raise ModelNotReadyError("TensorFlow model identity is not available")

    cv_request = CvAnalysisModelCoreRequest(
        requestId=request.requestId,
        inputVersion=MODEL_CORE_CV_ANALYZER_INPUT_VERSION,
        language=request.language,
        inputMode="UPLOAD",
        compareSource="JOB_SEARCH",
        profile=request.profileFeatures,
        jobCandidates=request.jobCandidates,
        maxRecommendations=request.maxRecommendations,
        rankingPolicy=request.rankingPolicy,
    )
    total_started_at = perf_counter()
    embedding_started_at = perf_counter()
    candidate_vectors = build_feature_vectors_for_request(
        cv_request,
        feature_config=feature_config,
        embedding_backend=embedding_backend,
        environment=environment,
    )
    embedding_latency_ms = _latency_ms(embedding_started_at)
    tensorflow_started_at = perf_counter()
    recommendations = service.predict_recommendations(
        tuple(vector.normalized for vector in candidate_vectors),
        max_recommendations=len(candidate_vectors),
        calibration_policy=calibration_policy,
        timeout_ms=timeout_ms,
        allow_internal_full_ranking=True,
    )
    recommendations = _limit_recommendations(
        _apply_deterministic_evidence_floor(recommendations, request.profileFeatures, request.jobCandidates, candidate_vectors),
        request.maxRecommendations,
    )
    tensorflow_latency_ms = _latency_ms(tensorflow_started_at)
    payload: dict[str, object] = {
        "requestId": request.requestId,
        "schemaVersion": MODEL_CORE_CANDIDATE_RERANKING_SCHEMA_VERSION,
        "candidateSetId": request.candidateSetId,
        "language": request.language,
        "recommendations": _recommendation_payloads_with_evidence(recommendations, request.profileFeatures, request.jobCandidates, candidate_vectors),
        "model": _model_identity_payload(state.model_identity),
        "rankedAt": utc_now_iso(),
    }
    _attach_observability(
        payload,
        requestId=request.requestId,
        modelVersion=state.model_identity.version,
        artifactHash=state.model_identity.artifact_sha256,
        candidateCount=len(request.jobCandidates),
        parseQuality="preparsed_profile_features",
        parseLatencyMs=0,
        embeddingLatencyMs=embedding_latency_ms,
        tensorflowLatencyMs=tensorflow_latency_ms,
        wrapperLatencyMs=0,
        totalLatencyMs=_latency_ms(total_started_at),
    )
    validate_model_core_payload(payload, {candidate.jobId for candidate in request.jobCandidates}, request.maxRecommendations)
    return payload


def _auth_error() -> dict[str, object]:
    return {
        "success": False,
        "message": "Unauthorized Model API request",
        "data": None,
        "error": {"code": "MODEL_API_UNAUTHORIZED", "details": ["valid bearer service token required"]},
    }


def _authorize_internal_request(headers: Any, config: RuntimeConfig) -> dict[str, object] | None:
    if config.environment.lower() in {"local", "test"} and config.allow_unauthenticated_local and not config.service_token:
        return None
    expected = config.service_token
    authorization = headers.get("authorization") if hasattr(headers, "get") else None
    if not expected or not isinstance(authorization, str):
        return _auth_error()
    expected_header = f"Bearer {expected}"
    if not compare_digest(authorization, expected_header):
        return _auth_error()
    return None


def _json_form_value(value: str | None, field_name: str) -> Any:
    if value is None or not str(value).strip():
        if field_name == "rankingPolicy":
            return None
        raise ContractValidationError([f"$.{field_name} is required"])
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ContractValidationError([f"$.{field_name} must be valid JSON: {exc.msg}"]) from exc


def _form_string_values(form: Any, field_name: str) -> list[str]:
    if hasattr(form, "getlist"):
        values = list(form.getlist(field_name))
    else:
        value = form.get(field_name) if hasattr(form, "get") else None
        values = value if isinstance(value, list) else ([] if value is None else [value])
    return [value for value in values if isinstance(value, str)]


def _job_roles_from_multipart(form: Any) -> list[str]:
    raw_values = _form_string_values(form, "jobRoles")
    if len(raw_values) == 1 and raw_values[0].lstrip().startswith("["):
        parsed = _json_form_value(raw_values[0], "jobRoles")
        raw_values = parsed if isinstance(parsed, list) else []
    roles: list[str] = []
    errors: list[str] = []
    for index, value in enumerate(raw_values):
        if not isinstance(value, str) or not value.strip():
            errors.append(f"$.jobRoles[{index}] must be non-empty string")
            continue
        role = value.strip()
        if len(role) > MAX_JOB_ROLE_CHARS:
            errors.append(f"$.jobRoles[{index}] max length {MAX_JOB_ROLE_CHARS}; actual={len(role)}")
        roles.append(role)
    if not roles:
        errors.append("$.jobRoles must contain at least 1 role")
    if len(roles) > MAX_JOB_ROLES:
        errors.append(f"$.jobRoles max items {MAX_JOB_ROLES}; actual={len(roles)}")
    if errors:
        raise ContractValidationError(errors)
    return roles


def _candidate_requirement_hints(values: Any) -> list[str]:
    hints: list[str] = []
    if not isinstance(values, list):
        return hints
    for value in values:
        if isinstance(value, str):
            hints.append(value)
        elif isinstance(value, dict) and isinstance(value.get("value"), str):
            hints.append(value["value"])
    return hints


def _validate_pdf_upload_bytes(pdf_bytes: bytes, runtime_config: RuntimeConfig) -> None:
    if len(pdf_bytes) > runtime_config.max_pdf_bytes:
        raise ContractValidationError(["cvFile exceeds MODEL_API_MAX_PDF_BYTES"])
    if not pdf_bytes.lstrip().startswith(b"%PDF"):
        raise ContractValidationError(["cvFile must start with PDF magic bytes"])


def _build_cv_payload_from_multipart(form: Any, pdf_bytes: bytes, runtime_config: RuntimeConfig) -> dict[str, object]:
    _validate_pdf_upload_bytes(pdf_bytes, runtime_config)
    job_candidates = _json_form_value(form.get("jobCandidates"), "jobCandidates")
    ranking_policy = _json_form_value(form.get("rankingPolicy"), "rankingPolicy")
    candidate_skill_hints: list[str] = []
    if isinstance(job_candidates, list):
        for candidate in job_candidates:
            if not isinstance(candidate, dict):
                continue
            scoring = candidate.get("scoringInput") if isinstance(candidate.get("scoringInput"), dict) else candidate
            if not isinstance(scoring, dict):
                continue
            required_skills = scoring.get("requiredSkills")
            if isinstance(required_skills, list):
                candidate_skill_hints.extend(str(value) for value in required_skills if isinstance(value, str))
            candidate_skill_hints.extend(_candidate_requirement_hints(scoring.get("requirements")))

    request_id = str(form.get("requestId") or "")
    SERVER_LOGGER.info("Model API cv-analysis PDF parse started request_id=%s", request_id)
    parse_started_at = perf_counter()
    parsed_pdf = parse_pdf_bytes(pdf_bytes, max_bytes=runtime_config.max_pdf_bytes, max_pages=runtime_config.max_pdf_pages)
    SERVER_LOGGER.info(
        "Model API cv-analysis PDF parse completed request_id=%s parse_latency_ms=%s text_length=%s page_count=%s",
        request_id,
        _latency_ms(parse_started_at),
        len(parsed_pdf.text),
        parsed_pdf.page_count,
    )
    parse_latency_ms = _latency_ms(parse_started_at)
    ats_score, detected_issues, ats_fallback = ats_score_from_pdf_evidence(parsed_pdf)
    target_roles = _job_roles_from_multipart(form)
    profile_text = parsed_pdf.text[:2000] if parsed_pdf.text.strip() else "target roles: " + ", ".join(target_roles)
    profile = {
        "cvText": parsed_pdf.text,
        "profileText": profile_text,
        "targetRoles": target_roles,
        "normalizedSkills": list(normalized_skills_from_text(parsed_pdf.text, candidate_skill_hints)),
        "experienceYears": parsed_pdf.estimated_experience_years,
        "detectedCvSectionNames": list(parsed_pdf.section_names),
    }
    payload: dict[str, object] = {
        "requestId": str(form.get("requestId") or ""),
        "inputVersion": MODEL_CORE_CV_ANALYZER_INPUT_VERSION,
        "language": str(form.get("language") or ""),
        "inputMode": str(form.get("inputMode") or "UPLOAD"),
        "compareSource": str(form.get("compareSource") or ""),
        "profile": profile,
        "jobCandidates": job_candidates,
        "rankingPolicy": ranking_policy,
        "maxRecommendations": (ranking_policy or {}).get("maxRecommendations", MAX_RECOMMENDATIONS) if isinstance(ranking_policy, dict) else MAX_RECOMMENDATIONS,
    }
    payload["_parsedPdfEvidence"] = {
        "pageCount": parsed_pdf.page_count,
        "parseQuality": parsed_pdf.parse_quality,
        "atsScore": ats_score,
        "detectedIssues": list(detected_issues),
        "hasQuantifiedImpact": parsed_pdf.has_quantified_impact,
        "estimatedExperienceYears": parsed_pdf.estimated_experience_years,
        "wordCount": parsed_pdf.word_count,
        "roleTitles": list(parsed_pdf.role_titles),
        "companyNames": list(parsed_pdf.company_names),
        "hasEducationSignal": parsed_pdf.has_education_signal,
        "hasCertificationSignal": parsed_pdf.has_certification_signal,
        "languageSignals": list(parsed_pdf.language_signals),
        "seniorityHints": list(parsed_pdf.seniority_hints),
        "fallback": ats_fallback,
        "parseLatencyMs": parse_latency_ms,
    }
    return payload


def create_app(
    config: RuntimeConfig | None = None,
    service: InferenceService | None = None,
    embedding_backend: TextEmbeddingBackend | None = None,
):
    """Create HTTP app, verify artifacts, and load model in the background during lifespan startup.

    Raises a clear error when FastAPI is not installed instead of failing during
    package import.
    """

    try:
        from fastapi import Body, FastAPI, Request
        from fastapi.responses import JSONResponse
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional runtime deps
        raise RuntimeError("FastAPI dependency missing; install serving requirements after Step 26.13") from exc

    globals()["Request"] = Request
    runtime_config = config or RuntimeConfig.from_env()
    runtime_config.validate_security()
    artifact_report = verify_runtime_artifacts(runtime_config.artifact_paths)
    feature_config = TensorFlowFeatureConfig.from_path(runtime_config.artifact_paths.tensorflow_feature_config_path)
    embedding_contract = validate_runtime_embedding_contract(
        paths=runtime_config.artifact_paths,
        artifact_report=artifact_report,
        feature_config=feature_config,
        expected_embedding_model=runtime_config.expected_embedding_model,
    )
    calibration_policy = ScoreCalibrationPolicy.from_path(runtime_config.artifact_paths.score_calibration_path)
    inference_service = service or InferenceService()
    runtime_embedding_backend = embedding_backend or SentenceTransformerE5Embedder(embedding_contract.embedding_model)
    validate_e5_backend(runtime_embedding_backend, runtime_config.environment, expected_model_name=embedding_contract.embedding_model)
    warmup_state: dict[str, object] = {"completed": False, "inProgress": False, "latencyMs": None, "error": None}
    inference_gate = BoundedSemaphore(runtime_config.max_concurrent_inference)

    def _load_runtime_once() -> RuntimeState:
        state = inference_service.load_once(runtime_config.artifact_paths, artifact_report)
        if runtime_config.warmup_on_startup and state.ready:
            try:
                warmup_state.update({"completed": False, "inProgress": True, "error": None})
                LOGGER.info("Model API startup warmup started")
                warmup_state.update(
                    _run_cv_analysis_warmup(
                        service=inference_service,
                        feature_config=feature_config,
                        calibration_policy=calibration_policy,
                        embedding_backend=runtime_embedding_backend,
                        environment=runtime_config.environment,
                        timeout_ms=runtime_config.timeout_ms,
                    )
                )
                LOGGER.info("Model API startup warmup completed latencyMs=%s", warmup_state.get("latencyMs"))
            except Exception as exc:  # pragma: no cover - depends on live TensorFlow/E5 runtime
                warmup_state.update({"completed": False, "error": repr(exc)})
                LOGGER.exception("Model API startup warmup failed")
            finally:
                warmup_state["inProgress"] = False
        return inference_service.state

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime_config = runtime_config
        app.state.artifact_report = artifact_report
        app.state.inference_service = inference_service
        app.state.feature_config = feature_config
        app.state.calibration_policy = calibration_policy
        app.state.embedding_contract = embedding_contract
        load_task = asyncio.create_task(asyncio.to_thread(_load_runtime_once))
        app.state.model_load_task = load_task
        yield

    app = FastAPI(title="Bisakerja Model API", version="0.1.0-phase26.4", lifespan=lifespan)

    @app.exception_handler(ModelNotReadyError)
    async def model_not_ready_handler(_request: Request, exc: ModelNotReadyError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "TensorFlow model is not ready",
                "data": None,
                "error": exc.to_error_payload(),
            },
        )

    @app.exception_handler(ContractValidationError)
    async def contract_validation_error_handler(_request: Request, exc: ContractValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"success": False, "message": "Invalid model-core request/response", "data": None, "error": exc.to_error_payload()},
        )

    @app.exception_handler(FeatureBuildError)
    async def feature_build_error_handler(_request: Request, exc: FeatureBuildError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"success": False, "message": "Invalid feature inputs", "data": None, "error": exc.to_error_payload()},
        )

    @app.exception_handler(InferenceTimeoutError)
    async def inference_timeout_handler(_request: Request, exc: InferenceTimeoutError) -> JSONResponse:
        return JSONResponse(
            status_code=504,
            content={"success": False, "message": "TensorFlow inference timeout", "data": None, "error": exc.to_error_payload()},
        )

    @app.exception_handler(ModelLoadError)
    async def model_load_error_handler(_request: Request, exc: ModelLoadError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"success": False, "message": "TensorFlow inference failed", "data": None, "error": exc.to_error_payload()},
        )

    @app.exception_handler(UnsupportedArtifactVersionError)
    async def unsupported_artifact_version_handler(_request: Request, exc: UnsupportedArtifactVersionError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Unsupported artifact version", "data": None, "error": exc.to_error_payload()},
        )

    @app.exception_handler(ArtifactError)
    async def artifact_error_handler(_request: Request, exc: ArtifactError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Artifact verification failed", "data": None, "error": exc.to_error_payload()},
        )

    @app.exception_handler(ModelApiError)
    async def model_api_error_handler(_request: Request, exc: ModelApiError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Model API error", "data": None, "error": exc.to_error_payload()},
        )

    @app.get("/live")
    async def live() -> dict[str, object]:
        return {
            "service": runtime_config.service_name,
            "environment": runtime_config.environment,
            "live": True,
            "message": "alive",
        }

    @app.get("/")
    async def root() -> dict[str, object]:
        state = inference_service.state
        return {
            "service": runtime_config.service_name,
            "environment": runtime_config.environment,
            "message": "Bisakerja Model API is running",
            "ready": state.ready,
            "status": state.message,
            "endpoints": {
                "live": "/live",
                "health": "/health",
                "ready": "/ready",
                "modelInfo": "/model-info",
                "cvAnalysis": "/inference/cv-analysis",
            },
        }

    @app.get("/health")
    async def health() -> dict[str, object]:
        state = inference_service.state
        identity = state.model_identity
        return {
            "service": runtime_config.service_name,
            "environment": runtime_config.environment,
            "ready": state.ready,
            "message": state.message,
            "modelName": None if identity is None else identity.name,
            "modelVersion": None if identity is None else identity.version,
            "artifactHash": None if identity is None else identity.artifact_sha256,
        }

    @app.get("/ready")
    async def ready() -> dict[str, object]:
        state = inference_service.state
        identity = state.model_identity
        checks = {
            "artifactsVerified": bool(artifact_report.artifact_hashes),
            "tensorflowModelLoaded": state.ready and identity is not None,
            "embeddingModelDeclared": bool(embedding_contract.embedding_model),
            "embeddingModelMatchesBackend": getattr(runtime_embedding_backend, "model_name", "") == embedding_contract.embedding_model,
            "e5BackendConfigured": getattr(runtime_embedding_backend, "backend_name", "") != "local-hash",
            "pdfParserAvailable": callable(parse_pdf_bytes),
            "serviceTokenConfigured": (not runtime_config.requires_service_token) or bool(runtime_config.service_token),
            "warmupCompleted": (not runtime_config.warmup_required) or bool(warmup_state.get("completed")),
        }
        return {
            "service": runtime_config.service_name,
            "environment": runtime_config.environment,
            "ready": all(checks.values()),
            "checks": checks,
            "warmup": {
                "required": runtime_config.warmup_required,
                "onStartup": runtime_config.warmup_on_startup,
                "completed": bool(warmup_state.get("completed")),
                "inProgress": bool(warmup_state.get("inProgress")),
                "latencyMs": warmup_state.get("latencyMs"),
                "error": warmup_state.get("error"),
            },
            "modelVersion": None if identity is None else identity.version,
            "embeddingModel": embedding_contract.embedding_model,
            "artifactPhase": artifact_report.manifest.phase_id,
            "artifactHash": None if identity is None else identity.artifact_sha256,
        }

    @app.get("/model-info")
    def model_info(http_request: Request) -> dict[str, object]:
        auth_error = _authorize_internal_request(http_request.headers, runtime_config)
        if auth_error is not None:
            return JSONResponse(status_code=401, content=auth_error)
        state = inference_service.state
        return {
            "ready": state.ready,
            "readiness": _runtime_state_payload(state),
            "model": _model_identity_payload(state.model_identity, include_artifact=True),
            "artifacts": runtime_config.artifact_paths.as_dict(),
            "artifactVerification": _artifact_verification_payload(artifact_report),
            "embeddingPolicy": embedding_contract.as_dict(),
            "openrouter": {
                "baseUrl": runtime_config.openrouter.base_url,
                "modelsUrl": runtime_config.openrouter.models_url,
                "model": runtime_config.openrouter.model,
                "enabled": runtime_config.openrouter.enabled,
            },
        }

    @app.post("/inference/cv-analysis")
    def cv_analysis(http_request: Request, payload: Any = Body(...)) -> dict[str, object]:
        auth_error = _authorize_internal_request(http_request.headers, runtime_config)
        if auth_error is not None:
            return JSONResponse(status_code=401, content=auth_error)
        request = parse_cv_analysis_model_core_request(payload)
        include_observability = http_request.headers.get("x-model-api-include-observability") == "true"
        with inference_gate:
            data = build_cv_analysis_response_payload(
                request,
                service=inference_service,
                feature_config=feature_config,
                calibration_policy=calibration_policy,
                embedding_backend=runtime_embedding_backend,
                environment=runtime_config.environment,
                timeout_ms=runtime_config.timeout_ms,
                include_observability=include_observability,
            )
        observability = data.get("observability") if isinstance(data, dict) else None
        warmup_state.update(
            {
                "completed": True,
                "inProgress": False,
                "latencyMs": observability.get("totalLatencyMs") if isinstance(observability, dict) else None,
                "error": None,
            }
        )
        return {"success": True, "message": "Model-core inference completed", "data": data, "error": None}

    @app.post("/inference/candidate-reranking")
    def candidate_reranking(http_request: Request, payload: Any = Body(...)) -> dict[str, object]:
        auth_error = _authorize_internal_request(http_request.headers, runtime_config)
        if auth_error is not None:
            return JSONResponse(status_code=401, content=auth_error)
        request = parse_candidate_reranking_core_request(payload)
        with inference_gate:
            data = build_candidate_reranking_response_payload(
                request,
                service=inference_service,
                feature_config=feature_config,
                calibration_policy=calibration_policy,
                embedding_backend=runtime_embedding_backend,
                environment=runtime_config.environment,
                timeout_ms=runtime_config.timeout_ms,
            )
        return {"success": True, "message": "Model-core candidate reranking completed", "data": data, "error": None}

    @app.post("/internal/model/cv-analysis")
    async def internal_model_cv_analysis(http_request: Request):
        route_started_at = perf_counter()
        request_id = http_request.headers.get("x-request-id", "")
        SERVER_LOGGER.info("Model API cv-analysis request received request_id=%s", request_id)
        auth_error = _authorize_internal_request(http_request.headers, runtime_config)
        if auth_error is not None:
            return JSONResponse(status_code=401, content=auth_error)
        content_type = http_request.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            raise ContractValidationError(["Content-Type must be multipart/form-data"])
        form_started_at = perf_counter()
        form = await http_request.form()
        form_latency_ms = _latency_ms(form_started_at)
        SERVER_LOGGER.info("Model API cv-analysis multipart parsed request_id=%s form_latency_ms=%s", request_id, form_latency_ms)
        file_values = [value for value in form.values() if hasattr(value, "filename") and hasattr(value, "read")]
        if len(file_values) != 1 or "cvFile" not in form:
            raise ContractValidationError(["multipart request must include exactly one cvFile"])
        upload = form["cvFile"]
        if getattr(upload, "content_type", None) not in {"application/pdf", "application/octet-stream"}:
            raise ContractValidationError(["cvFile content type must be application/pdf"])
        read_started_at = perf_counter()
        pdf_bytes = await upload.read()
        read_latency_ms = _latency_ms(read_started_at)
        SERVER_LOGGER.info(
            "Model API cv-analysis file read request_id=%s cv_bytes=%s read_latency_ms=%s",
            request_id,
            len(pdf_bytes),
            read_latency_ms,
        )
        _validate_pdf_upload_bytes(pdf_bytes, runtime_config)
        payload = _build_cv_payload_from_multipart(form, pdf_bytes, runtime_config)
        parsed_pdf_evidence = payload.pop("_parsedPdfEvidence")
        parse_latency_ms = int(parsed_pdf_evidence.get("parseLatencyMs", 0)) if isinstance(parsed_pdf_evidence, dict) else 0
        job_candidates = payload.get("jobCandidates")
        candidate_count = len(job_candidates) if isinstance(job_candidates, list) else None
        SERVER_LOGGER.info(
            "Model API cv-analysis payload built request_id=%s cv_bytes=%s candidate_count=%s parse_latency_ms=%s",
            request_id,
            len(pdf_bytes),
            candidate_count,
            parse_latency_ms,
        )
        request = parse_cv_analysis_model_core_request(payload)
        include_observability = http_request.headers.get("x-model-api-include-observability") == "true"
        queue_started_at = perf_counter()
        SERVER_LOGGER.info(
            "Model API cv-analysis inference queued request_id=%s max_concurrent_inference=%s",
            request_id,
            runtime_config.max_concurrent_inference,
        )

        def _run_inference_with_gate() -> tuple[dict[str, object], int, int]:
            with inference_gate:
                queue_latency_ms = _latency_ms(queue_started_at)
                SERVER_LOGGER.info(
                    "Model API cv-analysis inference started request_id=%s queue_latency_ms=%s",
                    request_id,
                    queue_latency_ms,
                )
                inference_started_at = perf_counter()
                data = build_cv_analysis_response_payload(
                    request,
                    service=inference_service,
                    feature_config=feature_config,
                    calibration_policy=calibration_policy,
                    embedding_backend=runtime_embedding_backend,
                    environment=runtime_config.environment,
                    timeout_ms=runtime_config.timeout_ms,
                    include_observability=True,
                    parse_latency_ms=parse_latency_ms,
                )
                return data, queue_latency_ms, _latency_ms(inference_started_at)

        data, queue_latency_ms, inference_latency_ms = await asyncio.to_thread(_run_inference_with_gate)
        observability = data.get("observability") if isinstance(data, dict) else None
        warmup_state.update(
            {
                "completed": True,
                "inProgress": False,
                "latencyMs": observability.get("totalLatencyMs") if isinstance(observability, dict) else inference_latency_ms,
                "error": None,
            }
        )
        if isinstance(data.get("atsFriendliness"), dict) and isinstance(parsed_pdf_evidence, dict):
            parse_quality = parsed_pdf_evidence["parseQuality"]
            extraction_evidence = [
                "deterministic_pdf_parser",
                f"parseLatencyMs={parsed_pdf_evidence['parseLatencyMs']}",
                f"wordCount={parsed_pdf_evidence.get('wordCount', 0)}",
                f"quantifiedImpact={bool(parsed_pdf_evidence.get('hasQuantifiedImpact'))}",
                f"experienceYears={parsed_pdf_evidence.get('estimatedExperienceYears')}",
            ]
            for key, label in (
                ("roleTitles", "roleTitleSignals"),
                ("companyNames", "companySignals"),
                ("languageSignals", "languageSignals"),
                ("seniorityHints", "seniorityHints"),
            ):
                values = parsed_pdf_evidence.get(key)
                if isinstance(values, list) and values:
                    extraction_evidence.append(f"{label}={len(values)}")
            data["parsedCv"] = {
                "status": "parsed" if request.profile.cvText.strip() else "empty_text",
                "textLength": len(request.profile.cvText),
                "pageCount": parsed_pdf_evidence["pageCount"],
                "detectedSections": list(request.profile.detectedCvSectionNames),
                "extractionEvidence": _signal_list(extraction_evidence),
            }
            data["atsFriendliness"]["score"] = parsed_pdf_evidence["atsScore"]
            grounded_ats_issues = _grounded_ats_issues(parsed_pdf_evidence["detectedIssues"], request.profile)
            data["atsFriendliness"]["detectedIssues"] = grounded_ats_issues
            parse_quality_map = {"good": "high", "partial": "medium", "empty": "low", "failed": "failed"}
            data["atsFriendliness"]["parseQuality"] = parse_quality_map.get(str(parse_quality), parse_quality if parse_quality in {"high", "medium", "low", "failed"} else "low")
            data["atsFriendliness"]["evidence"] = _signal_list([
                "deterministic_pdf_parser",
                f"sections={len(request.profile.detectedCvSectionNames)}",
                f"contact={bool(request.profile.cvText.strip()) and 'contact signal not detected' not in parsed_pdf_evidence.get('detectedIssues', [])}",
                f"timeline={'date or timeline signal not detected' not in parsed_pdf_evidence.get('detectedIssues', [])}",
                f"quantifiedImpact={bool(parsed_pdf_evidence.get('hasQuantifiedImpact'))}",
                f"formattingIssues={len(parsed_pdf_evidence.get('detectedIssues', []))}",
            ])
            if isinstance(data.get("overallImpression"), dict):
                recommendations_payload = (data.get("candidateReranking") or {}).get("recommendations", []) if isinstance(data.get("candidateReranking"), dict) else []
                top_job_id = recommendations_payload[0].get("jobId") if recommendations_payload and isinstance(recommendations_payload[0], dict) else None
                top_candidate = {candidate.jobId: candidate for candidate in request.jobCandidates}.get(top_job_id)
                matched_skills, missing_skills = _candidate_skill_evidence(request.profile, top_candidate) if top_candidate is not None else ((), ())
                data["overallImpression"]["evidence"] = _overall_impression_evidence(
                    request.profile,
                    top_candidate,
                    matched_skills,
                    missing_skills,
                    grounded_ats_issues,
                    data["atsFriendliness"]["parseQuality"],
                    parsed_pdf_evidence,
                )
        embedding_latency_ms = observability.get("embeddingLatencyMs") if isinstance(observability, dict) else None
        tensorflow_latency_ms = observability.get("tensorflowLatencyMs") if isinstance(observability, dict) else None
        if not include_observability:
            data.pop("observability", None)
        validate_model_core_payload(data, {candidate.jobId for candidate in request.jobCandidates}, request.maxRecommendations)
        SERVER_LOGGER.info(
            "Model API cv-analysis completed request_id=%s candidate_count=%s parse_latency_ms=%s embedding_latency_ms=%s tensorflow_latency_ms=%s inference_latency_ms=%s queue_latency_ms=%s total_latency_ms=%s",
            request_id,
            candidate_count,
            parse_latency_ms,
            embedding_latency_ms,
            tensorflow_latency_ms,
            inference_latency_ms,
            queue_latency_ms,
            _latency_ms(route_started_at),
        )
        return data

    return app
