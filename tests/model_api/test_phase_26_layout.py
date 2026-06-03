from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile

from model_api.app import build_cv_analysis_response_payload
from model_api.artifacts import load_manifest, verify_runtime_artifacts
from model_api.config import (
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
    ArtifactPaths,
    OpenRouterConfig,
    REQUIRED_RUNTIME_ARTIFACT_IDS,
    RuntimeConfig,
)
from model_api.contracts import load_backend_openapi_contract, load_prisma_cv_analysis_contract
from model_api.custom_objects import (
    PHASE25_CUSTOM_OBJECT_NAMES,
    PHASE25_REGISTERED_CUSTOM_OBJECT_NAMES,
    CustomObjectRegistrationError,
    get_phase25_custom_objects,
    load_phase25_keras_model,
    register_phase25_custom_objects,
)
from model_api.errors import (
    ContractValidationError,
    FeatureBuildError,
    InferenceTimeoutError,
    ModelLoadError,
    ModelNotReadyError,
    UnsupportedArtifactVersionError,
)
from model_api.features import (
    E5_JOB_PREFIX,
    E5_MODEL_NAME,
    E5_PROFILE_PREFIX,
    PHASE25_FEATURE_ORDER,
    FeatureBuilder,
    FeatureVector,
    TensorFlowFeatureConfig,
    assert_phase25_feature_order,
    build_candidate_raw_feature_map,
    build_ordered_feature_vector,
    normalize_feature_vector,
)
from model_api.inference import InferenceService, RuntimeState, ScoreCalibrationPolicy
from model_api.schemas import (
    BACKEND_CV_ANALYSIS_SCHEMA_VERSION,
    MODEL_CORE_CANDIDATE_RERANKING_REQUEST_VERSION,
    MODEL_CORE_CANDIDATE_RERANKING_SCHEMA_VERSION,
    MODEL_CORE_CV_ANALYSIS_SCHEMA_VERSION,
    CandidateBackendMetadata,
    CandidateJobInput,
    CandidateRerankingCoreRequest,
    CandidateScoringInput,
    CvAnalysisModelCoreRequest,
    ModelArtifactIdentity,
    ModelIdentity,
    RankingPolicy,
    SanitizedProfileInput,
    parse_candidate_reranking_core_request,
    parse_cv_analysis_model_core_request,
)
from model_api.validators import FORBIDDEN_MODEL_CORE_FIELDS, validate_model_core_payload


class Phase26LayoutTest(unittest.TestCase):
    def test_runtime_artifact_paths_match_phase_25_exports_and_backend_refs(self) -> None:
        paths = ArtifactPaths.from_env({})
        self.assertEqual(paths.model_path, Path("artifacts/phase_25_tensorflow_training_delivery/export/selected_jobfit_tf_phase25.keras"))
        self.assertEqual(paths.tensorflow_feature_config_path.name, "tensorflow_feature_config.json")
        self.assertEqual(paths.openapi_path, Path("references/docs/generated/openapi.json"))
        self.assertEqual(paths.prisma_schema_path, Path("references/prisma/schema.prisma"))

    def test_openrouter_config_is_openai_compatible_and_disabled_by_default(self) -> None:
        config = OpenRouterConfig.from_env({})
        self.assertEqual(config.base_url, DEFAULT_OPENROUTER_BASE_URL)
        self.assertEqual(config.chat_completions_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(config.model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(config.models_url, "https://openrouter.ai/models")
        self.assertFalse(config.enabled)
        enabled = OpenRouterConfig.from_env({"MODEL_API_ENABLE_GENAI_WRAPPER": "true", "OPENROUTER_MODEL": "openai/gpt-4o-mini"})
        self.assertTrue(enabled.enabled)
        self.assertEqual(enabled.model, "openai/gpt-4o-mini")

    def test_backend_openapi_contract_is_cv_analysis_v2(self) -> None:
        contract = load_backend_openapi_contract()
        self.assertEqual(contract.cv_analyzer_path, "/api/v1/ai/cv-analyzer")
        self.assertEqual(contract.analyze_cv_language_enum, ("id", "en"))
        self.assertEqual(contract.analyze_cv_input_mode_enum, ("UPLOAD", "REFERENCE"))
        self.assertEqual(contract.analyze_cv_compare_source_enum, ("BOOKMARK", "JOB_SEARCH", "DIRECT_JOB_DETAIL"))
        self.assertEqual(contract.cv_analysis_schema_version, BACKEND_CV_ANALYSIS_SCHEMA_VERSION)
        self.assertEqual(contract.cv_analysis_job_recommendations_max_items, 5)
        self.assertIn("topActionables", contract.backend_response_required_fields)
        self.assertIn("jobRecommendations", contract.backend_response_required_fields)

    def test_prisma_contract_matches_backend_persistence_enums(self) -> None:
        contract = load_prisma_cv_analysis_contract()
        self.assertEqual(contract.analysis_language_enum, ("ID", "EN"))
        self.assertEqual(contract.cv_input_mode_enum, ("UPLOAD", "REFERENCE"))
        self.assertEqual(contract.cv_compare_source_enum, ("BOOKMARK", "JOB_SEARCH", "DIRECT_JOB_DETAIL"))
        self.assertEqual(contract.job_recommendation_match_level_enum, ("STRONG", "GOOD", "STRETCH"))
        self.assertIn("jobRecommendations", contract.cv_analysis_result_json_fields)
        self.assertIn("nextSteps", contract.job_recommendation_item_json_fields)

    def test_model_core_request_keeps_backend_db_and_wrapper_boundaries(self) -> None:
        request = CvAnalysisModelCoreRequest(
            requestId="req_1",
            inputVersion="cv-analyzer-v1",
            language="id",
            inputMode="UPLOAD",
            compareSource="JOB_SEARCH",
            profile=SanitizedProfileInput(cvText="backend REST API", normalizedSkills=("typescript",)),
            jobCandidates=(CandidateJobInput(jobId="job-1", titleText="Backend Developer", requiredSkills=("typescript",)),),
        )
        self.assertEqual(request.jobCandidates[0].jobId, "job-1")
        self.assertEqual(request.jobCandidates[0].model_scoring_input.requiredSkills, ("typescript",))
        self.assertEqual(MODEL_CORE_CANDIDATE_RERANKING_REQUEST_VERSION, "model-core-candidate-reranking-request-v1")

    def test_strict_cv_analysis_request_schema_accepts_separated_scoring_and_metadata(self) -> None:
        request = parse_cv_analysis_model_core_request(
            {
                "requestId": "req_cv_1",
                "inputVersion": "cv-analyzer-v1",
                "language": "id",
                "inputMode": "UPLOAD",
                "compareSource": "JOB_SEARCH",
                "profile": {
                    "cvText": "Membangun REST API dan PostgreSQL",
                    "normalizedSkills": ["typescript", "postgresql", "rest api"],
                    "targetRoles": ["Backend Developer"],
                },
                "rankingPolicy": {
                    "maxRecommendations": 5,
                    "requireCandidateJobIds": True,
                    "deduplicateByJobId": True,
                    "backendOwnsHydration": True,
                },
                "jobCandidates": [
                    {
                        "jobId": "job-1",
                        "scoringInput": {
                            "titleText": "Backend Developer",
                            "requirementSummary": "Build REST APIs",
                            "requiredSkills": ["typescript", "postgresql"],
                            "roleFamily": "backend",
                        },
                        "backendMetadata": {
                            "title": "Backend Developer",
                            "companyName": "Nusantara Tech",
                            "location": {"display": "Jakarta"},
                            "workType": "REMOTE",
                            "experienceLevel": "ENTRY_LEVEL",
                        },
                    }
                ],
            }
        )
        self.assertEqual(request.language, "id")
        self.assertIsInstance(request.rankingPolicy, RankingPolicy)
        self.assertIsInstance(request.jobCandidates[0].scoringInput, CandidateScoringInput)
        self.assertIsInstance(request.jobCandidates[0].backendMetadata, CandidateBackendMetadata)
        self.assertEqual(request.jobCandidates[0].model_scoring_input.requiredSkills, ("typescript", "postgresql"))
        self.assertEqual(request.jobCandidates[0].backendMetadata.title, "Backend Developer")

    def test_strict_candidate_reranking_request_schema_accepts_phase25_fixture_shape(self) -> None:
        fixture = json.loads(
            Path("artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_fixtures.json").read_text(
                encoding="utf-8"
            )
        )["positive"]["candidateRerankingCoreRequest"]
        request = parse_candidate_reranking_core_request(fixture)
        self.assertIsInstance(request, CandidateRerankingCoreRequest)
        self.assertEqual(request.schemaVersion, MODEL_CORE_CANDIDATE_RERANKING_REQUEST_VERSION)
        self.assertEqual(request.language, "en")
        self.assertEqual(request.maxRecommendations, 5)
        self.assertEqual(len(request.jobCandidates), 3)
        self.assertEqual(request.jobCandidates[0].model_scoring_input.requirementCoverage, 1.0)
        self.assertEqual(request.jobCandidates[0].model_scoring_input.semanticSimilarity, 0.92)

    def test_strict_request_schemas_reject_non_object_root_payloads(self) -> None:
        with self.assertRaises(ContractValidationError) as cv_context:
            parse_cv_analysis_model_core_request([])
        self.assertEqual(cv_context.exception.errors, ["$ must be object"])

        with self.assertRaises(ContractValidationError) as reranking_context:
            parse_candidate_reranking_core_request([])
        self.assertEqual(reranking_context.exception.errors, ["$ must be object"])

    def test_strict_request_schemas_reject_invalid_language_unknown_fields_and_duplicates(self) -> None:
        valid_candidate = {
            "jobId": "job-1",
            "scoringInput": {"requiredSkills": ["typescript"], "roleFamily": "backend"},
        }
        with self.assertRaises(ContractValidationError) as language_context:
            parse_cv_analysis_model_core_request(
                {
                    "requestId": "req_cv_bad_lang",
                    "inputVersion": "cv-analyzer-v1",
                    "language": "jp",
                    "inputMode": "UPLOAD",
                    "compareSource": "JOB_SEARCH",
                    "profile": {"normalizedSkills": ["typescript"]},
                    "jobCandidates": [valid_candidate],
                }
            )
        self.assertIn("$.language must be one of ['en', 'id']", language_context.exception.errors)

        with self.assertRaises(ContractValidationError) as unknown_context:
            parse_cv_analysis_model_core_request(
                {
                    "requestId": "req_cv_unknown",
                    "inputVersion": "cv-analyzer-v1",
                    "language": "id",
                    "inputMode": "UPLOAD",
                    "compareSource": "JOB_SEARCH",
                    "profile": {"normalizedSkills": ["typescript"]},
                    "jobCandidates": [{**valid_candidate, "title": "must stay backendMetadata.title"}],
                }
            )
        self.assertIn("$.jobCandidates[0].title is not allowed", unknown_context.exception.errors)

        with self.assertRaises(ContractValidationError) as duplicate_context:
            parse_candidate_reranking_core_request(
                {
                    "requestId": "req_rerank_duplicate",
                    "schemaVersion": MODEL_CORE_CANDIDATE_RERANKING_REQUEST_VERSION,
                    "candidateSetId": "candidate-set-1",
                    "language": "en",
                    "profileFeatures": {"normalizedSkills": ["typescript"]},
                    "jobCandidates": [valid_candidate, valid_candidate],
                }
            )
        self.assertIn("$.jobCandidates[1].jobId duplicate: 'job-1'", duplicate_context.exception.errors)

    def test_strict_request_schemas_reject_empty_candidates_and_missing_signals(self) -> None:
        with self.assertRaises(ContractValidationError) as error_context:
            parse_cv_analysis_model_core_request(
                {
                    "requestId": "req_cv_missing_inputs",
                    "inputVersion": "cv-analyzer-v1",
                    "language": "id",
                    "inputMode": "UPLOAD",
                    "compareSource": "JOB_SEARCH",
                    "profile": {},
                    "jobCandidates": [],
                }
            )
        self.assertIn("$.profile must include sanitized cv/profile text or extracted signals", error_context.exception.errors)
        self.assertIn("$.jobCandidates must contain at least 1 candidate", error_context.exception.errors)

    def test_manifest_exposes_required_runtime_artifact_ids(self) -> None:
        manifest = load_manifest(ArtifactPaths.from_env({}))
        required = manifest.required_runtime_entries(REQUIRED_RUNTIME_ARTIFACT_IDS)
        self.assertEqual(len(required), len(REQUIRED_RUNTIME_ARTIFACT_IDS))
        self.assertTrue(all(entry.required_for_inference for entry in required))
        self.assertIn("final_keras_model", {entry.artifact_id for entry in required})
        self.assertIn("tensorflow_artifact_export", {entry.artifact_id for entry in required})

    def test_runtime_artifact_verification_checks_hashes_and_sizes(self) -> None:
        report = verify_runtime_artifacts(
            ArtifactPaths.from_env({}),
            artifact_ids=("score_calibration", "feature_config"),
        )
        calibration_path = Path("artifacts/phase_25_tensorflow_training_delivery/score_calibration.json")
        self.assertEqual(report.runtime_artifact_ids, ("score_calibration", "feature_config"))
        self.assertEqual(report.artifact_sizes["score_calibration"], calibration_path.stat().st_size)
        self.assertEqual(len(report.artifact_hashes["feature_config"]), 64)

    def test_runtime_artifact_verification_fails_on_stale_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            artifact = root / "feature_config.json"
            artifact.write_text("{}", encoding="utf-8")
            manifest = {
                "schema_version": "phase-25-artifact-manifest-v1",
                "phase_id": "phase_25_tensorflow_training_delivery",
                "artifacts": [
                    {
                        "artifact_id": "feature_config",
                        "path": str(artifact),
                        "required_for_inference": True,
                        "sha256": hashlib.sha256(b"stale").hexdigest(),
                        "size_bytes": artifact.stat().st_size,
                    }
                ],
            }
            manifest_path = root / "artifact_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            paths = ArtifactPaths.from_env(
                {
                    "MODEL_API_ARTIFACT_ROOT": str(root),
                    "MODEL_API_ARTIFACT_MANIFEST": str(manifest_path),
                    "MODEL_API_FEATURE_CONFIG": str(artifact),
                }
            )
            with self.assertRaisesRegex(Exception, "sha256 mismatch"):
                verify_runtime_artifacts(paths, artifact_ids=("feature_config",))

    def test_runtime_artifact_verification_fails_on_missing_hash_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            artifact = root / "feature_config.json"
            artifact.write_text("{}", encoding="utf-8")
            manifest = {
                "schema_version": "phase-25-artifact-manifest-v1",
                "phase_id": "phase_25_tensorflow_training_delivery",
                "artifacts": [
                    {
                        "artifact_id": "feature_config",
                        "path": str(artifact),
                        "required_for_inference": True,
                        "sha256": None,
                        "size_bytes": artifact.stat().st_size,
                    }
                ],
            }
            manifest_path = root / "artifact_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            paths = ArtifactPaths.from_env(
                {
                    "MODEL_API_ARTIFACT_ROOT": str(root),
                    "MODEL_API_ARTIFACT_MANIFEST": str(manifest_path),
                    "MODEL_API_FEATURE_CONFIG": str(artifact),
                }
            )
            with self.assertRaisesRegex(Exception, "missing sha256"):
                verify_runtime_artifacts(paths, artifact_ids=("feature_config",))

    def test_runtime_artifact_verification_rejects_unsupported_manifest_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "artifact_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "phase-24-artifact-manifest-v1",
                        "phase_id": "phase_24_final_gate",
                        "artifacts": [],
                    }
                ),
                encoding="utf-8",
            )
            paths = ArtifactPaths.from_env(
                {"MODEL_API_ARTIFACT_ROOT": str(root), "MODEL_API_ARTIFACT_MANIFEST": str(manifest_path)}
            )
            with self.assertRaises(UnsupportedArtifactVersionError) as error_context:
                load_manifest(paths)
            self.assertIn("artifact_manifest.json schema_version must be", str(error_context.exception))
            self.assertEqual(error_context.exception.code, "unsupported_artifact_version")

    def test_feature_layout_is_phase_25_order_with_e5_prefixes(self) -> None:
        self.assertEqual(
            PHASE25_FEATURE_ORDER,
            (
                "e5_cosine",
                "skill_overlap",
                "requirement_coverage",
                "role_match",
                "experience_match",
                "experience_gap_years_clipped",
            ),
        )
        self.assertEqual(E5_PROFILE_PREFIX, "query:")
        self.assertEqual(E5_JOB_PREFIX, "passage:")
        assert_phase25_feature_order(PHASE25_FEATURE_ORDER)
        with self.assertRaises(FeatureBuildError):
            assert_phase25_feature_order(tuple(reversed(PHASE25_FEATURE_ORDER)))

    def test_feature_vector_build_and_normalize(self) -> None:
        raw = {name: float(index + 1) for index, name in enumerate(PHASE25_FEATURE_ORDER)}
        vector = build_ordered_feature_vector("job-1", raw)
        self.assertEqual(vector.as_model_row(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        normalized = normalize_feature_vector(
            vector,
            mean={name: 1.0 for name in PHASE25_FEATURE_ORDER},
            std={name: 2.0 for name in PHASE25_FEATURE_ORDER},
        )
        self.assertEqual(normalized.as_model_row(), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    def test_tensorflow_feature_config_loads_phase25_normalization_contract(self) -> None:
        config = TensorFlowFeatureConfig.from_path(
            Path("artifacts/phase_25_tensorflow_training_delivery/tensorflow_feature_config.json")
        )
        self.assertEqual(config.approved_features, PHASE25_FEATURE_ORDER)
        self.assertEqual(config.schema_version, "phase-25-tensorflow-architecture-v1")
        self.assertGreater(config.std["e5_cosine"], 0.0)
        self.assertIn("skill_overlap", config.mean)

    def test_feature_builder_uses_e5_prefixes_and_phase25_raw_feature_order(self) -> None:
        class FakeE5Backend:
            model_name = E5_MODEL_NAME
            backend_name = "sentence-transformers"

            def __init__(self) -> None:
                self.texts: list[str] = []

            def encode(self, texts):
                self.texts.extend(texts)
                return ([1.0, 0.0], [0.8, 0.6])

        backend = FakeE5Backend()
        profile = SanitizedProfileInput(
            cvText="Built REST API with PostgreSQL",
            normalizedSkills=("typescript", "postgresql"),
            targetRoles=("Backend Developer",),
            roleFamily="backend",
            experienceYears=1.5,
            experienceBand="junior",
        )
        candidate = CandidateJobInput(
            jobId="job-1",
            scoringInput=CandidateScoringInput(
                titleText="Backend Developer",
                requirementSummary="Build APIs",
                requiredSkills=("typescript", "docker"),
                roleFamily="backend",
                experienceLevel="ENTRY_LEVEL",
            ),
        )
        raw = build_candidate_raw_feature_map(profile, candidate, backend, environment="staging")
        vector = build_ordered_feature_vector(candidate.jobId, raw)
        self.assertEqual(tuple(raw), PHASE25_FEATURE_ORDER)
        self.assertTrue(backend.texts[0].startswith(f"{E5_PROFILE_PREFIX} "))
        self.assertTrue(backend.texts[1].startswith(f"{E5_JOB_PREFIX} "))
        self.assertEqual(vector.as_model_row(), [0.8, 1.0 / 3.0, 0.5, 1.0, 0.75, 1.5])

    def test_feature_builder_normalizes_with_tensorflow_feature_config(self) -> None:
        class FakeE5Backend:
            model_name = E5_MODEL_NAME
            backend_name = "sentence-transformers"

            def encode(self, texts):
                return ([1.0, 0.0], [1.0, 0.0])

        config = TensorFlowFeatureConfig(
            approved_features=PHASE25_FEATURE_ORDER,
            mean={name: 0.0 for name in PHASE25_FEATURE_ORDER},
            std={name: 1.0 for name in PHASE25_FEATURE_ORDER},
        )
        builder = FeatureBuilder(config, FakeE5Backend(), environment="production")
        result = builder.build_candidate(
            SanitizedProfileInput(
                profileText="Backend TypeScript developer",
                normalizedSkills=("typescript",),
                roleFamily="backend",
                experienceBand="entry",
            ),
            CandidateJobInput(
                jobId="job-2",
                scoringInput=CandidateScoringInput(
                    titleText="Backend Engineer",
                    requiredSkills=("typescript",),
                    roleFamily="backend",
                    experienceLevel="ENTRY_LEVEL",
                ),
            ),
        )
        self.assertEqual(result.raw.as_model_row(), result.normalized.as_model_row())
        self.assertEqual(result.as_model_row(), [1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

    def test_feature_builder_rejects_non_e5_or_fallback_embedding_backend_in_staging(self) -> None:
        class LocalHashBackend:
            model_name = E5_MODEL_NAME
            backend_name = "local-hash-fallback"

            def encode(self, texts):
                return ([1.0], [1.0])

        class WrongModelBackend(LocalHashBackend):
            model_name = "local/tfidf"
            backend_name = "sentence-transformers"

        profile = SanitizedProfileInput(profileText="Backend", normalizedSkills=("typescript",))
        candidate = CandidateJobInput(
            jobId="job-3",
            scoringInput=CandidateScoringInput(titleText="Backend", requiredSkills=("typescript",)),
        )
        with self.assertRaises(FeatureBuildError):
            build_candidate_raw_feature_map(profile, candidate, LocalHashBackend(), environment="staging")
        with self.assertRaises(FeatureBuildError):
            build_candidate_raw_feature_map(profile, candidate, WrongModelBackend(), environment="local")

    def test_contract_boundaries_reject_backend_wrapper_owned_fields(self) -> None:
        self.assertIn("topActionables", FORBIDDEN_MODEL_CORE_FIELDS)
        with self.assertRaises(ContractValidationError) as error_context:
            validate_model_core_payload(
                {
                    "schemaVersion": MODEL_CORE_CV_ANALYSIS_SCHEMA_VERSION,
                    "language": "id",
                    "topActionables": [],
                    "jobFitAlignment": {"score": 80},
                    "overallImpression": {"score": 80, "summary": "ok"},
                    "candidateReranking": {"recommendations": [{"jobId": "job-1", "matchScore": 75, "title": "Backend"}]},
                },
                candidate_ids={"job-1"},
            )
        self.assertIn("$.topActionables is wrapper/backend-owned", error_context.exception.errors)
        self.assertIn("$.candidateReranking.recommendations[0].title is wrapper/backend-owned", error_context.exception.errors)

    def test_contract_boundaries_reject_backend_cv_analysis_v2_as_model_core_output(self) -> None:
        with self.assertRaises(ContractValidationError) as error_context:
            validate_model_core_payload(
                {"schemaVersion": BACKEND_CV_ANALYSIS_SCHEMA_VERSION, "language": "id", "jobFitAlignment": {"score": 78}},
                candidate_ids=set(),
            )
        self.assertIn("schemaVersion must be model-core schema", str(error_context.exception))

    def test_contract_validators_enforce_phase25_response_handoff_rules(self) -> None:
        valid_payload = {
            "schemaVersion": MODEL_CORE_CV_ANALYSIS_SCHEMA_VERSION,
            "language": "en",
            "jobFitAlignment": {"score": 80},
            "atsFriendliness": {"score": 72},
            "overallImpression": {"score": 76, "summary": "Evidence only."},
            "candidateReranking": {
                "schemaVersion": "model-core-candidate-reranking-v1",
                "language": "en",
                "recommendations": [
                    {"jobId": "job-1", "matchScore": 80, "matchLevel": "good"},
                    {"jobId": "job-2", "matchScore": 61, "matchLevel": "stretch"},
                ],
            },
        }
        validate_model_core_payload(valid_payload, {"job-1", "job-2"})

        invalid_payload = {
            **valid_payload,
            "id": "backend-analysis-id",
            "language": "EN",
            "jobFitAlignment": {"score": True},
            "atsFriendliness": {"score": 101},
            "overallImpression": {"score": -1, "summary": "bad"},
            "candidateReranking": {
                "schemaVersion": "wrong-version",
                "language": "id",
                "recommendations": [
                    {"jobId": "job-1", "matchScore": 80, "title": "Backend Developer"},
                    {"jobId": "job-1", "matchScore": 61},
                    {"jobId": "job-3", "matchScore": 50},
                    {"jobId": "job-4", "matchScore": 49},
                    {"jobId": "job-5", "matchScore": 48},
                    {"jobId": "job-6", "matchScore": 47},
                ],
            },
        }
        with self.assertRaises(ContractValidationError) as error_context:
            validate_model_core_payload(invalid_payload, {"job-1", "job-2"})
        errors = error_context.exception.errors
        self.assertIn("$.id is wrapper/backend-owned", errors)
        self.assertIn("language must be one of ['en', 'id']", errors)
        self.assertIn("$.jobFitAlignment.score must be integer 0-100; actual=True", errors)
        self.assertIn("$.atsFriendliness.score must be integer 0-100; actual=101", errors)
        self.assertIn("$.overallImpression.score must be integer 0-100; actual=-1", errors)
        self.assertIn("$.candidateReranking.schemaVersion must be 'model-core-candidate-reranking-v1'", errors)
        self.assertIn("$.candidateReranking.recommendations max items 5; actual=6", errors)
        self.assertIn("$.candidateReranking.recommendations[0].title is wrapper/backend-owned", errors)
        self.assertIn("$.candidateReranking.recommendations[1].jobId duplicate: 'job-1'", errors)
        self.assertIn("$.candidateReranking.recommendations[2].jobId not in candidate set: 'job-3'", errors)

    def test_contract_validators_reject_malformed_recommendation_payloads(self) -> None:
        with self.assertRaises(ContractValidationError) as non_array_context:
            validate_model_core_payload(
                {
                    "schemaVersion": MODEL_CORE_CANDIDATE_RERANKING_SCHEMA_VERSION,
                    "language": "id",
                    "recommendations": "job-1",
                },
                {"job-1"},
            )
        self.assertIn("recommendations must be array", non_array_context.exception.errors)

        with self.assertRaises(ContractValidationError) as item_context:
            validate_model_core_payload(
                {
                    "schemaVersion": MODEL_CORE_CANDIDATE_RERANKING_SCHEMA_VERSION,
                    "language": "id",
                    "recommendations": ["job-1"],
                },
                {"job-1"},
            )
        self.assertIn("recommendations[0] must be object", item_context.exception.errors)

    def test_runtime_layout_names_are_importable_without_tensorflow_or_fastapi(self) -> None:
        config = RuntimeConfig.from_env({"MODEL_API_ENV": "test"})
        self.assertEqual(config.environment, "test")
        self.assertEqual(config.max_recommendations, 5)
        self.assertEqual(config.openrouter.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(
            PHASE25_CUSTOM_OBJECT_NAMES,
            (
                "CosineInteractionLayer",
                "WeightedHuberLoss",
                "ProductionGateCallback",
                "HighRecallCalibrationLayer",
            ),
        )
        self.assertEqual(
            PHASE25_REGISTERED_CUSTOM_OBJECT_NAMES,
            (
                "BisakerjaPhase25>CosineInteractionLayer",
                "BisakerjaPhase25>WeightedHuberLoss",
                "BisakerjaPhase25>ProductionGateCallback",
                "BisakerjaPhase25>HighRecallCalibrationLayer",
            ),
        )

    def test_phase25_keras_file_references_registered_custom_objects(self) -> None:
        model_path = Path("artifacts/phase_25_tensorflow_training_delivery/export/selected_jobfit_tf_phase25.keras")
        with zipfile.ZipFile(model_path) as archive:
            config = json.loads(archive.read("config.json"))

        seen: set[str] = set()

        def walk(value: object) -> None:
            if isinstance(value, dict):
                registered_name = value.get("registered_name")
                if isinstance(registered_name, str) and registered_name.startswith("BisakerjaPhase25>"):
                    seen.add(registered_name)
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(config)
        self.assertIn("BisakerjaPhase25>CosineInteractionLayer", seen)
        self.assertIn("BisakerjaPhase25>HighRecallCalibrationLayer", seen)

    def test_phase25_custom_object_registration_is_lazy_and_stable(self) -> None:
        try:
            names = register_phase25_custom_objects()
        except CustomObjectRegistrationError as exc:
            self.assertIn("TensorFlow/Keras dependency missing", str(exc))
            return

        self.assertEqual(names, PHASE25_CUSTOM_OBJECT_NAMES)
        custom_objects = get_phase25_custom_objects()
        self.assertEqual(tuple(custom_objects), PHASE25_CUSTOM_OBJECT_NAMES)
        layer_config = custom_objects["CosineInteractionLayer"]().get_config()
        self.assertEqual(layer_config["cosine_index"], 0)
        self.assertEqual(layer_config["interaction_indices"], [1, 2, 3, 4])
        self.assertTrue(layer_config["include_original"])

    def test_phase25_model_loader_uses_compile_false_when_tensorflow_available(self) -> None:
        try:
            register_phase25_custom_objects()
        except CustomObjectRegistrationError as exc:
            self.skipTest(str(exc))

        model = load_phase25_keras_model(
            Path("artifacts/phase_25_tensorflow_training_delivery/export/selected_jobfit_tf_phase25.keras")
        )
        self.assertEqual(model.name, "bisakerja_jobfit_tf_functional_custom_v1_calibrated")
        self.assertFalse(hasattr(model, "optimizer") and model.optimizer is not None)

    def test_real_phase25_keras_artifact_matches_inference_smoke_fixture_predictions(self) -> None:
        if importlib.util.find_spec("tensorflow") is None:
            self.skipTest("TensorFlow dependency missing")
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("NumPy dependency missing")

        import numpy as np

        fixture_path = Path("artifacts/phase_25_tensorflow_training_delivery/export/inference_smoke_fixture.json")
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        training_features = np.load(
            "artifacts/phase_25_tensorflow_training_delivery/tensorflow_training_features_v1.npz",
            allow_pickle=True,
        )
        row_by_pair_id = {str(pair_id): index for index, pair_id in enumerate(training_features["pair_id"])}
        fixture_predictions = fixture["sample_predictions"]
        fixture_pair_ids = [item["pair_id"] for item in fixture_predictions]
        rows = [training_features["X_scaled"][row_by_pair_id[pair_id]] for pair_id in fixture_pair_ids]

        model = load_phase25_keras_model(
            Path("artifacts/phase_25_tensorflow_training_delivery/export/selected_jobfit_tf_phase25.keras")
        )
        predictions = model.predict(rows, verbose=0).tolist()
        scores = [_coerced_score[0] for _coerced_score in predictions]

        self.assertEqual(fixture["status"], "complete")
        self.assertEqual(fixture["input_shape"], [None, 6])
        self.assertEqual(fixture["output_shape"], [None, 1])
        self.assertEqual(len(scores), len(fixture_predictions))
        for score, expected in zip(scores, fixture_predictions, strict=True):
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertAlmostEqual(score, expected["score_0_1"], delta=0.0005)
            self.assertAlmostEqual(score * 100.0, expected["score_0_100"], delta=0.05)
        self.assertAlmostEqual(min(scores), fixture["score_bounds"]["min_0_1"], delta=0.0005)
        self.assertAlmostEqual(max(scores), fixture["score_bounds"]["max_0_1"], delta=0.0005)

    def test_phase25_handoff_fixtures_match_recorded_contract_validation_behavior(self) -> None:
        fixtures = json.loads(
            Path("artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_fixtures.json").read_text(
                encoding="utf-8"
            )
        )
        validation = json.loads(
            Path("artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_validation.json").read_text(
                encoding="utf-8"
            )
        )
        candidate_ids = {
            candidate["jobId"]
            for candidate in fixtures["positive"]["candidateRerankingCoreRequest"]["jobCandidates"]
        }

        validate_model_core_payload(fixtures["positive"]["candidateRerankingCoreOutput"], candidate_ids)
        validate_model_core_payload(fixtures["positive"]["cvAnalysisCoreOutput"], candidate_ids)
        self.assertEqual(validation["positive_validation_errors"]["candidate_reranking"], [])
        self.assertEqual(validation["positive_validation_errors"]["cv_core_output"], [])

        for fixture_name, expected_errors in validation["negative_validation_errors"].items():
            with self.subTest(fixture_name=fixture_name):
                with self.assertRaises(ContractValidationError) as error_context:
                    validate_model_core_payload(fixtures["negative"][fixture_name], candidate_ids)
                errors = error_context.exception.errors
                normalized_errors = {error.removeprefix("$.") for error in errors}
                for expected_error in expected_errors:
                    self.assertIn(expected_error.removeprefix("$."), normalized_errors)

    def test_startup_model_loader_builds_identity_and_loads_once(self) -> None:
        class FakeModel:
            name = "fake_runtime_model_name"

            def predict(self, values, verbose=0):
                return values

        paths = ArtifactPaths.from_env({})
        report = verify_runtime_artifacts(paths, artifact_ids=("final_keras_model", "model_card"))
        calls: list[Path] = []

        def fake_loader(path: Path) -> FakeModel:
            calls.append(path)
            return FakeModel()

        service = InferenceService(model_loader=fake_loader)
        state = service.load_once(paths, report)
        self.assertTrue(state.ready)
        self.assertEqual(service.load_count, 1)
        self.assertEqual(calls, [paths.model_path])
        self.assertEqual(state.model_identity.name, "bisakerja_jobfit_tf_functional_custom_v1")
        self.assertEqual(state.model_identity.version, "jobfit_tf_phase25_gradient_tape_v1")
        self.assertEqual(state.model_identity.artifact_sha256, report.artifact_hashes["final_keras_model"])
        self.assertIsNotNone(state.loaded_at)

        second_state = service.load_once(paths, report)
        self.assertIs(second_state, state)
        self.assertEqual(service.load_count, 1)

    def test_batch_inference_scores_sorts_clamps_calibrates_and_limits_recommendations(self) -> None:
        class FakeModel:
            name = "fake_runtime_model_name"

            def predict(self, values, verbose=0):
                self.values = values
                return [[0.764], [1.2], [-0.1], [0.421], [0.61], [0.9]]

        calibration = ScoreCalibrationPolicy.from_mapping(
            {
                "tables": [
                    {
                        "output": "recommendations[].matchScore",
                        "bucket": "61-80",
                        "min": 61,
                        "max": 80,
                        "mean_signed_error": 3.0,
                    }
                ]
            }
        )
        vectors = tuple(FeatureVector(f"job-{index}", (float(index),) * 6) for index in range(6))
        service = InferenceService(model=FakeModel())

        recommendations = service.predict_recommendations(vectors, max_recommendations=5, calibration_policy=calibration)

        self.assertEqual([item.jobId for item in recommendations], ["job-1", "job-5", "job-0", "job-4", "job-3"])
        self.assertEqual([item.matchScore for item in recommendations], [100, 90, 73, 58, 42])
        self.assertEqual([item.matchLevel for item in recommendations], ["strong", "strong", "good", "stretch", "stretch"])
        self.assertNotIn("job-2", [item.jobId for item in recommendations])

    def test_cv_analysis_response_payload_returns_model_core_only(self) -> None:
        class FakeModel:
            name = "fake_runtime_model_name"

            def predict(self, values, verbose=0):
                self.values = values
                return [[0.82], [0.64]]

        class FakeE5Backend:
            model_name = E5_MODEL_NAME
            backend_name = "sentence-transformers"

            def encode(self, texts):
                return ([1.0, 0.0], [1.0, 0.0])

        request = CvAnalysisModelCoreRequest(
            requestId="req_cv_endpoint",
            inputVersion="cv-analyzer-v1",
            language="en",
            inputMode="UPLOAD",
            compareSource="JOB_SEARCH",
            profile=SanitizedProfileInput(
                profileText="Backend TypeScript developer",
                normalizedSkills=("typescript",),
                targetRoles=("Backend Developer",),
                detectedCvSectionNames=("experience", "skills"),
            ),
            jobCandidates=(
                CandidateJobInput(
                    jobId="job-1",
                    scoringInput=CandidateScoringInput(
                        titleText="Backend Developer",
                        requiredSkills=("typescript", "postgresql"),
                        roleFamily="backend",
                    ),
                ),
                CandidateJobInput(
                    jobId="job-2",
                    scoringInput=CandidateScoringInput(
                        titleText="Frontend Developer",
                        requiredSkills=("react",),
                        roleFamily="frontend",
                    ),
                ),
            ),
        )
        config = TensorFlowFeatureConfig(
            approved_features=PHASE25_FEATURE_ORDER,
            mean={name: 0.0 for name in PHASE25_FEATURE_ORDER},
            std={name: 1.0 for name in PHASE25_FEATURE_ORDER},
        )
        service = InferenceService(
            model=FakeModel(),
            state=RuntimeState(
                ready=True,
                model_identity=ModelIdentity(
                    name="fake-runtime",
                    version="v1",
                    artifact=ModelArtifactIdentity(path="model.keras", sha256="abc"),
                ),
                message="ready",
            ),
        )

        payload = build_cv_analysis_response_payload(
            request,
            service=service,
            feature_config=config,
            calibration_policy=ScoreCalibrationPolicy(),
            embedding_backend=FakeE5Backend(),
            environment="staging",
        )

        self.assertEqual(payload["schemaVersion"], MODEL_CORE_CV_ANALYSIS_SCHEMA_VERSION)
        self.assertEqual(payload["jobFitAlignment"]["score"], 82)
        self.assertEqual(payload["atsFriendliness"]["score"], 85)
        self.assertEqual(payload["candidateReranking"]["recommendations"][0]["jobId"], "job-1")
        self.assertNotIn("topActionables", payload)
        self.assertNotIn("sectionReviews", payload)
        validate_model_core_payload(payload, {"job-1", "job-2"})

    def test_batch_inference_rejects_prediction_count_mismatch(self) -> None:
        class FakeModel:
            name = "fake_runtime_model_name"

            def predict(self, values, verbose=0):
                return [[0.5]]

        service = InferenceService(model=FakeModel())
        with self.assertRaisesRegex(Exception, "prediction count mismatch"):
            service.predict_recommendations(
                (
                    FeatureVector("job-1", (0.0,) * 6),
                    FeatureVector("job-2", (0.0,) * 6),
                )
            )

    def test_batch_inference_wraps_model_failure_and_rejects_timeout(self) -> None:
        class FailingModel:
            name = "fake_runtime_model_name"

            def predict(self, values, verbose=0):
                raise RuntimeError("predict boom")

        service = InferenceService(model=FailingModel())
        with self.assertRaises(ModelLoadError) as model_error_context:
            service.predict_recommendations((FeatureVector("job-1", (0.0,) * 6),))
        self.assertIn("TensorFlow model prediction failed: predict boom", str(model_error_context.exception))

        class NeverCalledModel:
            name = "fake_runtime_model_name"

            def predict(self, values, verbose=0):  # pragma: no cover - timeout occurs before call
                return [[0.5]]

        timeout_service = InferenceService(model=NeverCalledModel())
        with self.assertRaises(InferenceTimeoutError) as timeout_context:
            timeout_service.predict_recommendations((FeatureVector("job-1", (0.0,) * 6),), timeout_ms=0)
        self.assertEqual(timeout_context.exception.code, "inference_timeout")

    def test_startup_model_loader_records_failure_and_inference_is_not_ready(self) -> None:
        paths = ArtifactPaths.from_env({})
        report = verify_runtime_artifacts(paths, artifact_ids=("final_keras_model", "model_card"))

        def failing_loader(_path: Path):
            raise RuntimeError("boom")

        service = InferenceService(model_loader=failing_loader)
        state = service.load_once(paths, report)
        self.assertFalse(state.ready)
        self.assertEqual(state.error_code, "model_load_error")
        self.assertIn("TensorFlow model load failed: boom", state.message)
        with self.assertRaises(ModelNotReadyError) as error_context:
            service.require_ready()
        self.assertIn("TensorFlow model is not ready", str(error_context.exception))


if __name__ == "__main__":
    unittest.main()
