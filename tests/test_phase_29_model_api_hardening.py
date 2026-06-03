from __future__ import annotations

import unittest
import zlib

from model_api.app import (
    _authorize_internal_request,
    build_candidate_reranking_response_payload,
    build_cv_analysis_response_payload,
)
from model_api.config import RuntimeConfig
from model_api.errors import FeatureBuildError
from model_api.features import E5_MODEL_NAME, TensorFlowFeatureConfig, validate_e5_backend
from model_api.inference import InferenceService, RuntimeState, ScoreCalibrationPolicy
from model_api.pdf_parser import ats_score_from_pdf_evidence, parse_pdf_bytes
from model_api.schemas import (
    MODEL_CORE_CANDIDATE_RERANKING_REQUEST_VERSION,
    CandidateJobInput,
    CandidateRerankingCoreRequest,
    CandidateScoringInput,
    CvAnalysisModelCoreRequest,
    ModelArtifactIdentity,
    ModelIdentity,
    SanitizedProfileInput,
)


class FakeModel:
    name = "fake_phase25_model"

    def predict(self, values, verbose=0):
        return [[0.2], [0.9]][: len(values)]


class FakeE5Backend:
    model_name = E5_MODEL_NAME
    backend_name = "sentence-transformers"

    def encode(self, texts):
        return [[1.0, 0.0] if index % 2 == 0 else [1.0, 0.0] for index, _ in enumerate(texts)]


class Phase29ModelApiHardeningTest(unittest.TestCase):
    def feature_config(self) -> TensorFlowFeatureConfig:
        features = (
            "e5_cosine",
            "skill_overlap",
            "requirement_coverage",
            "role_match",
            "experience_match",
            "experience_gap_years_clipped",
        )
        return TensorFlowFeatureConfig(approved_features=features, mean={name: 0.0 for name in features}, std={name: 1.0 for name in features})

    def service(self) -> InferenceService:
        identity = ModelIdentity(
            name="phase25",
            version="test",
            artifact=ModelArtifactIdentity(path="artifacts/test.keras", sha256="0" * 64),
        )
        return InferenceService(
            model=FakeModel(),
            state=RuntimeState(ready=True, model_identity=identity, artifact_manifest_phase="phase_25", message="ready"),
        )

    def candidates(self) -> tuple[CandidateJobInput, CandidateJobInput]:
        return (
            CandidateJobInput(jobId="job-1", scoringInput=CandidateScoringInput(titleText="Frontend", requiredSkills=("react",))),
            CandidateJobInput(jobId="job-2", scoringInput=CandidateScoringInput(titleText="Backend", requiredSkills=("python", "sql"))),
        )

    def test_env_defaults_include_pdf_limits_auth_and_genai_disabled(self) -> None:
        config = RuntimeConfig.from_env({})
        self.assertEqual(config.environment, "local")
        self.assertEqual(config.max_pdf_bytes, 5_000_000)
        self.assertEqual(config.max_pdf_pages, 10)
        self.assertTrue(config.allow_unauthenticated_local)
        self.assertFalse(config.openrouter.enabled)

        prod_config = RuntimeConfig.from_env({"MODEL_API_ENV": "production"})
        with self.assertRaisesRegex(ValueError, "MODEL_API_SERVICE_TOKEN is required"):
            prod_config.validate_security()

    def test_auth_guard_requires_bearer_token_outside_local_bypass(self) -> None:
        config = RuntimeConfig.from_env(
            {"MODEL_API_ENV": "staging", "MODEL_API_SERVICE_TOKEN": "secret", "MODEL_API_ALLOW_UNAUTHENTICATED_LOCAL": "false"}
        )
        self.assertIsNotNone(_authorize_internal_request({}, config))
        self.assertIsNotNone(_authorize_internal_request({"authorization": "Bearer wrong"}, config))
        self.assertIsNone(_authorize_internal_request({"authorization": "Bearer secret"}, config))

    def test_pdf_parser_extracts_text_sections_contact_dates_and_ats_score(self) -> None:
        pdf = b"%PDF-1.4\n1 0 obj<</Type /Page>>stream\n(Summary Backend developer) Tj\n(Skills Python SQL) Tj\n(Email dev@example.com) Tj\n(Experience 2020) Tj\nendstream\n%%EOF"
        evidence = parse_pdf_bytes(pdf, max_bytes=5000, max_pages=5)
        score, issues, fallback = ats_score_from_pdf_evidence(evidence)

        self.assertIn("Backend developer", evidence.text)
        self.assertIn("summary", evidence.section_names)
        self.assertIn("skills", evidence.section_names)
        self.assertTrue(evidence.has_contact_signal)
        self.assertTrue(evidence.has_date_signal)
        self.assertGreaterEqual(score, 85)
        self.assertEqual(issues, ())
        self.assertFalse(fallback)

    def test_pdf_parser_extracts_text_from_flate_compressed_streams(self) -> None:
        stream = zlib.compress(b"BT (Summary Backend developer) Tj (Skills Python SQL) Tj (Experience 2020) Tj ET")
        pdf = b"%PDF-1.4\n1 0 obj<</Type /Page>>endobj\n2 0 obj<</Filter /FlateDecode /Length " + str(len(stream)).encode() + b">>stream\n" + stream + b"\nendstream\nendobj\n%%EOF"
        evidence = parse_pdf_bytes(pdf, max_bytes=5000, max_pages=5)

        self.assertIn("Backend developer", evidence.text)
        self.assertIn("skills", evidence.section_names)
        self.assertIn("experience", evidence.section_names)

    def test_pdf_parser_rejects_magic_bytes_and_scanned_empty_text_without_hallucination(self) -> None:
        bad = parse_pdf_bytes(b"not a pdf", max_bytes=100, max_pages=1)
        self.assertEqual(bad.parse_quality, "failed")
        self.assertEqual(bad.text, "")

        scanned = parse_pdf_bytes(b"%PDF-1.4\n1 0 obj<</Type /Page /XObject 2 0 R>>\n%%EOF", max_bytes=5000, max_pages=5)
        score, issues, fallback = ats_score_from_pdf_evidence(scanned)
        self.assertEqual(scanned.text, "")
        self.assertIn("no extractable PDF text; scanned or image-only CV suspected", issues)
        self.assertLess(score, 50)
        self.assertTrue(fallback)

    def test_top_candidate_evidence_comes_from_top_ranked_candidate_not_first_candidate(self) -> None:
        request = CvAnalysisModelCoreRequest(
            requestId="req-29",
            inputVersion="cv-analyzer-v1",
            language="en",
            inputMode="UPLOAD",
            compareSource="JOB_SEARCH",
            profile=SanitizedProfileInput(cvText="Python SQL backend", normalizedSkills=("python", "sql")),
            jobCandidates=self.candidates(),
        )
        payload = build_cv_analysis_response_payload(
            request,
            service=self.service(),
            feature_config=self.feature_config(),
            calibration_policy=ScoreCalibrationPolicy(),
            embedding_backend=FakeE5Backend(),
            environment="test",
        )

        self.assertEqual(payload["jobFitAlignment"]["score"], 90)
        self.assertEqual(payload["jobFitAlignment"]["matchedSkills"], ["python", "sql"])
        self.assertEqual(payload["candidateReranking"]["recommendations"][0]["jobId"], "job-2")
        self.assertEqual(payload["candidateReranking"]["recommendations"][0]["matchedSkills"], ["python", "sql"])

    def test_candidate_reranking_endpoint_payload_is_bounded_and_candidate_only(self) -> None:
        request = CandidateRerankingCoreRequest(
            requestId="req-rerank-29",
            schemaVersion=MODEL_CORE_CANDIDATE_RERANKING_REQUEST_VERSION,
            candidateSetId="set-1",
            language="en",
            profileFeatures=SanitizedProfileInput(cvText="Python SQL backend", normalizedSkills=("python", "sql")),
            jobCandidates=self.candidates(),
            maxRecommendations=1,
        )
        payload = build_candidate_reranking_response_payload(
            request,
            service=self.service(),
            feature_config=self.feature_config(),
            calibration_policy=ScoreCalibrationPolicy(),
            embedding_backend=FakeE5Backend(),
            environment="test",
        )

        self.assertEqual(payload["schemaVersion"], "model-core-candidate-reranking-v1")
        self.assertEqual(len(payload["recommendations"]), 1)
        self.assertEqual(payload["recommendations"][0]["jobId"], "job-2")
        self.assertLessEqual(payload["recommendations"][0]["matchScore"], 100)
        self.assertNotIn("title", str(payload))
        self.assertNotIn("companyName", str(payload))

    def test_fallback_embedding_backend_blocked_in_staging_and_production(self) -> None:
        class FallbackBackend(FakeE5Backend):
            backend_name = "local-hash"

        with self.assertRaises(FeatureBuildError):
            validate_e5_backend(FallbackBackend(), "production")
        self.assertIsNone(validate_e5_backend(FallbackBackend(), "local"))


if __name__ == "__main__":
    unittest.main()
