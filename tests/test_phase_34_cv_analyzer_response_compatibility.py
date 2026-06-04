from __future__ import annotations

import json
import unittest

from model_api.app import build_cv_analysis_response_payload, create_app
from model_api.config import RuntimeConfig
from model_api.features import TensorFlowFeatureConfig
from model_api.inference import InferenceService, RuntimeState, ScoreCalibrationPolicy
from model_api.schemas import (
    CandidateJobInput,
    CandidateScoringInput,
    CvAnalysisModelCoreRequest,
    ModelArtifactIdentity,
    ModelIdentity,
    SanitizedProfileInput,
)


class FakeModel:
    name = "fake_phase25_model"

    def predict(self, values, verbose=0):
        return [[0.91] for _ in values]


class FakeE5Backend:
    model_name = "intfloat/e5-base-v2"
    backend_name = "sentence-transformers"

    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class Phase34CvAnalyzerResponseCompatibilityTest(unittest.TestCase):
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
            artifact=ModelArtifactIdentity(path="artifacts/private.keras", sha256="0" * 64),
        )
        return InferenceService(
            model=FakeModel(),
            state=RuntimeState(ready=True, model_identity=identity, artifact_manifest_phase="phase_25", message="ready"),
        )

    def request(self) -> CvAnalysisModelCoreRequest:
        return CvAnalysisModelCoreRequest(
            requestId="req-phase34",
            inputVersion="cv-analyzer-v1",
            language="en",
            inputMode="UPLOAD",
            compareSource="JOB_SEARCH",
            profile=SanitizedProfileInput(
                cvText="Backend developer with Python and SQL",
                normalizedSkills=("python", "sql"),
                targetRoles=("Backend Developer",),
                detectedCvSectionNames=("summary", "skills"),
            ),
            jobCandidates=(
                CandidateJobInput(
                    jobId="job-34-001",
                    scoringInput=CandidateScoringInput(
                        titleText="Backend Developer",
                        requiredSkills=("python", "sql"),
                        requirements=("REST API",),
                    ),
                ),
            ),
        )

    def test_model_core_response_matches_backend_strict_shape(self) -> None:
        payload = build_cv_analysis_response_payload(
            self.request(),
            service=self.service(),
            feature_config=self.feature_config(),
            calibration_policy=ScoreCalibrationPolicy(),
            embedding_backend=FakeE5Backend(),
            environment="test",
        )

        self.assertEqual(
            set(payload),
            {"schemaVersion", "parsedCv", "jobFitAlignment", "atsFriendliness", "overallImpression", "candidateReranking", "model", "createdAt"},
        )
        self.assertEqual(set(payload["model"]), {"name", "version"})
        self.assertEqual(set(payload["parsedCv"]), {"status", "pageCount", "textLength", "detectedSections", "extractionEvidence"})
        self.assertEqual(set(payload["jobFitAlignment"]), {"score", "matchedSignals", "missingSignals", "matchedSkills", "missingSkills", "evidence"})
        self.assertEqual(set(payload["atsFriendliness"]), {"score", "detectedIssues", "parseQuality", "evidence"})
        self.assertEqual(set(payload["overallImpression"]), {"score", "evidence"})
        self.assertEqual(set(payload["candidateReranking"]), {"recommendations"})
        self.assertIsInstance(payload["jobFitAlignment"]["score"], int)
        self.assertEqual(payload["atsFriendliness"]["parseQuality"], "high")
        self.assertNotIn("artifact", json.dumps(payload))
        self.assertNotIn("summarySignals", json.dumps(payload))
        self.assertNotIn("confidenceNotes", json.dumps(payload))
        self.assertNotIn("topActionables", json.dumps(payload))
        self.assertEqual(payload["candidateReranking"]["recommendations"][0]["jobId"], "job-34-001")

    def test_long_requirement_text_is_bounded_for_backend_zod_contract(self) -> None:
        long_requirement = " ".join(["Build secure observable production APIs with PostgreSQL ownership"] * 8)
        request = CvAnalysisModelCoreRequest(
            requestId="req-phase34-long-requirement",
            inputVersion="cv-analyzer-v1",
            language="en",
            inputMode="UPLOAD",
            compareSource="JOB_SEARCH",
            profile=SanitizedProfileInput(
                cvText="Backend developer with Python",
                normalizedSkills=("python",),
                targetRoles=("Backend Developer",),
                detectedCvSectionNames=("summary",),
            ),
            jobCandidates=(
                CandidateJobInput(
                    jobId="job-34-long",
                    scoringInput=CandidateScoringInput(
                        titleText="Backend Developer",
                        requiredSkills=("python",),
                        requirements=(long_requirement,),
                    ),
                ),
            ),
        )

        payload = build_cv_analysis_response_payload(
            request,
            service=self.service(),
            feature_config=self.feature_config(),
            calibration_policy=ScoreCalibrationPolicy(),
            embedding_backend=FakeE5Backend(),
            environment="test",
        )

        recommendation = payload["candidateReranking"]["recommendations"][0]
        for value in (*payload["jobFitAlignment"]["matchedSkills"], *payload["jobFitAlignment"]["missingSkills"], *recommendation["matchedSkills"], *recommendation["missingSkills"]):
            self.assertLessEqual(len(value), 120)
        for value in (*payload["jobFitAlignment"]["evidence"], *recommendation["rankingSignals"]):
            self.assertLessEqual(len(value), 200)

    def test_internal_multipart_endpoint_returns_raw_model_core_payload(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("FastAPI serving dependencies are not installed")

        app = create_app(
            config=RuntimeConfig.from_env({"MODEL_API_ARTIFACT_ROOT": "artifacts/phase_25_tensorflow_training_delivery"}),
            service=self.service(),
            embedding_backend=FakeE5Backend(),
        )
        client = TestClient(app)
        pdf = b"%PDF-1.4\n1 0 obj<</Type /Page>>stream\n(Summary Backend developer) Tj\n(Skills Python SQL) Tj\n(Experience 2020) Tj\nendstream\n%%EOF"
        multipart = [
            ("requestId", (None, "req-phase34-endpoint")),
            ("language", (None, "en")),
            ("inputMode", (None, "UPLOAD")),
            ("compareSource", (None, "JOB_SEARCH")),
            ("jobRoles", (None, "Backend Developer")),
            (
                "jobCandidates",
                (
                    None,
                    json.dumps(
                        [
                            {
                                "jobId": "job-34-001",
                                "scoringInput": {"titleText": "Backend Developer", "requiredSkills": ["python", "sql"], "requirements": ["REST API"]},
                            }
                        ]
                    ),
                ),
            ),
            (
                "rankingPolicy",
                (None, json.dumps({"maxRecommendations": 10, "requireCandidateJobIds": True, "deduplicateByJobId": True, "backendOwnsHydration": True})),
            ),
            ("cvFile", ("cv.pdf", pdf, "application/pdf")),
        ]

        response = client.post("/internal/model/cv-analysis", files=multipart)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertNotIn("success", body)
        self.assertEqual(body["schemaVersion"], "model-core-cv-analysis-v1")
        self.assertEqual(body["parsedCv"]["status"], "parsed")
        self.assertIn("summary", body["parsedCv"]["detectedSections"])
        self.assertIn("skills", body["parsedCv"]["detectedSections"])
        self.assertIn(body["atsFriendliness"]["parseQuality"], {"high", "medium", "low", "failed"})
        self.assertEqual(body["candidateReranking"]["recommendations"][0]["jobId"], "job-34-001")


if __name__ == "__main__":
    unittest.main()
