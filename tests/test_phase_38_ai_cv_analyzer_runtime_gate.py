from __future__ import annotations

import json
import unittest

from model_api.app import build_cv_analysis_response_payload
from model_api.config import RuntimeConfig
from model_api.features import E5_MODEL_NAME, TensorFlowFeatureConfig
from model_api.inference import InferenceService, RuntimeState, ScoreCalibrationPolicy
from model_api.schemas import CandidateJobInput, CandidateScoringInput, CvAnalysisModelCoreRequest, ModelArtifactIdentity, ModelIdentity, SanitizedProfileInput
from scripts.verify_phase_38_ai_cv_analyzer_runtime_gate import REPORT_JSON_PATH, build_report, write_all


class FakeModel:
    name = "fake_phase25_model"

    def predict(self, values, verbose=0):
        return [[0.8] for _ in values]


class FakeE5Backend:
    model_name = E5_MODEL_NAME
    backend_name = "sentence-transformers"

    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class Phase38AiCvAnalyzerRuntimeGateTest(unittest.TestCase):
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

    def test_staging_requires_warmup_by_default_and_local_does_not(self) -> None:
        local = RuntimeConfig.from_env({})
        staging = RuntimeConfig.from_env({"MODEL_API_ENV": "staging", "MODEL_API_SERVICE_TOKEN": "token"})
        disabled = RuntimeConfig.from_env(
            {"MODEL_API_ENV": "staging", "MODEL_API_SERVICE_TOKEN": "token", "MODEL_API_WARMUP_REQUIRED": "false"}
        )

        self.assertFalse(local.warmup_required)
        self.assertTrue(staging.warmup_required)
        self.assertFalse(disabled.warmup_required)

    def test_cv_analysis_response_records_latency_breakdown_for_runtime_gate(self) -> None:
        request = CvAnalysisModelCoreRequest(
            requestId="req-phase38",
            inputVersion="cv-analyzer-v1",
            language="en",
            inputMode="UPLOAD",
            compareSource="JOB_SEARCH",
            profile=SanitizedProfileInput(
                cvText="Backend engineer Python SQL",
                normalizedSkills=("python", "sql"),
                detectedCvSectionNames=("summary", "skills"),
            ),
            jobCandidates=(
                CandidateJobInput(
                    jobId="job-1",
                    scoringInput=CandidateScoringInput(titleText="Backend Engineer", requiredSkills=("python", "sql")),
                ),
            ),
            maxRecommendations=1,
        )
        payload = build_cv_analysis_response_payload(
            request,
            service=self.service(),
            feature_config=self.feature_config(),
            calibration_policy=ScoreCalibrationPolicy(),
            embedding_backend=FakeE5Backend(),
            environment="test",
            timeout_ms=30_000,
            include_observability=True,
        )
        event = payload["observability"]

        self.assertEqual(event["requestId"], "req-phase38")
        self.assertIn("parseLatencyMs", event)
        self.assertIn("embeddingLatencyMs", event)
        self.assertIn("tensorflowLatencyMs", event)
        self.assertIn("totalLatencyMs", event)
        self.assertNotIn("rawCv", json.dumps(payload))
        self.assertNotIn("storageKey", json.dumps(payload))

    def test_phase_38_gate_report_is_go_and_writable(self) -> None:
        report = write_all()
        written = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(report["schema_version"], "phase-38-ai-cv-analyzer-runtime-gate-v1")
        self.assertEqual(report["final_decision"], "go")
        self.assertEqual(written["blockers"], [])
        self.assertTrue(all(build_report()["checks"].values()))
        self.assertIn("model_api_warmup", written["commands"])


if __name__ == "__main__":
    unittest.main()
