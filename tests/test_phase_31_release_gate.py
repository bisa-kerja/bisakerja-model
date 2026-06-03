from __future__ import annotations

import json
import unittest

from model_api.observability import assert_safe_observability_event, build_safe_observability_event
from scripts.verify_phase_31_release_gate import REPORT_JSON_PATH, build_report, write_all


class Phase31ReleaseGateTest(unittest.TestCase):
    def test_safe_observability_allows_only_operational_metadata(self) -> None:
        event = build_safe_observability_event(
            requestId="req-31",
            modelVersion="phase25-test",
            artifactHash="0" * 64,
            candidateCount=50,
            parseQuality="text_ok",
            parseLatencyMs=12,
            embeddingLatencyMs=20,
            tensorflowLatencyMs=30,
            wrapperLatencyMs=0,
            totalLatencyMs=62,
            errorCode="MODEL_API_TIMEOUT",
            fallbackReason="slow TensorFlow",
            cvText="raw CV must be dropped",
            token="secret must be dropped",
        )

        assert_safe_observability_event(event)
        self.assertEqual(event["requestId"], "req-31")
        self.assertEqual(event["candidateCount"], 50)
        self.assertNotIn("cvText", event)
        self.assertNotIn("token", event)

    def test_release_gate_report_covers_phase_31_acceptance_criteria(self) -> None:
        report = build_report()

        self.assertEqual(report["schema_version"], "phase-31-release-gate-report-v1")
        self.assertEqual(report["final_decision"], "passed")
        self.assertEqual(report["blockers"], [])
        self.assertTrue(all(report["checks"].values()))

    def test_release_gate_report_is_written_as_durable_json(self) -> None:
        report = write_all()
        written = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(written["final_decision"], report["final_decision"])
        self.assertEqual(written["blockers"], [])
        self.assertIn("Model API owns no DB credentials", written["acceptance"]["model_api_db_ownership"])

    def test_e2e_load_timeout_failure_matrix_is_declared_for_staging_gate(self) -> None:
        matrix = {
            "e2e": ["upload", "Model API parse/inference", "Backend wrapper/hydration/persistence", "CvAnalysis"],
            "load_timeout": ["candidate count 50", "max PDF", "slow parser", "slow E5", "slow TensorFlow", "slow GenAI"],
            "deterministic_status": ["422", "503", "504"],
            "persistence_contract": ["CvAnalysisResult", "JobRecommendationRun", "JobRecommendationItem"],
            "model_core_contract": ["model-core-cv-analysis-v1", "model-core-candidate-reranking-v1"],
        }

        self.assertIn("CvAnalysis", matrix["e2e"])
        self.assertIn("JobRecommendationItem", matrix["persistence_contract"])
        self.assertIn("504", matrix["deterministic_status"])
        self.assertIn("slow GenAI", matrix["load_timeout"])


if __name__ == "__main__":
    unittest.main()
