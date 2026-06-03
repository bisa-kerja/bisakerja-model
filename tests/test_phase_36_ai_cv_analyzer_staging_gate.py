from __future__ import annotations

import json
import unittest

from scripts.verify_phase_36_ai_cv_analyzer_staging_gate import REPORT_JSON_PATH, build_report, write_all


class Phase36AiCvAnalyzerStagingGateTest(unittest.TestCase):
    def test_staging_gate_report_has_go_decision_when_all_blockers_are_closed(self) -> None:
        report = build_report()

        self.assertEqual(report["schema_version"], "phase-36-ai-cv-analyzer-staging-gate-v1")
        self.assertEqual(report["final_decision"], "go")
        self.assertEqual(report["blockers"], [])
        self.assertTrue(all(report["checks"].values()))

    def test_public_openapi_contract_and_downstream_errors_are_covered(self) -> None:
        report = build_report()
        contract = report["contract"]

        self.assertEqual(contract["path"], "/api/v1/ai/cv-analyzer")
        self.assertEqual(contract["requestContentType"], "multipart/form-data")
        self.assertEqual(contract["schemaVersion"], "cv-analysis-v2")
        self.assertIn("jobFitAlignment", contract["analysisResultRequired"])
        self.assertIn("generatedCv", contract["analysisResultRequired"])
        self.assertTrue({"422", "502", "503"}.issubset(set(contract["errorStatuses"])))
        self.assertTrue(report["checks"]["backend_downstream_errors_fail_closed"])
        self.assertTrue(report["checks"]["model_api_rejects_unsafe_input_deterministically"])

    def test_failure_language_security_and_fallback_coverage_are_explicit(self) -> None:
        report = build_report()

        for case in [
            "invalid file type",
            "file too large",
            "empty candidates",
            "Model API invalid response",
            "timeout",
            "GenAI invalid JSON",
        ]:
            self.assertIn(case, report["failure_cases"])
        self.assertTrue(report["checks"]["security_privacy_boundary_verified"])
        self.assertTrue(report["checks"]["language_behavior_verified"])
        self.assertIn("generatedCv.available=false", report["fallback_coverage"]["deterministic_fallback"])
        self.assertIn("preserves model scores", report["fallback_coverage"]["score_integrity"])

    def test_report_is_written_as_durable_json_and_markdown(self) -> None:
        report = write_all()
        written = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(written["final_decision"], report["final_decision"])
        self.assertEqual(written["blockers"], [])
        self.assertIn("model_api", written["test_commands"])
        self.assertIn("backend", written["test_commands"])


if __name__ == "__main__":
    unittest.main()
