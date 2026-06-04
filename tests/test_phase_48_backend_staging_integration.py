from __future__ import annotations

import json
import unittest

from scripts.run_phase_48_backend_staging_shadow import compare_shadow_outputs, private_leak_tokens, redact, sanitize_url
from scripts.verify_phase_48_backend_staging_integration import REPORT_JSON_PATH, build_report, write_all


class Phase48BackendStagingIntegrationTest(unittest.TestCase):
    def test_redaction_removes_auth_tokens_from_evidence(self) -> None:
        payload = {
            "authorization": "Bearer secret-token",
            "nested": {"MODEL_API_SERVICE_TOKEN": "secret-token", "safe": "ok"},
            "url": "https://staging.example.test/model-info?token=secret-token",
        }

        redacted = redact(payload)

        self.assertEqual(redacted["authorization"], "<redacted>")
        self.assertEqual(redacted["nested"]["MODEL_API_SERVICE_TOKEN"], "<redacted>")
        self.assertEqual(redacted["nested"]["safe"], "ok")
        self.assertEqual(sanitize_url(payload["url"]), "https://staging.example.test/model-info")

    def test_private_leak_detection_flags_sensitive_payload_tokens(self) -> None:
        self.assertIn("rawCv", private_leak_tokens({"data": {"rawCv": "text"}}))
        self.assertIn("authorization", private_leak_tokens({"authorization": "Bearer secret-token"}))

    def test_shadow_compare_flags_score_delta_and_rank_swap(self) -> None:
        old_payload = {
            "body": {
                "schemaVersion": "model-core-cv-analysis-v1",
                "jobFitAlignment": {"score": 72, "matchedSkills": ["python"], "missingSkills": ["sql"]},
                "atsFriendliness": {"score": 90},
                "candidateReranking": {"recommendations": [{"jobId": "job-a", "matchScore": 72, "matchLevel": "good"}]},
            }
        }
        new_payload = {
            "body": {
                "schemaVersion": "model-core-cv-analysis-v1",
                "jobFitAlignment": {"score": 61, "matchedSkills": ["python"], "missingSkills": ["sql", "typescript"]},
                "atsFriendliness": {"score": 82},
                "candidateReranking": {"recommendations": [{"jobId": "job-b", "matchScore": 61, "matchLevel": "stretch"}]},
            }
        }

        comparison = compare_shadow_outputs(old_payload, new_payload, score_delta_threshold=5)

        self.assertEqual(comparison["decision"], "review")
        self.assertTrue(comparison["flags"]["score_delta_above_threshold"])
        self.assertTrue(comparison["flags"]["top_recommendation_rank_swap"])
        self.assertTrue(comparison["flags"]["ats_score_delta_above_threshold"])
        self.assertEqual(comparison["deltas"]["jobFitScore"], -11)

    def test_phase48_static_gate_writes_ready_report(self) -> None:
        report = write_all()
        written = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(report["schema_version"], "phase-48-backend-staging-integration-v1")
        self.assertEqual(written["status"], "ready_for_staging_execution")
        self.assertTrue(all(written["checks"].values()))
        self.assertTrue(written["operator_execution_required"])
        self.assertIn("phase48_shadow_compare", written["commands"])
        self.assertEqual(build_report()["status"], "ready_for_staging_execution")


if __name__ == "__main__":
    unittest.main()
