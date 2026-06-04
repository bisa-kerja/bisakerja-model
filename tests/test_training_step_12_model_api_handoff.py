from __future__ import annotations

import unittest

from scripts.verify_training_step_12_model_api_handoff import (
    HANDOFF_EXPORT_KEYS,
    build_report,
)


class TrainingStep12ModelApiHandoffTest(unittest.TestCase):
    def test_step_12_acceptance_is_explicit(self) -> None:
        report = build_report()

        self.assertEqual(report["schema_version"], "training-step-12-model-api-handoff-v1")
        self.assertIn(report["final_decision"], {"pass", "pass-with-documented-limitations", "blocked"})
        self.assertIn("training_exports_only_handoff_fixtures", report["acceptance"])
        self.assertIn("candidate_reranking_uses_backend_candidate_ids", report["acceptance"])
        self.assertIn("backend_behavior_remains_outside_training", report["acceptance"])

    def test_model_api_handoff_is_not_blocked(self) -> None:
        report = build_report()

        self.assertFalse(report["blocking_checks"])
        self.assertTrue(all(report["acceptance"].values()))

    def test_candidate_recommendations_are_backend_candidate_members(self) -> None:
        report = build_report()
        handoff = report["checks"]["phase_25_handoff_fixtures"]
        membership = handoff["candidate_membership"]

        self.assertGreater(membership["request_candidate_count"], 0)
        self.assertGreater(membership["response_recommendation_count"], 0)
        self.assertFalse(membership["unknown_job_ids"])
        self.assertFalse(membership["duplicate_job_ids"])

    def test_handoff_exports_are_limited_and_hash_checked(self) -> None:
        report = build_report()
        handoff = report["checks"]["phase_25_handoff_fixtures"]

        self.assertEqual(sorted(handoff["export_records"]), sorted(HANDOFF_EXPORT_KEYS))
        self.assertFalse(handoff["hash_mismatches"])
        self.assertFalse(handoff["positive_fixture_forbidden_paths"])
        self.assertFalse(handoff["score_violations"]["candidate_reranking"])
        self.assertFalse(handoff["score_violations"]["cv_analysis"])

    def test_backend_source_boundary_remains_external(self) -> None:
        report = build_report()
        boundary = report["checks"]["backend_source_boundary"]

        self.assertEqual(boundary["status"], "PASS")
        self.assertFalse(boundary["disallowed_backend_source_dirs"])
        self.assertTrue(boundary["root_readme_external_backend_boundary"])
        self.assertTrue(boundary["training_readme_source_boundary"])


if __name__ == "__main__":
    unittest.main()
