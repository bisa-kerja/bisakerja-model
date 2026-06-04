from __future__ import annotations

import unittest

from scripts.verify_training_step_1_scope_baseline import build_report


class TrainingStep1ScopeBaselineTest(unittest.TestCase):
    def test_scope_boundary_and_required_evidence_are_documented(self) -> None:
        report = build_report()

        self.assertEqual(report["scope_boundary"]["status"], "PASS")
        self.assertIn("jobFitAlignment", report["scope_boundary"]["training_owned_outputs"])
        self.assertIn("topActionables", report["scope_boundary"]["wrapper_owned_outputs"])
        self.assertIn("sectionReviews", report["scope_boundary"]["wrapper_owned_outputs"])
        self.assertTrue(report["acceptance"]["training_scope_explicit"])
        self.assertTrue(report["acceptance"]["existing_reports_reviewed"])
        self.assertTrue(report["acceptance"]["dirty_paths_documented"])

        required = report["evidence_review"]["required_evidence"]
        self.assertTrue(required)
        self.assertFalse([item for item in required if not item["exists"]])

    def test_current_release_blockers_stay_explicit_without_blocking_step_1(self) -> None:
        report = build_report()

        self.assertEqual(report["evidence_review"]["status"], "WARN")
        self.assertEqual(report["final_decision"], "implemented-with-documented-warnings")
        self.assertIn(
            "reports/phase_25_tensorflow_training_delivery.json",
            report["evidence_review"]["report_summaries"],
        )
        self.assertTrue(any("Phase 25 remains" in item for item in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
