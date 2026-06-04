from __future__ import annotations

import unittest

from scripts.verify_training_steps_3_8 import build_report


class TrainingSteps38AuditTest(unittest.TestCase):
    def test_steps_3_4_5_7_have_reusable_evidence(self) -> None:
        report = build_report()

        self.assertEqual(report["steps"]["3"]["status"], "PASS")
        self.assertEqual(report["steps"]["4"]["status"], "PASS")
        self.assertIn(report["steps"]["5"]["status"], {"PASS", "WARN"})
        self.assertEqual(report["steps"]["7"]["status"], "PASS")

        self.assertEqual(report["steps"]["3"]["notebook_count"], 27)
        self.assertFalse(report["steps"]["4"]["secret_scan_findings"])
        self.assertTrue(report["steps"]["5"]["checks"]["embedding_model"])
        self.assertTrue(report["steps"]["5"]["checks"]["profile_prefix"])
        self.assertTrue(report["steps"]["5"]["checks"]["job_prefix"])
        self.assertEqual(report["steps"]["7"]["leakage"]["leaking_profile_count"], 0)

    def test_label_and_selection_risks_stay_explicit(self) -> None:
        report = build_report()

        self.assertEqual(report["steps"]["6"]["status"], "WARN")
        self.assertFalse(report["steps"]["6"]["production_score_claims_allowed"])
        self.assertTrue(report["steps"]["6"]["production_claim_gate"]["production_score_claims_blocked_until_thresholds_pass"])
        self.assertEqual(report["steps"]["6"]["validation_slice_gate"]["missing_jobfit_slice_dimensions"], [])
        self.assertEqual(report["steps"]["6"]["validation_slice_gate"]["missing_ats_cases"], [])
        self.assertEqual(report["steps"]["8"]["status"], "WARN")
        self.assertEqual(report["steps"]["8"]["best_baseline"], "baseline_feature_regression")
        self.assertFalse(report["steps"]["8"]["phase25_production_selection_passed"])
        self.assertEqual(report["final_decision"], "implemented-with-warnings")


if __name__ == "__main__":
    unittest.main()
