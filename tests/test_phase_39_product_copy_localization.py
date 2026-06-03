from __future__ import annotations

import json
import unittest

from scripts.verify_phase_39_product_copy_localization import REPORT_JSON_PATH, build_report, write_all


class Phase39ProductCopyLocalizationTest(unittest.TestCase):
    def test_phase_39_gate_report_is_go_and_writable(self) -> None:
        report = write_all()
        written = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(report["schema_version"], "phase-39-product-copy-localization-v1")
        self.assertEqual(report["final_decision"], "go")
        self.assertEqual(written["blockers"], [])
        self.assertTrue(all(build_report()["checks"].values()))

    def test_language_policy_and_review_result_are_explicit(self) -> None:
        report = build_report()

        self.assertEqual(report["language_policy"]["staging_default"], "english_safe_copy")
        self.assertIn("English public copy", report["language_policy"]["requested_language_id"])
        self.assertEqual(report["review_result"]["fallback_copy_decision"], "accepted_for_staging_demo_with_english_safe_copy")
        self.assertIn("deferred", report["review_result"]["indonesian_localization"])

    def test_copy_safety_and_schema_compatibility_checks_pass(self) -> None:
        report = build_report()

        self.assertTrue(report["checks"]["copy_safety_filters_and_pii_tests_present"])
        self.assertTrue(report["checks"]["schema_length_limits_match_public_contract"])
        self.assertTrue(report["checks"]["approved_copy_has_no_localization_drift_or_sensitive_literals"])
        self.assertIn("backend_copy_tests", report["commands"])


if __name__ == "__main__":
    unittest.main()
