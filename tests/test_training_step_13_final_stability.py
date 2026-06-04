from __future__ import annotations

import unittest

from scripts.verify_training_step_13_final_stability import build_report


class TrainingStep13FinalStabilityTest(unittest.TestCase):
    def test_step_13_acceptance_is_explicit(self) -> None:
        report = build_report()

        self.assertEqual(report["schema_version"], "training-step-13-final-stability-v1")
        self.assertIn(report["final_decision"], {"pass", "pass-with-documented-limitations", "blocked"})
        self.assertIn("working_tree_only_intended_changes", report["acceptance"])
        self.assertIn("production_readiness_claim_supported_by_reports", report["acceptance"])
        self.assertIn("remaining_risks_are_explicit", report["acceptance"])

    def test_final_stability_is_not_blocked(self) -> None:
        report = build_report()

        self.assertFalse(report["blocking_checks"])
        self.assertTrue(all(report["acceptance"].values()))

    def test_working_tree_dirty_paths_are_intended(self) -> None:
        report = build_report()
        working_tree = report["checks"]["working_tree_intent"]

        self.assertFalse(working_tree["unexpected_dirty_paths"])
        self.assertGreaterEqual(working_tree["dirty_file_count"], 1)

    def test_sensitive_cache_and_private_data_are_not_pending(self) -> None:
        report = build_report()
        sensitive = report["checks"]["sensitive_cache_private_data"]

        self.assertIn(sensitive["status"], {"PASS", "WARN"})
        self.assertFalse(sensitive["pending_forbidden_paths"])

    def test_final_note_names_runtime_status_and_limitations(self) -> None:
        report = build_report()
        note = report["final_stability_note"]

        self.assertEqual(note["runtime"]["target_python"], "3.13.x")
        self.assertEqual(note["phase25_status"], "production-ready")
        self.assertIn(note["release_gate_status"], {"pass", "pass-with-documented-limitations"})
        self.assertEqual(note["model_api_smoke_status"], "blocked")
        self.assertGreaterEqual(len(note["limitations"]), 3)


if __name__ == "__main__":
    unittest.main()
