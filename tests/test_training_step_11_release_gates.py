from __future__ import annotations

import unittest

from scripts.verify_training_step_11_release_gates import (
    OUT_OF_SCOPE_REQUIREMENT_IDS,
    TRAINING_OWNED_REQUIREMENT_IDS,
    build_report,
)


class TrainingStep11ReleaseGatesTest(unittest.TestCase):
    def test_step_11_acceptance_is_explicit(self) -> None:
        report = build_report()

        self.assertEqual(report["schema_version"], "training-step-11-release-gates-v1")
        self.assertIn(report["final_decision"], {"pass", "pass-with-documented-limitations", "blocked"})
        self.assertIn("phase_27_1_and_27_2_passed", report["acceptance"])
        self.assertIn("phase_27_3_and_27_4_training_evidence_passed", report["acceptance"])
        self.assertIn("phase_27_10_training_owned_requirements_not_blocked", report["acceptance"])

    def test_training_owned_phase_27_gates_are_not_blocked(self) -> None:
        report = build_report()

        self.assertFalse(report["blocking_steps"])
        self.assertTrue(report["acceptance"]["phase_27_1_and_27_2_passed"])
        self.assertTrue(report["acceptance"]["phase_27_3_and_27_4_training_evidence_passed"])
        self.assertTrue(report["acceptance"]["phase_27_5_and_27_6_passed"])
        self.assertTrue(report["acceptance"]["phase_27_7_not_blocked"])
        self.assertTrue(report["acceptance"]["phase_27_8_complete"])

    def test_requirement_matrix_keeps_model_api_blockers_out_of_training_scope(self) -> None:
        report = build_report()
        matrix = report["steps"]["phase_27_10"]

        self.assertEqual(matrix["training_owned_requirement_ids"], list(TRAINING_OWNED_REQUIREMENT_IDS))
        self.assertEqual(matrix["out_of_scope_requirement_ids"], list(OUT_OF_SCOPE_REQUIREMENT_IDS))
        self.assertEqual(matrix["training_owned_satisfied_count"], len(TRAINING_OWNED_REQUIREMENT_IDS))
        self.assertFalse(matrix["blocked_training_rows"])
        self.assertTrue(matrix["blocked_out_of_scope_rows"])

    def test_label_release_limitation_is_documented_without_blocking_training_gate(self) -> None:
        report = build_report()
        label_gate = report["steps"]["phase_27_3_27_4"]

        self.assertEqual(label_gate["status"], "WARN")
        self.assertEqual(label_gate["phase_27_3_status"], "PASS")
        self.assertEqual(label_gate["phase_27_4_training_label_evidence_status"], "PASS")
        self.assertGreater(label_gate["label_release_blocker_count"], 0)
        self.assertFalse(label_gate["production_score_claims_allowed"])


if __name__ == "__main__":
    unittest.main()
