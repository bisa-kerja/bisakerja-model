from __future__ import annotations

import json
import unittest

from scripts.verify_phase_27_7_clean_kernel_export import REPORT_JSON_PATH, build_report, write_all


class Phase27CleanKernelExportTest(unittest.TestCase):
    def test_phase_25_clean_kernel_evidence_passes_required_tensorflow_gates(self) -> None:
        report = build_report()
        gates = {item["check"]: item for item in report["gates"]}

        self.assertEqual(gates["required_phase25_evidence_exists"]["status"], "PASS")
        self.assertEqual(gates["phase25_strict_gate_checks_pass"]["status"], "PASS")
        self.assertEqual(gates["phase25_required_step_reports_pass"]["status"], "PASS")
        self.assertEqual(gates["python_3_13_runtime_recorded"]["status"], "PASS")
        self.assertEqual(gates["tensorflow_functional_api"]["status"], "PASS")
        self.assertEqual(gates["custom_components_present"]["status"], "PASS")
        self.assertEqual(gates["gradient_tape_loop_no_model_fit"]["status"], "PASS")
        self.assertEqual(gates["mae_target_le_0_02"]["status"], "PASS")
        self.assertEqual(gates["tensorboard_events_recorded"]["status"], "PASS")
        self.assertEqual(gates["keras_export_and_inference_smoke"]["status"], "PASS")
        self.assertEqual(gates["phase25_notebook_has_no_saved_errors"]["status"], "PASS")

        self.assertLessEqual(gates["mae_target_le_0_02"]["evidence"]["validation_mae"], 0.02)
        self.assertFalse(gates["gradient_tape_loop_no_model_fit"]["evidence"]["uses_model_fit"])
        self.assertGreaterEqual(gates["tensorboard_events_recorded"]["evidence"]["event_file_count"], 1)

    def test_production_ready_passes_with_clean_git_and_phase25_production_status(self) -> None:
        report = build_report()
        gates = {item["check"]: item for item in report["gates"]}

        self.assertEqual(report["phase25"]["status"], "staging-ready")
        self.assertEqual(gates["phase25_final_status_production_ready"]["status"], "FAIL")
        self.assertEqual(gates["clean_git_state_now"]["status"], "FAIL")
        self.assertEqual(report["final_decision"], "blocked")
        self.assertIn("Phase 25 final report status is 'staging-ready', not 'production-ready'.", report["blockers"])

    def test_written_report_is_durable_json(self) -> None:
        report = write_all()
        written_report = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(written_report["schema_version"], "phase-27-7-clean-kernel-production-export-v1")
        self.assertEqual(written_report["final_decision"], report["final_decision"])
        self.assertEqual(written_report["policy"]["production_ready_requires_phase25_status"], "production-ready")
        self.assertEqual(written_report["policy"]["python_runtime"], "3.13.x")


if __name__ == "__main__":
    unittest.main()
