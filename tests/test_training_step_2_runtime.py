from __future__ import annotations

import unittest

from scripts.verify_training_step_2_runtime import (
    EXPECTED_KERAS,
    EXPECTED_KERNEL_NAME,
    EXPECTED_TENSORFLOW,
    build_report,
)


class TrainingStep2RuntimeTest(unittest.TestCase):
    def test_step_2_report_records_expected_runtime_contract(self) -> None:
        report = build_report()

        self.assertEqual(report["expected"]["python"], "3.13.x")
        self.assertEqual(report["expected"]["tensorflow"], EXPECTED_TENSORFLOW)
        self.assertEqual(report["expected"]["keras"], EXPECTED_KERAS)
        self.assertEqual(report["expected"]["kernel_name"], EXPECTED_KERNEL_NAME)
        self.assertIn(report["final_decision"], {"pass", "implemented-with-warnings", "blocked"})

    def test_step_2_acceptance_is_explicit_for_pass_or_blocked_state(self) -> None:
        report = build_report()

        self.assertIn("venv_uses_python_3_13", report["acceptance"])
        self.assertIn("pytest_available_in_venv", report["acceptance"])
        self.assertIn("tensorflow_imports_in_venv", report["acceptance"])
        self.assertIn("kernel_points_to_training_venv", report["acceptance"])

        if report["final_decision"] == "blocked":
            self.assertTrue(report["blockers"])
            self.assertTrue(
                any(
                    "python.exe could not execute" in item
                    or "Required imports could not be verified" in item
                    or "Kernel" in item
                    for item in report["blockers"]
                )
            )
        else:
            self.assertTrue(report["acceptance"]["venv_uses_python_3_13"])
            self.assertTrue(report["acceptance"]["pytest_available_in_venv"])
            self.assertTrue(report["acceptance"]["tensorflow_imports_in_venv"])


if __name__ == "__main__":
    unittest.main()
