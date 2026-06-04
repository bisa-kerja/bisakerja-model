from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.verify_phase_27_3_27_4_release_evidence import (
    LABEL_POLICY_PATH,
    REPORT_JSON_PATH,
    build_report,
)

ROOT = Path(__file__).resolve().parents[1]


class Phase27NotebookLabelGateTest(unittest.TestCase):
    def test_notebook_hygiene_gate_has_no_error_outputs_or_active_unexecuted_code(self) -> None:
        report = build_report()
        notebook_gate = report["steps"]["27.3"]["notebook_gate"]
        self.assertEqual(notebook_gate["status"], "PASS")
        self.assertEqual(notebook_gate["error_notebook_count"], 0)
        self.assertEqual(notebook_gate["active_unexecuted_notebook_count"], 0)

        notebooks = {item["path"]: item for item in notebook_gate["notebooks"]}
        self.assertEqual(notebooks["training/notebooks/phase_08_overall_impression_signals.ipynb"]["error_output_count"], 0)
        self.assertTrue(notebooks["training/notebooks/phase_13_data_snapshot_contract_freezing.ipynb"]["retired"])
        self.assertEqual(notebooks["training/notebooks/phase_13_data_snapshot_contract_freezing.ipynb"]["cell_counts"]["code"], 0)

    def test_label_gate_blocks_production_score_claims_without_release_scale_human_labels(self) -> None:
        report = build_report()
        label_gate = report["steps"]["27.4"]["label_gate"]
        self.assertEqual(label_gate["status"], "FAIL")
        self.assertFalse(label_gate["production_score_claims_allowed"])
        self.assertEqual(label_gate["unique_review_items"], 120)
        self.assertGreaterEqual(label_gate["reviewer_count"], 2)
        self.assertTrue(label_gate["weak_labels_allowed_only_as_bootstrap_training_support"])
        self.assertIn("language", label_gate["slice_dimensions"])
        self.assertIn("experience_band", label_gate["slice_dimensions"])
        self.assertTrue(label_gate["slice_dimensions"]["language"])
        self.assertTrue(label_gate["slice_dimensions"]["experience_band"])
        self.assertEqual(report["final_decision"], "blocked")

    def test_written_policy_and_report_are_durable_json(self) -> None:
        self.assertTrue(LABEL_POLICY_PATH.exists())
        self.assertTrue(REPORT_JSON_PATH.exists())
        policy = json.loads(LABEL_POLICY_PATH.read_text(encoding="utf-8"))
        written_report = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(policy["schema_version"], "phase-27-production-label-policy-v1")
        self.assertEqual(policy["minimum_release_validation_set"]["unique_items"], 600)
        self.assertEqual(written_report["schema_version"], "phase-27-3-27-4-notebook-label-gate-v1")


if __name__ == "__main__":
    unittest.main()
