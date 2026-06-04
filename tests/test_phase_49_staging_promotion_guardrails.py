from __future__ import annotations

import json
import unittest

from scripts.verify_phase_49_staging_promotion_guardrails import REPORT_JSON_PATH, build_report, write_all


class Phase49StagingPromotionGuardrailsTest(unittest.TestCase):
    def test_phase49_report_records_experiment_only_decision(self) -> None:
        report = build_report()

        self.assertEqual(report["schema_version"], "phase-49-staging-promotion-guardrails-v1")
        self.assertEqual(report["decision"], "staging-experiment-only")
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["operator_execution_required"])
        self.assertTrue(all(report["checks"].values()))

    def test_phase49_deployment_and_rollback_paths_are_explicit(self) -> None:
        report = build_report()

        self.assertEqual(
            report["deployment"]["default_staging"]["root"],
            "artifacts/phase_46_calibration_model_card_manifest_handoff_refresh",
        )
        self.assertEqual(report["deployment"]["default_staging"]["embedding_model"], "intfloat/multilingual-e5-small")
        self.assertEqual(report["deployment"]["rollback"]["root"], "artifacts/phase_25_tensorflow_training_delivery")
        self.assertEqual(report["deployment"]["rollback"]["embedding_model"], "intfloat/e5-base-v2")

    def test_phase49_production_blockers_remain_required(self) -> None:
        report = build_report()

        blockers = "\n".join(report["production_blockers"])
        self.assertIn("human/reviewer validation scale", blockers)
        self.assertIn("rollback drill", blockers)
        self.assertIn("privacy review", blockers)
        self.assertIn("cost/resource monitoring", blockers)

    def test_phase49_static_gate_writes_report(self) -> None:
        report = write_all()
        written = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "complete")
        self.assertEqual(written["decision"], "staging-experiment-only")
        self.assertTrue(written["checks"]["production_remains_blocked_by_live_evidence"])


if __name__ == "__main__":
    unittest.main()
