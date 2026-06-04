from __future__ import annotations

import json
import unittest

from scripts.verify_training_step_10_calibration_model_card_manifest import (
    ARTIFACT_MANIFEST_PATH,
    MODEL_CARD_PATH,
    REQUIRED_SCORE_OUTPUTS,
    build_report,
    sha256_file,
)


class TrainingStep10CalibrationModelCardManifestTest(unittest.TestCase):
    def test_step_10_acceptance_is_explicit(self) -> None:
        report = build_report()

        self.assertEqual(report["schema_version"], "training-step-10-calibration-model-card-manifest-v1")
        self.assertIn(report["final_decision"], {"pass", "implemented-with-warnings", "blocked"})
        self.assertIn("model_card_and_manifest_consistent", report["acceptance"])
        self.assertIn("tensorboard_release_evidence_recorded", report["acceptance"])
        self.assertIn("no_export_artifact_mutated_without_manifest_refresh", report["acceptance"])
        self.assertIn("phase_27_8_refresh_complete", report["acceptance"])

    def test_step_10_verifies_calibration_and_model_card_outputs(self) -> None:
        report = build_report()
        calibration = report["steps"]["calibration"]
        model_card = report["steps"]["model_card"]

        self.assertEqual(calibration["status"], "PASS")
        self.assertTrue(calibration["all_metric_leaves_passed"])
        for output in REQUIRED_SCORE_OUTPUTS:
            self.assertIn(output, calibration["required_outputs"])
            self.assertIn(output, model_card["score_semantics_outputs"])

    def test_step_10_verifies_manifest_and_tensorboard_evidence(self) -> None:
        report = build_report()
        manifest_step = report["steps"]["artifact_manifest"]
        tensorboard_step = report["steps"]["tensorboard"]
        phase_27_8 = report["steps"]["phase_27_8"]

        self.assertEqual(manifest_step["status"], "PASS")
        self.assertEqual(manifest_step["mismatch_count"], 0)
        self.assertEqual(tensorboard_step["status"], "PASS")
        self.assertGreaterEqual(tensorboard_step["event_file_count"], 1)
        self.assertFalse(tensorboard_step["release_git_ignore_check"]["release_event_file_ignored"])
        self.assertEqual(phase_27_8["final_decision"], "refresh-complete")

    def test_model_card_hash_is_recorded_in_manifest(self) -> None:
        manifest = json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))
        model_entry = next(entry for entry in manifest["artifacts"] if entry["artifact_id"] == "model_card")

        self.assertEqual(model_entry["sha256"], sha256_file(MODEL_CARD_PATH))


if __name__ == "__main__":
    unittest.main()
