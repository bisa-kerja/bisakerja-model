from __future__ import annotations

import importlib.util
import json
import unittest

if importlib.util.find_spec("numpy") is not None:
    import numpy as np
else:  # pragma: no cover - exercised only in minimal local envs.
    np = None

from scripts.verify_phase_45_multilingual_e5_small_training_delivery import (
    APPROVED_FEATURES,
    BASELINE_REPORT_PATH,
    EVALUATION_REPORT_PATH,
    FEATURE_CONFIG_PATH,
    PREDICTIONS_PATH,
    REPORT_JSON_PATH,
    TENSORBOARD_MANIFEST_PATH,
    TRAINING_FEATURES_PATH,
    build_report,
    write_all,
)


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "numpy is required for Phase 45 artifact matrix checks")
class Phase45MultilingualE5SmallTrainingDeliveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()
        if cls.report.get("status") != "complete":
            cls.report = write_all()

    def test_report_complete_and_selection_recorded(self) -> None:
        written = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(self.report["schema_version"], "phase-45-multilingual-e5-small-training-delivery-v1")
        self.assertEqual(written["status"], "complete")
        self.assertEqual(written["blockers"], [])
        self.assertIn(
            written["selection_decision"],
            {"select_for_staging_shadow_validation", "reject_keep_e5_base_phase25_selected"},
        )
        self.assertTrue(all(build_report()["checks"].values()))

    def test_feature_config_preserves_contract_and_refreshes_normalization(self) -> None:
        config = json.loads(FEATURE_CONFIG_PATH.read_text(encoding="utf-8"))
        matrix = np.load(TRAINING_FEATURES_PATH, allow_pickle=True)

        self.assertEqual(config["approved_features"], APPROVED_FEATURES)
        self.assertEqual(config["embedding_model_metadata"]["embedding_model"], "intfloat/multilingual-e5-small")
        self.assertEqual(config["embedding_model_metadata"]["embedding_dimension"], 384)
        self.assertEqual(config["embedding_model_metadata"]["profile_prefix"], "query:")
        self.assertEqual(config["embedding_model_metadata"]["job_prefix"], "passage:")
        self.assertEqual(list(matrix["feature_names"]), APPROVED_FEATURES)
        self.assertEqual(matrix["X_raw"].shape, (3600, 6))
        self.assertEqual(matrix["X_scaled"].shape, (3600, 6))
        self.assertTrue(np.isfinite(matrix["X_scaled"]).all())
        self.assertGreater(config["normalization"]["std"]["e5_cosine"], 0.0)

    def test_training_evaluation_contains_tensorflow_custom_loop_and_quality_metrics(self) -> None:
        evaluation = json.loads(EVALUATION_REPORT_PATH.read_text(encoding="utf-8"))
        predictions = np.load(PREDICTIONS_PATH, allow_pickle=True)

        self.assertTrue(evaluation["model"]["uses_gradient_tape"])
        self.assertFalse(evaluation["model"]["uses_model_fit"])
        self.assertTrue({"CosineInteractionLayer", "WeightedHuberLoss", "ProductionGateCallback"}.issubset(evaluation["model"]["custom_components"]))
        self.assertTrue(evaluation["model"]["clean_reload_smoke"]["passed"])
        for split in ["validation", "test"]:
            metrics = evaluation["metrics"][split]
            for key in ["mae", "rmse", "r2", "spearman", "score_band_agreement", "high_fit_recall"]:
                self.assertIn(key, metrics)
            ranking = evaluation["ranking_metrics"][split]
            for key in ["ndcg_at_5", "ndcg_at_10", "map_at_10"]:
                self.assertIn(key, ranking)
        self.assertTrue(np.isfinite(predictions["y_pred"]).all())
        self.assertGreaterEqual(float(predictions["y_pred"].min()), 0.0)
        self.assertLessEqual(float(predictions["y_pred"].max()), 1.0)

    def test_baselines_tensorboard_and_indonesian_behavior_are_recorded(self) -> None:
        baseline = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
        tensorboard = json.loads(TENSORBOARD_MANIFEST_PATH.read_text(encoding="utf-8"))
        evaluation = json.loads(EVALUATION_REPORT_PATH.read_text(encoding="utf-8"))

        for model_name in [
            "constant_train_mean",
            "constant_train_median",
            "skill_overlap_only_regression",
            "cosine_only_multilingual_e5_small_regression",
            "e5_base_phase25_scorer",
            "tensorflow_phase45_candidate",
        ]:
            self.assertIn(model_name, baseline["metrics"])
            self.assertIn("validation", baseline["metrics"][model_name])
        self.assertGreaterEqual(tensorboard["tensorboard"]["event_file_count"], 1)
        self.assertTrue(all(item["sha256"] and item["size_bytes"] for item in tensorboard["tensorboard"]["event_files"]))
        self.assertTrue(evaluation["indonesian_behavior_examples"].get("ID") or evaluation["indonesian_behavior_examples"].get("MIXED"))


if __name__ == "__main__":
    unittest.main()
