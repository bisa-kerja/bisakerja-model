from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.verify_phase_46_calibration_model_card_manifest_handoff_refresh import (
    ARTIFACT_MANIFEST_PATH,
    BUCKETS,
    FEATURE_CONFIG_PATH,
    HANDOFF_FIXTURES_PATH,
    HANDOFF_VALIDATION_PATH,
    INFERENCE_SMOKE_PATH,
    MODEL_CARD_PATH,
    REPORT_JSON_PATH,
    SCORE_CALIBRATION_PATH,
    TF_FEATURE_CONFIG_PATH,
    build_report,
)


class Phase46CalibrationModelCardManifestHandoffRefreshTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()

    def test_report_complete_and_acceptance_checks_pass(self) -> None:
        self.assertEqual(self.report["schema_version"], "phase-46-calibration-model-card-manifest-handoff-refresh-v1")
        self.assertEqual(self.report["status"], "complete")
        self.assertEqual(self.report["blockers"], [])
        self.assertTrue(all(self.report["checks"].values()))
        self.assertEqual(json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))["status"], "complete")

    def test_calibration_covers_required_outputs_metrics_buckets_and_slices(self) -> None:
        calibration = json.loads(SCORE_CALIBRATION_PATH.read_text(encoding="utf-8"))
        expected_buckets = {bucket["bucket"] for bucket in BUCKETS}

        self.assertEqual(calibration["schema_version"], "phase-46-score-calibration-v1")
        self.assertEqual(set(calibration["metrics"]), {"jobFitAlignment.score", "atsFriendliness.score", "recommendations[].matchScore"})
        for split in ["train", "validation", "test", "all"]:
            metrics = calibration["metrics"]["jobFitAlignment.score"][split]
            for key in ["ece_points", "mce_points", "mae_points", "within_10_points_rate", "score_band_agreement", "bucket_mae_points"]:
                self.assertIn(key, metrics)
            self.assertTrue(metrics["passed"])
        self.assertEqual({row["bucket"] for row in calibration["tables"] if row["output"] == "jobFitAlignment.score" and row["split"] == "validation"}, expected_buckets)
        self.assertIn("language", calibration["metrics"]["jobFitAlignment.score"]["slice_calibration"])

    def test_model_card_manifest_and_configs_are_self_contained_multilingual_e5_small(self) -> None:
        model_card = json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))
        manifest = json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))
        tf_config = json.loads(TF_FEATURE_CONFIG_PATH.read_text(encoding="utf-8"))
        feature_config = json.loads(FEATURE_CONFIG_PATH.read_text(encoding="utf-8"))

        self.assertEqual(model_card["model"]["embedding_model"], "intfloat/multilingual-e5-small")
        self.assertEqual(model_card["rollback_artifact"]["embedding_model"], "intfloat/e5-base-v2")
        self.assertFalse(model_card["readiness"]["phase25_mixed_metadata_used"])
        self.assertEqual(tf_config["embedding_model_metadata"]["embedding_model"], "intfloat/multilingual-e5-small")
        self.assertEqual(feature_config["embedding_model_metadata"]["embedding_model"], "intfloat/multilingual-e5-small")
        runtime_entries = [entry for entry in manifest["artifacts"] if entry["required_for_inference"]]
        self.assertGreaterEqual(len(runtime_entries), 5)
        for entry in runtime_entries:
            self.assertTrue(entry["sha256"])
            self.assertGreater(entry["size_bytes"], 0)
            self.assertFalse(entry["path"].startswith("artifacts/phase_25_tensorflow_training_delivery"))
            self.assertEqual(entry["embedding_model_metadata"]["embedding_model"], "intfloat/multilingual-e5-small")

    def test_handoff_fixtures_remain_model_core_only_and_backend_compatible(self) -> None:
        fixtures = json.loads(HANDOFF_FIXTURES_PATH.read_text(encoding="utf-8"))
        validation = json.loads(HANDOFF_VALIDATION_PATH.read_text(encoding="utf-8"))
        positive = fixtures["positive"]
        cv_output = positive["cvAnalysisCoreOutput"]
        rerank_output = positive["candidateRerankingCoreOutput"]

        self.assertEqual(validation["status"], "complete")
        self.assertTrue(all(validation["checks"].values()))
        self.assertEqual(cv_output["schemaVersion"], "model-core-cv-analysis-v1")
        self.assertEqual(rerank_output["schemaVersion"], "model-core-candidate-reranking-v1")
        self.assertLessEqual(len(rerank_output["recommendations"]), 5)
        self.assertEqual(len(validation["recommendation_ids"]), len(set(validation["recommendation_ids"])))
        for item in rerank_output["recommendations"]:
            self.assertIn(item["jobId"], positive["candidateRerankingCoreRequest"]["candidateJobIds"])
            self.assertGreaterEqual(item["matchScore"], 0)
            self.assertLessEqual(item["matchScore"], 100)
        self.assertEqual(validation["leaked_backend_owned_fields"], [])

    def test_clean_reload_smoke_records_registered_custom_objects(self) -> None:
        smoke = json.loads(INFERENCE_SMOKE_PATH.read_text(encoding="utf-8"))

        self.assertTrue(smoke["clean_reload_with_registered_custom_objects"])
        self.assertTrue(smoke["score_bounds_passed"])
        self.assertTrue({"CosineInteractionLayer", "WeightedHuberLoss", "ProductionGateCallback"}.issubset(set(smoke["custom_objects"])))
        self.assertEqual(smoke["prediction_count"], 8)
        self.assertGreaterEqual(smoke["prediction_min"], 0.0)
        self.assertLessEqual(smoke["prediction_max"], 1.0)
        self.assertTrue(Path(smoke["model"]["artifact"]["path"]).exists())


if __name__ == "__main__":
    unittest.main()
