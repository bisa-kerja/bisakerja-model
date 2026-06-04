from __future__ import annotations

import json
import unittest

from scripts.verify_phase_44_embedding_compatibility_audit import (
    DECISION_PATH,
    DRIFT_REPORT_PATH,
    METADATA_PATH,
    NORMALIZATION_IMPACT_PATH,
    PAIR_FEATURES_PATH,
    REPORT_JSON_PATH,
    build_report,
    write_all,
)


class Phase44EmbeddingCompatibilityAuditTest(unittest.TestCase):
    def test_phase_44_report_is_complete_and_blocks_direct_swap(self) -> None:
        report = write_all()
        written = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(report["schema_version"], "phase-44-embedding-compatibility-audit-v1")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(written["blockers"], [])
        self.assertEqual(written["final_decision"], "retrain_required_recalibration_required_direct_swap_blocked")
        self.assertTrue(all(build_report()["checks"].values()))

    def test_metadata_verifies_multilingual_e5_small_contract(self) -> None:
        write_all()
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

        self.assertEqual(metadata["embedding_model"], "intfloat/multilingual-e5-small")
        self.assertEqual(metadata["embedding_dimension"], 384)
        self.assertEqual(metadata["profile_prefix"], "query:")
        self.assertEqual(metadata["job_prefix"], "passage:")
        self.assertFalse(metadata["fallback_backend_used"])
        self.assertTrue(metadata["pair_embedding_check"]["query_finite_values"])
        self.assertTrue(metadata["pair_embedding_check"]["passage_finite_values"])
        self.assertLessEqual(metadata["representative_embedding_check"]["deterministic_max_abs_delta"], 1e-6)

    def test_drift_report_contains_required_slices_and_correlations(self) -> None:
        write_all()
        drift = json.loads(DRIFT_REPORT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(drift["status"], "complete")
        self.assertEqual(drift["overall"]["count"], 3600)
        self.assertLess(drift["overall"]["pearson"], 0.98)
        self.assertLess(drift["overall"]["spearman"], 0.98)
        for key in ["by_split", "by_language", "by_role_family", "by_pair_type", "by_score_band"]:
            self.assertTrue(drift[key], key)
        self.assertTrue({"ID", "EN", "MIXED", "UNKNOWN"}.issubset(drift["by_language"].keys()))
        self.assertGreaterEqual(len(drift["worst_drift_examples"]), 5)
        self.assertNotIn("profile_text", drift["worst_drift_examples"][0])
        self.assertNotIn("job_text", drift["worst_drift_examples"][0])

    def test_feature_records_and_decision_require_retraining_recalibration(self) -> None:
        write_all()
        features = json.loads(PAIR_FEATURES_PATH.read_text(encoding="utf-8"))["records"]
        normalization = json.loads(NORMALIZATION_IMPACT_PATH.read_text(encoding="utf-8"))
        decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))

        self.assertEqual(len(features), 3600)
        self.assertEqual(features[0]["source_embedding_model"], "intfloat/e5-base-v2")
        self.assertEqual(features[0]["embedding_model"], "intfloat/multilingual-e5-small")
        self.assertTrue(normalization["material_normalization_shift"])
        self.assertEqual(normalization["normalization_decision"], "phase25_mean_std_invalid_for_multilingual_e5_small")
        self.assertFalse(decision["direct_artifact_swap_allowed"])
        self.assertTrue(decision["retrain_required"])
        self.assertTrue(decision["recalibration_required"])


if __name__ == "__main__":
    unittest.main()
