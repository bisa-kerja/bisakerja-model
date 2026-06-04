from __future__ import annotations

import json
import unittest

from scripts.verify_phase_43_multilingual_e5_small_migration import (
    DECISION_PATH,
    QUALITY_BASELINE_PATH,
    REPORT_JSON_PATH,
    RUNTIME_BASELINE_PATH,
    build_report,
    write_all,
)


class Phase43MultilingualE5SmallMigrationTest(unittest.TestCase):
    def test_phase_43_report_is_complete_and_staging_only(self) -> None:
        report = write_all()
        written = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(report["schema_version"], "phase-43-multilingual-e5-small-migration-decision-v1")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["final_decision"], "approved_for_staging_experiment_only")
        self.assertEqual(written["blockers"], [])
        self.assertTrue(all(build_report()["checks"].values()))

    def test_runtime_baseline_preserves_e5_base_artifact_identity(self) -> None:
        write_all()
        runtime = json.loads(RUNTIME_BASELINE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(runtime["baseline_model"]["embedding_model"], "intfloat/e5-base-v2")
        self.assertEqual(runtime["baseline_model"]["artifact_hash"], "734b8c22f618e05b4b90ba24f4f0d01b475da4674c94e81fc3ed545fae40063d")
        self.assertIn("live", runtime["runtime_commands"])
        self.assertIn("warm_internal_cv_analysis", runtime["endpoint_latency_baseline"])
        self.assertIn("local_live_probe_2026_06_04", runtime["endpoint_latency_baseline"])
        self.assertNotEqual(runtime["resource_baseline"].get("e5_base_hf_cache_size_bytes"), 0)

    def test_quality_baseline_contains_metrics_calibration_and_examples(self) -> None:
        write_all()
        quality = json.loads(QUALITY_BASELINE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(quality["embedding_contract"]["embedding_model"], "intfloat/e5-base-v2")
        self.assertLessEqual(quality["jobfit_metrics"]["validation"]["mae"], 0.02)
        self.assertIn("jobFitAlignment.score", quality["calibration_metrics"])
        self.assertIn("pair_score_band_counts", quality["score_distribution"])
        self.assertGreaterEqual(len(quality["candidate_reranking_examples"]), 1)

    def test_decision_blocks_direct_runtime_swap_and_reserves_new_namespace(self) -> None:
        write_all()
        decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))

        self.assertEqual(decision["decision"], "approved_for_staging_experiment_only")
        self.assertEqual(decision["artifact_namespace"]["reserved_root"], "artifacts/phase_43_multilingual_e5_small_migration/")
        self.assertNotEqual(decision["artifact_namespace"]["reserved_root"], decision["artifact_namespace"]["forbidden_root"])
        self.assertIn("No production/staging env default switch to intfloat/multilingual-e5-small.", decision["stakeholder_decision_record"]["blocked_changes_now"])
        self.assertIn("runtime", decision["go_no_go_thresholds"])
        self.assertIn("quality", decision["go_no_go_thresholds"])
        self.assertIn("rollback", decision["go_no_go_thresholds"])


if __name__ == "__main__":
    unittest.main()
