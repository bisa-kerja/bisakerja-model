from __future__ import annotations

import json
import sys
import unittest

from scripts.verify_phase_27_8_model_card_manifest_refresh import (
    ARTIFACT_MANIFEST_PATH,
    MODEL_CARD_PATH,
    REPORT_JSON_PATH as REPORT_27_8_JSON_PATH,
    build_report as build_27_8_report,
    sha256_file,
    write_all as write_27_8_all,
)
from scripts.verify_phase_27_9_model_api_production_smoke import (
    REPORT_JSON_PATH as REPORT_27_9_JSON_PATH,
    build_report as build_27_9_report,
    write_all as write_27_9_all,
)


class Phase27ModelCardRuntimeSmokeTest(unittest.TestCase):
    def test_phase_27_8_refresh_writes_hashes_and_score_semantics(self) -> None:
        report = write_27_8_all()
        self.assertEqual(report["schema_version"], "phase-27-8-model-card-manifest-refresh-v1")
        self.assertEqual(report["final_decision"], "refresh-complete")

        model_card = json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))
        manifest = json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))
        model_entry = next(entry for entry in manifest["artifacts"] if entry["artifact_id"] == "model_card")

        self.assertIn("phase_27_8_release_refresh", model_card)
        self.assertIn("phase_27_8_release_refresh", manifest)
        self.assertIn("jobFitAlignment.score", model_card["score_semantics"])
        self.assertIn("atsFriendliness.score", model_card["score_semantics"])
        self.assertIn("recommendations[].matchScore", model_card["score_semantics"])
        self.assertFalse(model_card["phase_27_8_release_refresh"]["production_claim_allowed"])
        self.assertEqual(model_entry["sha256"], sha256_file(MODEL_CARD_PATH))

        written = json.loads(REPORT_27_8_JSON_PATH.read_text(encoding="utf-8"))
        gates = {item["check"]: item["status"] for item in written["gates"]}
        self.assertEqual(gates["manifest_model_card_hash_matches_current_file"], "PASS")
        self.assertEqual(gates["production_claim_remains_blocked_until_all_phase27_gates"], "PASS")

    def test_phase_27_8_read_only_report_detects_refreshed_manifest(self) -> None:
        write_27_8_all()
        report = build_27_8_report(write=False)
        gates = {item["check"]: item["status"] for item in report["gates"]}
        self.assertEqual(gates["model_card_contains_phase_27_8_release_refresh"], "PASS")
        self.assertEqual(gates["artifact_manifest_contains_phase_27_8_release_refresh"], "PASS")
        self.assertEqual(report["final_decision"], "refresh-complete")

    def test_phase_27_9_blocks_without_real_python_313_serving_smoke(self) -> None:
        report = build_27_9_report(run_live=False, run_tests=True)
        gates = {item["check"]: item for item in report["gates"]}

        self.assertEqual(report["schema_version"], "phase-27-9-model-api-production-smoke-v1")
        self.assertEqual(gates["root_requirements_pin_serving_runtime"]["status"], "PASS")
        self.assertIn("python_3_13_serving_runtime", gates)
        self.assertIn("live_fastapi_health_model_info_inference_smoke", gates)
        self.assertEqual(report["final_decision"], "blocked")
        if sys.version_info[:2] != (3, 13):
            self.assertEqual(gates["python_3_13_serving_runtime"]["status"], "FAIL")

    def test_phase_27_9_written_report_is_durable_json(self) -> None:
        report = write_27_9_all(run_live=False)
        written = json.loads(REPORT_27_9_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(written["schema_version"], "phase-27-9-model-api-production-smoke-v1")
        self.assertEqual(written["policy"]["fake_model_or_fake_embedding_allowed_for_production_smoke"], False)
        self.assertEqual(written["final_decision"], report["final_decision"])
        self.assertIn("real_keras_artifact_required", written["policy"])


if __name__ == "__main__":
    unittest.main()
