from __future__ import annotations

import json
import unittest

from scripts.verify_phase_27_10_requirement_matrix import (
    REPORT_JSON_PATH,
    REQUIREMENT_IDS,
    build_report,
    write_all,
)


class Phase27RequirementMatrixTest(unittest.TestCase):
    def test_requirement_matrix_covers_all_required_sections(self) -> None:
        report = build_report()
        rows = {item["requirement_id"]: item for item in report["matrix"]}

        self.assertEqual(report["schema_version"], "phase-27-10-requirement-matrix-v1")
        self.assertEqual(set(rows), set(REQUIREMENT_IDS))
        self.assertEqual(report["coverage"]["missing_ids"], [])
        self.assertIn("1.1", rows)
        self.assertIn("deliverables", rows)
        self.assertTrue(rows["1.1"]["checks"]["tensorflow_functional_api_gate_passed"])
        self.assertTrue(rows["1.5"]["checks"]["mae_target_le_0_02_gate_passed"])

    def test_requirement_matrix_blocks_until_real_runtime_smoke_and_tracked_release_evidence(self) -> None:
        report = build_report()
        rows = {item["requirement_id"]: item for item in report["matrix"]}

        self.assertEqual(report["final_decision"], "blocked")
        self.assertFalse(rows["3.2"]["checks"]["live_fastapi_health_model_info_inference_smoke_passed"])
        self.assertIn(rows["3.2"]["status"], {"failed-gate", "tracking-blocked"})
        self.assertFalse(rows["deliverables"]["checks"]["runtime_smoke_production_passed"])
        self.assertTrue(any(item["requirement_id"] == "3.2" for item in report["blockers"]))

    def test_written_report_is_durable_json_with_evidence_hashes(self) -> None:
        report = write_all()
        written = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))
        row_11 = next(item for item in written["matrix"] if item["requirement_id"] == "1.1")
        requirement_evidence = next(item for item in row_11["evidence"] if item["path"] == "REQUIREMENT.md")

        self.assertEqual(written["schema_version"], "phase-27-10-requirement-matrix-v1")
        self.assertEqual(written["final_decision"], report["final_decision"])
        self.assertTrue(requirement_evidence["exists"])
        self.assertTrue(requirement_evidence["tracked"])
        self.assertRegex(requirement_evidence["sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(requirement_evidence["size_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
