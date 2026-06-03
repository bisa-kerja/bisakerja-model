from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest

from scripts.verify_phase_27_5_27_6_validation_expansion import (
    ATS_FIXTURE_PATH,
    POLICY_PATH,
    REPORT_JSON_PATH,
    REQUIRED_ATS_CASES,
    REQUIRED_LANGUAGES,
    RERANK_FIXTURE_PATH,
    build_report,
)


class Phase27ValidationExpansionTest(unittest.TestCase):
    def test_ats_release_fixture_covers_required_sanitized_cv_cases(self) -> None:
        report = build_report()
        ats = report["steps"]["27.5"]["ats_gate"]
        self.assertEqual(ats["status"], "PASS")
        self.assertGreaterEqual(ats["document_count"], 72)
        for case in REQUIRED_ATS_CASES:
            self.assertGreaterEqual(ats["case_counts"].get(case, 0), 6)
        for language in REQUIRED_LANGUAGES:
            self.assertGreater(ats["language_counts"].get(language, 0), 0)
        self.assertEqual(ats["source_counts"], {"sanitized_document": 72})
        self.assertGreaterEqual(ats["parse_coverage"], 0.80)
        self.assertGreaterEqual(ats["bucket_agreement"], 0.85)
        self.assertGreaterEqual(ats["macro_issue_precision"], 0.85)
        self.assertGreaterEqual(ats["macro_issue_recall"], 0.85)
        self.assertTrue(all(item["safe_fallback"] for item in ats["safe_fallback_cases"]))

        with ATS_FIXTURE_PATH.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), ats["document_count"])
        self.assertIn("scanned_pdf", {row["case_family"] for row in rows})
        self.assertIn("docx", {row["file_family"] for row in rows})

    def test_recommendation_release_fixture_is_larger_than_small_handoff_fixture(self) -> None:
        report = build_report()
        rerank = report["steps"]["27.6"]["recommendation_gate"]
        self.assertEqual(rerank["status"], "PASS")
        self.assertGreaterEqual(rerank["candidate_set_count"], 10)
        self.assertGreaterEqual(rerank["candidate_count"], 100)
        self.assertEqual(rerank["constraint_violation_rate"], 0.0)
        self.assertGreaterEqual(rerank["metrics"]["model"]["ndcg_at_5"], 0.85)
        self.assertGreaterEqual(rerank["metrics"]["model"]["ndcg_at_10"], 0.85)
        self.assertGreaterEqual(rerank["metrics"]["model"]["map_at_10"], 0.80)
        self.assertGreaterEqual(rerank["metrics"]["uplift"]["ndcg_at_10"], 0.05)

        fixture = json.loads(RERANK_FIXTURE_PATH.read_text(encoding="utf-8"))
        for candidate_set in fixture["candidate_sets"]:
            candidate_ids = {item["jobId"] for item in candidate_set["jobCandidates"]}
            per_set = next(item for item in rerank["per_set"] if item["candidateSetId"] == candidate_set["candidateSetId"])
            self.assertTrue(set(per_set["returned_job_ids"]).issubset(candidate_ids))

    def test_policy_and_report_are_written_as_durable_json(self) -> None:
        self.assertTrue(POLICY_PATH.exists())
        self.assertTrue(REPORT_JSON_PATH.exists())
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        written_report = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(policy["schema_version"], "phase-27-5-27-6-release-validation-policy-v1")
        self.assertEqual(written_report["schema_version"], "phase-27-5-27-6-validation-expansion-v1")
        self.assertEqual(written_report["final_decision"], "production-ready")


if __name__ == "__main__":
    unittest.main()
