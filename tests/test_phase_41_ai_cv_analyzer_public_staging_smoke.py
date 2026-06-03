from __future__ import annotations

import json
import unittest
from argparse import Namespace

from scripts.smoke_ai_cv_analyzer_public_staging import build_checks
from scripts.verify_phase_41_ai_cv_analyzer_public_staging_smoke import REPORT_JSON_PATH, build_report, write_all


class Phase41AiCvAnalyzerPublicStagingSmokeTest(unittest.TestCase):
    def test_public_smoke_checks_accept_safe_contract(self) -> None:
        body = {
            "success": True,
            "message": "CV analysis completed",
            "data": {
                "analysisResult": {
                    "schemaVersion": "cv-analysis-v2",
                    "jobRecommendations": [
                        {
                            "jobId": "seed-job-001",
                            "title": "Backend Developer",
                            "companyName": "Nusantara Tech",
                            "matchScore": 88,
                        }
                    ],
                    "generatedCv": {"available": False},
                    "model": {"name": "cv-analyzer-model", "version": "phase25"},
                }
            },
            "meta": {"requestId": "phase41"},
        }
        args = Namespace(latency_budget_ms=5000, persist_result=False)

        checks = build_checks(200, body, 1250.0, args)

        self.assertTrue(all(checks.values()))

    def test_public_smoke_checks_reject_private_fields(self) -> None:
        body = {
            "success": True,
            "message": "CV analysis completed",
            "data": {
                "analysisResult": {
                    "schemaVersion": "cv-analysis-v2",
                    "jobRecommendations": [],
                    "generatedCv": {"available": False},
                    "model": {"name": "cv-analyzer-model", "version": "phase25"},
                    "debug": {"storageKey": "cv/user/file.pdf"},
                }
            },
            "meta": {},
        }
        args = Namespace(latency_budget_ms=5000, persist_result=True)

        checks = build_checks(200, body, 100.0, args)

        self.assertFalse(checks["private_fields_not_returned"])

    def test_phase_41_gate_report_is_go_and_writable(self) -> None:
        report = write_all()
        written = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(report["schema_version"], "phase-41-ai-cv-analyzer-public-staging-smoke-v1")
        self.assertEqual(report["final_decision"], "go")
        self.assertEqual(written["blockers"], [])
        self.assertTrue(all(build_report()["checks"].values()))
        self.assertEqual(written["phase_38_6_status"], "superseded_by_phase_41_public_staging_smoke")


if __name__ == "__main__":
    unittest.main()
