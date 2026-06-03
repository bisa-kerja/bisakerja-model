from __future__ import annotations

import json
import unittest

from scripts.verify_phase_28_contract_realignment import (
    FIXTURE_PATH,
    MATRIX_PATH,
    REPORT_JSON_PATH,
    build_report,
    write_all,
)


class Phase28ContractRealignmentTest(unittest.TestCase):
    def test_internal_contract_fixtures_cover_required_modes_sources_and_failures(self) -> None:
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        positive_sources = {case["form"]["compareSource"] for case in fixture["positiveCases"]}
        positive_modes = {case["form"]["inputMode"] for case in fixture["positiveCases"]}
        negative_ids = {case["caseId"] for case in fixture["negativeCases"]}

        self.assertEqual(fixture["contractRoute"], "POST /internal/model/cv-analysis")
        self.assertEqual(fixture["requestContentType"], "multipart/form-data")
        self.assertEqual(positive_sources, {"BOOKMARK", "JOB_SEARCH", "DIRECT_JOB_DETAIL"})
        self.assertEqual(positive_modes, {"UPLOAD", "REFERENCE"})
        self.assertEqual(negative_ids, {"duplicate-job-ids", "empty-pdf-parse", "missing-candidate-evidence"})

    def test_model_core_response_excludes_backend_owned_public_fields(self) -> None:
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        response = fixture["modelCoreResponseExample"]
        response_text = json.dumps(response)

        self.assertEqual(response["schemaVersion"], "model-core-cv-analysis-v1")
        self.assertIn("parsedCv", response)
        self.assertIn("candidateReranking", response)
        for forbidden in ["topActionables", "sectionReviews", "generatedCv", "companyName", "nextStep", "reason"]:
            self.assertNotIn(f'"{forbidden}"', response_text)

    def test_owner_matrix_maps_openapi_prisma_and_model_fields(self) -> None:
        matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
        entities = {row["entity"] for row in matrix["ownerMatrix"]}

        self.assertEqual(matrix["enumMappings"]["language"]["mapping"], {"id": "ID", "en": "EN"})
        self.assertEqual(
            matrix["enumMappings"]["matchLevel"]["mapping"],
            {"strong": "STRONG", "good": "GOOD", "stretch": "STRETCH"},
        )
        self.assertEqual(
            entities,
            {
                "CvAnalysis.analysisResult",
                "CvAnalysisResult",
                "JobRecommendationRun",
                "JobRecommendationItem",
                "JobListing",
                "JobRequirement",
                "JobSkill",
            },
        )

    def test_phase_28_report_passes_acceptance_criteria(self) -> None:
        report = build_report()

        self.assertEqual(report["schema_version"], "phase-28-contract-realignment-report-v1")
        self.assertEqual(report["final_decision"], "passed")
        self.assertEqual(report["blockers"], [])
        self.assertTrue(all(report["checks"].values()))

    def test_written_report_is_durable_json(self) -> None:
        report = write_all()
        written = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(written["final_decision"], report["final_decision"])
        self.assertEqual(written["fixtures"]["positive_case_ids"], report["fixtures"]["positive_case_ids"])
        self.assertEqual(written["blockers"], [])


if __name__ == "__main__":
    unittest.main()
