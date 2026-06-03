from __future__ import annotations

import json
import unittest

from scripts.verify_phase_32_contract_drift_audit import DRIFT_IDS, REPORT_JSON_PATH, build_report, write_all


class Phase32ContractDriftAuditTest(unittest.TestCase):
    def test_public_openapi_contract_is_extracted_from_generated_snapshot(self) -> None:
        report = build_report()
        public = report["publicOpenApiContract"]

        self.assertEqual(public["path"], "/api/v1/ai/cv-analyzer")
        self.assertEqual(public["requestContentType"], "multipart/form-data")
        self.assertEqual(public["analysisResult"]["schemaVersion"], "cv-analysis-v2")
        self.assertIn("analysisResult", public["publicDataSchema"]["required"])
        self.assertIn("generatedCv", public["analysisResult"]["required"])
        self.assertEqual(public["analysisResult"]["jobRecommendations"], "max 5; strict objects; jobId/companyName nullable; matchScore 0-100")

    def test_backend_and_model_api_contracts_capture_current_drift_sources(self) -> None:
        report = build_report()
        backend = report["backendModelApiClientContract"]
        actual = report["currentModelApiContract"]

        self.assertEqual(backend["modelApiEndpoint"], "POST /internal/model/cv-analysis")
        self.assertIn("jobRoles[]", backend["multipartRequest"]["fields"])
        self.assertIn("raw strict", backend["expectedModelCoreResponse"]["shape"])
        self.assertEqual(actual["responseBehavior"]["envelope"], "returns { success: true, message, data, error: null }")
        self.assertIn("form.get('jobRoles')", actual["multipartBehavior"]["jobRoles"])

    def test_drift_matrix_covers_required_payload_language_error_and_privacy_items(self) -> None:
        report = build_report()
        drift_ids = {item["id"] for item in report["driftMatrix"]}

        self.assertEqual(drift_ids, set(DRIFT_IDS))
        for required in [
            "response-envelope",
            "requirements-object-vs-string",
            "numericSignals-vs-numericFeatures",
            "error-envelope-mapping",
            "language-default-policy",
            "security-privacy-fields",
        ]:
            self.assertIn(required, drift_ids)

    def test_canonical_contract_freezes_owner_boundary_before_implementation(self) -> None:
        report = build_report()
        canonical = report["canonicalContractDecision"]

        self.assertEqual(report["final_decision"], "review_required")
        self.assertEqual(report["blockers"], [])
        self.assertTrue(all(report["checks"].values()))
        self.assertIn("raw model-core JSON", canonical["internalResponseShape"])
        self.assertIn("Backend owns public cv-analysis-v2", canonical["backendOwner"])
        self.assertIn("English", canonical["languagePolicy"])
        self.assertIn("separate audit scope", canonical["aiCvGenerateScope"])
        self.assertIn("No request/response implementation phase", canonical["reviewGate"])

    def test_report_is_written_as_durable_json_and_markdown(self) -> None:
        report = write_all()
        written = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(written["schema_version"], "phase-32-ai-cv-analyzer-contract-drift-audit-v1")
        self.assertEqual(written["final_decision"], report["final_decision"])
        self.assertEqual(written["blockers"], [])
        self.assertIn("driftMatrix", written)


if __name__ == "__main__":
    unittest.main()
