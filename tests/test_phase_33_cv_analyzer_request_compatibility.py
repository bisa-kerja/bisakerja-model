from __future__ import annotations

import json
import unittest

from model_api.app import _build_cv_payload_from_multipart, create_app
from model_api.config import RuntimeConfig
from model_api.errors import ContractValidationError
from model_api.inference import InferenceService, RuntimeState
from model_api.schemas import ModelArtifactIdentity, ModelIdentity, parse_cv_analysis_model_core_request


class FakeModel:
    name = "fake_phase25_model"

    def predict(self, values, verbose=0):
        return [[0.9] for _ in values]


class FakeE5Backend:
    model_name = "intfloat/e5-base-v2"
    backend_name = "sentence-transformers"

    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeMultipartForm(dict):
    def __init__(self, values):
        super().__init__()
        self._values = values
        for key, value in values.items():
            if isinstance(value, list):
                if value:
                    self[key] = value[-1]
            else:
                self[key] = value

    def get(self, key, default=None):
        return dict.get(self, key, default)

    def getlist(self, key):
        value = self._values.get(key, [])
        return value if isinstance(value, list) else [value]


class Phase33CvAnalyzerRequestCompatibilityTest(unittest.TestCase):
    def backend_candidate(self) -> dict[str, object]:
        return {
            "jobId": "job-33-001",
            "scoringInput": {
                "titleText": "Backend Developer",
                "descriptionText": "Build REST APIs and PostgreSQL services.",
                "requirementSummary": "TypeScript, PostgreSQL, API testing.",
                "requiredSkills": ["typescript", "postgresql", "rest api"],
                "requirements": [
                    {"type": "SKILL", "value": "Build REST APIs", "priority": "MUST_HAVE"},
                    {"type": "EXPERIENCE", "value": "Maintain PostgreSQL schema", "priority": "IMPORTANT"},
                ],
                "roleFamily": "backend",
                "experienceLevel": "ENTRY_LEVEL",
                "workType": "REMOTE",
                "numericSignals": {"requirement_coverage": 0.75, "jobExperienceYears": 1.0},
            },
            "backendMetadata": {
                "title": "Backend Developer",
                "companyName": "Nusantara Tech",
                "locationDisplay": "Jakarta, Indonesia",
                "sourceUpdatedAt": "2026-06-03T00:00:00.000Z",
            },
        }

    def base_payload(self) -> dict[str, object]:
        return {
            "requestId": "req-phase33",
            "inputVersion": "cv-analyzer-v1",
            "language": "en",
            "inputMode": "UPLOAD",
            "compareSource": "JOB_SEARCH",
            "profile": {"cvText": "Backend developer with TypeScript and PostgreSQL", "targetRoles": ["Backend Developer"]},
            "jobCandidates": [self.backend_candidate()],
            "rankingPolicy": {
                "maxRecommendations": 5,
                "requireCandidateJobIds": True,
                "deduplicateByJobId": True,
                "backendOwnsHydration": True,
            },
            "maxRecommendations": 5,
        }

    def test_backend_requirement_objects_numeric_signals_and_location_display_are_accepted(self) -> None:
        request = parse_cv_analysis_model_core_request(self.base_payload())
        candidate = request.jobCandidates[0]
        scoring = candidate.model_scoring_input

        self.assertEqual(scoring.requirements, ("Build REST APIs", "Maintain PostgreSQL schema"))
        self.assertEqual(scoring.numericFeatures["requirement_coverage"], 0.75)
        self.assertEqual(scoring.numericFeatures["jobExperienceYears"], 1.0)
        self.assertEqual(candidate.backendMetadata.locationDisplay, "Jakarta, Indonesia")

    def test_backend_modes_and_compare_sources_parse_without_contract_drift(self) -> None:
        cases = (("UPLOAD", "JOB_SEARCH"), ("REFERENCE", "BOOKMARK"), ("UPLOAD", "DIRECT_JOB_DETAIL"))
        for input_mode, compare_source in cases:
            with self.subTest(input_mode=input_mode, compare_source=compare_source):
                payload = self.base_payload()
                payload["inputMode"] = input_mode
                payload["compareSource"] = compare_source
                request = parse_cv_analysis_model_core_request(payload)
                self.assertEqual(request.inputMode, input_mode)
                self.assertEqual(request.compareSource, compare_source)

    def test_repeated_multipart_job_roles_parse_like_backend_formdata_append(self) -> None:
        form = FakeMultipartForm(
            {
                "requestId": "req-phase33-form",
                "language": "en",
                "inputMode": "UPLOAD",
                "compareSource": "JOB_SEARCH",
                "jobRoles": ["Backend Developer", "Software Engineer"],
                "jobCandidates": json.dumps([self.backend_candidate()]),
                "rankingPolicy": json.dumps(
                    {
                        "maxRecommendations": 5,
                        "requireCandidateJobIds": True,
                        "deduplicateByJobId": True,
                        "backendOwnsHydration": True,
                    }
                ),
            }
        )
        pdf = b"%PDF-1.4\n1 0 obj<</Type /Page>>stream\n(Summary Backend developer) Tj\n(Skills TypeScript PostgreSQL REST API) Tj\n(Experience 2020) Tj\nendstream\n%%EOF"

        payload = _build_cv_payload_from_multipart(form, pdf, RuntimeConfig.from_env({}))
        payload.pop("_parsedPdfEvidence")
        request = parse_cv_analysis_model_core_request(payload)

        self.assertEqual(request.profile.targetRoles, ("Backend Developer", "Software Engineer"))
        self.assertEqual(request.jobCandidates[0].model_scoring_input.requirements, ("Build REST APIs", "Maintain PostgreSQL schema"))

    def test_unsafe_requirement_shapes_numeric_keys_and_metadata_fail_closed(self) -> None:
        payload = self.base_payload()
        scoring = payload["jobCandidates"][0]["scoringInput"]
        scoring["requirements"] = [{"type": "SKILL", "value": "Python", "priority": "MUST_HAVE", "secret": "x"}]
        scoring["numericSignals"] = {"arbitrary_model_feature": 1.0}
        scoring["numericFeatures"] = {"requirement_coverage": 0.5}
        payload["jobCandidates"][0]["backendMetadata"]["storageKey"] = "private/object/key"

        with self.assertRaises(ContractValidationError) as ctx:
            parse_cv_analysis_model_core_request(payload)

        message = str(ctx.exception)
        self.assertIn("requirements[0].secret is not allowed", message)
        self.assertIn("numericSignals.arbitrary_model_feature is not an approved Phase 25 numeric signal", message)
        self.assertIn("must provide numericSignals or numericFeatures, not both", message)
        self.assertIn("backendMetadata.storageKey is not allowed", message)

    def test_candidate_policy_rejects_duplicates_empty_candidates_and_bad_ranking_policy(self) -> None:
        duplicate_payload = self.base_payload()
        duplicate_payload["jobCandidates"] = [self.backend_candidate(), self.backend_candidate()]
        with self.assertRaisesRegex(ContractValidationError, "jobId duplicate"):
            parse_cv_analysis_model_core_request(duplicate_payload)

        empty_payload = self.base_payload()
        empty_payload["jobCandidates"] = []
        with self.assertRaisesRegex(ContractValidationError, "must contain at least 1 candidate"):
            parse_cv_analysis_model_core_request(empty_payload)

        oversized_payload = self.base_payload()
        oversized_payload["jobCandidates"] = [dict(self.backend_candidate(), jobId=f"job-{index}") for index in range(51)]
        with self.assertRaisesRegex(ContractValidationError, "jobCandidates max items 50"):
            parse_cv_analysis_model_core_request(oversized_payload)

        zero_recommendations_payload = self.base_payload()
        zero_recommendations_payload["rankingPolicy"]["maxRecommendations"] = 0
        zero_recommendations_payload["maxRecommendations"] = 0
        self.assertEqual(parse_cv_analysis_model_core_request(zero_recommendations_payload).maxRecommendations, 0)

        bad_policy = self.base_payload()
        bad_policy["rankingPolicy"]["backendOwnsHydration"] = False
        with self.assertRaisesRegex(ContractValidationError, "backendOwnsHydration must be true"):
            parse_cv_analysis_model_core_request(bad_policy)

        too_many_recommendations = self.base_payload()
        too_many_recommendations["rankingPolicy"]["maxRecommendations"] = 6
        too_many_recommendations["maxRecommendations"] = 6
        with self.assertRaisesRegex(ContractValidationError, "maxRecommendations must be integer 0-5"):
            parse_cv_analysis_model_core_request(too_many_recommendations)

    def test_malformed_oversized_empty_pdf_and_empty_roles_fail_closed(self) -> None:
        form = FakeMultipartForm(
            {
                "requestId": "req-phase33-empty",
                "language": "en",
                "inputMode": "UPLOAD",
                "compareSource": "JOB_SEARCH",
                "jobRoles": [],
                "jobCandidates": json.dumps([self.backend_candidate()]),
                "rankingPolicy": json.dumps({"maxRecommendations": 5, "requireCandidateJobIds": True, "deduplicateByJobId": True, "backendOwnsHydration": True}),
            }
        )
        pdf = b"%PDF-1.4\n1 0 obj<</Type /Page /XObject 2 0 R>>\n%%EOF"

        with self.assertRaisesRegex(ContractValidationError, "no extractable PDF text"):
            _build_cv_payload_from_multipart(form, pdf, RuntimeConfig.from_env({}))

        valid_pdf = b"%PDF-1.4\n1 0 obj<</Type /Page>>stream\n(Summary Backend developer) Tj\n(Skills TypeScript PostgreSQL REST API) Tj\n(Experience 2020) Tj\nendstream\n%%EOF"
        with self.assertRaisesRegex(ContractValidationError, "jobRoles must contain at least 1 role"):
            _build_cv_payload_from_multipart(form, valid_pdf, RuntimeConfig.from_env({}))
        with self.assertRaisesRegex(ContractValidationError, "cvFile must start with PDF magic bytes"):
            _build_cv_payload_from_multipart(form, b"not a pdf", RuntimeConfig.from_env({}))
        with self.assertRaisesRegex(ContractValidationError, "cvFile exceeds MODEL_API_MAX_PDF_BYTES"):
            _build_cv_payload_from_multipart(form, valid_pdf, RuntimeConfig(max_pdf_bytes=10))

        payload = self.base_payload()
        payload["profile"]["targetRoles"] = ["Role"] * 11
        with self.assertRaisesRegex(ContractValidationError, "targetRoles max items 10"):
            parse_cv_analysis_model_core_request(payload)

    def test_fastapi_multipart_endpoint_accepts_backend_repeated_job_roles_when_serving_deps_exist(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("FastAPI serving dependencies are not installed")

        identity = ModelIdentity(
            name="phase25",
            version="test",
            artifact=ModelArtifactIdentity(path="artifacts/test.keras", sha256="0" * 64),
        )
        service = InferenceService(
            model=FakeModel(),
            state=RuntimeState(ready=True, model_identity=identity, artifact_manifest_phase="phase_25", message="ready"),
        )
        app = create_app(config=RuntimeConfig.from_env({}), service=service, embedding_backend=FakeE5Backend())
        client = TestClient(app)
        pdf = b"%PDF-1.4\n1 0 obj<</Type /Page>>stream\n(Summary Backend developer) Tj\n(Skills TypeScript PostgreSQL REST API) Tj\n(Experience 2020) Tj\nendstream\n%%EOF"
        multipart = [
            ("requestId", (None, "req-phase33-endpoint")),
            ("language", (None, "en")),
            ("inputMode", (None, "UPLOAD")),
            ("compareSource", (None, "JOB_SEARCH")),
            ("jobRoles", (None, "Backend Developer")),
            ("jobRoles", (None, "Software Engineer")),
            ("jobCandidates", (None, json.dumps([self.backend_candidate()]))),
            (
                "rankingPolicy",
                (
                    None,
                    json.dumps(
                        {
                            "maxRecommendations": 5,
                            "requireCandidateJobIds": True,
                            "deduplicateByJobId": True,
                            "backendOwnsHydration": True,
                        }
                    ),
                ),
            ),
            ("cvFile", ("cv.pdf", pdf, "application/pdf")),
        ]

        response = client.post("/internal/model/cv-analysis", files=multipart)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["schemaVersion"], "model-core-cv-analysis-v1")
        self.assertEqual(body["candidateReranking"]["recommendations"][0]["jobId"], "job-33-001")


if __name__ == "__main__":
    unittest.main()
