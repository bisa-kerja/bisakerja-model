from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
import unittest

from model_api.app import _signal_list, build_cv_analysis_response_payload
from model_api.features import E5_MODEL_NAME, TensorFlowFeatureConfig
from model_api.inference import InferenceService, RuntimeState, ScoreCalibrationPolicy
from model_api.pdf_parser import ats_score_from_pdf_evidence, normalized_skills_from_text, parse_pdf_bytes
from model_api.schemas import (
    CandidateJobInput,
    CandidateScoringInput,
    CvAnalysisModelCoreRequest,
    ModelArtifactIdentity,
    ModelIdentity,
    SanitizedProfileInput,
)
from model_api.validators import validate_model_core_payload

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "artifacts/phase_51_ai_cv_analyzer_benchmark_matrix/benchmark_matrix.json"
FORBIDDEN_INDONESIAN_OUTPUT_PHRASES = (
    "laporan keuangan",
    "menganalisis",
    "pengambilan keputusan",
    "manajemen proyek",
    "analisis data",
    "kemampuan komunikasi",
    "pengalaman kerja",
    "pendidikan",
    "sertifikasi",
    "keahlian",
    "keterampilan",
    "kontak tidak terdeteksi",
    "sinyal dampak terukur",
)
FORBIDDEN_GENERIC_OUTPUT_PHRASES = (
    "model-core evidence prepared for backend genai wrapper",
    "generic improvement text",
    "hiring outcome",
    "raw cv",
)
FORBIDDEN_RAW_CV_KEYS = ("rawCvText", "raw_cv_text")
FORBIDDEN_BACKEND_OWNED_FIELDS = ("topActionables", "sectionReviews", "reason", "nextStep", "generatedCv", "companyName")
BENCHMARK_LATENCY_BUDGET_MS = 30_000


class FakeModel:
    name = "fake_phase51_model"

    def predict(self, values, verbose=0):
        return [[0.64] for _ in values]


class FakeE5Backend:
    model_name = E5_MODEL_NAME
    backend_name = "sentence-transformers"

    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class Phase51AiCvAnalyzerBenchmarkOutputQualityTest(unittest.TestCase):
    def fixture(self) -> dict:
        return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def feature_config(self) -> TensorFlowFeatureConfig:
        features = (
            "e5_cosine",
            "skill_overlap",
            "requirement_coverage",
            "role_match",
            "experience_match",
            "experience_gap_years_clipped",
        )
        return TensorFlowFeatureConfig(approved_features=features, mean={name: 0.0 for name in features}, std={name: 1.0 for name in features})

    def service(self) -> InferenceService:
        identity = ModelIdentity(
            name="phase51",
            version="test",
            artifact=ModelArtifactIdentity(path="artifacts/test.keras", sha256="0" * 64),
        )
        return InferenceService(
            model=FakeModel(),
            state=RuntimeState(ready=True, model_identity=identity, artifact_manifest_phase="phase_25", message="ready"),
        )

    def benchmark_request_for_case(self, case_id: str, target_role: str | None = None, evidence=None) -> CvAnalysisModelCoreRequest:
        fixture = self.fixture()
        cv_case = next(item for item in fixture["benchmarkFiles"] if item["caseId"] == case_id)
        roles = [role for role in fixture["targetRoles"] if target_role is None or role["role"] == target_role]
        role_skills = [skill for role in roles for skill in role["scoringInput"]["requiredSkills"]]
        if evidence is None:
            evidence = parse_pdf_bytes((ROOT / cv_case["path"]).read_bytes(), max_bytes=5_000_000, max_pages=10)
        candidates = tuple(
            CandidateJobInput(
                jobId=role["candidateJobId"],
                scoringInput=CandidateScoringInput(**role["scoringInput"]),
            )
            for role in roles
        )
        return CvAnalysisModelCoreRequest(
            requestId=f"req-51-{case_id}" + (f"-{target_role}" if target_role else ""),
            inputVersion="cv-analyzer-v1",
            language="en",
            inputMode="UPLOAD",
            compareSource="JOB_SEARCH",
            profile=SanitizedProfileInput(
                cvText=evidence.text,
                profileText=evidence.text[:2000],
                normalizedSkills=normalized_skills_from_text(evidence.text, role_skills),
                targetRoles=tuple(role["role"] for role in roles),
                experienceYears=evidence.estimated_experience_years,
                detectedCvSectionNames=evidence.section_names,
            ),
            jobCandidates=candidates,
            maxRecommendations=min(3, len(candidates)),
        )

    def collect_strings(self, value) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            output: list[str] = []
            for child in value.values():
                output.extend(self.collect_strings(child))
            return output
        if isinstance(value, (list, tuple)):
            output: list[str] = []
            for child in value:
                output.extend(self.collect_strings(child))
            return output
        return []

    def assert_english_only_policy(self, payload: dict) -> None:
        checked = {
            "overallImpression": payload["overallImpression"],
            "jobFitAlignment": payload["jobFitAlignment"],
            "atsFriendliness": payload["atsFriendliness"],
            "candidateReranking": payload["candidateReranking"],
        }
        text = "\n".join(self.collect_strings(checked)).casefold()
        for phrase in FORBIDDEN_INDONESIAN_OUTPUT_PHRASES:
            self.assertNotIn(phrase, text)

    def assert_no_raw_cv_leakage(self, payload: dict, source_text: str) -> None:
        serialized = json.dumps(payload, sort_keys=True)
        folded = serialized.casefold()
        for key in (*FORBIDDEN_RAW_CV_KEYS, *FORBIDDEN_BACKEND_OWNED_FIELDS):
            self.assertNotIn(f'"{key}"'.casefold(), folded)
        compact_source = " ".join(source_text.split())
        snippets = [compact_source[index : index + 120] for index in (0, max(0, len(compact_source) // 2 - 60))]
        for snippet in snippets:
            if len(snippet) >= 80:
                self.assertNotIn(snippet.casefold(), folded)

    def assert_non_generic_copy(self, payload: dict) -> None:
        text = "\n".join(self.collect_strings(payload)).casefold()
        for phrase in FORBIDDEN_GENERIC_OUTPUT_PHRASES:
            self.assertNotIn(phrase, text)
        evidence = payload["overallImpression"]["evidence"]
        self.assertTrue(any(item.startswith("summary:") for item in evidence))
        self.assertTrue(any("next improvement:" in item for item in evidence))

    def assert_ats_issue_precision(self, payload: dict) -> None:
        for issue in payload["atsFriendliness"]["detectedIssues"]:
            self.assertIn(":", issue)
            self.assertIn("Fix:", issue)
            self.assertNotEqual(issue, "section evidence not provided")

    def test_benchmark_matrix_locks_required_upload_cv_role_cases_and_parser_snapshots(self) -> None:
        fixture = self.fixture()
        self.assertEqual(fixture["schemaVersion"], "phase-51-ai-cv-analyzer-benchmark-matrix-v1")
        self.assertEqual(fixture["inputMode"], "UPLOAD")
        self.assertEqual(len(fixture["benchmarkFiles"]), 3)
        self.assertEqual(len(fixture["targetRoles"]), 3)
        self.assertEqual(fixture["matrixExpectations"]["requiredCaseCount"], 9)
        self.assertEqual(len(fixture["benchmarkFiles"]) * len(fixture["targetRoles"]), 9)

        for cv_case in fixture["benchmarkFiles"]:
            path = ROOT / cv_case["path"]
            pdf_bytes = path.read_bytes()
            self.assertEqual(hashlib.sha256(pdf_bytes).hexdigest(), cv_case["sha256"])
            self.assertEqual(path.stat().st_size, cv_case["sizeBytes"])

            evidence = parse_pdf_bytes(pdf_bytes, max_bytes=5_000_000, max_pages=10)
            ats_score, issues, _fallback = ats_score_from_pdf_evidence(evidence)
            snapshot = cv_case["parserSnapshot"]

            self.assertEqual(evidence.page_count, snapshot["expectedPageCount"])
            self.assertEqual(evidence.parse_quality, snapshot["expectedParseQuality"])
            self.assertGreaterEqual(len(evidence.text), snapshot["minTextLength"])
            self.assertGreaterEqual(evidence.word_count, snapshot["wordCountRange"][0])
            self.assertLessEqual(evidence.word_count, snapshot["wordCountRange"][1])
            for section in snapshot["expectedSections"]:
                self.assertIn(section, evidence.section_names)
            for title in snapshot.get("expectedRoleTitlesInclude", []):
                self.assertIn(title, evidence.role_titles)
            for company in snapshot.get("expectedCompaniesInclude", []):
                self.assertIn(company, evidence.company_names)
            for issue in snapshot["expectedIssuesInclude"]:
                self.assertIn(issue, issues)
            self.assertGreaterEqual(ats_score, snapshot["atsScoreRange"][0])
            self.assertLessEqual(ats_score, snapshot["atsScoreRange"][1])
            skills = normalized_skills_from_text(evidence.text, [skill for role in fixture["targetRoles"] for skill in role["scoringInput"]["requiredSkills"]])
            for skill in cv_case["expectedSkillsInclude"]:
                self.assertIn(skill, skills)

    def test_job_fit_evidence_changes_by_role_without_unrequested_finance_gaps(self) -> None:
        payload = build_cv_analysis_response_payload(
            self.benchmark_request_for_case("salman-abdurrahman-ats"),
            service=self.service(),
            feature_config=self.feature_config(),
            calibration_policy=ScoreCalibrationPolicy(),
            embedding_backend=FakeE5Backend(),
            environment="test",
        )
        recommendations = {item["jobId"]: item for item in payload["candidateReranking"]["recommendations"]}
        software = recommendations["phase51-software-engineer"]
        product = recommendations["phase51-product-manager"]
        data = recommendations["phase51-data-analyst"]

        self.assertIn("react", software["matchedSkills"])
        self.assertIn("node.js", software["matchedSkills"])
        self.assertIn("sql", software["matchedSkills"])
        self.assertIn("product management", product["missingSkills"])
        self.assertIn("project management", product["missingSkills"])
        self.assertIn("sql", data["matchedSkills"])
        self.assertIn("power bi", data["missingSkills"])
        self.assertNotEqual((software["matchedSkills"], software["missingSkills"]), (product["matchedSkills"], product["missingSkills"]))
        self.assertNotEqual((software["matchedSkills"], software["missingSkills"]), (data["matchedSkills"], data["missingSkills"]))
        role_text = json.dumps(payload).casefold()
        self.assertNotIn("forecasting", role_text)
        self.assertNotIn("variance analysis", role_text)
        self.assertIn("roleFamily=software_engineering", software["rankingSignals"])

    def test_real_pdf_parser_repairs_compact_role_company_contact_and_confidence_evidence(self) -> None:
        evidence = parse_pdf_bytes((ROOT / "cv_examples/CV Salman Abdurrahman ATS.pdf").read_bytes(), max_bytes=5_000_000, max_pages=10)
        ats_score, issues, fallback = ats_score_from_pdf_evidence(evidence)

        self.assertTrue(evidence.has_contact_signal)
        self.assertTrue(evidence.has_date_signal)
        self.assertTrue(evidence.has_education_signal)
        self.assertIn("Full Stack Developer Intern", evidence.role_titles)
        self.assertIn("AMIKOM Computer Club", evidence.company_names)
        self.assertGreaterEqual(ats_score, 60)
        self.assertTrue(fallback)
        self.assertNotIn("contact signal not detected", issues)
        self.assertNotIn("role title evidence not detected", issues)
        self.assertNotIn("company evidence not detected", issues)
        self.assertNotIn("education credential detail not detected", issues)

    def test_model_core_outputs_canonical_english_for_indonesian_signal_labels(self) -> None:
        request = CvAnalysisModelCoreRequest(
            requestId="req-51-english-policy",
            inputVersion="cv-analyzer-v1",
            language="id",
            inputMode="UPLOAD",
            compareSource="JOB_SEARCH",
            profile=SanitizedProfileInput(
                cvText="Finance analyst menyusun laporan keuangan dan menganalisis data operasional.",
                profileText="Pengalaman kerja finance analyst dengan dashboard keuangan.",
                normalizedSkills=("laporan keuangan", "menganalisis data operasional", "dashboard keuangan"),
                targetRoles=("Data Analyst",),
                experienceYears=2.0,
                detectedCvSectionNames=("summary", "experience", "skills"),
            ),
            jobCandidates=(
                CandidateJobInput(
                    jobId="phase51-data-analyst",
                    scoringInput=CandidateScoringInput(
                        titleText="Data Analyst",
                        requiredSkills=(
                            "menyusun laporan keuangan",
                            "menganalisis data operasional",
                            "mendukung pengambilan keputusan bisnis",
                            "kemampuan komunikasi",
                        ),
                        requirements=("analisis data", "manajemen proyek"),
                        roleFamily="data",
                        experienceBand="junior",
                    ),
                ),
            ),
            maxRecommendations=1,
        )
        payload = build_cv_analysis_response_payload(
            request,
            service=self.service(),
            feature_config=self.feature_config(),
            calibration_policy=ScoreCalibrationPolicy(),
            embedding_backend=FakeE5Backend(),
            environment="test",
        )

        self.assertIn("financial reporting", payload["jobFitAlignment"]["matchedSkills"])
        self.assertIn("operational analysis", payload["jobFitAlignment"]["matchedSkills"])
        self.assertIn("business decision support", payload["jobFitAlignment"]["missingSkills"])
        self.assertIn("communication skills", payload["jobFitAlignment"]["missingSkills"])
        self.assert_english_only_policy(payload)

    def test_ats_issue_normalizer_translates_indonesian_labels_to_english(self) -> None:
        normalized = _signal_list([
            "kontak tidak terdeteksi",
            "sinyal dampak terukur tidak terdeteksi",
            "judul peran tidak terdeteksi",
            "bukti perusahaan tidak terdeteksi",
        ])

        self.assertEqual(
            normalized,
            [
                "contact signal not detected",
                "quantified impact signal not detected",
                "role title evidence not detected",
                "company evidence not detected",
            ],
        )

    def test_ats_issue_copy_is_grouped_grounded_and_actionable(self) -> None:
        request = CvAnalysisModelCoreRequest(
            requestId="req-51-ats-copy",
            inputVersion="cv-analyzer-v1",
            language="en",
            inputMode="UPLOAD",
            compareSource="JOB_SEARCH",
            profile=SanitizedProfileInput(
                cvText="Backend developer with Python but no clear section headings.",
                normalizedSkills=("python",),
                targetRoles=("Backend Developer",),
                detectedCvSectionNames=(),
            ),
            jobCandidates=(
                CandidateJobInput(
                    jobId="phase51-backend",
                    scoringInput=CandidateScoringInput(titleText="Backend Developer", requiredSkills=("python", "sql"), requirements=("Build APIs",)),
                ),
            ),
            maxRecommendations=1,
        )
        payload = build_cv_analysis_response_payload(
            request,
            service=self.service(),
            feature_config=self.feature_config(),
            calibration_policy=ScoreCalibrationPolicy(),
            embedding_backend=FakeE5Backend(),
            environment="test",
        )

        issues = payload["atsFriendliness"]["detectedIssues"]
        self.assertTrue(any(issue.startswith("Section structure weak:") for issue in issues))
        self.assertTrue(all("Fix:" in issue for issue in issues))
        self.assertNotIn("section evidence not provided", issues)
        self.assert_english_only_policy(payload)

    def test_overall_impression_uses_grounded_template_not_generic_placeholder(self) -> None:
        payload = build_cv_analysis_response_payload(
            self.benchmark_request_for_case("salman-abdurrahman-ats"),
            service=self.service(),
            feature_config=self.feature_config(),
            calibration_policy=ScoreCalibrationPolicy(),
            embedding_backend=FakeE5Backend(),
            environment="test",
        )

        evidence = payload["overallImpression"]["evidence"]
        joined = " ".join(evidence).casefold()
        self.assertTrue(evidence[0].startswith("summary: For Software Engineer"))
        self.assertIn("matched skills", joined)
        self.assertIn("cv sections", joined)
        self.assertIn("next improvement", joined)
        self.assertNotIn("model-core evidence prepared for backend genai wrapper", joined)
        self.assertNotIn("hiring outcome", joined)
        self.assert_english_only_policy(payload)

    def test_phase51_7_quality_gates_cover_full_benchmark_matrix(self) -> None:
        fixture = self.fixture()
        roles = [role["role"] for role in fixture["targetRoles"]]
        scores_by_case: dict[str, list[int]] = {}
        evidence_by_case: dict[str, list[tuple[tuple[str, ...], tuple[str, ...]]]] = {}
        all_scores: list[int] = []

        for cv_case in fixture["benchmarkFiles"]:
            parser_evidence = parse_pdf_bytes((ROOT / cv_case["path"]).read_bytes(), max_bytes=5_000_000, max_pages=10)
            _ats_score, parser_issues, _fallback = ats_score_from_pdf_evidence(parser_evidence)
            if parser_evidence.has_contact_signal:
                self.assertNotIn("contact signal not detected", parser_issues)
            self.assertTrue(all("not detected" in issue or "image content present" in issue for issue in parser_issues))

            for role in roles:
                request = self.benchmark_request_for_case(cv_case["caseId"], target_role=role, evidence=parser_evidence)
                started_at = perf_counter()
                payload = build_cv_analysis_response_payload(
                    request,
                    service=self.service(),
                    feature_config=self.feature_config(),
                    calibration_policy=ScoreCalibrationPolicy(),
                    embedding_backend=FakeE5Backend(),
                    environment="test",
                    include_observability=True,
                )
                latency_ms = round((perf_counter() - started_at) * 1000)
                self.assertLessEqual(latency_ms, BENCHMARK_LATENCY_BUDGET_MS)
                validate_model_core_payload(payload, {candidate.jobId for candidate in request.jobCandidates}, request.maxRecommendations)
                self.assert_english_only_policy(payload)
                self.assert_no_raw_cv_leakage(payload, parser_evidence.text)
                self.assert_non_generic_copy(payload)
                self.assert_ats_issue_precision(payload)

                recommendation = payload["candidateReranking"]["recommendations"][0]
                scores_by_case.setdefault(cv_case["caseId"], []).append(payload["jobFitAlignment"]["score"])
                evidence_by_case.setdefault(cv_case["caseId"], []).append((tuple(recommendation["matchedSkills"]), tuple(recommendation["missingSkills"])))
                all_scores.append(payload["jobFitAlignment"]["score"])

        self.assertGreaterEqual(len(set(all_scores)), 2)
        self.assertGreater(len(set(scores_by_case["salman-abdurrahman-ats"])), 1)
        for case_id, role_evidence in evidence_by_case.items():
            self.assertEqual(len(set(role_evidence)), len(roles), case_id)

    def test_phase51_7_quality_gate_uses_requirement_driven_non_benchmark_role(self) -> None:
        request = CvAnalysisModelCoreRequest(
            requestId="req-51-non-benchmark-role",
            inputVersion="cv-analyzer-v1",
            language="en",
            inputMode="UPLOAD",
            compareSource="JOB_SEARCH",
            profile=SanitizedProfileInput(
                cvText="Security analyst with Python SQL log review and incident triage experience.",
                profileText="Security analyst with Python SQL log review and incident triage experience.",
                normalizedSkills=("python", "sql", "log review", "incident triage"),
                targetRoles=("Cybersecurity Analyst",),
                experienceYears=2.0,
                detectedCvSectionNames=("summary", "experience", "skills"),
            ),
            jobCandidates=(
                CandidateJobInput(
                    jobId="phase51-cybersecurity-analyst",
                    scoringInput=CandidateScoringInput(
                        titleText="Cybersecurity Analyst",
                        requiredSkills=("python", "sql", "threat modeling", "incident response"),
                        requirements=("Review security logs", "Coordinate incident response"),
                        roleFamily="security",
                        experienceBand="junior",
                    ),
                ),
            ),
            maxRecommendations=1,
        )
        payload = build_cv_analysis_response_payload(
            request,
            service=self.service(),
            feature_config=self.feature_config(),
            calibration_policy=ScoreCalibrationPolicy(),
            embedding_backend=FakeE5Backend(),
            environment="test",
        )
        recommendation = payload["candidateReranking"]["recommendations"][0]
        serialized = json.dumps(payload).casefold()

        validate_model_core_payload(payload, {"phase51-cybersecurity-analyst"}, 1)
        self.assertIn("python", recommendation["matchedSkills"])
        self.assertIn("sql", recommendation["matchedSkills"])
        self.assertIn("threat modeling", recommendation["missingSkills"])
        self.assertIn("roleFamily=security", recommendation["rankingSignals"])
        self.assertTrue(payload["overallImpression"]["evidence"][0].startswith("summary: For Cybersecurity Analyst"))
        self.assertNotIn("phase51-software-engineer", serialized)
        self.assertNotIn("phase51-product-manager", serialized)
        self.assertNotIn("phase51-data-analyst", serialized)
        self.assert_english_only_policy(payload)
        self.assert_non_generic_copy(payload)


if __name__ == "__main__":
    unittest.main()
