from __future__ import annotations

import json
import unittest

from model_api.app import build_cv_analysis_response_payload
from model_api.features import E5_MODEL_NAME, TensorFlowFeatureConfig
from model_api.inference import InferenceService, RuntimeState, ScoreCalibrationPolicy
from model_api.pdf_parser import ats_score_from_pdf_evidence, parse_pdf_bytes
from model_api.schemas import (
    CandidateJobInput,
    CandidateScoringInput,
    CvAnalysisModelCoreRequest,
    ModelArtifactIdentity,
    ModelIdentity,
    SanitizedProfileInput,
)


class FakeModel:
    name = "fake_phase50_model"

    def predict(self, values, verbose=0):
        return [[0.88], [0.41]][: len(values)]


class LowScoreModel:
    name = "low_score_phase50_model"

    def predict(self, values, verbose=0):
        return [[0.04] for _ in values]


class FakeE5Backend:
    model_name = E5_MODEL_NAME
    backend_name = "sentence-transformers"

    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class Phase50AiCvAnalyzerIntelligenceTest(unittest.TestCase):
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

    def service(self, model=None) -> InferenceService:
        identity = ModelIdentity(
            name="phase50",
            version="test",
            artifact=ModelArtifactIdentity(path="artifacts/test.keras", sha256="0" * 64),
        )
        return InferenceService(
            model=model or FakeModel(),
            state=RuntimeState(ready=True, model_identity=identity, artifact_manifest_phase="phase_25", message="ready"),
        )

    def test_pdf_parser_extracts_richer_role_company_language_and_seniority_signals(self) -> None:
        pdf = b"""%PDF-1.4
1 0 obj<</Type /Page>>stream
(Summary Senior Backend Engineer with 6+ years of experience. Email dev@example.com) Tj
(Skills Python SQL Docker AWS English) Tj
(Experience Senior Backend Engineer at Acme Digital 2020 improved latency by 35% for 10000 users.) Tj
(Education Bachelor of Computer Science) Tj
(Certifications AWS Certified Developer) Tj
endstream
%%EOF"""
        evidence = parse_pdf_bytes(pdf, max_bytes=5000, max_pages=5)
        score, issues, fallback = ats_score_from_pdf_evidence(evidence)

        self.assertIn("experience", evidence.section_names)
        self.assertIn("Backend Engineer", evidence.role_titles)
        self.assertIn("Acme Digital", evidence.company_names)
        self.assertTrue(evidence.has_education_signal)
        self.assertTrue(evidence.has_certification_signal)
        self.assertIn("english", evidence.language_signals)
        self.assertIn("senior", evidence.seniority_hints)
        self.assertTrue(evidence.has_quantified_impact)
        self.assertGreaterEqual(score, 90)
        self.assertFalse(fallback)
        self.assertNotIn("quantified impact signal not detected", issues)

    def test_pdf_parser_repairs_subset_font_glyph_text_for_ats_signals(self) -> None:
        glyph_map = {" ": 0x03, "@": 0x23, ".": 0x11, **{str(value): 0x13 + value for value in range(10)}}
        glyph_text = bytes(glyph_map.get(char, ord(char) - 29) for char in "SUMMARY SKILLS EXPERIENCE EDUCATION BACHELOR EMAIL DEV@EXAMPLE.COM 2025")
        pdf = b"%PDF-1.4\n1 0 obj<</Type /Page>>stream\n<" + glyph_text.hex().encode() + b"> Tj\nendstream\n%%EOF"
        evidence = parse_pdf_bytes(pdf, max_bytes=5000, max_pages=5)
        score, issues, fallback = ats_score_from_pdf_evidence(evidence)

        self.assertIn("SUMMARY", evidence.text)
        self.assertIn("summary", evidence.section_names)
        self.assertIn("skills", evidence.section_names)
        self.assertIn("experience", evidence.section_names)
        self.assertTrue(evidence.has_contact_signal)
        self.assertTrue(evidence.has_date_signal)
        self.assertGreaterEqual(score, 85)
        self.assertNotIn("standard CV sections not detected", issues)
        self.assertNotIn("contact signal not detected", issues)
        self.assertNotIn("date or timeline signal not detected", issues)
        self.assertFalse(fallback)

    def test_feature_based_ats_penalizes_missing_metrics_and_formatting_risk(self) -> None:
        pdf = ("%PDF-1.4\n1 0 obj<</Type /Page /XObject 2 0 R /Columns 2>>stream\n(" + " ".join(
            ["Summary Skills Experience Education Python SQL backend developer delivery"] * 12
        ) + " 2021 dev@example.com) Tj\nendstream\n%%EOF").encode()
        evidence = parse_pdf_bytes(pdf, max_bytes=20000, max_pages=5)
        score, issues, fallback = ats_score_from_pdf_evidence(evidence)

        self.assertLess(score, 90)
        self.assertTrue(fallback)
        self.assertIn("quantified impact signal not detected", issues)
        self.assertIn("image content present", issues)
        self.assertIn("multi-column layout marker present", issues)

    def test_model_core_response_adds_evidence_without_backend_owned_fields_or_raw_cv_leakage(self) -> None:
        request = CvAnalysisModelCoreRequest(
            requestId="req-50",
            inputVersion="cv-analyzer-v1",
            language="en",
            inputMode="UPLOAD",
            compareSource="JOB_SEARCH",
            profile=SanitizedProfileInput(
                cvText="Senior Backend Engineer. Skills Python SQL Docker. Improved latency 35%.",
                profileText="Senior Backend Engineer",
                normalizedSkills=("python", "sql", "docker"),
                targetRoles=("Backend Engineer",),
                experienceYears=6.0,
                detectedCvSectionNames=("summary", "skills", "experience"),
            ),
            jobCandidates=(
                CandidateJobInput(
                    jobId="job-1",
                    scoringInput=CandidateScoringInput(requiredSkills=("python", "sql", "kubernetes"), experienceBand="senior"),
                ),
                CandidateJobInput(jobId="job-2", scoringInput=CandidateScoringInput(requiredSkills=("react",))),
            ),
        )
        payload = build_cv_analysis_response_payload(
            request,
            service=self.service(),
            feature_config=self.feature_config(),
            calibration_policy=ScoreCalibrationPolicy(),
            embedding_backend=FakeE5Backend(),
            environment="test",
        )
        serialized = json.dumps(payload).lower()

        self.assertIn("matched required skills: python, sql", payload["jobFitAlignment"]["evidence"])
        self.assertIn("missing required skills: kubernetes", payload["jobFitAlignment"]["evidence"])
        self.assertIn("experienceYears=6; candidateBand=senior", payload["candidateReranking"]["recommendations"][0]["rankingSignals"])
        self.assertNotIn("topActionables", payload)
        self.assertNotIn("sectionReviews", payload)
        self.assertNotIn("senior backend engineer. skills python", serialized)

    def test_finance_gap_score_uses_deterministic_evidence_floor_instead_of_pathological_four_percent(self) -> None:
        request = CvAnalysisModelCoreRequest(
            requestId="req-finance-low-score",
            inputVersion="cv-analyzer-v1",
            language="id",
            inputMode="UPLOAD",
            compareSource="JOB_SEARCH",
            profile=SanitizedProfileInput(
                cvText=(
                    "Pengalaman finance analyst menyusun laporan keuangan bulanan, "
                    "membuat dashboard keuangan, financial analysis, dan menganalisis data operasional."
                ),
                profileText="Finance analyst dengan pengalaman reporting dan dashboard.",
                normalizedSkills=("laporan keuangan", "dashboard keuangan", "financial analysis", "analisis operasional"),
                targetRoles=("Finance Analyst",),
                experienceYears=2.0,
                detectedCvSectionNames=("summary", "experience", "skills"),
            ),
            jobCandidates=(
                CandidateJobInput(
                    jobId="job-finance-analyst",
                    scoringInput=CandidateScoringInput(
                        titleText="Finance Analyst",
                        requiredSkills=(
                            "drafting laporan keuangan",
                            "dashboard",
                            "financial analysis",
                            "menganalisis data keuangan",
                            "menganalisis data operasional",
                            "mendukung pengambilan keputusan bisnis",
                        ),
                        experienceBand="junior",
                    ),
                ),
            ),
            maxRecommendations=1,
        )
        payload = build_cv_analysis_response_payload(
            request,
            service=self.service(LowScoreModel()),
            feature_config=self.feature_config(),
            calibration_policy=ScoreCalibrationPolicy(),
            embedding_backend=FakeE5Backend(),
            environment="test",
        )

        recommendation = payload["candidateReranking"]["recommendations"][0]
        self.assertGreaterEqual(payload["jobFitAlignment"]["score"], 45)
        self.assertGreaterEqual(recommendation["matchScore"], 45)
        self.assertIn("financial analysis", recommendation["matchedSkills"])
        self.assertNotEqual(payload["jobFitAlignment"]["score"], 4)


if __name__ == "__main__":
    unittest.main()
