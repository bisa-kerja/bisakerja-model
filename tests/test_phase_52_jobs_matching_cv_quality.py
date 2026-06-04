from __future__ import annotations

import unittest

from model_api.app import build_cv_analysis_response_payload
from model_api.features import E5_MODEL_NAME, PHASE25_FEATURE_ORDER, TensorFlowFeatureConfig
from model_api.inference import InferenceService, RuntimeState, ScoreCalibrationPolicy
from model_api.schemas import (
    CandidateJobInput,
    CandidateScoringInput,
    CvAnalysisModelCoreRequest,
    ModelArtifactIdentity,
    ModelIdentity,
    SanitizedProfileInput,
)


class RankingModelWithFalseHigh:
    name = "phase52_false_high_model"

    def predict(self, values, verbose=0):
        return [[0.85], [0.10], [0.50]][: len(values)]


class DirectionalE5Backend:
    model_name = E5_MODEL_NAME
    backend_name = "sentence-transformers"

    def encode(self, texts):
        vectors = [[1.0, 0.0]]
        for text in texts[1:]:
            folded = text.casefold()
            if "frontend" in folded:
                vectors.append([0.0, 1.0])
            elif "backend" in folded:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.7, 0.3])
        return vectors


class Phase52JobsMatchingCvQualityTest(unittest.TestCase):
    def feature_config(self) -> TensorFlowFeatureConfig:
        return TensorFlowFeatureConfig(
            approved_features=PHASE25_FEATURE_ORDER,
            mean={name: 0.0 for name in PHASE25_FEATURE_ORDER},
            std={name: 1.0 for name in PHASE25_FEATURE_ORDER},
        )

    def service(self) -> InferenceService:
        identity = ModelIdentity(
            name="phase52",
            version="test",
            artifact=ModelArtifactIdentity(path="artifacts/test.keras", sha256="0" * 64),
        )
        return InferenceService(
            model=RankingModelWithFalseHigh(),
            state=RuntimeState(ready=True, model_identity=identity, artifact_manifest_phase="phase_25", message="ready"),
        )

    def test_jobs_matching_cv_uses_skill_experience_evidence_before_public_limit(self) -> None:
        request = CvAnalysisModelCoreRequest(
            requestId="req-52-jobs-matching-cv",
            inputVersion="cv-analyzer-v1",
            language="en",
            inputMode="UPLOAD",
            compareSource="JOB_SEARCH",
            profile=SanitizedProfileInput(
                cvText="Backend engineer with Python SQL Docker experience from 2021 to 2024.",
                profileText="Backend engineer with Python SQL Docker.",
                normalizedSkills=("python", "sql", "docker"),
                targetRoles=("Backend Engineer",),
                experienceYears=3.0,
                detectedCvSectionNames=("summary", "experience", "skills"),
            ),
            jobCandidates=(
                CandidateJobInput(
                    jobId="frontend-false-high",
                    scoringInput=CandidateScoringInput(
                        titleText="Frontend Developer",
                        requiredSkills=("react", "typescript"),
                        roleFamily="frontend",
                        experienceBand="mid",
                    ),
                ),
                CandidateJobInput(
                    jobId="backend-evidence-fit",
                    scoringInput=CandidateScoringInput(
                        titleText="Backend Engineer",
                        requiredSkills=("python", "sql", "docker"),
                        roleFamily="backend",
                        experienceBand="mid",
                    ),
                ),
                CandidateJobInput(
                    jobId="data-partial-fit",
                    scoringInput=CandidateScoringInput(
                        titleText="Data Analyst",
                        requiredSkills=("sql", "power bi"),
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
            embedding_backend=DirectionalE5Backend(),
            environment="test",
        )

        recommendations = payload["candidateReranking"]["recommendations"]
        self.assertEqual(len(recommendations), 1)
        top = recommendations[0]
        self.assertEqual(top["jobId"], "backend-evidence-fit")
        self.assertGreaterEqual(top["matchScore"], 75)
        self.assertIn("python", top["matchedSkills"])
        self.assertIn("sql", top["matchedSkills"])
        self.assertIn("docker", top["matchedSkills"])
        self.assertIn("skillCoverage=100%", top["rankingSignals"])
        self.assertIn("experienceMatch=83%", top["rankingSignals"])
        self.assertIn("calibratedMatchScore=78", top["rankingSignals"])
        self.assertEqual(payload["jobFitAlignment"]["score"], top["matchScore"])


if __name__ == "__main__":
    unittest.main()
