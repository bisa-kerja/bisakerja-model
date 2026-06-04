from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from model_api.app import create_app, validate_runtime_embedding_contract
from model_api.artifacts import ArtifactManifest, ArtifactVerificationReport, verify_runtime_artifacts
from model_api.config import ArtifactPaths, RuntimeConfig
from model_api.errors import ArtifactError, FeatureBuildError
from model_api.features import E5_MODEL_NAME, EmbeddingModelMetadata, TensorFlowFeatureConfig, validate_e5_backend
from model_api.inference import InferenceService, RuntimeState
from model_api.schemas import ModelArtifactIdentity, ModelIdentity
from scripts.verify_phase_47_model_api_runtime_support import REPORT_JSON_PATH, build_report, write_all

PHASE46_ROOT = "artifacts/phase_46_calibration_model_card_manifest_handoff_refresh"
PHASE46_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class FakeModel:
    name = "fake_phase47_model"

    def predict(self, values, verbose=0):
        return [[0.7] for _ in values]


class FakeE5BaseBackend:
    model_name = E5_MODEL_NAME
    backend_name = "sentence-transformers"

    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeMultilingualE5SmallBackend(FakeE5BaseBackend):
    model_name = PHASE46_EMBEDDING_MODEL


class FakeFallbackBackend(FakeMultilingualE5SmallBackend):
    backend_name = "local-hash"


class Phase47ModelApiRuntimeSupportTest(unittest.TestCase):
    def service(
        self,
        *,
        embedding_model: str = PHASE46_EMBEDDING_MODEL,
        artifact_phase: str = "phase_46_calibration_model_card_manifest_handoff_refresh",
    ) -> InferenceService:
        identity = ModelIdentity(
            name="phase47",
            version="test",
            artifact=ModelArtifactIdentity(path="artifacts/test.keras", sha256="0" * 64),
            artifact_phase=artifact_phase,
            embedding_model=embedding_model,
        )
        return InferenceService(
            model=FakeModel(),
            state=RuntimeState(ready=True, model_identity=identity, artifact_manifest_phase=artifact_phase, message="ready"),
        )

    def test_artifact_root_switch_selects_phase46_runtime_files(self) -> None:
        paths = ArtifactPaths.from_env({"MODEL_API_ARTIFACT_ROOT": PHASE46_ROOT})
        report = verify_runtime_artifacts(paths)

        self.assertEqual(paths.model_path, Path(PHASE46_ROOT) / "export/selected_jobfit_tf_phase46_multilingual_e5_small.keras")
        self.assertEqual(report.manifest.phase_id, "phase_46_calibration_model_card_manifest_handoff_refresh")
        self.assertEqual(
            set(report.runtime_artifact_ids),
            {"final_keras_model", "tensorflow_feature_config", "feature_config", "score_calibration", "model_card"},
        )

    def test_embedding_contract_loaded_from_versioned_feature_config_and_manifest(self) -> None:
        paths = ArtifactPaths.from_env({"MODEL_API_ARTIFACT_ROOT": PHASE46_ROOT})
        report = verify_runtime_artifacts(paths)
        feature_config = TensorFlowFeatureConfig.from_path(paths.tensorflow_feature_config_path)
        contract = validate_runtime_embedding_contract(paths=paths, artifact_report=report, feature_config=feature_config)

        self.assertEqual(feature_config.embedding_model_name, PHASE46_EMBEDDING_MODEL)
        self.assertEqual(contract.embedding_model, PHASE46_EMBEDDING_MODEL)
        self.assertEqual(contract.profile_prefix, "query:")
        self.assertEqual(contract.job_prefix, "passage:")
        self.assertTrue(contract.normalized_embeddings)
        self.assertEqual(contract.embedding_dimension, 384)

    def test_runtime_rejects_env_or_backend_embedding_mismatch_and_fallback(self) -> None:
        paths = ArtifactPaths.from_env({"MODEL_API_ARTIFACT_ROOT": PHASE46_ROOT})
        report = verify_runtime_artifacts(paths)
        feature_config = TensorFlowFeatureConfig.from_path(paths.tensorflow_feature_config_path)

        with self.assertRaisesRegex(ArtifactError, "MODEL_API_EXPECTED_EMBEDDING_MODEL mismatch"):
            validate_runtime_embedding_contract(
                paths=paths,
                artifact_report=report,
                feature_config=feature_config,
                expected_embedding_model=E5_MODEL_NAME,
            )
        with self.assertRaises(FeatureBuildError):
            validate_e5_backend(FakeE5BaseBackend(), "local", expected_model_name=PHASE46_EMBEDDING_MODEL)
        with self.assertRaises(FeatureBuildError):
            validate_e5_backend(FakeFallbackBackend(), "local", expected_model_name=PHASE46_EMBEDDING_MODEL)

    def test_model_info_and_ready_expose_deployment_metadata_without_contract_drift(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("FastAPI serving dependencies are not installed")

        config = RuntimeConfig.from_env({"MODEL_API_ARTIFACT_ROOT": PHASE46_ROOT})
        app = create_app(config=config, service=self.service(), embedding_backend=FakeMultilingualE5SmallBackend())
        client = TestClient(app)

        ready = client.get("/ready").json()
        model_info = client.get("/model-info").json()

        self.assertEqual(ready["embeddingModel"], PHASE46_EMBEDDING_MODEL)
        self.assertEqual(ready["artifactPhase"], "phase_46_calibration_model_card_manifest_handoff_refresh")
        self.assertTrue(ready["checks"]["embeddingModelDeclared"])
        self.assertTrue(ready["checks"]["embeddingModelMatchesBackend"])
        self.assertEqual(model_info["embeddingPolicy"]["embeddingModel"], PHASE46_EMBEDDING_MODEL)
        self.assertEqual(model_info["model"]["embeddingModel"], PHASE46_EMBEDDING_MODEL)
        self.assertEqual(model_info["model"]["artifactPhase"], "phase_46_calibration_model_card_manifest_handoff_refresh")
        self.assertNotIn("topActionables", json.dumps(model_info))
        self.assertNotIn("companyName", json.dumps(model_info))

    def test_missing_embedding_metadata_rejected_for_new_artifact_package(self) -> None:
        payload = {
            "approved_features": [
                "e5_cosine",
                "skill_overlap",
                "requirement_coverage",
                "role_match",
                "experience_match",
                "experience_gap_years_clipped",
            ],
            "feature_policy": {"uses_e5_derived_similarity": True},
            "normalization": {
                "mean": {
                    "e5_cosine": 0.0,
                    "skill_overlap": 0.0,
                    "requirement_coverage": 0.0,
                    "role_match": 0.0,
                    "experience_match": 0.0,
                    "experience_gap_years_clipped": 0.0,
                },
                "std": {
                    "e5_cosine": 1.0,
                    "skill_overlap": 1.0,
                    "requirement_coverage": 1.0,
                    "role_match": 1.0,
                    "experience_match": 1.0,
                    "experience_gap_years_clipped": 1.0,
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tf_path = root / "tensorflow_feature_config.json"
            feature_path = root / "feature_config.json"
            model_card_path = root / "model_card.json"
            for path in (tf_path, feature_path):
                path.write_text(json.dumps(payload), encoding="utf-8")
            model_card_path.write_text(json.dumps({"model": {"name": "missing-metadata"}}), encoding="utf-8")
            paths = SimpleNamespace(
                tensorflow_feature_config_path=tf_path,
                feature_config_path=feature_path,
                model_card_path=model_card_path,
            )
            report = ArtifactVerificationReport(
                manifest=ArtifactManifest(
                    schema_version="phase-46-artifact-manifest-v1",
                    phase_id="phase_46_calibration_model_card_manifest_handoff_refresh",
                    entries=(),
                ),
                verified_artifacts=(),
            )
            with self.assertRaisesRegex(ArtifactError, "Embedding metadata missing"):
                validate_runtime_embedding_contract(
                    paths=paths,
                    artifact_report=report,
                    feature_config=TensorFlowFeatureConfig.from_mapping(payload),
                )

    def test_embedding_metadata_comparison_rejects_prefix_drift(self) -> None:
        paths = ArtifactPaths.from_env({"MODEL_API_ARTIFACT_ROOT": PHASE46_ROOT})
        report = verify_runtime_artifacts(paths)
        feature_config = TensorFlowFeatureConfig.from_path(paths.tensorflow_feature_config_path)
        drifted = TensorFlowFeatureConfig(
            approved_features=feature_config.approved_features,
            mean=feature_config.mean,
            std=feature_config.std,
            embedding_metadata=EmbeddingModelMetadata(
                embedding_model=PHASE46_EMBEDDING_MODEL,
                profile_prefix="passage:",
                job_prefix="passage:",
                normalized_embeddings=True,
                embedding_dimension=384,
            ),
        )

        with self.assertRaisesRegex(ArtifactError, "profile_prefix"):
            validate_runtime_embedding_contract(paths=paths, artifact_report=report, feature_config=drifted)

    def test_phase47_report_is_complete_and_records_performance_smoke(self) -> None:
        report = write_all()
        written = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(report["schema_version"], "phase-47-model-api-runtime-support-v1")
        self.assertEqual(written["status"], "complete")
        self.assertTrue(all(written["checks"].values()))
        self.assertIn("phase_25_tensorflow_training_delivery", written["artifact_support"])
        self.assertIn("phase_46_calibration_model_card_manifest_handoff_refresh", written["artifact_support"])
        self.assertGreaterEqual(written["performance_smoke"]["firstInferenceMs"], 0)
        self.assertGreaterEqual(written["performance_smoke"]["warmInferenceMs"], 0)
        self.assertEqual(build_report()["status"], "complete")


if __name__ == "__main__":
    unittest.main()
