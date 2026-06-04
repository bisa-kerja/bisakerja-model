"""Runtime configuration for the Phase 26 serving API.

The config module owns path/env resolution only. Hash verification, model
loading, request validation, and inference are handled by sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Mapping

PHASE46_ARTIFACT_ROOT_NAME = "phase_46_calibration_model_card_manifest_handoff_refresh"
DEFAULT_ARTIFACT_ROOT = Path("artifacts") / PHASE46_ARTIFACT_ROOT_NAME
DEFAULT_EXPORT_DIR = DEFAULT_ARTIFACT_ROOT / "export"
DEFAULT_OPENAPI_PATH = Path("references/docs/generated/openapi.json")
DEFAULT_PRISMA_SCHEMA_PATH = Path("references/prisma/schema.prisma")
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODELS_URL = "https://openrouter.ai/models"
DEFAULT_OPENROUTER_MODEL = "~openai/gpt-latest"

REQUIRED_RUNTIME_ARTIFACT_IDS: tuple[str, ...] = (
    "score_calibration",
    "feature_config",
    "source_trained_candidate_model",
    "source_tensorflow_feature_config",
    "tensorflow_artifact_export",
    "final_keras_model",
    "model_api_handoff_fixtures",
    "model_api_handoff_validation",
    "training_only_genai_boundary",
    "genai_wrapper_handoff_contract",
    "model_card",
)


def default_model_path_for_root(root: Path, export_dir: Path) -> Path:
    """Return default Keras artifact for known versioned artifact roots."""

    if root.name == PHASE46_ARTIFACT_ROOT_NAME:
        return export_dir / "selected_jobfit_tf_phase46_multilingual_e5_small.keras"
    return export_dir / "selected_jobfit_tf_phase25.keras"


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolved runtime artifact paths for versioned artifact packages."""

    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    export_dir: Path = DEFAULT_EXPORT_DIR
    manifest_path: Path = DEFAULT_ARTIFACT_ROOT / "artifact_manifest.json"
    model_path: Path = DEFAULT_EXPORT_DIR / "selected_jobfit_tf_phase46_multilingual_e5_small.keras"
    tensorflow_feature_config_path: Path = DEFAULT_ARTIFACT_ROOT / "tensorflow_feature_config.json"
    feature_config_path: Path = DEFAULT_ARTIFACT_ROOT / "feature_config.json"
    score_calibration_path: Path = DEFAULT_ARTIFACT_ROOT / "score_calibration.json"
    model_card_path: Path = DEFAULT_ARTIFACT_ROOT / "model_card.json"
    source_trained_candidate_model_path: Path = DEFAULT_ARTIFACT_ROOT / "gradient_tape_trained_candidate.keras"
    tensorflow_artifact_export_path: Path = DEFAULT_ARTIFACT_ROOT / "tensorflow_artifact_export.json"
    training_only_genai_boundary_path: Path = DEFAULT_ARTIFACT_ROOT / "training_only_genai_boundary.json"
    handoff_fixtures_path: Path = DEFAULT_EXPORT_DIR / "model_api_handoff_fixtures.json"
    handoff_validation_path: Path = DEFAULT_EXPORT_DIR / "model_api_handoff_validation.json"
    genai_contract_path: Path = DEFAULT_EXPORT_DIR / "genai_wrapper_handoff_contract.json"
    openapi_path: Path = DEFAULT_OPENAPI_PATH
    prisma_schema_path: Path = DEFAULT_PRISMA_SCHEMA_PATH

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ArtifactPaths":
        """Build paths from env without touching the filesystem."""

        data = os.environ if env is None else env
        root = Path(data.get("MODEL_API_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT)))
        export_dir = Path(data.get("MODEL_API_EXPORT_DIR", str(root / "export")))
        default_model_path = default_model_path_for_root(root, export_dir)
        return cls(
            artifact_root=root,
            export_dir=export_dir,
            manifest_path=Path(data.get("MODEL_API_ARTIFACT_MANIFEST", str(root / "artifact_manifest.json"))),
            model_path=Path(data.get("MODEL_API_MODEL_PATH", str(default_model_path))),
            tensorflow_feature_config_path=Path(
                data.get("MODEL_API_TENSORFLOW_FEATURE_CONFIG", str(root / "tensorflow_feature_config.json"))
            ),
            feature_config_path=Path(data.get("MODEL_API_FEATURE_CONFIG", str(root / "feature_config.json"))),
            score_calibration_path=Path(data.get("MODEL_API_SCORE_CALIBRATION", str(root / "score_calibration.json"))),
            model_card_path=Path(data.get("MODEL_API_MODEL_CARD", str(root / "model_card.json"))),
            source_trained_candidate_model_path=Path(
                data.get("MODEL_API_SOURCE_TRAINED_CANDIDATE_MODEL", str(root / "gradient_tape_trained_candidate.keras"))
            ),
            tensorflow_artifact_export_path=Path(
                data.get("MODEL_API_TENSORFLOW_ARTIFACT_EXPORT", str(root / "tensorflow_artifact_export.json"))
            ),
            training_only_genai_boundary_path=Path(
                data.get("MODEL_API_TRAINING_ONLY_GENAI_BOUNDARY", str(root / "training_only_genai_boundary.json"))
            ),
            handoff_fixtures_path=Path(
                data.get("MODEL_API_HANDOFF_FIXTURES", str(export_dir / "model_api_handoff_fixtures.json"))
            ),
            handoff_validation_path=Path(
                data.get("MODEL_API_HANDOFF_VALIDATION", str(export_dir / "model_api_handoff_validation.json"))
            ),
            genai_contract_path=Path(
                data.get("MODEL_API_GENAI_CONTRACT", str(export_dir / "genai_wrapper_handoff_contract.json"))
            ),
            openapi_path=Path(data.get("MODEL_API_OPENAPI_PATH", str(DEFAULT_OPENAPI_PATH))),
            prisma_schema_path=Path(data.get("MODEL_API_PRISMA_SCHEMA_PATH", str(DEFAULT_PRISMA_SCHEMA_PATH))),
        )

    def as_dict(self) -> dict[str, str]:
        """Return stable string paths for logs/model-info responses."""

        return {name: str(value) for name, value in self.__dict__.items()}


@dataclass(frozen=True)
class OpenRouterConfig:
    """OpenAI-compatible OpenRouter settings for wrapper-owned GenAI prose."""

    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    models_url: str = DEFAULT_OPENROUTER_MODELS_URL
    model: str = DEFAULT_OPENROUTER_MODEL
    api_key_env_name: str = "OPENROUTER_API_KEY"
    enabled: bool = False
    timeout_ms: int = 10_000

    @property
    def chat_completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "OpenRouterConfig":
        data = os.environ if env is None else env
        enabled_value = data.get("MODEL_API_ENABLE_GENAI_WRAPPER", "false").lower()
        return cls(
            base_url=data.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
            models_url=data.get("OPENROUTER_MODELS_URL", DEFAULT_OPENROUTER_MODELS_URL),
            model=data.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
            api_key_env_name=data.get("OPENROUTER_API_KEY_ENV_NAME", "OPENROUTER_API_KEY"),
            enabled=enabled_value == "true",
            timeout_ms=int(data.get("OPENROUTER_TIMEOUT_MS", data.get("MODEL_API_TIMEOUT_MS", "10000"))),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Serving runtime settings used by app, loader, features, and inference."""

    service_name: str = "bisakerja-model-api"
    environment: str = "local"
    max_recommendations: int = 5
    timeout_ms: int = 10_000
    service_token: str | None = None
    allow_unauthenticated_local: bool = True
    max_pdf_bytes: int = 5_000_000
    max_pdf_pages: int = 10
    warmup_on_startup: bool = False
    warmup_required: bool = False
    artifact_paths: ArtifactPaths = field(default_factory=ArtifactPaths.from_env)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig.from_env)
    expected_embedding_model: str | None = None

    @property
    def requires_service_token(self) -> bool:
        return self.environment.lower() in {"staging", "production", "prod"}

    def validate_security(self) -> None:
        if self.requires_service_token and not self.service_token:
            raise ValueError("MODEL_API_SERVICE_TOKEN is required in staging/production")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RuntimeConfig":
        data = os.environ if env is None else env
        environment = data.get("MODEL_API_ENV", "local")
        warmup_required_default = "true" if environment.lower() in {"staging", "production", "prod"} else "false"
        return cls(
            service_name=data.get("MODEL_API_SERVICE_NAME", "bisakerja-model-api"),
            environment=environment,
            max_recommendations=int(data.get("MODEL_API_MAX_RECOMMENDATIONS", "5")),
            timeout_ms=int(data.get("MODEL_API_TIMEOUT_MS", "10000")),
            service_token=data.get("MODEL_API_SERVICE_TOKEN") or None,
            allow_unauthenticated_local=data.get("MODEL_API_ALLOW_UNAUTHENTICATED_LOCAL", "true").lower() == "true",
            max_pdf_bytes=int(data.get("MODEL_API_MAX_PDF_BYTES", "5000000")),
            max_pdf_pages=int(data.get("MODEL_API_MAX_PDF_PAGES", "10")),
            warmup_on_startup=data.get("MODEL_API_WARMUP_ON_STARTUP", "false").lower() == "true",
            warmup_required=data.get("MODEL_API_WARMUP_REQUIRED", warmup_required_default).lower() == "true",
            artifact_paths=ArtifactPaths.from_env(data),
            openrouter=OpenRouterConfig.from_env(data),
            expected_embedding_model=(data.get("MODEL_API_EXPECTED_EMBEDDING_MODEL") or None),
        )
