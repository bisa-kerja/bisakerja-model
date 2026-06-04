"""Artifact loading and verification for Phase 25 runtime files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .config import REQUIRED_RUNTIME_ARTIFACT_IDS, ArtifactPaths
from .errors import ArtifactError, UnsupportedArtifactVersionError

EXPECTED_ARTIFACT_MANIFEST_SCHEMA_VERSION = "phase-25-artifact-manifest-v1"
EXPECTED_ARTIFACT_MANIFEST_PHASE_ID = "phase_25_tensorflow_training_delivery"
PHASE46_ARTIFACT_MANIFEST_SCHEMA_VERSION = "phase-46-artifact-manifest-v1"
PHASE46_ARTIFACT_MANIFEST_PHASE_ID = "phase_46_calibration_model_card_manifest_handoff_refresh"

SUPPORTED_ARTIFACT_MANIFESTS: dict[str, str] = {
    EXPECTED_ARTIFACT_MANIFEST_SCHEMA_VERSION: EXPECTED_ARTIFACT_MANIFEST_PHASE_ID,
    PHASE46_ARTIFACT_MANIFEST_SCHEMA_VERSION: PHASE46_ARTIFACT_MANIFEST_PHASE_ID,
}

PHASE_RUNTIME_REQUIRED_ARTIFACT_IDS: dict[str, tuple[str, ...]] = {
    EXPECTED_ARTIFACT_MANIFEST_PHASE_ID: REQUIRED_RUNTIME_ARTIFACT_IDS,
    PHASE46_ARTIFACT_MANIFEST_PHASE_ID: (
        "final_keras_model",
        "tensorflow_feature_config",
        "feature_config",
        "score_calibration",
        "model_card",
    ),
}


@dataclass(frozen=True)
class ArtifactManifestEntry:
    artifact_id: str
    path: Path
    required_for_inference: bool
    sha256: str | None
    size_bytes: int | None
    role: str | None = None
    schema_version: str | None = None
    embedding_model_metadata: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ArtifactManifestEntry":
        return cls(
            artifact_id=str(raw["artifact_id"]),
            path=Path(str(raw["path"])),
            required_for_inference=bool(raw.get("required_for_inference", False)),
            sha256=None if raw.get("sha256") is None else str(raw["sha256"]),
            size_bytes=None if raw.get("size_bytes") is None else int(raw["size_bytes"]),
            role=None if raw.get("role") is None else str(raw["role"]),
            schema_version=None if raw.get("schema_version") is None else str(raw["schema_version"]),
            embedding_model_metadata=raw.get("embedding_model_metadata") if isinstance(raw.get("embedding_model_metadata"), dict) else None,
        )


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: str
    phase_id: str
    entries: tuple[ArtifactManifestEntry, ...]
    embedding_model_metadata: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ArtifactManifest":
        entries = tuple(ArtifactManifestEntry.from_mapping(item) for item in raw.get("artifacts", []))
        return cls(
            schema_version=str(raw.get("schema_version", "")),
            phase_id=str(raw.get("phase_id", "")),
            entries=entries,
            embedding_model_metadata=raw.get("embedding_model_metadata") if isinstance(raw.get("embedding_model_metadata"), dict) else None,
        )

    def by_id(self) -> dict[str, ArtifactManifestEntry]:
        return {entry.artifact_id: entry for entry in self.entries}

    def required_runtime_entries(
        self,
        artifact_ids: Iterable[str] | None = None,
    ) -> tuple[ArtifactManifestEntry, ...]:
        selected_ids = tuple(artifact_ids) if artifact_ids is not None else PHASE_RUNTIME_REQUIRED_ARTIFACT_IDS.get(self.phase_id, ())
        indexed = self.by_id()
        missing = [artifact_id for artifact_id in selected_ids if artifact_id not in indexed]
        if missing:
            raise ArtifactError(f"Missing runtime artifacts in manifest: {missing}")
        return tuple(indexed[artifact_id] for artifact_id in selected_ids)

    def required_for_inference_entries(self) -> tuple[ArtifactManifestEntry, ...]:
        """Return every manifest entry flagged as runtime-required."""

        return tuple(entry for entry in self.entries if entry.required_for_inference)


@dataclass(frozen=True)
class VerifiedArtifact:
    artifact_id: str
    path: Path
    sha256: str
    size_bytes: int
    role: str | None = None
    schema_version: str | None = None


@dataclass(frozen=True)
class ArtifactVerificationReport:
    manifest: ArtifactManifest
    verified_artifacts: tuple[VerifiedArtifact, ...]

    @property
    def artifact_hashes(self) -> dict[str, str]:
        return {artifact.artifact_id: artifact.sha256 for artifact in self.verified_artifacts}

    @property
    def artifact_sizes(self) -> dict[str, int]:
        return {artifact.artifact_id: artifact.size_bytes for artifact in self.verified_artifacts}

    @property
    def runtime_artifact_ids(self) -> tuple[str, ...]:
        return tuple(artifact.artifact_id for artifact in self.verified_artifacts)


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON with deterministic ArtifactError wrapping."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError as exc:
        raise ArtifactError(f"Artifact file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"Artifact JSON is invalid: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ArtifactError(f"Artifact JSON root must be object: {path}")
    return raw


def load_manifest(paths: ArtifactPaths) -> ArtifactManifest:
    """Load supported runtime artifact manifest without verifying hashes yet."""

    manifest = ArtifactManifest.from_mapping(load_json(paths.manifest_path))
    expected_phase = SUPPORTED_ARTIFACT_MANIFESTS.get(manifest.schema_version)
    if expected_phase is None:
        supported = sorted(SUPPORTED_ARTIFACT_MANIFESTS)
        raise UnsupportedArtifactVersionError(
            "artifact_manifest.json schema_version must be one of "
            f"{supported!r}; actual={manifest.schema_version!r}"
        )
    if manifest.phase_id != expected_phase:
        raise UnsupportedArtifactVersionError(
            "artifact_manifest.json phase_id must match schema_version; "
            f"expected={expected_phase!r}; actual={manifest.phase_id!r}"
        )
    return manifest


def sha256_file(path: Path) -> str:
    """Hash file content in bounded chunks."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise ArtifactError(f"Artifact file not found: {path}") from exc
    return digest.hexdigest()


def configured_runtime_artifact_paths(paths: ArtifactPaths) -> dict[str, Path]:
    """Map manifest artifact IDs to env-resolved runtime paths."""

    return {
        "score_calibration": paths.score_calibration_path,
        "feature_config": paths.feature_config_path,
        "source_trained_candidate_model": paths.source_trained_candidate_model_path,
        "source_tensorflow_feature_config": paths.tensorflow_feature_config_path,
        "tensorflow_feature_config": paths.tensorflow_feature_config_path,
        "tensorflow_artifact_export": paths.tensorflow_artifact_export_path,
        "final_keras_model": paths.model_path,
        "model_api_handoff_fixtures": paths.handoff_fixtures_path,
        "handoff_fixtures": paths.handoff_fixtures_path,
        "model_api_handoff_validation": paths.handoff_validation_path,
        "handoff_validation": paths.handoff_validation_path,
        "training_only_genai_boundary": paths.training_only_genai_boundary_path,
        "genai_wrapper_handoff_contract": paths.genai_contract_path,
        "model_card": paths.model_card_path,
    }


def resolve_artifact_path(entry: ArtifactManifestEntry, paths: ArtifactPaths) -> Path:
    """Resolve path from env/config first, then manifest path."""

    configured = configured_runtime_artifact_paths(paths).get(entry.artifact_id)
    if configured is not None:
        return configured
    return entry.path if entry.path.is_absolute() else Path(entry.path)


def verify_manifest_entry(entry: ArtifactManifestEntry, paths: ArtifactPaths) -> VerifiedArtifact:
    """Verify one runtime artifact SHA-256 and byte size."""

    errors: list[str] = []
    expected_hash = (entry.sha256 or "").lower()
    if not expected_hash:
        errors.append(f"{entry.artifact_id}: missing sha256")
    elif len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        errors.append(f"{entry.artifact_id}: invalid sha256 format")
    if entry.size_bytes is None:
        errors.append(f"{entry.artifact_id}: missing size_bytes")
    elif entry.size_bytes < 0:
        errors.append(f"{entry.artifact_id}: invalid size_bytes")

    resolved_path = resolve_artifact_path(entry, paths)
    if errors:
        raise ArtifactError("; ".join(errors))

    try:
        actual_size = resolved_path.stat().st_size
    except FileNotFoundError as exc:
        raise ArtifactError(f"{entry.artifact_id}: artifact file not found: {resolved_path}") from exc

    if actual_size != entry.size_bytes:
        raise ArtifactError(
            f"{entry.artifact_id}: size mismatch for {resolved_path}: expected {entry.size_bytes}, got {actual_size}"
        )

    actual_hash = sha256_file(resolved_path)
    if actual_hash != expected_hash:
        raise ArtifactError(
            f"{entry.artifact_id}: sha256 mismatch for {resolved_path}: expected {expected_hash}, got {actual_hash}"
        )

    return VerifiedArtifact(
        artifact_id=entry.artifact_id,
        path=resolved_path,
        sha256=actual_hash,
        size_bytes=actual_size,
        role=entry.role,
        schema_version=entry.schema_version,
    )


def verify_runtime_artifacts(
    paths: ArtifactPaths,
    artifact_ids: Iterable[str] | None = None,
) -> ArtifactVerificationReport:
    """Load manifest and verify every runtime-required Phase 25 artifact.

    Default mode verifies all manifest entries with `required_for_inference=true`.
    Passing `artifact_ids` verifies that exact subset after manifest membership check.
    """

    manifest = load_manifest(paths)
    entries = (
        manifest.required_runtime_entries(artifact_ids)
        if artifact_ids is not None
        else manifest.required_for_inference_entries()
    )
    if not entries:
        raise ArtifactError("No runtime-required artifacts found in manifest")

    if artifact_ids is None:
        required_ids = PHASE_RUNTIME_REQUIRED_ARTIFACT_IDS.get(manifest.phase_id, ())
        configured_missing = [artifact_id for artifact_id in required_ids if artifact_id not in manifest.by_id()]
        if configured_missing:
            raise ArtifactError(f"Missing configured runtime artifacts in manifest: {configured_missing}")

    verified: list[VerifiedArtifact] = []
    errors: list[str] = []
    for entry in entries:
        try:
            verified.append(verify_manifest_entry(entry, paths))
        except ArtifactError as exc:
            errors.append(str(exc))

    if errors:
        raise ArtifactError("; ".join(errors))

    return ArtifactVerificationReport(manifest=manifest, verified_artifacts=tuple(verified))
