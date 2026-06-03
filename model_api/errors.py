"""Deterministic error types for Model API phases."""

from __future__ import annotations


class ModelApiError(Exception):
    """Base exception with stable machine-readable error code."""

    code = "model_api_error"

    def to_error_payload(self) -> dict[str, str]:
        return {"errorCode": self.code, "message": str(self)}


class ArtifactError(ModelApiError):
    code = "artifact_error"


class UnsupportedArtifactVersionError(ArtifactError):
    code = "unsupported_artifact_version"


class ModelNotReadyError(ModelApiError):
    code = "model_not_ready"


class ModelLoadError(ModelApiError):
    code = "model_load_error"


class InferenceTimeoutError(ModelApiError):
    code = "inference_timeout"


class FeatureBuildError(ModelApiError):
    code = "feature_build_error"


class ContractReferenceError(ModelApiError):
    code = "contract_reference_error"


class ContractValidationError(ModelApiError):
    code = "contract_validation_error"

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors

    def to_error_payload(self) -> dict[str, object]:
        return {"errorCode": self.code, "message": str(self), "errors": self.errors}
