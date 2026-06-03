# Artifacts

Durable generated files used by training, Model API serving, contract validation, and release gates.

## Purpose

This directory stores files that are too operationally important to hide inside notebooks only:

- exported TensorFlow/Keras models
- feature configs and calibration files
- artifact manifests with SHA-256 and byte-size checks
- Backend/Model API contract fixtures
- TensorBoard release evidence
- manual validation label files
- smoke-test fixtures and release-gate inputs

## Important Subdirectories

| Path                                     | Purpose                                                         |
| ---------------------------------------- | --------------------------------------------------------------- |
| `phase_25_tensorflow_training_delivery/` | Current production-track training export and manifest set.      |
| `backend_model_api_contract/`            | Backend/Model API internal contract fixtures and owner matrix.  |
| `phase_27_validation_expansion/`         | ATS benchmark and recommendation validation expansion evidence. |
| `manual_validation/`                     | Human-label governance files and frozen validation labels.      |
| `ats_benchmark/`                         | CV benchmark fixtures for ATS readiness checks.                 |

## Rules

- Prefer updating artifacts through notebooks or scripts, not manual edits.
- Keep relative paths stable because notebooks, tests, and manifests reference them.
- Do not write secrets, raw CV text, tokens, DB URLs, or unrelated PII.
- Runtime services may read artifacts but must not mutate them.
- If an artifact changes, update the owning manifest and run the matching verification script.
