# Phase 46 multilingual-E5-small Artifact Package

Self-contained runtime and release handoff package for `intfloat/multilingual-e5-small` TensorFlow scoring artifacts.

## Runtime-required files

- `export/selected_jobfit_tf_phase46_multilingual_e5_small.keras`
- `tensorflow_feature_config.json`
- `feature_config.json`
- `score_calibration.json`
- `model_card.json`
- `artifact_manifest.json`

## Handoff evidence

- `export/model_api_handoff_fixtures.json`
- `export/model_api_handoff_validation.json`
- `export/inference_smoke_fixture.json`
- `score_calibration_tables.csv`
- `label_manifest.json`
- `dataset_manifest.json`
- `tensorboard_monitoring_manifest.json`

Phase 25 remains rollback-only. Runtime loaders must not mix Phase 25 calibration/config hashes with this package.
