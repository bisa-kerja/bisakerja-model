# Phase 46 Calibration, Model Card, Artifact Manifest, and Handoff Fixtures Refresh

Status: `complete`

This report records a self-contained `intfloat/multilingual-e5-small` artifact package for Model API handoff.

## Summary

- Embedding model: `intfloat/multilingual-e5-small`
- Runtime-required artifacts: `5`
- Handoff validation: `complete`
- Validation calibration: `{'bucket_agreement': 0.9722, 'bucket_mae_points': {'0-20': 0.3796, '21-40': 0.5004, '41-60': 1.1248, '61-80': 2.3033, '81-100': 0.4679}, 'ece_points': 0.4144, 'mae_points': 0.7418, 'mce_points': 1.5912, 'passed': True, 'rmse_points': 2.9635, 'row_count': 540, 'score_band_agreement': 0.9722, 'within_10_points_rate': 0.9852}`
- Test calibration: `{'bucket_agreement': 0.9722, 'bucket_mae_points': {'0-20': 0.5085, '21-40': 0.4224, '41-60': 0.7402, '61-80': 1.6269, '81-100': 1.0984}, 'ece_points': 0.3842, 'mae_points': 0.7074, 'mce_points': 0.9083, 'passed': True, 'rmse_points': 2.2366, 'row_count': 540, 'score_band_agreement': 0.9722, 'within_10_points_rate': 0.987}`

## Checks

- PASS `notebook_exists_with_required_markdown`
- PASS `no_stale_phase25_runtime_references`
- PASS `model_card_declares_multilingual_e5_small`
- PASS `runtime_manifest_entries_hashed`
- PASS `calibration_outputs_complete`
- PASS `calibration_buckets_complete`
- PASS `clean_reload_smoke_passed`
- PASS `feature_configs_refreshed`
- PASS `handoff_validation_complete`
- PASS `model_core_contract_compatible`
- PASS `tensorboard_references_recorded`
- PASS `rollback_artifact_recorded`

## Artifacts

- `artifact_root`: `artifacts/phase_46_calibration_model_card_manifest_handoff_refresh`
- `model_card`: `artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/model_card.json`
- `artifact_manifest`: `artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/artifact_manifest.json`
- `score_calibration`: `artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/score_calibration.json`
- `handoff_fixtures`: `artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/export/model_api_handoff_fixtures.json`
- `handoff_validation`: `artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/export/model_api_handoff_validation.json`

## Commands

- Write evidence: `training/.tf-venv-3.13/bin/python scripts/verify_phase_46_calibration_model_card_manifest_handoff_refresh.py --write`
- Verify: `.venv/bin/python -m unittest tests.test_phase_46_calibration_model_card_manifest_handoff_refresh`
