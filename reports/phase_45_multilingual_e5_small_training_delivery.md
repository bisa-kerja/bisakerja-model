# Phase 45 multilingual-E5-small Training Delivery

Status: `complete`
Selection decision: `select_for_staging_shadow_validation`

This report records TensorFlow retraining evidence with `intfloat/multilingual-e5-small`-derived `e5_cosine` values while preserving the approved model-core contract.

## Summary

- Embedding model: `intfloat/multilingual-e5-small`
- Feature rows: `3600`
- Feature count: `6`
- Split counts: `{'test': 540, 'train': 2520, 'validation': 540}`
- Validation metrics: `{'high_fit_recall': 0.93333333, 'loss': 0.00016084, 'mae': 0.00741753, 'mae_0_100': 0.741753, 'predicted_high_count': 85, 'r2': 0.98998901, 'rmse': 0.02963539, 'rmse_0_100': 2.963539, 'row_count': 540, 'score_band_agreement': 0.98518519, 'spearman': 0.99543045, 'true_high_count': 90}`
- Test metrics: `{'high_fit_recall': 0.94444444, 'loss': 0.00014651, 'mae': 0.0070744, 'mae_0_100': 0.70744, 'predicted_high_count': 85, 'r2': 0.9941131, 'rmse': 0.022366, 'rmse_0_100': 2.2366, 'row_count': 540, 'score_band_agreement': 0.98888889, 'spearman': 0.9955746, 'true_high_count': 90}`

## Checks

- PASS `notebook_exists_with_required_markdown`
- PASS `phase25_artifacts_not_mutated_by_namespace`
- PASS `feature_config_records_multilingual_e5_small`
- PASS `approved_feature_contract_preserved`
- PASS `train_split_normalization_written`
- PASS `gradient_tape_training_used`
- PASS `custom_components_present`
- PASS `required_metrics_present`
- PASS `ranking_metrics_present`
- PASS `indonesian_behavior_recorded`
- PASS `tensorboard_manifest_has_event_file_hashes`
- PASS `selection_or_rejection_recorded`

## Artifacts

- `feature_config`: `artifacts/phase_45_multilingual_e5_small_training_delivery/tensorflow_feature_config.json`
- `feature_matrix`: `artifacts/phase_45_multilingual_e5_small_training_delivery/tensorflow_training_features_v1.npz`
- `training_evaluation`: `artifacts/phase_45_multilingual_e5_small_training_delivery/training_evaluation.json`
- `baseline_comparison`: `artifacts/phase_45_multilingual_e5_small_training_delivery/baseline_comparison.json`
- `tensorboard_manifest`: `artifacts/phase_45_multilingual_e5_small_training_delivery/tensorboard_monitoring_manifest.json`
- `selection_decision`: `artifacts/phase_45_multilingual_e5_small_training_delivery/selection_decision.json`
- `model`: `artifacts/phase_45_multilingual_e5_small_training_delivery/gradient_tape_trained_candidate.keras`

## Commands

- Train/write evidence: `training/.tf-venv-3.13/bin/python scripts/verify_phase_45_multilingual_e5_small_training_delivery.py --write`
- Verify: `.venv/bin/python -m unittest tests.test_phase_45_multilingual_e5_small_training_delivery`
