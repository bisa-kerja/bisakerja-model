# Phase 25 Final Human-Readable Report

Generated at: `2026-06-03T01:45:43.809468+00:00`

## Final decision

- Status: **staging-ready**
- Model version: `jobfit_tf_phase25_gradient_tape_v1`
- Model API: `Keras Functional API`
- Score scale: training `0-1`, API `0-100`
- Production cap: dirty worktree at export blocks production-ready claim

## Requirement compliance

| Requirement | Evidence |
|---|---|
| 1.1 TensorFlow architecture | Keras Functional API |
| 1.2 Custom component | CosineInteractionLayer, WeightedHuberLoss, ProductionGateCallback, HighRecallCalibrationLayer |
| 1.3 GradientTape loop | PASS |
| 1.4 TensorBoard | PASS (1 event files) |
| 1.5 MAE <= 0.02 | PASS (test=0.0086, validation=0.0084) |
| 2.1 TensorFlow export | PASS (.keras) |
| 2.2 Inference smoke | PASS |
| 3.x REST API boundary | handoff fixtures exported; server remains separate deliverable |
| 4.x GenAI boundary | wrapper contract exported; no external GenAI call in training |

## Key metrics

| Metric | Value |
|---|---|
| validation_mae_0_1 | 0.0084 |
| validation_mae_points | 0.844 |
| validation_r2 | 0.9874 |
| validation_spearman | 0.9955 |
| validation_high_fit_recall | 1.0000 |
| validation_score_band_agreement | 0.9889 |
| test_mae_0_1 | 0.0086 |
| test_mae_points | 0.860 |
| test_r2 | 0.9896 |
| test_spearman | 0.9954 |
| test_high_fit_recall | 1.0000 |
| test_score_band_agreement | 0.9944 |

## Gate status

| Gate | Status |
|---|---|
| final_status | staging-ready |
| all_strict_checks_passed | PASS |
| production_selection_passed | PASS |
| strict_failure_count | 0 |
| git_dirty_at_export | WARN |
| baseline_readiness_cap | production-ready |

## Calibration and manifest checks

| Check | Status |
|---|---|
| calibration_buckets | PASS |
| dataset_manifest_hash | PASS |
| label_manifest_hash | PASS |
| feature_config_hash | PASS |

## Artifact paths

| Artifact | Path |
|---|---|
| final_model | artifacts/phase_25_tensorflow_training_delivery/export/selected_jobfit_tf_phase25.keras |
| model_card | artifacts/phase_25_tensorflow_training_delivery/model_card.json |
| artifact_manifest | artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json |
| tensorboard_log_root | artifacts/tensorboard/phase_25_tensorflow_training_delivery |
| api_handoff_fixtures | artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_fixtures.json |
| api_handoff_validation | artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_validation.json |

## Model API handoff

| Validation | Status |
|---|---|
| cv_core_fixture_model_outputs_only | True |
| candidate_reranking_model_outputs_only | True |
| score_bounds_enforced | True |
| language_id_en_enforced | True |
| candidate_membership_enforced | True |
| wrapper_backend_fields_rejected | True |

Files:
- `artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_fixtures.json`
- `artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_validation.json`

## Known limitations

- Automated hiring decision, rejection, eligibility, salary, or protected-class inference.
- Production score claims without replacing fixture/weak-label evidence with validated labels.
- Backend-owned persistence, auth, job hydration, or GenAI wrapper output generation.

## Next action

- Commit or otherwise freeze the clean-run artifacts before claiming production-ready status.

