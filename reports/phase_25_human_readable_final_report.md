# Phase 25 Final Human-Readable Report

Generated at: `2026-06-04T08:20:00.979855+00:00`

## Final decision

- Status: **prototype-only**
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
| 1.5 MAE <= 0.02 | PASS (test=0.0150, validation=0.0119) |
| 2.1 TensorFlow export | PASS (.keras) |
| 2.2 Inference smoke | FAIL |
| 3.x REST API boundary | handoff fixtures exported; server remains separate deliverable |
| 4.x GenAI boundary | wrapper contract exported; no external GenAI call in training |

## Key metrics

| Metric | Value |
|---|---|
| validation_mae_0_1 | 0.0119 |
| validation_mae_points | 1.189 |
| validation_r2 | 0.9694 |
| validation_spearman | 0.9940 |
| validation_high_fit_recall | 1.0000 |
| validation_score_band_agreement | 0.9778 |
| test_mae_0_1 | 0.0150 |
| test_mae_points | 1.496 |
| test_r2 | 0.9515 |
| test_spearman | 0.9931 |
| test_high_fit_recall | 1.0000 |
| test_score_band_agreement | 0.9778 |

## Gate status

| Gate | Status |
|---|---|
| final_status | prototype-only |
| all_strict_checks_passed | FAIL |
| production_selection_passed | FAIL |
| strict_failure_count | 1 |
| git_dirty_at_export | WARN |
| baseline_readiness_cap | staging-ready |

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
| final_model | artifacts\phase_25_tensorflow_training_delivery\export\selected_jobfit_tf_phase25.keras |
| model_card | artifacts\phase_25_tensorflow_training_delivery\model_card.json |
| artifact_manifest | artifacts\phase_25_tensorflow_training_delivery\artifact_manifest.json |
| tensorboard_log_root | artifacts\tensorboard\phase_25_tensorflow_training_delivery |
| api_handoff_fixtures | artifacts\phase_25_tensorflow_training_delivery\export\model_api_handoff_fixtures.json |
| api_handoff_validation | artifacts\phase_25_tensorflow_training_delivery\export\model_api_handoff_validation.json |

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
- `artifacts\phase_25_tensorflow_training_delivery\export\model_api_handoff_fixtures.json`
- `artifacts\phase_25_tensorflow_training_delivery\export\model_api_handoff_validation.json`

## Known limitations

- Automated hiring decision, rejection, eligibility, salary, or protected-class inference.
- Production score claims without replacing fixture/weak-label evidence with validated labels.
- Backend-owned persistence, auth, job hydration, or GenAI wrapper output generation.

## Next action

- Commit or otherwise freeze the clean-run artifacts before claiming production-ready status.

