# Phase 43 multilingual-E5-small Migration Decision

Decision: `approved_for_staging_experiment_only`

Phase 43 freezes E5-base behavior before any multilingual-E5-small implementation. Phase 25 artifacts remain rollback baseline.

## Baseline Summary
- Embedding model: `intfloat/e5-base-v2`
- Model artifact hash: `734b8c22f618e05b4b90ba24f4f0d01b475da4674c94e81fc3ed545fae40063d`
- Phase 25 status: `staging-ready`
- Validation MAE: `0.84428` points
- Test MAE: `0.86032` points
- First CV analysis latency: `None` ms
- Warm internal latency: `733.644` ms
- E5-base cache size: `877935460` bytes

## Checks
- PASS `runtime_baseline_has_model_hash_environment_commands`
- PASS `runtime_baseline_records_required_endpoint_latency_fields`
- PASS `quality_baseline_has_metrics_calibration_distribution_examples`
- PASS `migration_hypothesis_and_risks_explicit`
- PASS `go_no_go_thresholds_cover_runtime_quality_contract_rollback`
- PASS `artifact_namespace_keeps_phase25_rollback_safe`
- PASS `model_version_names_reserved_and_schema_compatibility_noted`
- PASS `staging_experiment_only_decision_recorded`

## Reserved Namespace
- Artifact root: `artifacts/phase_43_multilingual_e5_small_migration/`
- Later training root: `artifacts/phase_45_multilingual_e5_small_training_delivery/`
- Forbidden overwrite root: `artifacts/phase_25_tensorflow_training_delivery/`

## Go/No-Go Threshold Groups
- `runtime`
- `quality`
- `contract`
- `rollback`

## Commands
- `phase43_gate`: `python scripts/verify_phase_43_multilingual_e5_small_migration.py --write`
- `phase44_next`: create embedding compatibility audit before changing runtime defaults.
