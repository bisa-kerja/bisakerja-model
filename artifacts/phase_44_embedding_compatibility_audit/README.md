# Phase 44 Embedding Compatibility Audit Artifacts

This directory stores multilingual-E5-small embedding compatibility evidence before TensorFlow retraining.

## Files

- `multilingual_e5_small_embedding_metadata.json` — model metadata, runtime dependency versions, prefix policy, dimension, normalization, finite-value checks, and cache metadata.
- `multilingual_e5_small_pair_embeddings_v1.npz` — query and passage embeddings for the frozen Phase 25 pair set using `intfloat/multilingual-e5-small`.
- `multilingual_e5_small_pair_features.json` — old E5-base cosine, new multilingual-E5-small cosine, and slice metadata per pair.
- `embedding_cosine_drift_report.json` — overall and slice-level drift statistics plus worst drift examples without raw CV text.
- `feature_normalization_impact.json` — impact on the approved six-feature vector and Phase 25 normalization stats.
- `retrain_vs_recalibrate_decision.json` — direct-swap rejection and required retraining/recalibration steps.

## Safety

These artifacts are audit evidence only. They must not be used as Model API defaults before retraining, recalibration, contract validation, staging shadow comparison, and rollback gates pass.
