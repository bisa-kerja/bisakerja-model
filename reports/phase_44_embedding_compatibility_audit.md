# Phase 44 Embedding Compatibility Audit

Status: `complete`
Decision: `retrain_required_recalibration_required_direct_swap_blocked`

Phase 44 measures embedding and feature drift for `intfloat/multilingual-e5-small` before TensorFlow retraining. It does not change Model API defaults.

## Summary
- Old embedding model: `intfloat/e5-base-v2`
- New embedding model: `intfloat/multilingual-e5-small`
- New embedding dimension: `384`
- Pair rows compared: `3600`
- Mean absolute cosine delta: `0.04841541873084174`
- Max absolute cosine delta: `0.12737274169921875`
- Pearson correlation: `0.8642641530224022`
- Spearman correlation: `0.864807930599793`
- Material normalization shift: `True`

## Checks
- PASS `metadata_verifies_model_dimension_prefix_norms_and_runtime`
- PASS `paired_embedding_cache_matches_frozen_phase25_pair_set`
- PASS `cosine_drift_quantified_by_required_slices`
- PASS `rank_correlation_and_worst_examples_recorded`
- PASS `multilingual_slices_cover_id_en_mixed_unknown_when_present`
- PASS `feature_normalization_impact_recorded`
- PASS `direct_replacement_rejected_with_retrain_recalibrate_decision`

## Language Slices
- `EN`: count `2260`, mean new cosine `0.8512039266328896`, mean abs delta `0.051058301857087464`
- `ID`: count `22`, mean new cosine `0.8467532775618813`, mean abs delta `0.03402738137678667`
- `MIXED`: count `14`, mean new cosine `0.84163698554039`, mean abs delta `0.06031307578086853`
- `UNKNOWN`: count `1304`, mean new cosine `0.8473376108824842`, mean abs delta `0.04394996915858216`

## Normalization Impact
- Phase 25 e5 cosine mean/std: `{'mean': 0.8016536831855774, 'std': 0.03460521623492241}`
- multilingual-E5-small train cosine stats: `{'count': 2520, 'mean': 0.849728303580057, 'std': 0.022915645295038273, 'min': 0.7859638929367065, 'p05': 0.8143214464187621, 'p25': 0.83119136095047, 'p50': 0.8506752848625183, 'p75': 0.8661491125822067, 'p95': 0.8878830850124358, 'max': 0.9068382382392883}`
- Decision: `phase25_mean_std_invalid_for_multilingual_e5_small`

## Next Steps
- Create Phase 45 training notebook with multilingual-E5-small-derived e5_cosine values.
- Write new train-split normalization stats for all approved features.
- Retrain TensorFlow scorer and compare against Phase 25 E5-base scorer and baselines.
- Recalibrate score buckets before any staging default switch.

## Commands
- Generate live cache: `training/.tf-venv-3.13/bin/python scripts/verify_phase_44_embedding_compatibility_audit.py --write --allow-download --regenerate`
- Verify: `python -m unittest tests.test_phase_44_embedding_compatibility_audit`
