# Phase 45 multilingual-E5-small Training Delivery Artifacts

This directory stores TensorFlow retraining evidence using `intfloat/multilingual-e5-small` features. Phase 25 artifacts remain rollback-safe and are not overwritten.

## Files

- `tensorflow_training_features_v1.npz` — approved six-feature matrix with multilingual-E5-small `e5_cosine` and frozen Phase 25 labels/splits.
- `tensorflow_feature_config.json` — feature order, train-split normalization stats, embedding metadata, and source hashes.
- `gradient_tape_training_history.csv` — manual `tf.GradientTape` training history.
- `gradient_tape_predictions_v1.npz` — bounded `0-1` predictions and targets for every frozen pair.
- `gradient_tape_trained_candidate.keras` — trained TensorFlow Functional API candidate model with registered custom components.
- `training_evaluation.json` — quality metrics, slice metrics, ranking metrics, clean reload smoke, and Indonesian behavior examples.
- `baseline_comparison.json` — constant, skill-only, cosine-only, Phase 25, previous-scorer, and Phase 45 comparisons.
- `tensorboard_monitoring_manifest.json` — bounded TensorBoard event file manifest with hashes and byte sizes.
- `selection_decision.json` — staging-shadow select/reject result. Model API defaults must not read Phase 45 artifacts until calibration and handoff refresh pass.

## Safety

These artifacts are training evidence only. Do not mix them with Phase 25 calibration or Model API handoff files.
