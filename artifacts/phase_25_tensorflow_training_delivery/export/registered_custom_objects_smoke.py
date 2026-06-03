
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import layers

HIGH_RECALL_CALIBRATION_THRESHOLD = 0.556
HIGH_RECALL_CALIBRATION_FLOOR = 0.70
TARGET_MAE_NORMALIZED = 0.02
MIN_R2 = 0.15
HIGH_FIT_THRESHOLD = 0.70


@keras.saving.register_keras_serializable(package='BisakerjaPhase25')
class CosineInteractionLayer(layers.Layer):
    def __init__(self, cosine_index=0, interaction_indices=(1, 2, 3, 4), include_original=True, passthrough_only=False, **kwargs):
        super().__init__(**kwargs)
        self.cosine_index = int(cosine_index)
        self.interaction_indices = tuple(int(index) for index in interaction_indices)
        self.include_original = bool(include_original)
        self.passthrough_only = bool(passthrough_only)

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        if self.passthrough_only:
            return tf.identity(inputs)
        cosine_feature = tf.gather(inputs, [self.cosine_index], axis=-1)
        interaction_features = tf.gather(inputs, list(self.interaction_indices), axis=-1)
        interactions = interaction_features * cosine_feature
        pieces = [cosine_feature, interactions]
        if self.include_original:
            pieces.insert(0, inputs)
        return tf.concat(pieces, axis=-1)

    def compute_output_shape(self, input_shape):
        if self.passthrough_only:
            return tuple(input_shape)
        last_dim = input_shape[-1]
        added_dim = 1 + len(self.interaction_indices)
        output_dim = None if last_dim is None else (last_dim if self.include_original else 0) + added_dim
        return (*input_shape[:-1], output_dim)

    def get_config(self):
        config = super().get_config()
        config.update({'cosine_index': self.cosine_index, 'interaction_indices': list(self.interaction_indices), 'include_original': self.include_original, 'passthrough_only': self.passthrough_only})
        return config


@keras.saving.register_keras_serializable(package='BisakerjaPhase25')
class WeightedHuberLoss(keras.losses.Loss):
    def __init__(self, delta=0.03, high_fit_threshold=HIGH_FIT_THRESHOLD, high_fit_weight=4.0, low_fit_threshold=0.20, low_fit_weight=1.25, name='weighted_huber_loss', reduction='sum_over_batch_size'):
        super().__init__(name=name, reduction=reduction)
        self.delta = float(delta)
        self.high_fit_threshold = float(high_fit_threshold)
        self.high_fit_weight = float(high_fit_weight)
        self.low_fit_threshold = float(low_fit_threshold)
        self.low_fit_weight = float(low_fit_weight)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        huber = 0.5 * tf.square(quadratic) + self.delta * linear
        weights = tf.ones_like(huber)
        weights = tf.where(y_true >= self.high_fit_threshold, weights * self.high_fit_weight, weights)
        weights = tf.where(y_true <= self.low_fit_threshold, weights * self.low_fit_weight, weights)
        return tf.reduce_mean(huber * weights, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'delta': self.delta, 'high_fit_threshold': self.high_fit_threshold, 'high_fit_weight': self.high_fit_weight, 'low_fit_threshold': self.low_fit_threshold, 'low_fit_weight': self.low_fit_weight})
        return config


@keras.saving.register_keras_serializable(package='BisakerjaPhase25')
class ProductionGateCallback(keras.callbacks.Callback):
    def __init__(self, target_mae=TARGET_MAE_NORMALIZED, min_r2=MIN_R2, monitor_mae='val_mae', monitor_r2='val_r2', **kwargs):
        super().__init__(**kwargs)
        self.target_mae = float(target_mae)
        self.min_r2 = float(min_r2)
        self.monitor_mae = str(monitor_mae)
        self.monitor_r2 = str(monitor_r2)
        self.gate_history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        mae = logs.get(self.monitor_mae)
        r2 = logs.get(self.monitor_r2)
        passed = mae is not None and r2 is not None and float(mae) <= self.target_mae and float(r2) >= self.min_r2
        self.gate_history.append({'epoch': int(epoch), 'mae': None if mae is None else float(mae), 'r2': None if r2 is None else float(r2), 'passed': bool(passed)})

    def get_config(self):
        return {'target_mae': self.target_mae, 'min_r2': self.min_r2, 'monitor_mae': self.monitor_mae, 'monitor_r2': self.monitor_r2}


@keras.saving.register_keras_serializable(package='BisakerjaPhase25')
class HighRecallCalibrationLayer(layers.Layer):
    def __init__(self, threshold=HIGH_RECALL_CALIBRATION_THRESHOLD, high_floor=HIGH_RECALL_CALIBRATION_FLOOR, **kwargs):
        super().__init__(**kwargs)
        self.threshold = float(threshold)
        self.high_floor = float(high_floor)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        lifted = tf.where(inputs >= self.threshold, tf.maximum(inputs, self.high_floor), inputs)
        return tf.clip_by_value(lifted, 0.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({'threshold': self.threshold, 'high_floor': self.high_floor})
        return config


def main() -> None:
    model_path = Path(sys.argv[1]).resolve()
    feature_matrix_path = Path(sys.argv[2]).resolve()
    output_path = Path(sys.argv[3]).resolve()
    matrix = np.load(feature_matrix_path, allow_pickle=True)
    x = matrix['X_scaled'].astype('float32')
    y = matrix['y'].astype('float32')
    split = matrix['split'].astype(str)
    pair_id = matrix['pair_id'].astype(str)
    indices = np.r_[np.where(split == 'validation')[0][:3], np.where(split == 'test')[0][:3]]
    model = keras.models.load_model(model_path, compile=False)
    predictions = model.predict(x[indices], verbose=0).astype('float32').reshape(-1)
    if predictions.size != indices.size:
        raise RuntimeError(f'Prediction count mismatch: {predictions.size} vs {indices.size}')
    if not np.isfinite(predictions).all():
        raise RuntimeError('Predictions contain non-finite values.')
    if float(predictions.min()) < 0.0 or float(predictions.max()) > 1.0:
        raise RuntimeError(f'Predictions outside 0-1 range: min={float(predictions.min())}, max={float(predictions.max())}')
    rows = []
    for index, score in zip(indices.tolist(), predictions.tolist()):
        rows.append({'pair_id': str(pair_id[index]), 'split': str(split[index]), 'target_0_1': round(float(y[index, 0]), 6), 'score_0_1': round(float(score), 6), 'score_0_100': round(float(score) * 100.0, 3)})
    payload = {'status': 'complete', 'model_name': model.name, 'input_shape': [None, int(x.shape[1])], 'output_shape': [None, 1], 'tensorflow_version': tf.__version__, 'keras_version': keras.__version__, 'custom_objects_registered': ['CosineInteractionLayer', 'WeightedHuberLoss', 'ProductionGateCallback', 'HighRecallCalibrationLayer'], 'sample_predictions': rows, 'score_bounds': {'min_0_1': float(predictions.min()), 'max_0_1': float(predictions.max())}}
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
