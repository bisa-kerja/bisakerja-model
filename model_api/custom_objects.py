"""Phase 25 Keras custom-object registration boundary.

This module is importable without TensorFlow/Keras so lightweight layout tests can
run before serving dependencies are installed. Runtime code calls
``register_phase25_custom_objects()`` before loading the Phase 25 ``.keras`` file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import ModelApiError

PHASE25_KERAS_PACKAGE = "BisakerjaPhase25"
PHASE45_KERAS_PACKAGE = "BisakerjaPhase45"
PHASE25_CUSTOM_OBJECT_NAMES: tuple[str, ...] = (
    "CosineInteractionLayer",
    "WeightedHuberLoss",
    "ProductionGateCallback",
    "HighRecallCalibrationLayer",
)
PHASE25_REGISTERED_CUSTOM_OBJECT_NAMES: tuple[str, ...] = tuple(
    f"{PHASE25_KERAS_PACKAGE}>{name}" for name in PHASE25_CUSTOM_OBJECT_NAMES
)
PHASE45_REGISTERED_CUSTOM_OBJECT_NAMES: tuple[str, ...] = tuple(
    f"{PHASE45_KERAS_PACKAGE}>{name}" for name in PHASE25_CUSTOM_OBJECT_NAMES
)

HIGH_RECALL_CALIBRATION_THRESHOLD = 0.556
HIGH_RECALL_CALIBRATION_FLOOR = 0.70
TARGET_MAE_NORMALIZED = 0.02
MIN_R2 = 0.15
HIGH_FIT_THRESHOLD = 0.70

_CUSTOM_OBJECTS: dict[str, type[Any]] | None = None


class CustomObjectRegistrationError(ModelApiError):
    """Raised when Phase 25 Keras custom objects cannot be registered."""

    code = "custom_object_registration_error"


def _import_keras_runtime() -> tuple[Any, Any, Any, Any]:
    """Import TensorFlow and Keras lazily with stable error text."""

    try:
        import tensorflow as tf  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime deps
        raise CustomObjectRegistrationError(
            "TensorFlow/Keras dependency missing; install serving requirements before loading supported Keras model"
        ) from exc

    try:
        import keras  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
        keras = tf.keras

    layers = keras.layers
    saving = getattr(keras, "saving", None)
    register = getattr(saving, "register_keras_serializable", None) if saving is not None else None
    if register is None:
        register = keras.utils.register_keras_serializable
    return tf, keras, layers, register


def _build_custom_objects() -> dict[str, type[Any]]:
    """Define and register Phase 25 custom Keras classes once."""

    tf, keras, layers, register = _import_keras_runtime()

    @register(package=PHASE25_KERAS_PACKAGE)
    class CosineInteractionLayer(layers.Layer):  # type: ignore[misc, valid-type]
        """Append cosine-driven interaction features to six-feature input rows."""

        def __init__(
            self,
            cosine_index: int = 0,
            interaction_indices: tuple[int, ...] | list[int] = (1, 2, 3, 4),
            include_original: bool = True,
            passthrough_only: bool = False,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.cosine_index = int(cosine_index)
            self.interaction_indices = tuple(int(index) for index in interaction_indices)
            self.include_original = bool(include_original)
            self.passthrough_only = bool(passthrough_only)

        def call(self, inputs: Any) -> Any:
            values = tf.convert_to_tensor(inputs)
            if self.passthrough_only:
                return tf.identity(values)
            cosine_feature = tf.gather(values, [self.cosine_index], axis=-1)
            interaction_features = tf.gather(values, list(self.interaction_indices), axis=-1)
            interactions = interaction_features * cosine_feature
            pieces = [cosine_feature, interactions]
            if self.include_original:
                pieces.insert(0, values)
            return tf.concat(pieces, axis=-1)

        def compute_output_shape(self, input_shape: Any) -> Any:
            if self.passthrough_only:
                return tuple(input_shape)
            last_dim = input_shape[-1]
            added_dim = 1 + len(self.interaction_indices)
            output_dim = None if last_dim is None else (last_dim if self.include_original else 0) + added_dim
            return (*input_shape[:-1], output_dim)

        def get_config(self) -> dict[str, Any]:
            config = super().get_config()
            config.update(
                {
                    "cosine_index": self.cosine_index,
                    "interaction_indices": list(self.interaction_indices),
                    "include_original": self.include_original,
                    "passthrough_only": self.passthrough_only,
                }
            )
            return config

    register(package=PHASE45_KERAS_PACKAGE)(CosineInteractionLayer)

    @register(package=PHASE25_KERAS_PACKAGE)
    class WeightedHuberLoss(keras.losses.Loss):  # type: ignore[misc, valid-type]
        """Huber loss with extra weight for high-fit and low-fit examples."""

        def __init__(
            self,
            delta: float = 0.03,
            high_fit_threshold: float = HIGH_FIT_THRESHOLD,
            high_fit_weight: float = 4.0,
            low_fit_threshold: float = 0.20,
            low_fit_weight: float = 1.25,
            name: str = "weighted_huber_loss",
            reduction: str = "sum_over_batch_size",
        ) -> None:
            super().__init__(name=name, reduction=reduction)
            self.delta = float(delta)
            self.high_fit_threshold = float(high_fit_threshold)
            self.high_fit_weight = float(high_fit_weight)
            self.low_fit_threshold = float(low_fit_threshold)
            self.low_fit_weight = float(low_fit_weight)

        def call(self, y_true: Any, y_pred: Any) -> Any:
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

        def get_config(self) -> dict[str, Any]:
            config = super().get_config()
            config.update(
                {
                    "delta": self.delta,
                    "high_fit_threshold": self.high_fit_threshold,
                    "high_fit_weight": self.high_fit_weight,
                    "low_fit_threshold": self.low_fit_threshold,
                    "low_fit_weight": self.low_fit_weight,
                }
            )
            return config

    register(package=PHASE45_KERAS_PACKAGE)(WeightedHuberLoss)

    @register(package=PHASE25_KERAS_PACKAGE)
    class ProductionGateCallback(keras.callbacks.Callback):  # type: ignore[misc, valid-type]
        """Record whether validation metrics pass Phase 25 production gates."""

        def __init__(
            self,
            target_mae: float = TARGET_MAE_NORMALIZED,
            min_r2: float = MIN_R2,
            monitor_mae: str = "val_mae",
            monitor_r2: str = "val_r2",
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.target_mae = float(target_mae)
            self.min_r2 = float(min_r2)
            self.monitor_mae = str(monitor_mae)
            self.monitor_r2 = str(monitor_r2)
            self.gate_history: list[dict[str, object]] = []

        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            logs = logs or {}
            mae = logs.get(self.monitor_mae)
            r2 = logs.get(self.monitor_r2)
            passed = mae is not None and r2 is not None and float(mae) <= self.target_mae and float(r2) >= self.min_r2
            self.gate_history.append(
                {
                    "epoch": int(epoch),
                    "mae": None if mae is None else float(mae),
                    "r2": None if r2 is None else float(r2),
                    "passed": bool(passed),
                }
            )

        def get_config(self) -> dict[str, Any]:
            return {
                "target_mae": self.target_mae,
                "min_r2": self.min_r2,
                "monitor_mae": self.monitor_mae,
                "monitor_r2": self.monitor_r2,
            }

    register(package=PHASE45_KERAS_PACKAGE)(ProductionGateCallback)

    @register(package=PHASE25_KERAS_PACKAGE)
    class HighRecallCalibrationLayer(layers.Layer):  # type: ignore[misc, valid-type]
        """Lift high-recall predictions above configured floor, then clamp 0..1."""

        def __init__(
            self,
            threshold: float = HIGH_RECALL_CALIBRATION_THRESHOLD,
            high_floor: float = HIGH_RECALL_CALIBRATION_FLOOR,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.threshold = float(threshold)
            self.high_floor = float(high_floor)

        def call(self, inputs: Any) -> Any:
            values = tf.cast(inputs, tf.float32)
            lifted = tf.where(values >= self.threshold, tf.maximum(values, self.high_floor), values)
            return tf.clip_by_value(lifted, 0.0, 1.0)

        def get_config(self) -> dict[str, Any]:
            config = super().get_config()
            config.update({"threshold": self.threshold, "high_floor": self.high_floor})
            return config

    register(package=PHASE45_KERAS_PACKAGE)(HighRecallCalibrationLayer)

    custom_objects: dict[str, type[Any]] = {
        "CosineInteractionLayer": CosineInteractionLayer,
        "WeightedHuberLoss": WeightedHuberLoss,
        "ProductionGateCallback": ProductionGateCallback,
        "HighRecallCalibrationLayer": HighRecallCalibrationLayer,
    }
    for package in (PHASE25_KERAS_PACKAGE, PHASE45_KERAS_PACKAGE):
        for name, custom_object in tuple(custom_objects.items()):
            custom_objects[f"{package}>{name}"] = custom_object
    globals().update(custom_objects)
    return custom_objects


def get_phase25_custom_objects() -> dict[str, type[Any]]:
    """Return registered Phase 25 custom classes keyed by class name."""

    global _CUSTOM_OBJECTS
    if _CUSTOM_OBJECTS is None:
        _CUSTOM_OBJECTS = _build_custom_objects()
    return dict(_CUSTOM_OBJECTS)


def register_phase25_custom_objects() -> tuple[str, ...]:
    """Register Phase 25 custom objects with Keras serialization registry.

    Returns class names in the stable order expected by Phase 25 reports.
    """

    custom_objects = get_phase25_custom_objects()
    missing = [name for name in PHASE25_CUSTOM_OBJECT_NAMES if name not in custom_objects]
    if missing:
        raise CustomObjectRegistrationError(f"Missing Phase 25 custom object definitions: {missing}")
    return PHASE25_CUSTOM_OBJECT_NAMES


def load_phase25_keras_model(model_path: str | Path) -> Any:
    """Register custom objects, then load supported Phase 25/45/46 Keras model with compile=False."""

    custom_objects = get_phase25_custom_objects()
    _, keras, _, _ = _import_keras_runtime()
    return keras.models.load_model(str(Path(model_path)), compile=False, custom_objects=custom_objects)


def __getattr__(name: str) -> Any:
    """Expose lazy class attributes after runtime deps are available."""

    if name in PHASE25_CUSTOM_OBJECT_NAMES:
        return get_phase25_custom_objects()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CustomObjectRegistrationError",
    "PHASE25_CUSTOM_OBJECT_NAMES",
    "PHASE25_REGISTERED_CUSTOM_OBJECT_NAMES",
    "PHASE45_REGISTERED_CUSTOM_OBJECT_NAMES",
    "PHASE25_KERAS_PACKAGE",
    "PHASE45_KERAS_PACKAGE",
    "register_phase25_custom_objects",
    "get_phase25_custom_objects",
    "load_phase25_keras_model",
    *PHASE25_CUSTOM_OBJECT_NAMES,
]
