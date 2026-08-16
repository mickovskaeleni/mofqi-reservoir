"""Core calculations for batch-mode fitted Q-iteration."""

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mofqi_reservoir.transitions import TransitionBatch


FloatArray = NDArray[np.float64]


class Predictor(Protocol):
    """Interface required from a fitted action-value regressor."""

    def predict(self, features: FloatArray) -> ArrayLike:
        """Predict action values for state-action features."""
        ...


def _as_action_grid(values: ArrayLike) -> FloatArray:
    """Convert candidate actions to a finite two-dimensional array."""
    actions = np.asarray(values, dtype=float)

    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)

    if actions.ndim != 2:
        raise ValueError("candidate_actions must be one- or two-dimensional.")

    if actions.shape[0] == 0:
        raise ValueError("candidate_actions must contain at least one action.")

    if not np.isfinite(actions).all():
        raise ValueError("candidate_actions must contain only finite values.")

    return actions


def _candidate_features(
    states: FloatArray,
    candidate_actions: FloatArray,
) -> FloatArray:
    """Create every state-action combination used in the Bellman maximum."""
    repeated_states = np.repeat(states, candidate_actions.shape[0], axis=0)
    tiled_actions = np.tile(candidate_actions, (states.shape[0], 1))
    return np.hstack((repeated_states, tiled_actions))


def build_bellman_targets(
    batch: TransitionBatch,
    candidate_actions: ArrayLike,
    gamma: float,
    previous_model: Predictor | None = None,
) -> FloatArray:
    """Construct one FQI iteration's regression targets."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")

    actions = _as_action_grid(candidate_actions)

    if previous_model is None:
        return np.asarray(batch.rewards, dtype=float).copy()

    features = _candidate_features(batch.next_states, actions)
    predictions = np.asarray(previous_model.predict(features), dtype=float)

    expected_size = batch.n_samples * actions.shape[0]
    if predictions.ndim != 1 or predictions.shape[0] != expected_size:
        raise ValueError(
            "The previous model returned an unexpected number of predictions."
        )

    if not np.isfinite(predictions).all():
        raise ValueError("The previous model returned nonfinite predictions.")

    maximum_next_values = predictions.reshape(
        batch.n_samples,
        actions.shape[0],
    ).max(axis=1)

    return np.asarray(batch.rewards) + gamma * maximum_next_values