"""Batch-mode fitted Q-iteration using Extremely Randomized Trees."""

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import NotFittedError

from mofqi_reservoir.transitions import TransitionBatch


FloatArray = NDArray[np.float64]


class Predictor(Protocol):
    """Interface required from a fitted action-value regressor."""

    def predict(self, features: FloatArray) -> ArrayLike:
        """Predict action values for state-action features."""
        ...


def _as_feature_matrix(values: ArrayLike, name: str) -> FloatArray:
    """Convert feature values to a finite two-dimensional array."""
    matrix = np.asarray(values, dtype=float)

    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    if matrix.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")

    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")

    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")

    return matrix


def _as_action_grid(values: ArrayLike) -> FloatArray:
    """Convert candidate actions to a validated matrix."""
    return _as_feature_matrix(values, "candidate_actions")


def _aligned_features(states: ArrayLike, actions: ArrayLike) -> FloatArray:
    """Combine corresponding state and action rows."""
    state_matrix = _as_feature_matrix(states, "states")
    action_matrix = _as_feature_matrix(actions, "actions")

    if state_matrix.shape[0] != action_matrix.shape[0]:
        raise ValueError("states and actions must have the same number of rows.")

    return np.hstack((state_matrix, action_matrix))


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


class FittedQIteration:
    """Batch-mode FQI with standard scikit-learn Extra Trees."""

    def __init__(
        self,
        candidate_actions: ArrayLike,
        gamma: float = 1.0,
        n_iterations: int = 10,
        n_estimators: int = 100,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")

        if n_iterations < 1:
            raise ValueError("n_iterations must be at least 1.")

        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")

        self.candidate_actions = _as_action_grid(candidate_actions)
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, batch: TransitionBatch) -> "FittedQIteration":
        """Fit one Extra Trees ensemble at each FQI iteration."""
        if self.candidate_actions.shape[1] != batch.action_dim:
            raise ValueError(
                "candidate_actions and training actions must have the same dimension."
            )

        training_features = np.hstack((batch.states, batch.actions))
        previous_model = None
        models = []

        for iteration in range(self.n_iterations):
            targets = build_bellman_targets(
                batch=batch,
                candidate_actions=self.candidate_actions,
                gamma=self.gamma,
                previous_model=previous_model,
            )

            seed = (
                None
                if self.random_state is None
                else self.random_state + iteration
            )

            model = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                max_features=1.0,
                bootstrap=False,
                random_state=seed,
                n_jobs=self.n_jobs,
            )
            model.fit(training_features, targets)

            models.append(model)
            previous_model = model

        self.models_ = models
        self.model_ = models[-1]
        self.state_dim_ = batch.state_dim
        self.action_dim_ = batch.action_dim

        return self

    def predict(
        self,
        states: ArrayLike,
        actions: ArrayLike,
    ) -> FloatArray:
        """Predict action values for corresponding state-action pairs."""
        if not hasattr(self, "model_"):
            raise NotFittedError(
                "FittedQIteration must be fitted before prediction."
            )

        features = _aligned_features(states, actions)

        if features.shape[1] != self.state_dim_ + self.action_dim_:
            raise ValueError(
                "Prediction features do not match the training dimensions."
            )

        return np.asarray(self.model_.predict(features), dtype=float)
    
    def select_actions(
        self,
        states: ArrayLike,
    ) -> FloatArray:
        """Select the candidate action with the highest predicted value."""
        if not hasattr(self, "model_"):
            raise NotFittedError(
                "FittedQIteration must be fitted before action selection."
            )

        state_matrix = _as_feature_matrix(states, "states")

        if state_matrix.shape[1] != self.state_dim_:
            raise ValueError("states do not match the training dimension.")

        n_states = state_matrix.shape[0]
        n_actions = self.candidate_actions.shape[0]

        repeated_states = np.repeat(
            state_matrix,
            n_actions,
            axis=0,
        )
        tiled_actions = np.tile(
            self.candidate_actions,
            (n_states, 1),
        )

        values = self.predict(
            states=repeated_states,
            actions=tiled_actions,
        ).reshape(n_states, n_actions)

        best_indices = np.argmax(values, axis=1)
        return self.candidate_actions[best_indices].copy()
    