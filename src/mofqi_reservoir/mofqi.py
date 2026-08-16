"""Weight augmentation and reward scalarisation for MOFQI."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mofqi_reservoir.transitions import (
    MultiObjectiveTransitionBatch,
    TransitionBatch,
)

from sklearn.exceptions import NotFittedError

from mofqi_reservoir.fqi import FittedQIteration


FloatArray = NDArray[np.float64]


def validate_preference_weights(
    weights: ArrayLike,
    n_objectives: int,
) -> FloatArray:
    """Validate objective weights on the unit simplex."""
    if n_objectives < 2:
        raise ValueError("n_objectives must be at least 2.")

    matrix = np.asarray(weights, dtype=float)

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    if matrix.ndim != 2:
        raise ValueError("weights must be one- or two-dimensional.")

    if matrix.shape[0] == 0:
        raise ValueError("weights must contain at least one row.")

    if matrix.shape[1] != n_objectives:
        raise ValueError("Each weight vector must match the number of objectives.")

    if not np.isfinite(matrix).all():
        raise ValueError("weights must contain only finite values.")

    if np.any(matrix < 0.0):
        raise ValueError("weights must be nonnegative.")

    if not np.allclose(matrix.sum(axis=1), 1.0):
        raise ValueError("Each weight vector must sum to 1.")

    return matrix


def sample_preference_weights(
    n_weights: int,
    n_objectives: int,
    random_state: int | None = None,
) -> FloatArray:
    """Sample reproducible weights uniformly from the unit simplex."""
    if n_weights < 1:
        raise ValueError("n_weights must be at least 1.")

    if n_objectives < 2:
        raise ValueError("n_objectives must be at least 2.")

    generator = np.random.default_rng(random_state)
    return generator.dirichlet(
        alpha=np.ones(n_objectives),
        size=n_weights,
    )


def scalarize_rewards(
    rewards: ArrayLike,
    weights: ArrayLike,
) -> FloatArray:
    """Calculate the weighted sum of vector-valued rewards."""
    reward_matrix = np.asarray(rewards, dtype=float)

    if reward_matrix.ndim != 2:
        raise ValueError("rewards must be a two-dimensional array.")

    if not np.isfinite(reward_matrix).all():
        raise ValueError("rewards must contain only finite values.")

    weight_matrix = validate_preference_weights(
        weights,
        n_objectives=reward_matrix.shape[1],
    )

    if weight_matrix.shape[0] == 1:
        weight_matrix = np.repeat(
            weight_matrix,
            reward_matrix.shape[0],
            axis=0,
        )
    elif weight_matrix.shape[0] != reward_matrix.shape[0]:
        raise ValueError(
            "weights must contain either one row or one row per reward."
        )

    return np.einsum("ij,ij->i", reward_matrix, weight_matrix)


def augment_transitions(
    batch: MultiObjectiveTransitionBatch,
    weights: ArrayLike,
) -> TransitionBatch:
    """Create the weight-augmented scalar transition dataset for MOFQI."""
    weight_matrix = validate_preference_weights(
        weights,
        n_objectives=batch.n_objectives,
    )
    n_weights = weight_matrix.shape[0]

    repeated_states = np.repeat(batch.states, n_weights, axis=0)
    repeated_actions = np.repeat(batch.actions, n_weights, axis=0)
    repeated_next_states = np.repeat(batch.next_states, n_weights, axis=0)
    repeated_rewards = np.repeat(batch.rewards, n_weights, axis=0)

    tiled_weights = np.tile(weight_matrix, (batch.n_samples, 1))
    weight_coordinates = tiled_weights[:, :-1]

    augmented_states = np.hstack(
        (repeated_states, weight_coordinates)
    )
    augmented_next_states = np.hstack(
        (repeated_next_states, weight_coordinates)
    )
    scalar_rewards = scalarize_rewards(
        repeated_rewards,
        tiled_weights,
    )

    return TransitionBatch(
        states=augmented_states,
        actions=repeated_actions,
        next_states=augmented_next_states,
        rewards=scalar_rewards,
    )

def _as_prediction_matrix(
    values: ArrayLike,
    name: str,
) -> FloatArray:
    """Convert prediction inputs to a finite feature matrix."""
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


class MultiObjectiveFittedQIteration:
    """Weight-conditioned MOFQI using one FQI training process."""

    def __init__(
        self,
        candidate_actions: ArrayLike,
        gamma: float = 1.0,
        n_iterations: int = 10,
        n_estimators: int = 100,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.candidate_actions = candidate_actions
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(
        self,
        batch: MultiObjectiveTransitionBatch,
        weights: ArrayLike,
    ) -> "MultiObjectiveFittedQIteration":
        """Fit one FQI model over the augmented state-weight space."""
        weight_matrix = validate_preference_weights(
            weights,
            n_objectives=batch.n_objectives,
        )
        augmented_batch = augment_transitions(batch, weight_matrix)

        learner = FittedQIteration(
            candidate_actions=self.candidate_actions,
            gamma=self.gamma,
            n_iterations=self.n_iterations,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        learner.fit(augmented_batch)

        self.learner_ = learner
        self.model_ = learner.model_
        self.models_ = learner.models_
        self.augmented_batch_ = augmented_batch
        self.weights_ = weight_matrix
        self.state_dim_ = batch.state_dim
        self.action_dim_ = batch.action_dim
        self.n_objectives_ = batch.n_objectives

        return self

    def predict(
        self,
        states: ArrayLike,
        weights: ArrayLike,
        actions: ArrayLike,
    ) -> FloatArray:
        """Predict values for aligned state-preference-action inputs."""
        if not hasattr(self, "learner_"):
            raise NotFittedError(
                "MultiObjectiveFittedQIteration must be fitted before prediction."
            )

        state_matrix = _as_prediction_matrix(states, "states")
        action_matrix = _as_prediction_matrix(actions, "actions")

        if state_matrix.shape[0] != action_matrix.shape[0]:
            raise ValueError(
                "states and actions must have the same number of rows."
            )

        if state_matrix.shape[1] != self.state_dim_:
            raise ValueError("states do not match the training dimension.")

        if action_matrix.shape[1] != self.action_dim_:
            raise ValueError("actions do not match the training dimension.")

        weight_matrix = validate_preference_weights(
            weights,
            n_objectives=self.n_objectives_,
        )

        if weight_matrix.shape[0] == 1:
            weight_matrix = np.repeat(
                weight_matrix,
                state_matrix.shape[0],
                axis=0,
            )
        elif weight_matrix.shape[0] != state_matrix.shape[0]:
            raise ValueError(
                "weights must contain one row or one row per state."
            )

        augmented_states = np.hstack(
            (state_matrix, weight_matrix[:, :-1])
        )

        return self.learner_.predict(
            states=augmented_states,
            actions=action_matrix,
        )