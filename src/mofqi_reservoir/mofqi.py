"""Weight augmentation and reward scalarisation for MOFQI."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mofqi_reservoir.transitions import (
    MultiObjectiveTransitionBatch,
    TransitionBatch,
)


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