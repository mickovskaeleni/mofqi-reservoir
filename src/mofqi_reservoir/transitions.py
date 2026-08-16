"""Validated transition data for fitted Q-iteration."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_feature_matrix(values: ArrayLike, name: str) -> FloatArray:
    """Convert state or action values to a finite two-dimensional array."""
    array = np.asarray(values, dtype=float)

    if array.ndim == 1:
        array = array.reshape(-1, 1)

    if array.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array.")

    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")

    return array


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    """Batch of offline state-action transitions used by FQI."""

    states: ArrayLike
    actions: ArrayLike
    next_states: ArrayLike
    rewards: ArrayLike

    def __post_init__(self) -> None:
        states = _as_feature_matrix(self.states, "states")
        actions = _as_feature_matrix(self.actions, "actions")
        next_states = _as_feature_matrix(self.next_states, "next_states")
        rewards = np.asarray(self.rewards, dtype=float)

        if rewards.ndim != 1:
            raise ValueError("rewards must be a one-dimensional array.")

        sample_counts = {
            states.shape[0],
            actions.shape[0],
            next_states.shape[0],
            rewards.shape[0],
        }
        if len(sample_counts) != 1:
            raise ValueError("All transition arrays must have the same number of samples.")

        if states.shape[1] != next_states.shape[1]:
            raise ValueError("states and next_states must have the same dimension.")

        if not np.isfinite(rewards).all():
            raise ValueError("rewards must contain only finite values.")

        object.__setattr__(self, "states", states)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "next_states", next_states)
        object.__setattr__(self, "rewards", rewards)

    @property
    def n_samples(self) -> int:
        """Return the number of transitions."""
        return self.states.shape[0]

    @property
    def state_dim(self) -> int:
        """Return the number of state variables."""
        return self.states.shape[1]

    @property
    def action_dim(self) -> int:
        """Return the number of action variables."""
        return self.actions.shape[1]

@dataclass(frozen=True, slots=True)
class MultiObjectiveTransitionBatch:
    """Offline transitions containing one reward per objective."""

    states: ArrayLike
    actions: ArrayLike
    next_states: ArrayLike
    rewards: ArrayLike

    def __post_init__(self) -> None:
        states = _as_feature_matrix(self.states, "states")
        actions = _as_feature_matrix(self.actions, "actions")
        next_states = _as_feature_matrix(self.next_states, "next_states")
        rewards = np.asarray(self.rewards, dtype=float)

        if rewards.ndim != 2:
            raise ValueError("rewards must be a two-dimensional array.")

        if rewards.shape[1] < 2:
            raise ValueError("Multi-objective rewards require at least two objectives.")

        sample_counts = {
            states.shape[0],
            actions.shape[0],
            next_states.shape[0],
            rewards.shape[0],
        }
        if len(sample_counts) != 1:
            raise ValueError("All transition arrays must have the same number of samples.")

        if states.shape[1] != next_states.shape[1]:
            raise ValueError("states and next_states must have the same dimension.")

        if not np.isfinite(rewards).all():
            raise ValueError("rewards must contain only finite values.")

        object.__setattr__(self, "states", states)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "next_states", next_states)
        object.__setattr__(self, "rewards", rewards)

    @property
    def n_samples(self) -> int:
        """Return the number of transitions."""
        return self.states.shape[0]

    @property
    def state_dim(self) -> int:
        """Return the number of state variables."""
        return self.states.shape[1]

    @property
    def action_dim(self) -> int:
        """Return the number of action variables."""
        return self.actions.shape[1]

    @property
    def n_objectives(self) -> int:
        """Return the number of reward objectives."""
        return self.rewards.shape[1]