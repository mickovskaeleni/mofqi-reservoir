"""Simulation-based evaluation of learned policies."""

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sklearn.exceptions import NotFittedError

from mofqi_reservoir.fqi import FittedQIteration
from mofqi_reservoir.mofqi import (
    MultiObjectiveFittedQIteration,
    validate_preference_weights,
)

FloatArray = NDArray[np.float64]
Policy = Callable[[FloatArray], ArrayLike]


class MultiObjectiveEnvironment(Protocol):
    """Interface required for policy simulation."""

    def reset(self) -> ArrayLike:
        """Reset the environment and return its initial state."""
        ...

    def step(
        self,
        action: FloatArray,
    ) -> tuple[ArrayLike, ArrayLike, bool]:
        """Apply an action and return state, vector reward and termination."""
        ...


def _as_vector(values: ArrayLike, name: str) -> FloatArray:
    """Convert values to a finite one-dimensional array."""
    vector = np.asarray(values, dtype=float)

    if vector.ndim == 0:
        vector = vector.reshape(1)

    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if vector.size == 0:
        raise ValueError(f"{name} must contain at least one value.")

    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values.")

    return vector


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """Trajectory and objective-specific returns from one simulation."""

    states: FloatArray
    actions: FloatArray
    rewards: FloatArray
    objective_returns: FloatArray

    @property
    def n_steps(self) -> int:
        """Return the number of simulated decisions."""
        return self.actions.shape[0]

    @property
    def n_objectives(self) -> int:
        """Return the number of evaluated objectives."""
        return self.objective_returns.shape[0]

    def scalarized_return(self, weights: ArrayLike) -> float:
        """Combine objective returns using one preference vector."""
        weight_matrix = validate_preference_weights(
            weights,
            n_objectives=self.n_objectives,
        )

        if weight_matrix.shape[0] != 1:
            raise ValueError(
                "scalarized_return requires exactly one weight vector."
            )

        return float(weight_matrix[0] @ self.objective_returns)


def evaluate_policy(
    environment: MultiObjectiveEnvironment,
    policy: Policy,
    max_steps: int,
    gamma: float = 1.0,
) -> PolicyEvaluation:
    """Simulate a policy and calculate discounted objective returns."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")

    state = _as_vector(environment.reset(), "initial state")
    states = [state.copy()]
    actions = []
    rewards = []

    action_dim = None
    reward_dim = None
    objective_returns = None

    for step_index in range(max_steps):
        action = _as_vector(
            policy(state.copy()),
            "policy action",
        )

        if action_dim is None:
            action_dim = action.shape[0]
        elif action.shape[0] != action_dim:
            raise ValueError(
                "policy actions must have a consistent dimension."
            )

        next_state_values, reward_values, terminated = environment.step(
            action.copy()
        )

        next_state = _as_vector(next_state_values, "next state")
        reward = _as_vector(reward_values, "reward")

        if next_state.shape[0] != state.shape[0]:
            raise ValueError(
                "environment states must have a consistent dimension."
            )

        if reward.shape[0] < 2:
            raise ValueError(
                "policy evaluation requires at least two objectives."
            )

        if reward_dim is None:
            reward_dim = reward.shape[0]
            objective_returns = np.zeros(reward_dim, dtype=float)
        elif reward.shape[0] != reward_dim:
            raise ValueError(
                "environment rewards must have a consistent dimension."
            )

        if not isinstance(terminated, (bool, np.bool_)):
            raise ValueError("terminated must be a boolean value.")

        objective_returns += (gamma**step_index) * reward

        actions.append(action.copy())
        rewards.append(reward.copy())
        states.append(next_state.copy())
        state = next_state

        if terminated:
            break

    return PolicyEvaluation(
        states=np.vstack(states),
        actions=np.vstack(actions),
        rewards=np.vstack(rewards),
        objective_returns=objective_returns,
    )

def evaluate_fqi_policy(
    environment: MultiObjectiveEnvironment,
    learner: FittedQIteration,
    max_steps: int,
    gamma: float = 1.0,
) -> PolicyEvaluation:
    """Evaluate a greedy policy learned by standard FQI."""
    if not hasattr(learner, "model_"):
        raise NotFittedError(
            "FittedQIteration must be fitted before policy evaluation."
        )

    def policy(state: FloatArray) -> FloatArray:
        return learner.select_actions(
            states=state.reshape(1, -1),
        )[0]

    return evaluate_policy(
        environment=environment,
        policy=policy,
        max_steps=max_steps,
        gamma=gamma,
    )


def evaluate_mofqi_policy(
    environment: MultiObjectiveEnvironment,
    learner: MultiObjectiveFittedQIteration,
    weights: ArrayLike,
    max_steps: int,
    gamma: float = 1.0,
) -> PolicyEvaluation:
    """Evaluate a MOFQI policy for one objective preference."""
    if not hasattr(learner, "learner_"):
        raise NotFittedError(
            "MultiObjectiveFittedQIteration must be fitted "
            "before policy evaluation."
        )

    weight_matrix = validate_preference_weights(
        weights,
        n_objectives=learner.n_objectives_,
    )

    if weight_matrix.shape[0] != 1:
        raise ValueError(
            "Policy evaluation requires exactly one weight vector."
        )

    def policy(state: FloatArray) -> FloatArray:
        return learner.select_actions(
            states=state.reshape(1, -1),
            weights=weight_matrix,
        )[0]

    return evaluate_policy(
        environment=environment,
        policy=policy,
        max_steps=max_steps,
        gamma=gamma,
    )