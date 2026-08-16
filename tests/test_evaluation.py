import numpy as np

from mofqi_reservoir import evaluate_policy


class ThreeStepEnvironment:
    """Deterministic environment used to test policy evaluation."""

    def reset(self):
        self.time = 0
        return [0.0]

    def step(self, action):
        self.time += 1

        next_state = [float(self.time)]
        reward = [1.0, -2.0]
        terminated = self.time >= 3

        return next_state, reward, terminated


def increasing_action_policy(state):
    """Select an action using the current state."""
    return [state[0] + 1.0]


def test_policy_evaluation_records_complete_trajectory():
    """Evaluation records states, actions and vector rewards."""
    result = evaluate_policy(
        environment=ThreeStepEnvironment(),
        policy=increasing_action_policy,
        max_steps=10,
        gamma=0.5,
    )

    assert result.n_steps == 3
    assert result.n_objectives == 2

    np.testing.assert_allclose(
        result.states.ravel(),
        [0.0, 1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        result.actions.ravel(),
        [1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        result.rewards,
        [
            [1.0, -2.0],
            [1.0, -2.0],
            [1.0, -2.0],
        ],
    )
    np.testing.assert_allclose(
        result.objective_returns,
        [1.75, -3.5],
    )

    assert np.isclose(
        result.scalarized_return([0.25, 0.75]),
        -2.1875,
    )


def test_policy_evaluation_respects_maximum_steps():
    """Simulation stops at the requested horizon."""
    result = evaluate_policy(
        environment=ThreeStepEnvironment(),
        policy=increasing_action_policy,
        max_steps=2,
    )

    assert result.n_steps == 2
    np.testing.assert_allclose(
        result.objective_returns,
        [2.0, -4.0],
    )