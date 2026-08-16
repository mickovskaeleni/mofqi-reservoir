import numpy as np

from mofqi_reservoir import (
    FittedQIteration,
    MultiObjectiveFittedQIteration,
    MultiObjectiveTransitionBatch,
    TransitionBatch,
    evaluate_fqi_policy,
    evaluate_mofqi_policy,
    evaluate_policy,
)


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

class ActionRewardEnvironment:
    """Two-objective environment whose rewards depend on the action."""

    def reset(self):
        self.time = 0
        return [0.0]

    def step(self, action):
        self.time += 1
        action_value = float(action[0])

        next_state = [0.0]
        reward = [
            1.0 - action_value,
            action_value,
        ]
        terminated = self.time >= 2

        return next_state, reward, terminated


def test_evaluate_fqi_policy_uses_greedy_actions():
    """FQI evaluation applies the learner's greedy policy."""
    batch = TransitionBatch(
        states=[0.0, 0.0],
        actions=[0.0, 1.0],
        next_states=[0.0, 0.0],
        rewards=[0.0, 1.0],
    )

    learner = FittedQIteration(
        candidate_actions=[0.0, 1.0],
        gamma=0.0,
        n_iterations=1,
        n_estimators=20,
        random_state=42,
    ).fit(batch)

    result = evaluate_fqi_policy(
        environment=ActionRewardEnvironment(),
        learner=learner,
        max_steps=2,
    )

    np.testing.assert_allclose(
        result.actions.ravel(),
        [1.0, 1.0],
    )
    np.testing.assert_allclose(
        result.objective_returns,
        [0.0, 2.0],
    )


def test_evaluate_mofqi_policy_uses_requested_preference():
    """MOFQI evaluation changes policy with objective preference."""
    batch = MultiObjectiveTransitionBatch(
        states=[0.0, 0.0],
        actions=[0.0, 1.0],
        next_states=[0.0, 0.0],
        rewards=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
    )

    learner = MultiObjectiveFittedQIteration(
        candidate_actions=[0.0, 1.0],
        gamma=0.0,
        n_iterations=1,
        n_estimators=20,
        random_state=42,
    ).fit(
        batch,
        weights=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
    )

    first_objective = evaluate_mofqi_policy(
        environment=ActionRewardEnvironment(),
        learner=learner,
        weights=[1.0, 0.0],
        max_steps=2,
    )
    second_objective = evaluate_mofqi_policy(
        environment=ActionRewardEnvironment(),
        learner=learner,
        weights=[0.0, 1.0],
        max_steps=2,
    )

    np.testing.assert_allclose(
        first_objective.actions.ravel(),
        [0.0, 0.0],
    )
    np.testing.assert_allclose(
        first_objective.objective_returns,
        [2.0, 0.0],
    )
    np.testing.assert_allclose(
        second_objective.actions.ravel(),
        [1.0, 1.0],
    )
    np.testing.assert_allclose(
        second_objective.objective_returns,
        [0.0, 2.0],
    )