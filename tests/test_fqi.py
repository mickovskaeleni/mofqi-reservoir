import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import NotFittedError

from mofqi_reservoir import (
    FittedQIteration,
    TransitionBatch,
    build_bellman_targets,
)


class LinearQ:
    """Deterministic action-value model used to test target construction."""

    def predict(self, features):
        states = features[:, 0]
        actions = features[:, 1]
        return states + 2.0 * actions


@pytest.fixture
def transition_batch():
    return TransitionBatch(
        states=[0.0, 0.0],
        actions=[0.0, 1.0],
        next_states=[1.0, 2.0],
        rewards=[0.5, -1.0],
    )


def test_first_iteration_uses_immediate_rewards(transition_batch):
    """The first FQI iteration has no future-value estimate."""
    targets = build_bellman_targets(
        transition_batch,
        candidate_actions=[0.0, 3.0],
        gamma=0.5,
    )

    np.testing.assert_allclose(targets, [0.5, -1.0])


def test_later_iteration_adds_maximum_future_value(transition_batch):
    """Later targets use the best predicted next-state action value."""
    targets = build_bellman_targets(
        transition_batch,
        candidate_actions=[0.0, 3.0],
        gamma=0.5,
        previous_model=LinearQ(),
    )

    np.testing.assert_allclose(targets, [4.0, 3.0])


@pytest.mark.parametrize("gamma", [-0.1, 1.1])
def test_bellman_targets_reject_invalid_discount(transition_batch, gamma):
    """The discount factor must remain within its mathematical bounds."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        build_bellman_targets(
            transition_batch,
            candidate_actions=[0.0, 1.0],
            gamma=gamma,
        )


@pytest.fixture
def fqi_training_batch():
    return TransitionBatch(
        states=[0.0, 0.0, 1.0, 1.0],
        actions=[0.0, 1.0, 0.0, 1.0],
        next_states=[0.0, 1.0, 0.0, 1.0],
        rewards=[0.0, 1.0, 0.0, 1.0],
    )


def test_fqi_trains_one_ensemble_per_iteration(fqi_training_batch):
    """Each FQI iteration fits a new Extra Trees approximation."""
    learner = FittedQIteration(
        candidate_actions=[0.0, 1.0],
        gamma=0.9,
        n_iterations=3,
        n_estimators=20,
        random_state=42,
    )

    result = learner.fit(fqi_training_batch)

    assert result is learner
    assert len(learner.models_) == 3
    assert isinstance(learner.model_, ExtraTreesRegressor)
    assert learner.model_.n_estimators == 20
    assert learner.model_.max_features == 1.0


def test_fqi_predicts_one_value_per_pair(fqi_training_batch):
    """The fitted model predicts aligned state-action pairs."""
    learner = FittedQIteration(
        candidate_actions=[0.0, 1.0],
        n_iterations=2,
        n_estimators=20,
        random_state=42,
    ).fit(fqi_training_batch)

    predictions = learner.predict(
        states=[0.0, 1.0],
        actions=[0.0, 1.0],
    )

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()


def test_fqi_requires_training_before_prediction():
    """Prediction is unavailable before the FQI loop is fitted."""
    learner = FittedQIteration(candidate_actions=[0.0, 1.0])

    with pytest.raises(NotFittedError, match="fitted before prediction"):
        learner.predict(states=[0.0], actions=[0.0])