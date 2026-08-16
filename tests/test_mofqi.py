import numpy as np
import pytest

from mofqi_reservoir import (
    MultiObjectiveTransitionBatch,
    augment_transitions,
    sample_preference_weights,
    scalarize_rewards,
)


def test_multiobjective_batch_preserves_vector_rewards():
    """Each transition retains one reward value per objective."""
    batch = MultiObjectiveTransitionBatch(
        states=[10.0, 20.0],
        actions=[2.0, 4.0],
        next_states=[12.0, 24.0],
        rewards=[
            [-1.0, -3.0],
            [-2.0, -1.0],
        ],
    )

    assert batch.rewards.shape == (2, 2)
    assert batch.n_samples == 2
    assert batch.n_objectives == 2
    assert batch.state_dim == 1
    assert batch.action_dim == 1


def test_multiobjective_batch_rejects_scalar_rewards():
    """MOFQI requires a reward vector rather than one scalar reward."""
    with pytest.raises(ValueError, match="two-dimensional"):
        MultiObjectiveTransitionBatch(
            states=[10.0, 20.0],
            actions=[2.0, 4.0],
            next_states=[12.0, 24.0],
            rewards=[-1.0, -2.0],
        )


def test_multiobjective_batch_requires_multiple_objectives():
    """A multi-objective batch must contain at least two objectives."""
    with pytest.raises(ValueError, match="at least two objectives"):
        MultiObjectiveTransitionBatch(
            states=[10.0, 20.0],
            actions=[2.0, 4.0],
            next_states=[12.0, 24.0],
            rewards=[[-1.0], [-2.0]],
        )

@pytest.fixture
def two_objective_batch():
    return MultiObjectiveTransitionBatch(
        states=[10.0, 20.0],
        actions=[2.0, 4.0],
        next_states=[12.0, 24.0],
        rewards=[
            [-1.0, -3.0],
            [-2.0, -1.0],
        ],
    )


def test_scalarize_rewards_applies_weighted_sum():
    """Vector rewards are combined using objective preferences."""
    rewards = [
        [-10.0, -2.0],
        [-4.0, -8.0],
    ]

    values = scalarize_rewards(rewards, weights=[0.25, 0.75])

    np.testing.assert_allclose(values, [-4.0, -7.0])


def test_augmentation_reuses_transitions_for_each_weight(
    two_objective_batch,
):
    """Every original transition is repeated for each preference."""
    weights = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ]

    augmented = augment_transitions(two_objective_batch, weights)

    np.testing.assert_allclose(
        augmented.states,
        [
            [10.0, 1.0],
            [10.0, 0.0],
            [10.0, 0.5],
            [20.0, 1.0],
            [20.0, 0.0],
            [20.0, 0.5],
        ],
    )
    np.testing.assert_allclose(
        augmented.next_states,
        [
            [12.0, 1.0],
            [12.0, 0.0],
            [12.0, 0.5],
            [24.0, 1.0],
            [24.0, 0.0],
            [24.0, 0.5],
        ],
    )
    np.testing.assert_allclose(
        augmented.actions.ravel(),
        [2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
    )
    np.testing.assert_allclose(
        augmented.rewards,
        [-1.0, -3.0, -2.0, -2.0, -1.0, -1.5],
    )


def test_sampled_weights_are_reproducible():
    """A fixed seed produces the same simplex samples."""
    first = sample_preference_weights(
        n_weights=5,
        n_objectives=3,
        random_state=42,
    )
    second = sample_preference_weights(
        n_weights=5,
        n_objectives=3,
        random_state=42,
    )

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first.sum(axis=1), 1.0)
    assert np.all(first >= 0.0)


@pytest.mark.parametrize(
    "weights",
    [
        [[0.2, 0.2]],
        [[1.1, -0.1]],
    ],
)
def test_augmentation_rejects_invalid_weights(
    two_objective_batch,
    weights,
):
    """Preference vectors must belong to the unit simplex."""
    with pytest.raises(ValueError):
        augment_transitions(two_objective_batch, weights)