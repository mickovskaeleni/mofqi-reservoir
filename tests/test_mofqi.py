import pytest

from mofqi_reservoir import MultiObjectiveTransitionBatch


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