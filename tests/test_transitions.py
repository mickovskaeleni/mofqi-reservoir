import numpy as np
import pytest

from mofqi_reservoir import TransitionBatch


def test_transition_batch_normalizes_scalar_features():
    """One-dimensional state and action inputs become column matrices."""
    batch = TransitionBatch(
        states=[0.0, 1.0],
        actions=[0.0, 1.0],
        next_states=[1.0, 2.0],
        rewards=[1.0, 2.0],
    )

    assert batch.states.shape == (2, 1)
    assert batch.actions.shape == (2, 1)
    assert batch.next_states.shape == (2, 1)
    assert batch.rewards.shape == (2,)
    assert batch.n_samples == 2
    assert batch.state_dim == 1
    assert batch.action_dim == 1


def test_transition_batch_rejects_inconsistent_lengths():
    """Every component must describe the same number of transitions."""
    with pytest.raises(ValueError, match="same number of samples"):
        TransitionBatch(
            states=[0.0, 1.0],
            actions=[0.0],
            next_states=[1.0, 2.0],
            rewards=[1.0, 2.0],
        )


def test_transition_batch_rejects_nonfinite_values():
    """Training data cannot contain NaN or infinite values."""
    with pytest.raises(ValueError, match="finite"):
        TransitionBatch(
            states=[0.0, np.nan],
            actions=[0.0, 1.0],
            next_states=[1.0, 2.0],
            rewards=[1.0, 2.0],
        )