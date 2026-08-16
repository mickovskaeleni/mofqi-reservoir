import numpy as np
import pytest

from mofqi_reservoir import (
    SyntheticReservoir,
    sample_reservoir_transitions,
)


def test_reservoir_calculates_mass_balance_and_rewards():
    """A feasible release updates storage and both objectives."""
    environment = SyntheticReservoir(
        initial_storage=80.0,
        inflow_mean=20.0,
        inflow_std=0.0,
        episode_length=2,
    )

    initial_state = environment.reset()
    next_state, rewards, terminated = environment.step([30.0])

    np.testing.assert_allclose(initial_state, [80.0])
    np.testing.assert_allclose(next_state, [70.0])
    np.testing.assert_allclose(rewards, [-20.0, -20.0])

    assert environment.last_inflow_ == 20.0
    assert environment.last_release_ == 30.0
    assert not terminated


@pytest.mark.parametrize(
    (
        "initial_storage",
        "release_decision",
        "expected_release",
        "expected_storage",
    ),
    [
        (120.0, 0.0, 20.0, 100.0),
        (30.0, 100.0, 30.0, 0.0),
    ],
)
def test_reservoir_enforces_feasible_release(
    initial_storage,
    release_decision,
    expected_release,
    expected_storage,
):
    """Release remains between storage-dependent limits."""
    environment = SyntheticReservoir(
        initial_storage=initial_storage,
        inflow_mean=0.0,
        inflow_std=0.0,
    )

    next_state, _, _ = environment.step([release_decision])

    assert environment.last_release_ == expected_release
    np.testing.assert_allclose(next_state, [expected_storage])


def test_reservoir_terminates_at_episode_length():
    """The environment stops after the configured horizon."""
    environment = SyntheticReservoir(
        initial_storage=50.0,
        inflow_mean=0.0,
        inflow_std=0.0,
        episode_length=2,
    )

    _, _, first_terminated = environment.step([0.0])
    _, _, second_terminated = environment.step([0.0])

    assert not first_terminated
    assert second_terminated

    with pytest.raises(RuntimeError, match="episode has terminated"):
        environment.step([0.0])


def test_reservoir_inflows_are_reproducible():
    """Identical seeds produce identical stochastic transitions."""
    first = SyntheticReservoir(random_state=42)
    second = SyntheticReservoir(random_state=42)

    first_transition = first.step([20.0])
    second_transition = second.step([20.0])

    np.testing.assert_allclose(
        first_transition[0],
        second_transition[0],
    )
    np.testing.assert_allclose(
        first_transition[1],
        second_transition[1],
    )
    assert first.last_inflow_ == second.last_inflow_

def test_sample_reservoir_transitions_returns_vector_rewards():
    """Offline reservoir samples contain both objectives."""
    batch = sample_reservoir_transitions(
        n_samples=50,
        random_state=42,
    )

    assert batch.n_samples == 50
    assert batch.state_dim == 1
    assert batch.action_dim == 1
    assert batch.n_objectives == 2
    assert batch.rewards.shape == (50, 2)
    assert np.isfinite(batch.states).all()
    assert np.isfinite(batch.next_states).all()
    assert np.all(batch.rewards <= 0.0)


def test_sample_reservoir_transitions_is_reproducible():
    """A fixed seed reproduces the complete offline dataset."""
    first = sample_reservoir_transitions(
        n_samples=20,
        random_state=42,
    )
    second = sample_reservoir_transitions(
        n_samples=20,
        random_state=42,
    )

    np.testing.assert_allclose(first.states, second.states)
    np.testing.assert_allclose(first.actions, second.actions)
    np.testing.assert_allclose(
        first.next_states,
        second.next_states,
    )
    np.testing.assert_allclose(first.rewards, second.rewards)