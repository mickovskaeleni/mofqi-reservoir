import numpy as np

from mofqi_reservoir import (
    MultiObjectiveFittedQIteration,
    MultiObjectiveTransitionBatch,
    TrainingComparison,
    train_fqi_mofqi_comparison,
)


def test_comparison_trains_repeated_fqi_and_single_mofqi():
    """One FQI model is trained per weight but only one MOFQI model."""
    batch = MultiObjectiveTransitionBatch(
        states=[0.0, 0.0, 1.0, 1.0],
        actions=[0.0, 1.0, 0.0, 1.0],
        next_states=[0.0, 1.0, 0.0, 1.0],
        rewards=[
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
    )
    weights = [
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 1.0],
    ]

    result = train_fqi_mofqi_comparison(
        batch=batch,
        weights=weights,
        candidate_actions=[0.0, 1.0],
        gamma=0.0,
        n_iterations=1,
        n_estimators=5,
        random_state=42,
    )

    assert isinstance(result, TrainingComparison)
    assert result.n_weights == 3
    assert len(result.fqi_learners) == 3
    assert isinstance(
        result.mofqi_learner,
        MultiObjectiveFittedQIteration,
    )

    assert all(
        len(learner.models_) == 1
        for learner in result.fqi_learners
    )
    assert len(result.mofqi_learner.models_) == 1
    assert result.mofqi_learner.augmented_batch_.n_samples == 12

    np.testing.assert_allclose(result.weights, weights)
    assert result.fqi_training_seconds >= 0.0
    assert result.mofqi_training_seconds >= 0.0