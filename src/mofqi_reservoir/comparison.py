"""Training comparison between repeated FQI and single-process MOFQI."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mofqi_reservoir.fqi import FittedQIteration
from mofqi_reservoir.mofqi import (
    MultiObjectiveFittedQIteration,
    scalarize_rewards,
    validate_preference_weights,
)
from mofqi_reservoir.transitions import (
    MultiObjectiveTransitionBatch,
    TransitionBatch,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TrainingComparison:
    """Models and training times produced by an FQI–MOFQI comparison."""

    weights: FloatArray
    fqi_learners: tuple[FittedQIteration, ...]
    mofqi_learner: MultiObjectiveFittedQIteration
    fqi_training_seconds: float
    mofqi_training_seconds: float

    @property
    def n_weights(self) -> int:
        """Return the number of compared preference vectors."""
        return self.weights.shape[0]


def train_fqi_mofqi_comparison(
    batch: MultiObjectiveTransitionBatch,
    weights: ArrayLike,
    candidate_actions: ArrayLike,
    gamma: float = 1.0,
    n_iterations: int = 10,
    n_estimators: int = 100,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> TrainingComparison:
    """Train repeated scalar FQI models and one MOFQI model."""
    weight_matrix = validate_preference_weights(
        weights,
        n_objectives=batch.n_objectives,
    )

    fqi_learners = []
    fqi_start = perf_counter()

    for index, weight in enumerate(weight_matrix):
        scalar_batch = TransitionBatch(
            states=batch.states,
            actions=batch.actions,
            next_states=batch.next_states,
            rewards=scalarize_rewards(
                batch.rewards,
                weight,
            ),
        )

        seed = (
            None
            if random_state is None
            else random_state + index
        )

        learner = FittedQIteration(
            candidate_actions=candidate_actions,
            gamma=gamma,
            n_iterations=n_iterations,
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=n_jobs,
        ).fit(scalar_batch)

        fqi_learners.append(learner)

    fqi_training_seconds = perf_counter() - fqi_start

    mofqi_start = perf_counter()

    mofqi_learner = MultiObjectiveFittedQIteration(
        candidate_actions=candidate_actions,
        gamma=gamma,
        n_iterations=n_iterations,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    ).fit(
        batch,
        weights=weight_matrix,
    )

    mofqi_training_seconds = perf_counter() - mofqi_start

    return TrainingComparison(
        weights=weight_matrix,
        fqi_learners=tuple(fqi_learners),
        mofqi_learner=mofqi_learner,
        fqi_training_seconds=fqi_training_seconds,
        mofqi_training_seconds=mofqi_training_seconds,
    )