"""Multi-objective fitted Q-iteration for reservoir operation."""

from mofqi_reservoir.fqi import FittedQIteration, build_bellman_targets
from mofqi_reservoir.transitions import (
    MultiObjectiveTransitionBatch,
    TransitionBatch,
)
from mofqi_reservoir.mofqi import (
    augment_transitions,
    sample_preference_weights,
    scalarize_rewards,
    validate_preference_weights,
)

__version__ = "0.1.0"
__all__ = [
    "FittedQIteration",
    "MultiObjectiveTransitionBatch",
    "TransitionBatch",
    "build_bellman_targets",
    "augment_transitions",
    "sample_preference_weights",
    "scalarize_rewards",
    "validate_preference_weights",
]