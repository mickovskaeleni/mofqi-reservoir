"""Multi-objective fitted Q-iteration for reservoir operation."""

from mofqi_reservoir.fqi import (
    FittedQIteration,
    build_bellman_targets,
)
from mofqi_reservoir.mofqi import (
    MultiObjectiveFittedQIteration,
    augment_transitions,
    sample_preference_weights,
    scalarize_rewards,
    validate_preference_weights,
)
from mofqi_reservoir.transitions import (
    MultiObjectiveTransitionBatch,
    TransitionBatch,
)
from mofqi_reservoir.evaluation import (
    PolicyEvaluation,
    evaluate_fqi_policy,
    evaluate_mofqi_policy,
    evaluate_policy,
)
from mofqi_reservoir.reservoir import SyntheticReservoir

__version__ = "0.1.0"

__all__ = [
    "FittedQIteration",
    "MultiObjectiveFittedQIteration",
    "MultiObjectiveTransitionBatch",
    "TransitionBatch",
    "augment_transitions",
    "build_bellman_targets",
    "sample_preference_weights",
    "scalarize_rewards",
    "validate_preference_weights",
    "PolicyEvaluation",
    "evaluate_policy",
    "evaluate_fqi_policy",
    "evaluate_mofqi_policy",
    "SyntheticReservoir",
]