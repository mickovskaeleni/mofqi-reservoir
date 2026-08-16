"""Multi-objective fitted Q-iteration for reservoir operation."""

from mofqi_reservoir.fqi import build_bellman_targets
from mofqi_reservoir.transitions import TransitionBatch

__version__ = "0.1.0"
__all__ = ["TransitionBatch", "build_bellman_targets"]