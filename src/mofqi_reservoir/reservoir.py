"""Synthetic two-objective reservoir environment."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class SyntheticReservoir:
    """Stationary reservoir environment with flood and irrigation objectives."""

    def __init__(
        self,
        initial_storage: float = 50.0,
        inflow_mean: float = 40.0,
        inflow_std: float = 10.0,
        capacity: float = 100.0,
        flood_threshold: float = 50.0,
        irrigation_demand: float = 50.0,
        surface_area: float = 1.0,
        episode_length: int = 10,
        random_state: int | None = None,
    ) -> None:
        parameters = {
            "initial_storage": initial_storage,
            "inflow_mean": inflow_mean,
            "inflow_std": inflow_std,
            "capacity": capacity,
            "flood_threshold": flood_threshold,
            "irrigation_demand": irrigation_demand,
            "surface_area": surface_area,
        }

        for name, value in parameters.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")

        if initial_storage < 0.0:
            raise ValueError("initial_storage must be nonnegative.")

        if inflow_std < 0.0:
            raise ValueError("inflow_std must be nonnegative.")

        if capacity <= 0.0:
            raise ValueError("capacity must be positive.")

        if flood_threshold < 0.0:
            raise ValueError("flood_threshold must be nonnegative.")

        if irrigation_demand < 0.0:
            raise ValueError("irrigation_demand must be nonnegative.")

        if surface_area <= 0.0:
            raise ValueError("surface_area must be positive.")

        if episode_length < 1:
            raise ValueError("episode_length must be at least 1.")

        self.initial_storage = float(initial_storage)
        self.inflow_mean = float(inflow_mean)
        self.inflow_std = float(inflow_std)
        self.capacity = float(capacity)
        self.flood_threshold = float(flood_threshold)
        self.irrigation_demand = float(irrigation_demand)
        self.surface_area = float(surface_area)
        self.episode_length = episode_length

        self._rng = np.random.default_rng(random_state)
        self.reset()

    def reset(self, seed: int | None = None) -> FloatArray:
        """Reset the reservoir and return its initial storage."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.storage = self.initial_storage
        self.step_count = 0
        self.terminated = False
        self.last_inflow_: float | None = None
        self.last_release_: float | None = None

        return np.array([self.storage], dtype=float)

    def step(
        self,
        action: ArrayLike,
    ) -> tuple[FloatArray, FloatArray, bool]:
        """Apply a release decision and advance the reservoir one step."""
        if self.terminated:
            raise RuntimeError(
                "The episode has terminated; call reset before stepping again."
            )

        action_vector = np.asarray(action, dtype=float)

        if action_vector.ndim == 0:
            action_vector = action_vector.reshape(1)

        if action_vector.shape != (1,):
            raise ValueError("action must contain exactly one release decision.")

        if not np.isfinite(action_vector).all():
            raise ValueError("action must contain only finite values.")

        release_decision = float(action_vector[0])

        minimum_release = max(self.storage - self.capacity, 0.0)
        maximum_release = max(self.storage, 0.0)
        actual_release = max(
            minimum_release,
            min(release_decision, maximum_release),
        )

        inflow = float(
            self._rng.normal(
                loc=self.inflow_mean,
                scale=self.inflow_std,
            )
        )

        next_storage = self.storage + inflow - actual_release
        next_level = next_storage / self.surface_area

        flood_reward = -max(
            next_level - self.flood_threshold,
            0.0,
        )
        irrigation_reward = -max(
            self.irrigation_demand - actual_release,
            0.0,
        )

        self.storage = float(next_storage)
        self.last_inflow_ = inflow
        self.last_release_ = actual_release
        self.step_count += 1
        self.terminated = self.step_count >= self.episode_length

        state = np.array([self.storage], dtype=float)
        rewards = np.array(
            [flood_reward, irrigation_reward],
            dtype=float,
        )

        return state, rewards, self.terminated