from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PlannerResult:
    controls: np.ndarray
    ego_trajectory: np.ndarray
    cost: float
    min_obstacle_distance: float
    collision_free: bool
    runtime_seconds: float
    converged: bool
    method: str
