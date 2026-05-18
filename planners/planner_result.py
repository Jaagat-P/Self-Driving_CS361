# shared result container for all planner variants
# each planner returns one of these after optimization

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PlannerResult:
    controls: np.ndarray          # (T, 2) optimal [accel, steer] per step
    ego_trajectory: np.ndarray    # (T+1, 4) rolled-out [x, y, heading, speed]
    cost: float                   # final objective value
    min_obstacle_distance: float  # closest we ever got to any obstacle
    collision_free: bool          # did we stay outside the safe radius?
    runtime_seconds: float        # wall clock time for optimization
    converged: bool               # did the optimizer converge?
    method: str                   # e.g. "plan_on_mean_sqp", "worst_case_cmaes"
