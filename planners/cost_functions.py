"""
Shared cost functions and collision utilities for trajectory optimization.

Cost function (per proposal):
    J(u) = Σ_t (a_t² + κ·j_t² + ρ·d_t²)

where a_t is acceleration, j_t is jerk (finite diff of accel), and d_t is
lane deviation from the nearest lanelet center.

Collision checking uses smooth Euclidean distance so that gradient-based
solvers (SLSQP) get useful derivative information.
"""

from __future__ import annotations

import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork


def driving_cost(controls: np.ndarray, dt: float,
                 kappa: float = 1.0) -> float:
    """
    Acceleration + jerk cost from the control sequence.

    Parameters
    ----------
    controls : (T, 2) array — [accel, steer] per step
    dt       : time step [s]
    kappa    : weight on jerk term

    Returns
    -------
    float — Σ_t a_t² + κ Σ_t j_t²
    """
    accels = controls[:, 0]
    accel_cost = np.sum(accels ** 2)

    jerk = np.diff(accels) / dt
    jerk_cost = np.sum(jerk ** 2)

    return accel_cost + kappa * jerk_cost


def _build_center_segments(lanelet_network: LaneletNetwork):
    """
    Precompute all lanelet center-line segments into flat arrays for fast
    vectorized distance queries.  Returns (starts, diffs, lengths_sq).
    """
    starts_list = []
    diffs_list = []
    for lanelet in lanelet_network.lanelets:
        cv = lanelet.center_vertices
        starts_list.append(cv[:-1])
        diffs_list.append(cv[1:] - cv[:-1])
    if not starts_list:
        return None, None, None
    starts = np.vstack(starts_list)
    diffs = np.vstack(diffs_list)
    lengths_sq = np.sum(diffs ** 2, axis=1)
    return starts, diffs, lengths_sq


_segment_cache: dict[int, tuple] = {}


def _get_segments(lanelet_network: LaneletNetwork):
    """Cache segment arrays by network id to avoid recomputation."""
    key = id(lanelet_network)
    if key not in _segment_cache:
        _segment_cache[key] = _build_center_segments(lanelet_network)
    return _segment_cache[key]


def _point_to_segments_distance(point: np.ndarray,
                                starts: np.ndarray,
                                diffs: np.ndarray,
                                lengths_sq: np.ndarray) -> float:
    """Min distance from a point to any precomputed segment (vectorized)."""
    t = np.sum((point - starts) * diffs, axis=1)
    safe = np.where(lengths_sq > 0, lengths_sq, 1.0)
    t = np.clip(t / safe, 0.0, 1.0)
    proj = starts + t[:, None] * diffs
    dists_sq = np.sum((proj - point) ** 2, axis=1)
    return float(np.sqrt(np.min(dists_sq)))


def lane_deviation_cost(ego_traj: np.ndarray,
                        lanelet_network: LaneletNetwork) -> float:
    """
    Σ_t d_t² where d_t is the distance from ego position to the nearest
    lanelet center polyline.

    Parameters
    ----------
    ego_traj        : (T+1, 4) array — [x, y, psi, v] per step
    lanelet_network : CommonRoad LaneletNetwork

    Returns
    -------
    float
    """
    starts, diffs, lengths_sq = _get_segments(lanelet_network)
    if starts is None:
        return 0.0

    cost = 0.0
    for t in range(ego_traj.shape[0]):
        d = _point_to_segments_distance(ego_traj[t, :2], starts, diffs, lengths_sq)
        cost += d * d
    return cost


def obstacle_distances(ego_traj: np.ndarray,
                       obstacle_positions: dict[int, np.ndarray]
                       ) -> dict[int, np.ndarray]:
    """
    Euclidean distance from ego to each obstacle at each timestep.

    Parameters
    ----------
    ego_traj           : (T+1, 4) array — ego states
    obstacle_positions : {obs_id → (T, 2)} collapsed positions

    Returns
    -------
    {obs_id → (T,)} distance array
    """
    result = {}
    for obs_id, pos in obstacle_positions.items():
        T = pos.shape[0]
        ego_xy = ego_traj[1:T+1, :2]
        result[obs_id] = np.linalg.norm(ego_xy - pos, axis=1)
    return result


def min_obstacle_distance(ego_traj: np.ndarray,
                          obstacle_positions: dict[int, np.ndarray]
                          ) -> float:
    """Minimum distance to any obstacle across all timesteps."""
    dists = obstacle_distances(ego_traj, obstacle_positions)
    if not dists:
        return float('inf')
    return float(min(d.min() for d in dists.values()))


def collision_constraint_value(ego_traj: np.ndarray,
                               obstacle_positions: dict[int, np.ndarray],
                               safe_radius: float = 3.0) -> float:
    """
    Smooth collision constraint: min_distance - safe_radius.

    Returns ≥ 0 when feasible (no collision), < 0 when violated.
    """
    return min_obstacle_distance(ego_traj, obstacle_positions) - safe_radius


def collision_penalty(ego_traj: np.ndarray,
                      obstacle_positions: dict[int, np.ndarray],
                      safe_radius: float = 3.0) -> float:
    """
    Quadratic penalty for collision constraint violation.
    Returns 0 when feasible, positive when violated.
    Used by CMA-ES (unconstrained solver).
    """
    dists = obstacle_distances(ego_traj, obstacle_positions)
    penalty = 0.0
    for d in dists.values():
        violations = np.maximum(0.0, safe_radius - d)
        penalty += np.sum(violations ** 2)
    return penalty


def total_cost(controls: np.ndarray,
               ego_traj: np.ndarray,
               obstacle_positions: dict[int, np.ndarray],
               lanelet_network: LaneletNetwork,
               dt: float,
               kappa: float = 1.0,
               rho: float = 1.0,
               collision_weight: float = 0.0,
               safe_radius: float = 3.0) -> float:
    """
    Combined objective for unconstrained solvers (CMA-ES).

    J = driving_cost + ρ·lane_deviation + w_col·collision_penalty
    """
    j = driving_cost(controls, dt, kappa)
    j += rho * lane_deviation_cost(ego_traj, lanelet_network)
    if collision_weight > 0:
        j += collision_weight * collision_penalty(
            ego_traj, obstacle_positions, safe_radius)
    return j


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Cost Functions – Self-Test ===\n")

    T, dt = 40, 0.1
    controls = np.column_stack([
        np.sin(np.linspace(0, 2*np.pi, T)) * 0.5,
        np.zeros(T)
    ])
    cost_d = driving_cost(controls, dt, kappa=1.0)
    print(f"Driving cost (sinusoidal accel): {cost_d:.4f}")

    ego_traj = np.zeros((T+1, 4))
    ego_traj[:, 0] = np.linspace(0, 50, T+1)

    obs_pos = {1: np.column_stack([
        np.linspace(10, 60, T),
        np.ones(T) * 5.0
    ])}
    dists = obstacle_distances(ego_traj, obs_pos)
    print(f"Obstacle distances shape: {dists[1].shape}, "
          f"min={dists[1].min():.2f}, max={dists[1].max():.2f}")

    min_d = min_obstacle_distance(ego_traj, obs_pos)
    print(f"Min obstacle distance: {min_d:.2f}")

    cv = collision_constraint_value(ego_traj, obs_pos, safe_radius=3.0)
    print(f"Collision constraint value: {cv:.2f} "
          f"({'feasible' if cv >= 0 else 'VIOLATED'})")

    cp = collision_penalty(ego_traj, obs_pos, safe_radius=3.0)
    print(f"Collision penalty: {cp:.4f}")

    print("\nAll cost function tests passed.")
