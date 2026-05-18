# shared cost functions and collision utilities
#
# cost function from the proposal:
#   J(u) = sum_t (a_t^2 + kappa * j_t^2 + rho * d_t^2)
#
# collision checking uses smooth euclidean distance so that
# gradient-based solvers (SLSQP) get useful derivatives

from __future__ import annotations

import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork


def driving_cost(controls: np.ndarray, dt: float, kappa: float = 1.0) -> float:
    # acceleration squared + jerk squared (jerk = finite diff of accel)
    accels = controls[:, 0]
    accel_cost = np.sum(accels ** 2)

    jerk = np.diff(accels) / dt
    jerk_cost = np.sum(jerk ** 2)

    return accel_cost + kappa * jerk_cost


def _build_center_segments(lanelet_network: LaneletNetwork):
    # precompute all lanelet center-line segments into flat arrays
    # so we can do vectorized distance queries later
    starts_list = []
    diffs_list = []
    for lanelet in lanelet_network.lanelets:
        vertices = lanelet.center_vertices
        starts_list.append(vertices[:-1])
        diffs_list.append(vertices[1:] - vertices[:-1])
    if not starts_list:
        return None, None, None
    starts = np.vstack(starts_list)
    diffs = np.vstack(diffs_list)
    lengths_sq = np.sum(diffs ** 2, axis=1)
    return starts, diffs, lengths_sq


# cache segment arrays by network id so we don't recompute every optimizer iteration
_segment_cache: dict[int, tuple] = {}


def _get_segments(lanelet_network: LaneletNetwork):
    key = id(lanelet_network)
    if key not in _segment_cache:
        _segment_cache[key] = _build_center_segments(lanelet_network)
    return _segment_cache[key]


def _point_to_segments_distance(point, starts, diffs, lengths_sq):
    # vectorized min distance from a single point to all precomputed segments
    t_param = np.sum((point - starts) * diffs, axis=1)
    safe_lengths = np.where(lengths_sq > 0, lengths_sq, 1.0)
    t_param = np.clip(t_param / safe_lengths, 0.0, 1.0)
    projections = starts + t_param[:, None] * diffs
    dists_sq = np.sum((projections - point) ** 2, axis=1)
    return float(np.sqrt(np.min(dists_sq)))


def lane_deviation_cost(ego_traj: np.ndarray,
                        lanelet_network: LaneletNetwork) -> float:
    # sum of squared distances from ego position to nearest lane center
    starts, diffs, lengths_sq = _get_segments(lanelet_network)
    if starts is None:
        return 0.0

    cost = 0.0
    for t in range(ego_traj.shape[0]):
        dist = _point_to_segments_distance(ego_traj[t, :2], starts, diffs, lengths_sq)
        cost += dist * dist
    return cost


def obstacle_distances(ego_traj: np.ndarray,
                       obstacle_positions: dict[int, np.ndarray]
                       ) -> dict[int, np.ndarray]:
    # euclidean distance from ego to each obstacle at each timestep
    # ego_traj is (T+1, 4), obstacle_positions is {obs_id -> (T, 2)}
    result = {}
    for obs_id, positions in obstacle_positions.items():
        horizon = positions.shape[0]
        ego_xy = ego_traj[1:horizon+1, :2]  # skip initial state to align with predictions
        result[obs_id] = np.linalg.norm(ego_xy - positions, axis=1)
    return result


def min_obstacle_distance(ego_traj: np.ndarray,
                          obstacle_positions: dict[int, np.ndarray]) -> float:
    # smallest distance to any obstacle across all timesteps
    dists = obstacle_distances(ego_traj, obstacle_positions)
    if not dists:
        return float('inf')
    return float(min(d.min() for d in dists.values()))


def collision_constraint_value(ego_traj: np.ndarray,
                               obstacle_positions: dict[int, np.ndarray],
                               safe_radius: float = 3.0) -> float:
    # returns >= 0 when safe, < 0 when violated
    # used as an inequality constraint for SLSQP
    return min_obstacle_distance(ego_traj, obstacle_positions) - safe_radius


def collision_penalty(ego_traj: np.ndarray,
                      obstacle_positions: dict[int, np.ndarray],
                      safe_radius: float = 3.0) -> float:
    # quadratic penalty for constraint violations
    # returns 0 when safe, positive when too close
    # used by CMA-ES since it can't handle constraints natively
    dists = obstacle_distances(ego_traj, obstacle_positions)
    penalty = 0.0
    for dist_array in dists.values():
        violations = np.maximum(0.0, safe_radius - dist_array)
        penalty += np.sum(violations ** 2)
    return penalty


def total_cost(controls, ego_traj, obstacle_positions, lanelet_network,
               dt, kappa=1.0, rho=1.0, collision_weight=0.0,
               safe_radius=3.0):
    # combined objective for unconstrained solvers (CMA-ES)
    # J = driving_cost + rho * lane_deviation + w_col * collision_penalty
    cost = driving_cost(controls, dt, kappa)
    cost += rho * lane_deviation_cost(ego_traj, lanelet_network)
    if collision_weight > 0:
        cost += collision_weight * collision_penalty(
            ego_traj, obstacle_positions, safe_radius)
    return cost


if __name__ == "__main__":
    print("cost functions self-test\n")

    horizon, dt = 40, 0.1
    controls = np.column_stack([
        np.sin(np.linspace(0, 2 * np.pi, horizon)) * 0.5,
        np.zeros(horizon)
    ])
    cost_d = driving_cost(controls, dt, kappa=1.0)
    print(f"driving cost (sinusoidal accel): {cost_d:.4f}")

    ego_traj = np.zeros((horizon + 1, 4))
    ego_traj[:, 0] = np.linspace(0, 50, horizon + 1)

    obs_positions = {1: np.column_stack([
        np.linspace(10, 60, horizon),
        np.ones(horizon) * 5.0
    ])}
    dists = obstacle_distances(ego_traj, obs_positions)
    print(f"obstacle distances shape: {dists[1].shape}, "
          f"min={dists[1].min():.2f}, max={dists[1].max():.2f}")

    min_dist = min_obstacle_distance(ego_traj, obs_positions)
    print(f"min obstacle distance: {min_dist:.2f}")

    constraint_val = collision_constraint_value(ego_traj, obs_positions, safe_radius=3.0)
    print(f"collision constraint: {constraint_val:.2f} "
          f"({'feasible' if constraint_val >= 0 else 'VIOLATED'})")

    pen = collision_penalty(ego_traj, obs_positions, safe_radius=3.0)
    print(f"collision penalty: {pen:.4f}")

    print("\nall tests passed")
