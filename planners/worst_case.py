"""
Worst-Case trajectory planner.

For each obstacle at each timestep, selects the prediction mode that places
the obstacle closest to the ego vehicle (dynamic per-timestep minimax).
This creates the most conservative baseline — the ego must avoid the
worst-case future at every instant.

Two backends: SQP (scipy SLSQP) and CMA-ES.
"""

from __future__ import annotations

import time
import numpy as np
import scipy.optimize
import cma

from planners.vehicle_dynamics import (
    VehicleState, VehicleParams, rollout, states_to_array,
)
from planners.cost_functions import (
    driving_cost, lane_deviation_cost, collision_constraint_value,
    collision_penalty, min_obstacle_distance, total_cost,
)
from planners.planner_result import PlannerResult
from planners.dataloader import ScenarioData


DEFAULT_PARAMS = VehicleParams()
SAFE_RADIUS = 3.0
KAPPA = 1.0
RHO = 1.0
COLLISION_WEIGHT = 1000.0


def select_worst_case(ego_traj: np.ndarray,
                      obstacle_predictions: dict[int, np.ndarray]
                      ) -> dict[int, np.ndarray]:
    """
    Dynamic per-timestep worst-case: for each obstacle at each t, pick the
    prediction mode whose position is closest to the ego.

    Parameters
    ----------
    ego_traj             : (T+1, 4) ego state array
    obstacle_predictions : {obs_id → (K, T, 2)}

    Returns
    -------
    {obs_id → (T, 2)} worst-case positions
    """
    worst = {}
    for obs_id, preds in obstacle_predictions.items():
        K, T, _ = preds.shape
        worst_pos = np.zeros((T, 2))
        for t in range(T):
            ego_xy = ego_traj[t + 1, :2]
            dists = np.linalg.norm(preds[:, t, :] - ego_xy, axis=1)
            worst_pos[t] = preds[np.argmin(dists), t]
        worst[obs_id] = worst_pos
    return worst


def _build_bounds(T: int, params: VehicleParams) -> list[tuple[float, float]]:
    bounds = []
    for _ in range(T):
        bounds.append((params.min_accel, params.max_accel))
        bounds.append((-params.max_steer, params.max_steer))
    return bounds


def _unpack_controls(u_flat: np.ndarray, T: int) -> np.ndarray:
    return u_flat.reshape(T, 2)


def _rollout_from_flat(u_flat: np.ndarray, T: int,
                       ego_init: VehicleState, dt: float,
                       params: VehicleParams) -> np.ndarray:
    controls = _unpack_controls(u_flat, T)
    states = rollout(ego_init, controls, dt=dt, params=params)
    return states_to_array(states)


# -----------------------------------------------------------------------
# SQP backend
# -----------------------------------------------------------------------

def worst_case_sqp(scenario: ScenarioData,
                   kappa: float = KAPPA,
                   rho: float = RHO,
                   safe_radius: float = SAFE_RADIUS,
                   maxiter: int = 200) -> PlannerResult:
    T = scenario.T
    dt = scenario.dt
    ego_init = scenario.ego_initial_state
    params = DEFAULT_PARAMS
    obs_preds = scenario.obstacle_predictions

    cache = {"key": None, "traj": None, "wc": None}

    def _get_cached(u_flat):
        key = u_flat.tobytes()
        if cache["key"] != key:
            traj = _rollout_from_flat(u_flat, T, ego_init, dt, params)
            cache["key"] = key
            cache["traj"] = traj
            cache["wc"] = select_worst_case(traj, obs_preds)
        return cache["traj"], cache["wc"]

    def objective(u_flat):
        controls = _unpack_controls(u_flat, T)
        traj, _ = _get_cached(u_flat)
        j = driving_cost(controls, dt, kappa)
        j += rho * lane_deviation_cost(traj, scenario.lanelet_network)
        return j

    def collision_constr(u_flat):
        traj, wc = _get_cached(u_flat)
        return collision_constraint_value(traj, wc, safe_radius)

    bounds = _build_bounds(T, params)
    u0 = np.zeros(2 * T)

    constraints = [{"type": "ineq", "fun": collision_constr}]

    t0 = time.time()
    result = scipy.optimize.minimize(
        objective, u0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": 1e-6, "disp": False},
    )
    elapsed = time.time() - t0

    controls = _unpack_controls(result.x, T)
    traj = _rollout_from_flat(result.x, T, ego_init, dt, params)
    wc = select_worst_case(traj, obs_preds)
    min_dist = min_obstacle_distance(traj, wc)

    return PlannerResult(
        controls=controls,
        ego_trajectory=traj,
        cost=float(result.fun),
        min_obstacle_distance=min_dist,
        collision_free=(min_dist >= safe_radius - 0.01),
        runtime_seconds=elapsed,
        converged=result.success,
        method="worst_case_sqp",
    )


# -----------------------------------------------------------------------
# CMA-ES backend
# -----------------------------------------------------------------------

def worst_case_cmaes(scenario: ScenarioData,
                     kappa: float = KAPPA,
                     rho: float = RHO,
                     safe_radius: float = SAFE_RADIUS,
                     collision_weight: float = COLLISION_WEIGHT,
                     maxfevals: int = 5000,
                     sigma0: float = 0.5) -> PlannerResult:
    T = scenario.T
    dt = scenario.dt
    ego_init = scenario.ego_initial_state
    params = DEFAULT_PARAMS
    obs_preds = scenario.obstacle_predictions

    lower = np.tile([params.min_accel, -params.max_steer], T)
    upper = np.tile([params.max_accel, params.max_steer], T)

    def objective(u_flat):
        u_flat = np.asarray(u_flat)
        controls = _unpack_controls(u_flat, T)
        traj = _rollout_from_flat(u_flat, T, ego_init, dt, params)
        wc = select_worst_case(traj, obs_preds)
        return total_cost(
            controls, traj, wc, scenario.lanelet_network,
            dt, kappa, rho, collision_weight, safe_radius,
        )

    u0 = np.zeros(2 * T)

    opts = {
        "maxfevals": maxfevals,
        "bounds": [lower.tolist(), upper.tolist()],
        "verbose": -9,
        "seed": 42,
    }

    t0 = time.time()
    es = cma.CMAEvolutionStrategy(u0, sigma0, opts)
    es.optimize(objective)
    elapsed = time.time() - t0

    best = np.asarray(es.result.xbest)
    controls = _unpack_controls(best, T)
    traj = _rollout_from_flat(best, T, ego_init, dt, params)
    wc = select_worst_case(traj, obs_preds)
    min_dist = min_obstacle_distance(traj, wc)

    cost_val = driving_cost(controls, dt, kappa)
    cost_val += rho * lane_deviation_cost(traj, scenario.lanelet_network)

    return PlannerResult(
        controls=controls,
        ego_trajectory=traj,
        cost=cost_val,
        min_obstacle_distance=min_dist,
        collision_free=(min_dist >= safe_radius - 0.01),
        runtime_seconds=elapsed,
        converged=not es.result.xbest is None,
        method="worst_case_cmaes",
    )


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from planners.dataloader import load_scenario

    xml_path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/USA_Lanker-1_1_T-1.xml"
    scenario = load_scenario(xml_path)
    print(f"Scenario: {scenario.scenario_id}  "
          f"(T={scenario.T}, obstacles={scenario.num_obstacles})\n")

    print("--- Worst-Case SQP ---")
    r = worst_case_sqp(scenario)
    print(f"  Cost:         {r.cost:.4f}")
    print(f"  Min dist:     {r.min_obstacle_distance:.2f} m")
    print(f"  Collision-free: {r.collision_free}")
    print(f"  Converged:    {r.converged}")
    print(f"  Runtime:      {r.runtime_seconds:.2f} s\n")

    print("--- Worst-Case CMA-ES ---")
    r = worst_case_cmaes(scenario)
    print(f"  Cost:         {r.cost:.4f}")
    print(f"  Min dist:     {r.min_obstacle_distance:.2f} m")
    print(f"  Collision-free: {r.collision_free}")
    print(f"  Converged:    {r.converged}")
    print(f"  Runtime:      {r.runtime_seconds:.2f} s")
