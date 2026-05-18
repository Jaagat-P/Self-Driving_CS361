# worst-case (minimax) trajectory planner
#
# for each obstacle at each timestep, picks the prediction mode that puts
# the obstacle closest to ego (dynamic per-timestep minimax selection)
# this is the most conservative baseline -- ego has to dodge the scariest
# possible future at every instant
#
# two solver backends: SQP (scipy SLSQP) and CMA-ES

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


def select_worst_case(ego_traj, obstacle_predictions):
    # for each obstacle at each timestep, find which of the K modes
    # puts the obstacle closest to where ego is -- that's the worst case
    # note: this depends on the current ego trajectory, so it gets
    # recomputed every time the optimizer tries new controls
    worst_positions = {}
    for obs_id, preds in obstacle_predictions.items():
        num_modes, horizon, _ = preds.shape
        worst = np.zeros((horizon, 2))
        for t in range(horizon):
            ego_xy = ego_traj[t + 1, :2]
            dists = np.linalg.norm(preds[:, t, :] - ego_xy, axis=1)
            worst[t] = preds[np.argmin(dists), t]  # pick the closest mode
        worst_positions[obs_id] = worst
    return worst_positions


def _build_bounds(horizon, params):
    bounds = []
    for _ in range(horizon):
        bounds.append((params.min_accel, params.max_accel))
        bounds.append((-params.max_steer, params.max_steer))
    return bounds


def _unpack_controls(controls_flat, horizon):
    return controls_flat.reshape(horizon, 2)


def _rollout_from_flat(controls_flat, horizon, ego_init, dt, params):
    controls = _unpack_controls(controls_flat, horizon)
    states = rollout(ego_init, controls, dt=dt, params=params)
    return states_to_array(states)


# SQP backend
def worst_case_sqp(scenario: ScenarioData,
                   kappa=KAPPA, rho=RHO,
                   safe_radius=SAFE_RADIUS,
                   maxiter=200) -> PlannerResult:

    horizon = scenario.T
    dt = scenario.dt
    ego_init = scenario.ego_initial_state
    params = DEFAULT_PARAMS
    all_predictions = scenario.obstacle_predictions

    # cache to avoid redundant rollout + worst-case selection
    # SQP calls objective and constraint with the same controls_flat,
    # so we cache by the raw bytes of the input
    cache = {"key": None, "traj": None, "worst_obs": None}

    def _get_cached(controls_flat):
        key = controls_flat.tobytes()
        if cache["key"] != key:
            traj = _rollout_from_flat(controls_flat, horizon, ego_init, dt, params)
            cache["key"] = key
            cache["traj"] = traj
            cache["worst_obs"] = select_worst_case(traj, all_predictions)
        return cache["traj"], cache["worst_obs"]

    def objective(controls_flat):
        controls = _unpack_controls(controls_flat, horizon)
        traj, _ = _get_cached(controls_flat)
        cost = driving_cost(controls, dt, kappa)
        cost += rho * lane_deviation_cost(traj, scenario.lanelet_network)
        return cost

    def collision_constr(controls_flat):
        traj, worst_obs = _get_cached(controls_flat)
        return collision_constraint_value(traj, worst_obs, safe_radius)

    bounds = _build_bounds(horizon, params)
    initial_controls = np.zeros(2 * horizon)

    start_time = time.time()
    result = scipy.optimize.minimize(
        objective, initial_controls,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": collision_constr}],
        options={"maxiter": maxiter, "ftol": 1e-6, "disp": False},
    )
    elapsed = time.time() - start_time

    controls = _unpack_controls(result.x, horizon)
    traj = _rollout_from_flat(result.x, horizon, ego_init, dt, params)
    worst_obs = select_worst_case(traj, all_predictions)
    min_dist = min_obstacle_distance(traj, worst_obs)

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


# CMA-ES backend
def worst_case_cmaes(scenario: ScenarioData,
                     kappa=KAPPA, rho=RHO,
                     safe_radius=SAFE_RADIUS,
                     collision_weight=COLLISION_WEIGHT,
                     maxfevals=5000, sigma0=0.5) -> PlannerResult:

    horizon = scenario.T
    dt = scenario.dt
    ego_init = scenario.ego_initial_state
    params = DEFAULT_PARAMS
    all_predictions = scenario.obstacle_predictions

    lower = np.tile([params.min_accel, -params.max_steer], horizon)
    upper = np.tile([params.max_accel, params.max_steer], horizon)

    def objective(controls_flat):
        controls_flat = np.asarray(controls_flat)
        controls = _unpack_controls(controls_flat, horizon)
        traj = _rollout_from_flat(controls_flat, horizon, ego_init, dt, params)
        # worst-case obstacles change with every new ego trajectory
        worst_obs = select_worst_case(traj, all_predictions)
        return total_cost(
            controls, traj, worst_obs, scenario.lanelet_network,
            dt, kappa, rho, collision_weight, safe_radius,
        )

    initial_controls = np.zeros(2 * horizon)

    start_time = time.time()
    es = cma.CMAEvolutionStrategy(initial_controls, sigma0, {
        "maxfevals": maxfevals,
        "bounds": [lower.tolist(), upper.tolist()],
        "verbose": -9,
        "seed": 42,
    })
    es.optimize(objective)
    elapsed = time.time() - start_time

    best_controls_flat = np.asarray(es.result.xbest)
    controls = _unpack_controls(best_controls_flat, horizon)
    traj = _rollout_from_flat(best_controls_flat, horizon, ego_init, dt, params)
    worst_obs = select_worst_case(traj, all_predictions)
    min_dist = min_obstacle_distance(traj, worst_obs)

    # cost without the collision penalty for fair comparison
    final_cost = driving_cost(controls, dt, kappa)
    final_cost += rho * lane_deviation_cost(traj, scenario.lanelet_network)

    return PlannerResult(
        controls=controls,
        ego_trajectory=traj,
        cost=final_cost,
        min_obstacle_distance=min_dist,
        collision_free=(min_dist >= safe_radius - 0.01),
        runtime_seconds=elapsed,
        converged=es.result.xbest is not None,
        method="worst_case_cmaes",
    )


if __name__ == "__main__":
    import sys
    from planners.dataloader import load_scenario

    xml_path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/USA_Lanker-1_1_T-1.xml"
    scenario = load_scenario(xml_path)
    print(f"scenario: {scenario.scenario_id}  "
          f"(T={scenario.T}, obstacles={scenario.num_obstacles})\n")

    print("worst-case SQP")
    result = worst_case_sqp(scenario)
    print(f"  cost:           {result.cost:.4f}")
    print(f"  min dist:       {result.min_obstacle_distance:.2f} m")
    print(f"  collision free: {result.collision_free}")
    print(f"  converged:      {result.converged}")
    print(f"  runtime:        {result.runtime_seconds:.2f} s\n")

    print("worst-case CMA-ES")
    result = worst_case_cmaes(scenario)
    print(f"  cost:           {result.cost:.4f}")
    print(f"  min dist:       {result.min_obstacle_distance:.2f} m")
    print(f"  collision free: {result.collision_free}")
    print(f"  converged:      {result.converged}")
    print(f"  runtime:        {result.runtime_seconds:.2f} s")
