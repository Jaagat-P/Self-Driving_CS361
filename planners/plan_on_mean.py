# plan-on-mean trajectory planner
#
# collapses K obstacle prediction modes into one mean trajectory per obstacle,
# then optimizes the ego controls against that single deterministic prediction
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


def _collapse_mean(obstacle_predictions):
    # average K prediction modes into one (T, 2) trajectory per obstacle
    return {
        obs_id: preds.mean(axis=0)
        for obs_id, preds in obstacle_predictions.items()
    }


def _build_bounds(horizon, params):
    # box bounds for flattened decision variable [a0, d0, a1, d1, ...]
    bounds = []
    for _ in range(horizon):
        bounds.append((params.min_accel, params.max_accel))
        bounds.append((-params.max_steer, params.max_steer))
    return bounds


def _unpack_controls(controls_flat, horizon):
    # reshape (2T,) back to (T, 2)
    return controls_flat.reshape(horizon, 2)


def _rollout_from_flat(controls_flat, horizon, ego_init, dt, params):
    # run the bicycle model forward and return (T+1, 4) state array
    controls = _unpack_controls(controls_flat, horizon)
    states = rollout(ego_init, controls, dt=dt, params=params)
    return states_to_array(states)


# SQP backend
def plan_on_mean_sqp(scenario: ScenarioData,
                     kappa=KAPPA, rho=RHO,
                     safe_radius=SAFE_RADIUS,
                     maxiter=200) -> PlannerResult:

    horizon = scenario.T
    dt = scenario.dt
    ego_init = scenario.ego_initial_state
    params = DEFAULT_PARAMS

    # collapse all K modes into one mean prediction per obstacle
    mean_obstacles = _collapse_mean(scenario.obstacle_predictions)

    def objective(controls_flat):
        controls = _unpack_controls(controls_flat, horizon)
        traj = _rollout_from_flat(controls_flat, horizon, ego_init, dt, params)
        cost = driving_cost(controls, dt, kappa)
        cost += rho * lane_deviation_cost(traj, scenario.lanelet_network)
        return cost

    def collision_constr(controls_flat):
        # SLSQP needs this >= 0 for feasibility
        traj = _rollout_from_flat(controls_flat, horizon, ego_init, dt, params)
        return collision_constraint_value(traj, mean_obstacles, safe_radius)

    bounds = _build_bounds(horizon, params)
    initial_controls = np.zeros(2 * horizon)  # start with zero controls (coast)

    start_time = time.time()
    result = scipy.optimize.minimize(
        objective, initial_controls,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": collision_constr}],
        options={"maxiter": maxiter, "ftol": 1e-6, "disp": False},
    )
    elapsed = time.time() - start_time

    # extract final trajectory from the best controls found
    controls = _unpack_controls(result.x, horizon)
    traj = _rollout_from_flat(result.x, horizon, ego_init, dt, params)
    min_dist = min_obstacle_distance(traj, mean_obstacles)

    return PlannerResult(
        controls=controls,
        ego_trajectory=traj,
        cost=float(result.fun),
        min_obstacle_distance=min_dist,
        collision_free=(min_dist >= safe_radius - 0.01),
        runtime_seconds=elapsed,
        converged=result.success,
        method="plan_on_mean_sqp",
    )


# CMA-ES backend
def plan_on_mean_cmaes(scenario: ScenarioData,
                       kappa=KAPPA, rho=RHO,
                       safe_radius=SAFE_RADIUS,
                       collision_weight=COLLISION_WEIGHT,
                       maxfevals=5000, sigma0=0.5) -> PlannerResult:

    horizon = scenario.T
    dt = scenario.dt
    ego_init = scenario.ego_initial_state
    params = DEFAULT_PARAMS

    mean_obstacles = _collapse_mean(scenario.obstacle_predictions)

    # CMA-ES box bounds as [lower_list, upper_list]
    lower = np.tile([params.min_accel, -params.max_steer], horizon)
    upper = np.tile([params.max_accel, params.max_steer], horizon)

    def objective(controls_flat):
        # CMA-ES is unconstrained, so collisions go in as a big penalty term
        controls_flat = np.asarray(controls_flat)
        controls = _unpack_controls(controls_flat, horizon)
        traj = _rollout_from_flat(controls_flat, horizon, ego_init, dt, params)
        return total_cost(
            controls, traj, mean_obstacles, scenario.lanelet_network,
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

    # pull out the best solution CMA-ES found
    best_controls_flat = np.asarray(es.result.xbest)
    controls = _unpack_controls(best_controls_flat, horizon)
    traj = _rollout_from_flat(best_controls_flat, horizon, ego_init, dt, params)
    min_dist = min_obstacle_distance(traj, mean_obstacles)

    # recompute cost without the collision penalty for a fair comparison
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
        method="plan_on_mean_cmaes",
    )


if __name__ == "__main__":
    import sys
    from planners.dataloader import load_scenario

    xml_path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/USA_Lanker-1_1_T-1.xml"
    scenario = load_scenario(xml_path)
    print(f"scenario: {scenario.scenario_id}  "
          f"(T={scenario.T}, obstacles={scenario.num_obstacles})\n")

    print("plan-on-mean SQP")
    result = plan_on_mean_sqp(scenario)
    print(f"  cost:           {result.cost:.4f}")
    print(f"  min dist:       {result.min_obstacle_distance:.2f} m")
    print(f"  collision free: {result.collision_free}")
    print(f"  converged:      {result.converged}")
    print(f"  runtime:        {result.runtime_seconds:.2f} s\n")

    print("plan-on-mean CMA-ES")
    result = plan_on_mean_cmaes(scenario)
    print(f"  cost:           {result.cost:.4f}")
    print(f"  min dist:       {result.min_obstacle_distance:.2f} m")
    print(f"  collision free: {result.collision_free}")
    print(f"  converged:      {result.converged}")
    print(f"  runtime:        {result.runtime_seconds:.2f} s")
