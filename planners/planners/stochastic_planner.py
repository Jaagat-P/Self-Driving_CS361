from __future__ import annotations
import time
import numpy as np
import scipy.optimize
import cma
from planners.vehicle_dynamics import VehicleParams, rollout, states_to_array
from planners.cost_functions import (
    driving_cost, lane_deviation_cost, collision_penalty, min_obstacle_distance,
)
from planners.planner_result import PlannerResult
from planners.dataloader import ScenarioData


DEFAULT_PARAMS = VehicleParams()
SAFE_RADIUS = 3.0
KAPPA = 1.0
RHO = 1.0
COLLISION_WEIGHT = 1000.0
EPSILON = 0.1          # default chance-constraint tolerance 
SIGMOID_SHARPNESS = 5.0  # steepness of the smooth collision indicator


def _mode_obstacles(obstacle_predictions: dict, k: int) -> dict:
    """Extract the k-th prediction mode for all obstacles → {obs_id: (T, 2)}."""
    return {obs_id: preds[k] for obs_id, preds in obstacle_predictions.items()}


def _build_bounds(horizon: int, params: VehicleParams) -> list:
    bounds = []
    for _ in range(horizon):
        bounds.append((params.min_accel, params.max_accel))
        bounds.append((-params.max_steer, params.max_steer))
    return bounds


def _unpack_controls(controls_flat: np.ndarray, horizon: int) -> np.ndarray:
    return controls_flat.reshape(horizon, 2)


def _rollout_from_flat(controls_flat, horizon, ego_init, dt, params):
    controls = _unpack_controls(controls_flat, horizon)
    states = rollout(ego_init, controls, dt=dt, params=params)
    return states_to_array(states)


def _collision_fraction_smooth(ego_traj, all_mode_obs, safe_radius, sharpness=SIGMOID_SHARPNESS):
   
    K = len(all_mode_obs)
    total = 0.0
    for mode_obs in all_mode_obs:
        min_dist = min_obstacle_distance(ego_traj, mode_obs)
        total += 1.0 / (1.0 + np.exp(-sharpness * (safe_radius - min_dist)))
    return total / K


def _collision_fraction_hard(ego_traj, all_mode_obs, safe_radius):
    """Exact (non-smooth) fraction of modes in collision — used for final reporting."""
    K = len(all_mode_obs)
    count = sum(
        1 for mode_obs in all_mode_obs
        if min_obstacle_distance(ego_traj, mode_obs) < safe_radius
    )
    return count / K


# SQP backend
def stochastic_sqp(scenario: ScenarioData,
                   kappa: float = KAPPA,
                   rho: float = RHO,
                   safe_radius: float = SAFE_RADIUS,
                   epsilon: float = EPSILON,
                   maxiter: int = 200) -> PlannerResult:
    horizon = scenario.T
    dt = scenario.dt
    ego_init = scenario.ego_initial_state
    params = DEFAULT_PARAMS
    K = scenario.K

    all_mode_obs = [_mode_obstacles(scenario.obstacle_predictions, k) for k in range(K)]

    # shared cache: objective and constraint are called with the same controls
    # during the same SLSQP iteration, so avoid re-running the rollout twice
    cache: dict = {"key": None, "traj": None}

    def _get_traj(controls_flat):
        key = controls_flat.tobytes()
        if cache["key"] != key:
            cache["traj"] = _rollout_from_flat(controls_flat, horizon, ego_init, dt, params)
            cache["key"] = key
        return cache["traj"]

    def objective(controls_flat):
        controls = _unpack_controls(controls_flat, horizon)
        traj = _get_traj(controls_flat)
        cost = driving_cost(controls, dt, kappa)
        cost += rho * lane_deviation_cost(traj, scenario.lanelet_network)
        return cost

    def chance_constraint(controls_flat):
        # SLSQP requires this to be >= 0 for feasibility
        traj = _get_traj(controls_flat)
        collision_frac = _collision_fraction_smooth(traj, all_mode_obs, safe_radius)
        return epsilon - collision_frac

    bounds = _build_bounds(horizon, params)
    initial_controls = np.zeros(2 * horizon)

    start_time = time.time()
    result = scipy.optimize.minimize(
        objective, initial_controls,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": chance_constraint}],
        options={"maxiter": maxiter, "ftol": 1e-6, "disp": False},
    )
    elapsed = time.time() - start_time

    controls = _unpack_controls(result.x, horizon)
    traj = _rollout_from_flat(result.x, horizon, ego_init, dt, params)

    collision_frac = _collision_fraction_hard(traj, all_mode_obs, safe_radius)
    min_dist = min(min_obstacle_distance(traj, mode_obs) for mode_obs in all_mode_obs)

    return PlannerResult(
        controls=controls,
        ego_trajectory=traj,
        cost=float(result.fun),
        min_obstacle_distance=min_dist,
        collision_free=(collision_frac <= epsilon),
        runtime_seconds=elapsed,
        converged=result.success,
        method="stochastic_sqp",
    )


# CMA-ES backend
def stochastic_cmaes(scenario: ScenarioData,
                     kappa: float = KAPPA,
                     rho: float = RHO,
                     safe_radius: float = SAFE_RADIUS,
                     epsilon: float = EPSILON,
                     collision_weight: float = COLLISION_WEIGHT,
                     maxfevals: int = 5000,
                     sigma0: float = 0.5) -> PlannerResult:
 
    horizon = scenario.T
    dt = scenario.dt
    ego_init = scenario.ego_initial_state
    params = DEFAULT_PARAMS
    K = scenario.K

    all_mode_obs = [_mode_obstacles(scenario.obstacle_predictions, k) for k in range(K)]

    lower = np.tile([params.min_accel, -params.max_steer], horizon)
    upper = np.tile([params.max_accel, params.max_steer], horizon)

    def objective(controls_flat):
        controls_flat = np.asarray(controls_flat)
        controls = _unpack_controls(controls_flat, horizon)
        traj = _rollout_from_flat(controls_flat, horizon, ego_init, dt, params)

        cost = driving_cost(controls, dt, kappa)
        cost += rho * lane_deviation_cost(traj, scenario.lanelet_network)

        # expected collision penalty: average quadratic violation across K modes
        expected_pen = sum(
            collision_penalty(traj, mode_obs, safe_radius) for mode_obs in all_mode_obs
        ) / K
        cost += collision_weight * expected_pen

        # additional penalty when the chance constraint is violated
        collision_frac = _collision_fraction_hard(traj, all_mode_obs, safe_radius)
        if collision_frac > epsilon:
            cost += collision_weight * (collision_frac - epsilon) ** 2

        return cost

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

    collision_frac = _collision_fraction_hard(traj, all_mode_obs, safe_radius)
    min_dist = min(min_obstacle_distance(traj, mode_obs) for mode_obs in all_mode_obs)

    # report cost without collision penalty so it's comparable to other planners
    final_cost = driving_cost(controls, dt, kappa)
    final_cost += rho * lane_deviation_cost(traj, scenario.lanelet_network)

    return PlannerResult(
        controls=controls,
        ego_trajectory=traj,
        cost=final_cost,
        min_obstacle_distance=min_dist,
        collision_free=(collision_frac <= epsilon),
        runtime_seconds=elapsed,
        converged=es.result.xbest is not None,
        method="stochastic_cmaes",
    )


if __name__ == "__main__":
    import sys
    from planners.dataloader import load_scenario

    xml_path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/USA_Lanker-1_1_T-1.xml"
    scenario = load_scenario(xml_path)
    print(f"scenario: {scenario.scenario_id}  "
          f"(T={scenario.T}, K={scenario.K}, obstacles={scenario.num_obstacles})\n")

    print("stochastic SQP")
    result = stochastic_sqp(scenario)
    print(f"  cost:           {result.cost:.4f}")
    print(f"  min dist:       {result.min_obstacle_distance:.2f} m")
    print(f"  collision free: {result.collision_free}")
    print(f"  converged:      {result.converged}")
    print(f"  runtime:        {result.runtime_seconds:.2f} s\n")

    print("stochastic CMA-ES")
    result = stochastic_cmaes(scenario)
    print(f"  cost:           {result.cost:.4f}")
    print(f"  min dist:       {result.min_obstacle_distance:.2f} m")
    print(f"  collision free: {result.collision_free}")
    print(f"  converged:      {result.converged}")
    print(f"  runtime:        {result.runtime_seconds:.2f} s")
