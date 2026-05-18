import time

def comfort_cost(controls):
    accel = controls[:, 0]
    steer = controls[:, 1]

    J = np.sum(accel ** 2)
    J += 5.0 * np.sum(steer ** 2)

    if len(accel) > 1:
        J += 0.5 * np.sum(np.diff(accel) ** 2)
        J += 2.0 * np.sum(np.diff(steer) ** 2)

    return J


def goal_cost(ego_traj):
    return -10.0 * ego_traj[-1, 0]


def stochastic_objective(u_flat, scenario, epsilon=0.05):
    T = scenario.T
    controls = u_flat.reshape(T, 2)

    ego_traj = rollout_trajectory(
        scenario.ego_initial_state,
        controls,
        scenario.dt,
    )

    J = 0.0
    J += comfort_cost(controls)
    J += 100.0 * expected_collision_cost(
        ego_traj,
        scenario.obstacle_predictions,
    )
    J += goal_cost(ego_traj)

    p_collision = compute_collision_probability(
        ego_traj,
        scenario.obstacle_predictions,
    )

    if p_collision > epsilon:
        J += 1e6 * (p_collision - epsilon) ** 2

    return J


def plan_stochastic(scenario, epsilon=0.05):
    T = scenario.T

    u0 = np.zeros((T, 2))

    bounds = []
    for _ in range(T):
        bounds.append((-3.0, 2.0))   
        bounds.append((-0.5, 0.5))   

    t0 = time.time()

    result = minimize(
        stochastic_objective,
        u0.flatten(),
        args=(scenario, epsilon),
        method="SLSQP",
        bounds=bounds,
        options={
            "maxiter": 100,
            "ftol": 1e-3,
            "disp": True,
        },
    )

    runtime = time.time() - t0

    controls = result.x.reshape(T, 2)

    trajectory = rollout_trajectory(
        scenario.ego_initial_state,
        controls,
        scenario.dt,
    )

    collision_probability = compute_collision_probability(
        trajectory,
        scenario.obstacle_predictions,
    )

    return {
        "controls": controls,
        "trajectory": trajectory,
        "objective": float(result.fun),
        "success": bool(result.success),
        "message": result.message,
        "iterations": int(result.nit),
        "runtime": runtime,
        "collision_probability": collision_probability,
    }
