# planner result visualization
#
# overlays the optimized ego trajectory, obstacle predictions, and
# collision proximity on top of the commonroad road network
#
# generates three plots per run:
#   trajectory.png  -- ego path + collapsed obstacle paths on road
#   predictions.png -- ego path + all K prediction modes (color per mode)
#   distances.png   -- min obstacle distance over time with safe-radius line
#
# usage:
#   python -m visualization.plot_planner scenarios/USA_Lanker-1_1_T-1.xml --planner plan_on_mean_sqp
#   python -m visualization.plot_planner scenarios/USA_Lanker-1_1_T-1.xml --planner worst_case_cmaes

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from commonroad.visualization.mp_renderer import MPRenderer

from planners.dataloader import ScenarioData
from planners.planner_result import PlannerResult


PREDICTION_COLORS = ["#00e5ff", "#ffea00", "#aa00ff", "#ff6d00", "#ffffff", "#76ff03"]
MODE_LABELS = ["Const. velocity", "Braking", "Lane change L", "Lane change R"]


def _output_dir(scenario_id, method):
    out = os.path.join("visualization", "planner_results", scenario_id, method)
    os.makedirs(out, exist_ok=True)
    return out


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def _draw_road(ax, scenario, skip_obstacles=False):
    # render the commonroad road network (and optionally the obstacles)
    # skip_obstacles=True draws only the lanelets/traffic signs, not the blue blocks
    rnd = MPRenderer(ax=ax)
    if skip_obstacles:
        scenario.raw_scenario.lanelet_network.draw(rnd)
    else:
        scenario.raw_scenario.draw(rnd)
    rnd.render()


def _zoom_to_ego(ax, ego, margin=20.0):
    # crop the view to a box around the ego trajectory so it's actually visible
    x_min, x_max = ego[:, 0].min() - margin, ego[:, 0].max() + margin
    y_min, y_max = ego[:, 1].min() - margin, ego[:, 1].max() + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def _draw_ego(ax, ego):
    # bright magenta so it's unmissable on the grey road
    ax.plot(ego[:, 0], ego[:, 1], "-", color="#ff00ff", linewidth=4.0,
            alpha=0.9, zorder=10, label="Ego (optimized)")
    ax.plot(ego[:, 0], ego[:, 1], "o", color="white", markersize=4,
            zorder=10, markeredgecolor="#ff00ff", markeredgewidth=1.5)
    ax.plot(ego[0, 0], ego[0, 1], "s", color="#00ff00", markersize=12,
            zorder=11, label="Ego start", markeredgecolor="black", markeredgewidth=1.5)
    ax.plot(ego[-1, 0], ego[-1, 1], "*", color="#ff0000", markersize=16,
            zorder=11, label="Ego end", markeredgecolor="black", markeredgewidth=1)


def plot_trajectory(scenario, result, collapsed_obs=None):
    # ego trajectory + collapsed obstacle trajectories on the road
    fig, ax = plt.subplots(figsize=(14, 9))
    _draw_road(ax, scenario, skip_obstacles=True)

    ego = result.ego_trajectory
    _draw_ego(ax, ego)

    # draw the collapsed (mean or worst-case) obstacle paths
    # only show the ones closest to ego so it's not cluttered
    if collapsed_obs is not None:
        ranked = _rank_obstacles_by_proximity(ego, scenario.obstacle_predictions)
        nearby_ids = [obs_id for obs_id, _ in ranked[:6]]

        for i, obs_id in enumerate(nearby_ids):
            if obs_id not in collapsed_obs:
                continue
            positions = collapsed_obs[obs_id]
            color = PREDICTION_COLORS[i % len(PREDICTION_COLORS)]
            ax.plot(positions[:, 0], positions[:, 1], "--", color=color,
                    linewidth=3.0, alpha=0.9, zorder=9, label=f"Obs {obs_id} (collapsed)")
            ax.plot(positions[0, 0], positions[0, 1], "o", color=color,
                    markersize=8, zorder=9, markeredgecolor="black", markeredgewidth=1)
            # label at the start position
            ax.annotate(f"Obs {obs_id}", xy=(positions[0, 0], positions[0, 1]),
                        fontsize=7, fontweight="bold", color="black",
                        ha="center", va="bottom",
                        xytext=(0, 8), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
                        zorder=15)

    ax.set_title(f"{result.method} — {scenario.scenario_id}\n"
                 f"cost={result.cost:.2f}  min_dist={result.min_obstacle_distance:.2f}m  "
                 f"collision_free={result.collision_free}  (vs collapsed obstacles only)",
                 fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _zoom_to_ego(ax, ego)
    leg = ax.legend(fontsize=7, loc="upper right", ncol=2, framealpha=1.0)
    leg.set_zorder(100)

    out = os.path.join(_output_dir(str(scenario.scenario_id), result.method),
                       "trajectory.png")
    _save(fig, out)
    return out


def _rank_obstacles_by_proximity(ego_traj, obstacle_predictions):
    # rank obstacles by how close they ever get to ego (across all modes)
    # returns list of (obs_id, min_distance) sorted closest first
    rankings = []
    for obs_id, preds in obstacle_predictions.items():
        num_modes, horizon, _ = preds.shape
        best_dist = np.inf
        for k in range(num_modes):
            length = min(ego_traj.shape[0] - 1, horizon)
            for t in range(length):
                dist = np.linalg.norm(ego_traj[t + 1, :2] - preds[k, t, :2])
                best_dist = min(best_dist, dist)
        rankings.append((obs_id, best_dist))
    rankings.sort(key=lambda x: x[1])
    return rankings


def plot_all_predictions(scenario, result, max_obstacles=6):
    # ego trajectory + all K prediction modes for the closest obstacles
    # skips commonroad's blue obstacle blocks to avoid clutter
    fig, ax = plt.subplots(figsize=(14, 9))
    _draw_road(ax, scenario, skip_obstacles=True)

    ego = result.ego_trajectory
    _draw_ego(ax, ego)

    # pick the obstacles that actually get close to ego
    ranked = _rank_obstacles_by_proximity(ego, scenario.obstacle_predictions)
    nearby_ids = [obs_id for obs_id, _ in ranked[:max_obstacles]]

    # legend entries for the prediction mode colors
    handles = [mpatches.Patch(color="#ff00ff", label="Ego trajectory")]
    for k in range(min(4, scenario.K)):
        label = MODE_LABELS[k] if k < len(MODE_LABELS) else f"Mode {k}"
        handles.append(mpatches.Patch(color=PREDICTION_COLORS[k], label=label))

    # draw all K modes for each nearby obstacle
    for obs_id in nearby_ids:
        preds = scenario.obstacle_predictions[obs_id]
        num_modes = preds.shape[0]
        for k in range(num_modes):
            color = PREDICTION_COLORS[k % len(PREDICTION_COLORS)]
            ax.plot(preds[k, :, 0], preds[k, :, 1], "-", color=color,
                    linewidth=1.5, alpha=0.6)
            # mark the start of each prediction with a dot
            ax.plot(preds[k, 0, 0], preds[k, 0, 1], "o", color=color,
                    markersize=5, alpha=0.8)
        # label the obstacle at its starting position
        start_pos = preds[0, 0]
        ax.annotate(f"Obs {obs_id}", xy=(start_pos[0], start_pos[1]),
                    fontsize=7, fontweight="bold", color="black",
                    ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
                    zorder=15)

    ax.set_title(f"{result.method} — K={scenario.K} prediction modes for "
                 f"{len(nearby_ids)} closest obstacles\n"
                 f"{scenario.scenario_id} ({scenario.num_obstacles} total obstacles)",
                 fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _zoom_to_ego(ax, ego)
    leg = ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=1.0)
    leg.set_zorder(100)

    out = os.path.join(_output_dir(str(scenario.scenario_id), result.method),
                       "predictions.png")
    _save(fig, out)
    return out


def plot_distance_profile(scenario, result, safe_radius=3.0):
    # min distance to any obstacle (across ALL K modes) at each timestep
    ego = result.ego_trajectory
    horizon = ego.shape[0] - 1
    times = np.arange(horizon) * scenario.dt

    # check every prediction mode for every obstacle at every timestep
    min_dists = np.full(horizon, np.inf)
    for obs_id, preds in scenario.obstacle_predictions.items():
        num_modes = preds.shape[0]
        for k in range(num_modes):
            length = min(horizon, preds.shape[1])
            for t in range(length):
                dist = np.linalg.norm(ego[t + 1, :2] - preds[k, t, :2])
                min_dists[t] = min(min_dists[t], dist)

    posthoc_safe = bool(min_dists.min() >= safe_radius)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, min_dists[:len(times)], "-o", color="#2c3e50", markersize=3,
            linewidth=2, label="Min distance to any obstacle")
    ax.axhline(safe_radius, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Safe radius = {safe_radius} m")

    # shade the collision zone in red if we dip below the safe radius
    violation_mask = min_dists[:len(times)] < safe_radius
    if np.any(violation_mask):
        ax.fill_between(times, 0, min_dists[:len(times)],
                        where=violation_mask, color="#e74c3c", alpha=0.15,
                        label="Collision zone")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Min obstacle distance [m]")
    ax.set_title(f"{result.method} — Distance profile (vs ALL K={scenario.K} futures)\n"
                 f"min={min_dists.min():.2f}m  safe post-hoc={posthoc_safe}"
                 f"  (planner believed collision_free={result.collision_free})",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    out = os.path.join(_output_dir(str(scenario.scenario_id), result.method),
                       "distances.png")
    _save(fig, out)
    return out


def plot_planner_result(scenario, result, collapsed_obs=None, safe_radius=3.0):
    # generate all three plots, return list of saved file paths
    return [
        plot_trajectory(scenario, result, collapsed_obs),
        plot_all_predictions(scenario, result),
        plot_distance_profile(scenario, result, safe_radius),
    ]


if __name__ == "__main__":
    import sys
    import argparse
    from planners.dataloader import load_scenario

    parser = argparse.ArgumentParser(description="visualize planner results")
    parser.add_argument("scenario", help="path to CommonRoad XML scenario file")
    parser.add_argument("--planner", default="plan_on_mean_sqp",
                        choices=["plan_on_mean_sqp", "plan_on_mean_cmaes",
                                 "worst_case_sqp", "worst_case_cmaes"],
                        help="which planner to run and visualize")
    args = parser.parse_args()

    print(f"loading scenario: {args.scenario}")
    scenario_data = load_scenario(args.scenario)
    print(f"  {scenario_data.scenario_id}: T={scenario_data.T}, "
          f"obstacles={scenario_data.num_obstacles}\n")

    collapsed = None
    print(f"running {args.planner} ...")

    if args.planner == "plan_on_mean_sqp":
        from planners.plan_on_mean import plan_on_mean_sqp, _collapse_mean
        result = plan_on_mean_sqp(scenario_data)
        collapsed = _collapse_mean(scenario_data.obstacle_predictions)

    elif args.planner == "plan_on_mean_cmaes":
        from planners.plan_on_mean import plan_on_mean_cmaes, _collapse_mean
        result = plan_on_mean_cmaes(scenario_data)
        collapsed = _collapse_mean(scenario_data.obstacle_predictions)

    elif args.planner == "worst_case_sqp":
        from planners.worst_case import worst_case_sqp, select_worst_case
        result = worst_case_sqp(scenario_data)
        collapsed = select_worst_case(result.ego_trajectory,
                                      scenario_data.obstacle_predictions)

    elif args.planner == "worst_case_cmaes":
        from planners.worst_case import worst_case_cmaes, select_worst_case
        result = worst_case_cmaes(scenario_data)
        collapsed = select_worst_case(result.ego_trajectory,
                                      scenario_data.obstacle_predictions)

    print(f"\nresult: cost={result.cost:.4f}  min_dist={result.min_obstacle_distance:.2f}m  "
          f"collision_free={result.collision_free}  runtime={result.runtime_seconds:.2f}s\n")

    print("generating plots ...")
    paths = plot_planner_result(scenario_data, result, collapsed)
    print(f"\ndone, {len(paths)} plots saved")
