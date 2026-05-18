"""
Visualization utilities for CommonRoad scenarios.

Figures are saved to:
    <scenarios_dir>/visualizations/<scenario_stem>/
        static.png          – road network + obstacle positions at t=0
        trajectories.png    – full obstacle paths over time
        animation.png       – grid of snapshots at evenly-spaced time steps
"""

import os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend – saves files, no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load(scenario_path: str):
    """Load and return (scenario, planning_problem_set) from an XML file."""
    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()
    return scenario, planning_problem_set


def _output_dir(scenario_path: str) -> str:
    """
    Derive the output folder from the scenario path.

    e.g. ../scenarios/USA_Lanker-1_1_T-1.xml
         → ../scenarios/visualizations/USA_Lanker-1_1_T-1/
    """
    scenarios_dir = os.path.dirname(os.path.abspath(scenario_path))
    stem = os.path.splitext(os.path.basename(scenario_path))[0]
    out = os.path.join(scenarios_dir, "visualizations", stem)
    os.makedirs(out, exist_ok=True)
    return out


def _obstacle_color_map(n: int):
    """Return a list of n distinct colors for obstacle trajectories."""
    cmap = matplotlib.colormaps.get_cmap("tab10")
    return [cmap(i % 10) for i in range(max(n, 1))]


def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_scenario_static(
    scenario_path: str,
    *,
    show_obstacle_ids: bool = True,
    title: str | None = None,
) -> str:
    """
    Draw a static snapshot of the scenario at time step 0.

    Returns the path of the saved PNG.
    """
    scenario, planning_problem_set = _load(scenario_path)

    fig, ax = plt.subplots(figsize=(14, 8))

    rnd = MPRenderer(ax=ax)
    scenario.draw(rnd)
    planning_problem_set.draw(rnd)
    rnd.render()

    if show_obstacle_ids:
        for obs in scenario.dynamic_obstacles:
            init_pos = obs.initial_state.position
            ax.text(
                init_pos[0],
                init_pos[1] + 0.8,
                str(obs.obstacle_id),
                fontsize=7,
                ha="center",
                color="darkred",
                fontweight="bold",
            )

    ax.set_title(title or f"Scenario: {scenario.scenario_id}", fontsize=13)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    out = os.path.join(_output_dir(scenario_path), "static.png")
    _save(fig, out)
    return out


def plot_obstacle_trajectories(
    scenario_path: str,
    *,
    max_obstacles: int = 10,
    title: str | None = None,
) -> str:
    """
    Plot the full predicted trajectories of dynamic obstacles as colored paths.

    Returns the path of the saved PNG.
    """
    scenario, planning_problem_set = _load(scenario_path)

    fig, ax = plt.subplots(figsize=(14, 8))

    rnd = MPRenderer(ax=ax)
    scenario.draw(rnd)
    rnd.render()

    obstacles = scenario.dynamic_obstacles[:max_obstacles]
    colors = _obstacle_color_map(len(obstacles))
    legend_handles = []

    for obs, color in zip(obstacles, colors):
        traj = obs.prediction.trajectory
        xs = [state.position[0] for state in traj.state_list]
        ys = [state.position[1] for state in traj.state_list]

        ax.plot(xs, ys, "-o", color=color, markersize=2, linewidth=1.5, alpha=0.85)
        ax.plot(xs[0], ys[0], "o", color=color, markersize=6)

        legend_handles.append(
            mpatches.Patch(color=color, label=f"Obstacle {obs.obstacle_id}")
        )

    for pp in planning_problem_set.planning_problem_dict.values():
        init = pp.initial_state.position
        ax.plot(init[0], init[1], "g*", markersize=14, label="Ego start", zorder=5)

    ax.set_title(title or f"Obstacle Trajectories – {scenario.scenario_id}", fontsize=13)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(handles=legend_handles, fontsize=7, loc="upper right", ncol=2)

    out = os.path.join(_output_dir(scenario_path), "trajectories.png")
    _save(fig, out)
    return out


def plot_scenario_animation(
    scenario_path: str,
    *,
    time_steps: list[int] | None = None,
    cols: int = 4,
) -> str:
    """
    Render a grid of snapshots at multiple time steps.

    Returns the path of the saved PNG.
    """
    scenario, planning_problem_set = _load(scenario_path)

    max_t = 0
    for obs in scenario.dynamic_obstacles:
        if hasattr(obs.prediction, "trajectory"):
            max_t = max(max_t, len(obs.prediction.trajectory.state_list) - 1)

    resolved_steps: list[int]
    if time_steps is None:
        n_steps = min(8, max_t + 1)
        resolved_steps = np.linspace(0, max_t, n_steps, dtype=int).tolist()
    else:
        resolved_steps = time_steps

    steps = resolved_steps
    rows = int(np.ceil(len(steps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten()

    for ax, t in zip(axes, steps):
        # MPDrawParams is the correct typed object; a plain dict causes an AttributeError
        params = MPDrawParams()
        params.time_begin = t
        params.time_end = t

        rnd = MPRenderer(ax=ax)
        scenario.draw(rnd, draw_params=params)
        planning_problem_set.draw(rnd)
        rnd.render()
        ax.set_title(f"t = {t}", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")

    for ax in axes[len(steps):]:
        ax.set_visible(False)

    fig.suptitle(f"Scenario: {scenario.scenario_id}", fontsize=13, y=1.01)
    plt.tight_layout()

    out = os.path.join(_output_dir(scenario_path), "animation.png")
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Quick test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "../scenarios/USA_Lanker-1_1_T-1.xml"
    print(f"Loading scenario: {path}")

    print("\n[1/3] Static snapshot at t=0 ...")
    plot_scenario_static(path)

    print("[2/3] Full obstacle trajectories ...")
    plot_obstacle_trajectories(path)

    print("[3/3] Time-step grid (animation frames) ...")
    plot_scenario_animation(path)

    print("\nDone. Check scenarios/visualizations/ for output files.")
