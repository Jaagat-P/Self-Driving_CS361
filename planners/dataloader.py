"""
Scenario dataloader for the chance-constrained trajectory optimization project.

Scans a directory of CommonRoad XML files and returns a list of ScenarioData
objects — one per scenario — each containing everything a planner needs:

  - ego initial state and planning problem
  - K predicted future trajectories per obstacle  (shape: K × T × 2)
  - lanelet network (for lane-deviation cost d_t)
  - scenario metadata (id, dt, horizon T)

Typical usage
-------------
    from planners.dataloader import load_scenarios

    dataset = load_scenarios("scenarios/", K=4, T=50)
    for sd in dataset:
        print(sd.scenario_id, sd.ego_initial_state)
        # sd.obstacle_predictions: dict { obs_id → np.ndarray (K, T, 2) }

ScenarioData fields
-------------------
    scenario_id          str
    dt                   float   time step [s]
    T                    int     planning horizon [steps]
    ego_initial_state    VehicleState
    planning_problem     CommonRoad PlanningProblem object
    lanelet_network      CommonRoad LaneletNetwork object
    obstacle_predictions dict[int, np.ndarray]   shape (K, T, 2)
    raw_scenario         CommonRoad Scenario object (kept for visualization)
"""

from __future__ import annotations

import os
import glob
import warnings
from dataclasses import dataclass, field

import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import Scenario

from planners.vehicle_dynamics import VehicleState
from planners.predictor import predict_all_obstacles, DEFAULT_MODES, PredictionMode


# ---------------------------------------------------------------------------
# ScenarioData container
# ---------------------------------------------------------------------------

@dataclass
class ScenarioData:
    """All planner-relevant data extracted from one CommonRoad scenario."""

    scenario_id: str
    dt: float
    T: int

    ego_initial_state: VehicleState
    planning_problem: PlanningProblem
    lanelet_network: LaneletNetwork

    # { obstacle_id → np.ndarray of shape (K, T, 2) }
    obstacle_predictions: dict[int, np.ndarray]

    # kept for visualization / debugging
    raw_scenario: Scenario

    @property
    def K(self) -> int:
        """Number of prediction modes."""
        if not self.obstacle_predictions:
            return 0
        return next(iter(self.obstacle_predictions.values())).shape[0]

    @property
    def num_obstacles(self) -> int:
        return len(self.obstacle_predictions)

    def __repr__(self) -> str:
        return (
            f"ScenarioData(id={self.scenario_id!r}, "
            f"T={self.T}, dt={self.dt}, K={self.K}, "
            f"obstacles={self.num_obstacles}, "
            f"ego={self.ego_initial_state})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_ego_state(planning_problem: PlanningProblem) -> VehicleState:
    """Pull the ego vehicle's initial state from the planning problem."""
    s = planning_problem.initial_state
    x, y = float(s.position[0]), float(s.position[1])
    psi  = float(s.orientation)
    v    = float(s.velocity) if hasattr(s, "velocity") else 0.0
    return VehicleState(x=x, y=y, psi=psi, v=v)


def _obstacle_state_dict(obs) -> dict:
    """Convert a CommonRoad dynamic obstacle to the dict format predictor expects."""
    s = obs.initial_state
    return {
        "id":      obs.obstacle_id,
        "x":       float(s.position[0]),
        "y":       float(s.position[1]),
        "heading": float(s.orientation),
        "speed":   float(s.velocity) if hasattr(s, "velocity") else 0.0,
    }


def _infer_dt(scenario: Scenario) -> float:
    """Read dt from the scenario; fall back to 0.1 s if unavailable."""
    try:
        return float(scenario.dt)
    except AttributeError:
        return 0.1


def _infer_horizon(scenario: Scenario, dt: float, default_T: int) -> int:
    """
    Use the longest obstacle trajectory in the scenario as the horizon.
    Falls back to default_T if no trajectory is present.
    """
    max_steps = 0
    for obs in scenario.dynamic_obstacles:
        if hasattr(obs.prediction, "trajectory"):
            max_steps = max(max_steps, len(obs.prediction.trajectory.state_list))
    return max_steps if max_steps > 0 else default_T


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_scenario(
    xml_path: str,
    *,
    K: int = 4,
    T: int | None = None,
    modes: list[PredictionMode] | None = None,
) -> ScenarioData:
    """
    Load a single CommonRoad XML file and return a ScenarioData object.

    Parameters
    ----------
    xml_path : str
        Path to the .xml scenario file.
    K : int
        Number of prediction modes.  Ignored if `modes` is provided.
    T : int or None
        Planning horizon in steps.  If None, inferred from scenario duration.
    modes : list[PredictionMode] or None
        Custom prediction modes.  Defaults to DEFAULT_MODES[:K].

    Returns
    -------
    ScenarioData
    """
    scenario, planning_problem_set = CommonRoadFileReader(xml_path).open()

    # pick the first planning problem (most scenarios have exactly one)
    pp_id   = next(iter(planning_problem_set.planning_problem_dict))
    pp      = planning_problem_set.planning_problem_dict[pp_id]

    dt      = _infer_dt(scenario)
    horizon = T if T is not None else _infer_horizon(scenario, dt, default_T=50)

    ego_state = _extract_ego_state(pp)

    if modes is None:
        modes = DEFAULT_MODES[:K]

    obs_dicts = [_obstacle_state_dict(obs) for obs in scenario.dynamic_obstacles]
    predictions = predict_all_obstacles(obs_dicts, T=horizon, dt=dt, modes=modes)

    return ScenarioData(
        scenario_id=str(scenario.scenario_id),
        dt=dt,
        T=horizon,
        ego_initial_state=ego_state,
        planning_problem=pp,
        lanelet_network=scenario.lanelet_network,
        obstacle_predictions=predictions,
        raw_scenario=scenario,
    )


def load_scenarios(
    scenarios_dir: str,
    *,
    K: int = 4,
    T: int | None = None,
    modes: list[PredictionMode] | None = None,
    pattern: str = "*.xml",
    verbose: bool = True,
) -> list[ScenarioData]:
    """
    Load all CommonRoad XML files from a directory.

    Parameters
    ----------
    scenarios_dir : str
        Path to the directory containing .xml scenario files.
    K : int
        Number of prediction modes per obstacle.
    T : int or None
        Planning horizon.  If None, inferred per scenario.
    modes : list[PredictionMode] or None
        Custom prediction modes.
    pattern : str
        Glob pattern for matching files inside `scenarios_dir`.
    verbose : bool
        Print a summary line per scenario.

    Returns
    -------
    list[ScenarioData]  – one entry per successfully loaded scenario.
    """
    xml_files = sorted(glob.glob(os.path.join(scenarios_dir, pattern)))

    if not xml_files:
        warnings.warn(f"No XML files found in {scenarios_dir!r} matching {pattern!r}")
        return []

    dataset: list[ScenarioData] = []

    for path in xml_files:
        try:
            sd = load_scenario(path, K=K, T=T, modes=modes)
            dataset.append(sd)
            if verbose:
                print(f"  Loaded: {sd}")
        except Exception as exc:
            warnings.warn(f"Skipping {path}: {exc}")

    if verbose:
        print(f"\nTotal: {len(dataset)} scenario(s) loaded from {scenarios_dir!r}")

    return dataset


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    scenarios_dir = sys.argv[1] if len(sys.argv) > 1 else "../scenarios"

    print(f"Loading all scenarios from: {scenarios_dir}\n")
    dataset = load_scenarios(scenarios_dir, K=4, verbose=True)

    if dataset:
        sd = dataset[0]
        print(f"\n--- First scenario details ---")
        print(f"  ID:              {sd.scenario_id}")
        print(f"  dt:              {sd.dt} s")
        print(f"  Horizon T:       {sd.T} steps ({sd.T * sd.dt:.1f} s)")
        print(f"  Ego start:       {sd.ego_initial_state}")
        print(f"  Obstacles:       {sd.num_obstacles}")
        print(f"  Prediction modes (K={sd.K}):")
        for obs_id, preds in sd.obstacle_predictions.items():
            print(f"    Obstacle {obs_id}: predictions.shape = {preds.shape}  (K, T, xy)")
            break  # just show one
