"""
Comparative evaluation of all three planners (6 variants total) on 20 scenarios.

Planners evaluated:
  plan_on_mean_sqp   — collapses K futures to mean, SQP solver
  plan_on_mean_cmaes — collapses K futures to mean, CMA-ES solver
  worst_case_sqp     — minimax over K futures, SQP solver
  worst_case_cmaes   — minimax over K futures, CMA-ES solver
  stochastic_sqp     — chance constraint over K futures, SQP solver
  stochastic_cmaes   — chance constraint over K futures, CMA-ES solver

Metrics per scenario × planner:
  collision_rate      fraction of K=4 modes with post-hoc collision (0–1)
  completion_rate     1 if converged AND collision_free, else 0
  expected_cost       J(u*) = driving + lane-deviation cost (no penalty term)
  runtime_s           wall-clock optimisation time [s]
  min_obstacle_dist   closest approach to any obstacle across all timesteps [m]
  converged           did the solver converge?
  collision_free      is the plan collision-free under the planner's own model?

Results are saved as:
  evaluation/results/results.csv   — one row per (scenario, planner)
  evaluation/results/results.json  — same data, plus per-planner summary stats

Usage (from repo root, with selfdriving-cs361 env active):
    python -m evaluation.run_evaluation
    python -m evaluation.run_evaluation --scenarios scenarios/eval --out evaluation/results
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback

import numpy as np

from planners.dataloader import load_scenarios, ScenarioData
from planners.cost_functions import min_obstacle_distance
from planners.planner_result import PlannerResult
from planners.plan_on_mean import plan_on_mean_sqp, plan_on_mean_cmaes
from planners.worst_case import worst_case_sqp, worst_case_cmaes
from planners.planners.stochastic_planner import stochastic_sqp, stochastic_cmaes

SAFE_RADIUS = 3.0

# (display_name, callable)
PLANNERS = [
    ("plan_on_mean_sqp",   plan_on_mean_sqp),
    ("plan_on_mean_cmaes", plan_on_mean_cmaes),
    ("worst_case_sqp",     worst_case_sqp),
    ("worst_case_cmaes",   worst_case_cmaes),
    ("stochastic_sqp",     stochastic_sqp),
    ("stochastic_cmaes",   stochastic_cmaes),
]

# Real CommonRoad scenario descriptions (NGSIM dataset)
SCENARIO_DESCRIPTIONS = {
    # Lankershim urban multi-lane intersection (Los Angeles, CA)
    "USA_Lanker-1_1_T-1":  "Lankershim map-1 cfg-01: urban intersection, 24 obs, T=40",
    "USA_Lanker-1_2_T-1":  "Lankershim map-1 cfg-02: urban intersection, 42 obs, T=55",
    "USA_Lanker-1_3_T-1":  "Lankershim map-1 cfg-03: urban intersection, 36 obs, T=40",
    "USA_Lanker-1_4_T-1":  "Lankershim map-1 cfg-04: urban intersection, 34 obs, T=15",
    "USA_Lanker-1_5_T-1":  "Lankershim map-1 cfg-05: urban intersection, 35 obs, T=10",
    "USA_Lanker-1_6_T-1":  "Lankershim map-1 cfg-06: urban intersection, 34 obs, T=10",
    "USA_Lanker-1_7_T-1":  "Lankershim map-1 cfg-07: urban intersection, 22 obs, T=15",
    "USA_Lanker-1_8_T-1":  "Lankershim map-1 cfg-08: urban intersection, 31 obs, T=15",
    "USA_Lanker-1_9_T-1":  "Lankershim map-1 cfg-09: urban intersection, 26 obs, T=20",
    "USA_Lanker-1_10_T-1": "Lankershim map-1 cfg-10: urban intersection, 32 obs, T=15",
    # US-101 highway (Hollywood Freeway, Los Angeles, CA)
    "USA_US101-1_1_T-1":   "US-101 map-01: highway, 2 obs, T=60",
    "USA_US101-2_1_T-1":   "US-101 map-02: highway dense, 40 obs, T=104",
    "USA_US101-3_1_T-1":   "US-101 map-03: highway dense, 34 obs, T=80",
    "USA_US101-4_1_T-1":   "US-101 map-04: highway, 22 obs, T=100",
    "USA_US101-5_1_T-1":   "US-101 map-05: highway, 25 obs, T=100",
    "USA_US101-6_1_T-1":   "US-101 map-06: highway dense, 29 obs, T=80",
    "USA_US101-8_1_T-1":   "US-101 map-08: highway, 27 obs, T=75",
    "USA_US101-9_1_T-1":   "US-101 map-09: highway dense, 30 obs, T=80",
    "USA_US101-10_1_T-1":  "US-101 map-10: highway dense, 44 obs, T=80",
    "USA_US101-11_1_T-1":  "US-101 map-11: highway, 32 obs, T=72",
}


# ---------------------------------------------------------------------------
# Post-hoc collision rate (evaluated against ALL K modes)
# ---------------------------------------------------------------------------

def posthoc_collision_rate(result: PlannerResult,
                           obstacle_predictions: dict,
                           safe_radius: float = SAFE_RADIUS) -> float:
    if not obstacle_predictions:
        return 0.0
    first = next(iter(obstacle_predictions.values()))
    K = first.shape[0]
    collisions = 0
    for k in range(K):
        mode_obs = {obs_id: preds[k] for obs_id, preds in obstacle_predictions.items()}
        if min_obstacle_distance(result.ego_trajectory, mode_obs) < safe_radius:
            collisions += 1
    return collisions / K


# ---------------------------------------------------------------------------
# Single-scenario evaluation
# ---------------------------------------------------------------------------

def evaluate_scenario(scenario: ScenarioData) -> list[dict]:
    rows = []
    for name, fn in PLANNERS:
        print(f"    [{name}] ...", end=" ", flush=True)
        try:
            result = fn(scenario)
            col_rate = posthoc_collision_rate(result, scenario.obstacle_predictions)
            completion = 1 if (result.converged and result.collision_free) else 0

            row = {
                "scenario_id":          scenario.scenario_id,
                "num_obstacles":         scenario.num_obstacles,
                "planner":               name,
                "collision_rate":        round(col_rate, 4),
                "completion_rate":       completion,
                "expected_cost":         round(result.cost, 6),
                "runtime_s":             round(result.runtime_seconds, 4),
                "min_obstacle_dist_m":   round(result.min_obstacle_distance, 4),
                "converged":             int(result.converged),
                "collision_free":        int(result.collision_free),
                "error":                 "",
            }
            print(f"done  cost={result.cost:.3f}  col_rate={col_rate:.0%}"
                  f"  dist={result.min_obstacle_distance:.2f}m"
                  f"  t={result.runtime_seconds:.1f}s")
        except Exception:
            tb = traceback.format_exc().strip().split("\n")[-1]
            print(f"FAILED: {tb}")
            row = {
                "scenario_id":         scenario.scenario_id,
                "num_obstacles":        scenario.num_obstacles,
                "planner":              name,
                "collision_rate":       None,
                "completion_rate":      None,
                "expected_cost":        None,
                "runtime_s":            None,
                "min_obstacle_dist_m":  None,
                "converged":            None,
                "collision_free":       None,
                "error":                tb,
            }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Summary stats across scenarios
# ---------------------------------------------------------------------------

def _safe_stats(values: list) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    arr = np.array(vals, dtype=float)
    return {
        "mean": round(float(arr.mean()), 4),
        "std":  round(float(arr.std()),  4),
        "min":  round(float(arr.min()),  4),
        "max":  round(float(arr.max()),  4),
        "n":    int(len(arr)),
    }


def compute_summary(rows: list[dict]) -> dict:
    planner_names = [p for p, _ in PLANNERS]
    summary = {}
    for planner in planner_names:
        pr = [r for r in rows if r["planner"] == planner]
        summary[planner] = {
            "collision_rate":      _safe_stats([r["collision_rate"]      for r in pr]),
            "completion_rate":     _safe_stats([r["completion_rate"]     for r in pr]),
            "expected_cost":       _safe_stats([r["expected_cost"]       for r in pr]),
            "runtime_s":           _safe_stats([r["runtime_s"]           for r in pr]),
            "min_obstacle_dist_m": _safe_stats([r["min_obstacle_dist_m"] for r in pr]),
            "num_scenarios":       len(pr),
            "num_errors":          sum(1 for r in pr if r["error"]),
        }
    return summary


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "scenario_id", "num_obstacles", "planner",
    "collision_rate", "completion_rate", "expected_cost",
    "runtime_s", "min_obstacle_dist_m", "converged", "collision_free", "error",
]


def save_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved  → {path}")


def save_json(rows: list[dict], summary: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "metadata": {
            "num_scenarios":     len({r["scenario_id"] for r in rows}),
            "num_planners":      len(PLANNERS),
            "safe_radius_m":     SAFE_RADIUS,
            "scenario_descriptions": SCENARIO_DESCRIPTIONS,
        },
        "summary": summary,
        "raw_rows": rows,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON saved → {path}")


def print_summary_table(summary: dict) -> None:
    header = (f"\n{'Planner':<22} {'Col.Rate':>9} {'±std':>6} "
              f"{'Compl.':>7} {'±std':>6} "
              f"{'Cost':>10} {'±std':>8} "
              f"{'Runtime(s)':>11} {'±std':>6} "
              f"{'MinDist(m)':>11} {'±std':>6}")
    print(header)
    print("-" * len(header))
    for planner, s in summary.items():
        cr  = s["collision_rate"]
        cp  = s["completion_rate"]
        ec  = s["expected_cost"]
        rt  = s["runtime_s"]
        md  = s["min_obstacle_dist_m"]

        def _fmt(d, pct=False):
            if d["mean"] is None:
                return f"{'N/A':>9} {'':>6}"
            fmt = f"{d['mean']*100:8.1f}%" if pct else f"{d['mean']:9.4f}"
            return f"{fmt} {d['std']:>6.4f}"

        print(f"{planner:<22} {_fmt(cr, pct=True)} {_fmt(cp, pct=True)} "
              f"{_fmt(ec)} {_fmt(rt)} {_fmt(md)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(scenarios_dir: str, out_dir: str) -> None:
    print(f"Loading scenarios from: {scenarios_dir}")
    dataset = load_scenarios(scenarios_dir, verbose=False)
    if not dataset:
        print("ERROR: no scenarios found.")
        sys.exit(1)

    print(f"Running {len(PLANNERS)} planners × {len(dataset)} scenarios "
          f"= {len(PLANNERS) * len(dataset)} optimisations\n")

    all_rows: list[dict] = []
    for i, sd in enumerate(dataset):
        desc = SCENARIO_DESCRIPTIONS.get(sd.scenario_id,
                                          f"{sd.scenario_id}  (T={sd.T}, obs={sd.num_obstacles})")
        print(f"[{i+1:02d}/{len(dataset)}]  {desc}  "
              f"(obstacles={sd.num_obstacles})")
        rows = evaluate_scenario(sd)
        all_rows.extend(rows)

    summary = compute_summary(all_rows)
    print_summary_table(summary)

    csv_path  = os.path.join(out_dir, "results.csv")
    json_path = os.path.join(out_dir, "results.json")
    save_csv(all_rows, csv_path)
    save_json(all_rows, summary, json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="scenarios/real")
    parser.add_argument("--out",       default="evaluation/results")
    args = parser.parse_args()
    run(args.scenarios, args.out)
