"""
Evaluation harness for comparing trajectory planners.

Runs plan-on-mean and worst-case planners (SQP + CMA-ES backends) on all
scenarios and reports metrics:
  - cost J(u*)
  - min obstacle distance
  - collision-free (against collapsed predictions)
  - collision rate (post-hoc against ALL K=4 prediction modes)
  - runtime
  - convergence
"""

from __future__ import annotations

import csv
import os
import sys
import numpy as np

from planners.dataloader import load_scenarios, ScenarioData
from planners.cost_functions import min_obstacle_distance
from planners.planner_result import PlannerResult
from planners.plan_on_mean import plan_on_mean_sqp, plan_on_mean_cmaes
from planners.worst_case import worst_case_sqp, worst_case_cmaes


SAFE_RADIUS = 3.0


def posthoc_collision_rate(result: PlannerResult,
                           obstacle_predictions: dict[int, np.ndarray],
                           safe_radius: float = SAFE_RADIUS) -> float:
    """
    Evaluate the optimized trajectory against ALL K prediction modes.
    Returns the fraction of modes that contain at least one collision.
    """
    if not obstacle_predictions:
        return 0.0

    first_preds = next(iter(obstacle_predictions.values()))
    K = first_preds.shape[0]
    T = first_preds.shape[1]

    collisions = 0
    for k in range(K):
        mode_positions = {
            obs_id: preds[k]
            for obs_id, preds in obstacle_predictions.items()
        }
        md = min_obstacle_distance(result.ego_trajectory, mode_positions)
        if md < safe_radius:
            collisions += 1

    return collisions / K


def run_evaluation(scenarios_dir: str,
                   output_csv: str | None = None) -> list[dict]:
    dataset = load_scenarios(scenarios_dir, verbose=True)
    if not dataset:
        print("No scenarios found.")
        return []

    planners = [
        ("plan_on_mean_sqp", plan_on_mean_sqp),
        ("plan_on_mean_cmaes", plan_on_mean_cmaes),
        ("worst_case_sqp", worst_case_sqp),
        ("worst_case_cmaes", worst_case_cmaes),
    ]

    rows = []
    for sd in dataset:
        print(f"\n{'='*60}")
        print(f"Scenario: {sd.scenario_id}  "
              f"(T={sd.T}, obstacles={sd.num_obstacles})")
        print(f"{'='*60}")

        for name, planner_fn in planners:
            print(f"\n  Running {name}...", end=" ", flush=True)
            try:
                result = planner_fn(sd)
                coll_rate = posthoc_collision_rate(
                    result, sd.obstacle_predictions)

                row = {
                    "scenario": sd.scenario_id,
                    "planner": name,
                    "cost": f"{result.cost:.4f}",
                    "min_dist": f"{result.min_obstacle_distance:.2f}",
                    "collision_free": result.collision_free,
                    "collision_rate_allK": f"{coll_rate:.2%}",
                    "converged": result.converged,
                    "runtime_s": f"{result.runtime_seconds:.2f}",
                }
                rows.append(row)

                print(f"done ({result.runtime_seconds:.1f}s)")
                print(f"    Cost={result.cost:.4f}  "
                      f"MinDist={result.min_obstacle_distance:.2f}m  "
                      f"CollFree={result.collision_free}  "
                      f"CollRate(K=4)={coll_rate:.0%}  "
                      f"Converged={result.converged}")

            except Exception as e:
                print(f"FAILED: {e}")
                rows.append({
                    "scenario": sd.scenario_id,
                    "planner": name,
                    "cost": "ERROR",
                    "min_dist": "ERROR",
                    "collision_free": "ERROR",
                    "collision_rate_allK": "ERROR",
                    "converged": False,
                    "runtime_s": "ERROR",
                })

    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'Scenario':<30} {'Planner':<22} {'Cost':>10} {'MinDist':>8} "
    header += f"{'CollFree':>9} {'CollRate':>9} {'Conv':>5} {'Time':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = (f"{r['scenario']:<30} {r['planner']:<22} "
                f"{r['cost']:>10} {r['min_dist']:>8} "
                f"{str(r['collision_free']):>9} "
                f"{r['collision_rate_allK']:>9} "
                f"{str(r['converged']):>5} {r['runtime_s']:>7}")
        print(line)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {output_csv}")

    return rows


if __name__ == "__main__":
    scenarios_dir = sys.argv[1] if len(sys.argv) > 1 else "scenarios"
    run_evaluation(scenarios_dir, output_csv="evaluation/results.csv")
