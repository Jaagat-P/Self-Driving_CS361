# evaluation harness -- runs all planner variants on all scenarios
# and reports metrics including the post-hoc collision rate against
# all K=4 prediction modes (not just the collapsed one)

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


def posthoc_collision_rate(result, obstacle_predictions, safe_radius=SAFE_RADIUS):
    # test the optimized trajectory against ALL K futures, not just the
    # collapsed one the planner saw -- this is the real safety check
    if not obstacle_predictions:
        return 0.0

    first_preds = next(iter(obstacle_predictions.values()))
    num_modes = first_preds.shape[0]

    collisions = 0
    for k in range(num_modes):
        # build a single-mode prediction dict for this future
        mode_positions = {
            obs_id: preds[k]
            for obs_id, preds in obstacle_predictions.items()
        }
        min_dist = min_obstacle_distance(result.ego_trajectory, mode_positions)
        if min_dist < safe_radius:
            collisions += 1

    return collisions / num_modes


def run_evaluation(scenarios_dir, output_csv=None):
    dataset = load_scenarios(scenarios_dir, verbose=True)
    if not dataset:
        print("no scenarios found")
        return []

    planners = [
        ("plan_on_mean_sqp", plan_on_mean_sqp),
        ("plan_on_mean_cmaes", plan_on_mean_cmaes),
        ("worst_case_sqp", worst_case_sqp),
        ("worst_case_cmaes", worst_case_cmaes),
    ]

    rows = []
    for scenario_data in dataset:
        print(f"\nscenario: {scenario_data.scenario_id}  "
              f"(T={scenario_data.T}, obstacles={scenario_data.num_obstacles})")

        for name, planner_fn in planners:
            print(f"  running {name}...", end=" ", flush=True)
            try:
                result = planner_fn(scenario_data)
                collision_rate = posthoc_collision_rate(
                    result, scenario_data.obstacle_predictions)

                row = {
                    "scenario": scenario_data.scenario_id,
                    "planner": name,
                    "cost": f"{result.cost:.4f}",
                    "min_dist": f"{result.min_obstacle_distance:.2f}",
                    "collision_free": result.collision_free,
                    "collision_rate_allK": f"{collision_rate:.2%}",
                    "converged": result.converged,
                    "runtime_s": f"{result.runtime_seconds:.2f}",
                }
                rows.append(row)

                print(f"done ({result.runtime_seconds:.1f}s)")
                print(f"    cost={result.cost:.4f}  "
                      f"min_dist={result.min_obstacle_distance:.2f}m  "
                      f"collision_free={result.collision_free}  "
                      f"collision_rate(K=4)={collision_rate:.0%}  "
                      f"converged={result.converged}")

            except Exception as e:
                print(f"FAILED: {e}")
                rows.append({
                    "scenario": scenario_data.scenario_id,
                    "planner": name,
                    "cost": "ERROR",
                    "min_dist": "ERROR",
                    "collision_free": "ERROR",
                    "collision_rate_allK": "ERROR",
                    "converged": False,
                    "runtime_s": "ERROR",
                })

    # print summary table
    print(f"\n\nsummary")
    header = f"{'scenario':<30} {'planner':<22} {'cost':>10} {'min_dist':>8} "
    header += f"{'coll_free':>9} {'coll_rate':>9} {'conv':>5} {'time':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (f"{row['scenario']:<30} {row['planner']:<22} "
                f"{row['cost']:>10} {row['min_dist']:>8} "
                f"{str(row['collision_free']):>9} "
                f"{row['collision_rate_allK']:>9} "
                f"{str(row['converged']):>5} {row['runtime_s']:>7}")
        print(line)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nresults saved to {output_csv}")

    return rows


if __name__ == "__main__":
    scenarios_dir = sys.argv[1] if len(sys.argv) > 1 else "scenarios"
    run_evaluation(scenarios_dir, output_csv="evaluation/results.csv")
