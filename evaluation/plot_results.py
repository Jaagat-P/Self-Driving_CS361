"""
Publication-quality comparison figures from evaluation results.

Reads evaluation/results/results.json and produces:

  Fig 1 – Main comparison bar chart  (2×2 grid)
          collision rate | completion rate | expected cost | runtime

  Fig 2 – Safety vs. efficiency scatter (collision rate vs. expected cost)

  Fig 3 – Per-scenario collision-rate heatmap (planners × scenarios)

  Fig 4 – Min obstacle distance box plots

All figures are saved as PDF + PNG to evaluation/results/figures/.

Usage:
    python -m evaluation.plot_results
    python -m evaluation.plot_results --results evaluation/results/results.json
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import numpy as np


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PLANNER_DISPLAY = {
    "plan_on_mean_sqp":   "Mean\n(SQP)",
    "plan_on_mean_cmaes": "Mean\n(CMA-ES)",
    "worst_case_sqp":     "Worst\n(SQP)",
    "worst_case_cmaes":   "Worst\n(CMA-ES)",
    "stochastic_sqp":     "Stochastic\n(SQP)",
    "stochastic_cmaes":   "Stochastic\n(CMA-ES)",
}

PLANNER_COLORS = {
    "plan_on_mean_sqp":   "#4C72B0",
    "plan_on_mean_cmaes": "#74A0C7",
    "worst_case_sqp":     "#DD8452",
    "worst_case_cmaes":   "#F0A97A",
    "stochastic_sqp":     "#55A868",
    "stochastic_cmaes":   "#8ACC9C",
}

PLANNER_ORDER = list(PLANNER_DISPLAY.keys())

def _short_id(scenario_id: str) -> str:
    """Shorten a CommonRoad scenario ID to a compact label for plots."""
    # e.g. "USA_Lanker-1_3_T-1" -> "Lnk-1_3"
    #      "USA_US101-4_1_T-1"  -> "101-4"
    sid = scenario_id.replace("_T-1", "").replace("USA_", "").replace("DEU_", "")
    sid = sid.replace("Lanker-1_", "Lnk-1/").replace("US101-", "101-")
    return sid

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi":      150,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(json_path: str) -> tuple[list[dict], dict]:
    with open(json_path) as f:
        payload = json.load(f)
    return payload["raw_rows"], payload["summary"]


def planner_values(rows: list[dict], planner: str, field: str) -> list[float]:
    return [r[field] for r in rows
            if r["planner"] == planner and r[field] is not None]


def scenario_ids(rows: list[dict]) -> list[str]:
    seen, out = set(), []
    for r in rows:
        if r["scenario_id"] not in seen:
            seen.add(r["scenario_id"])
            out.append(r["scenario_id"])
    return sorted(out, key=lambda s: int(s.split("-")[-1]))


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _bar(ax, means, stds, labels, colors, ylabel, title, ylim=None, pct=False):
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.6,
                  color=colors, edgecolor="white", linewidth=0.5,
                  error_kw=dict(elinewidth=1.0, ecolor="#333"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    if ylim is not None:
        ax.set_ylim(ylim)
    if pct:
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    return bars


# ---------------------------------------------------------------------------
# Figure 1: Main 2×2 bar chart
# ---------------------------------------------------------------------------

def fig_main_comparison(rows, summary, out_dir):
    planners = PLANNER_ORDER
    labels   = [PLANNER_DISPLAY[p] for p in planners]
    colors   = [PLANNER_COLORS[p]  for p in planners]

    def _mv(field):
        means = [summary[p][field]["mean"] or 0 for p in planners]
        stds  = [summary[p][field]["std"]  or 0 for p in planners]
        return means, stds

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("Planner Comparison across 20 Scenarios", fontsize=13, y=1.01)

    m, s = _mv("collision_rate")
    _bar(axes[0, 0], m, s, labels, colors,
         "Collision Rate", "(a) Post-hoc Collision Rate ↓",
         ylim=(0, 1.05), pct=True)

    m, s = _mv("completion_rate")
    _bar(axes[0, 1], m, s, labels, colors,
         "Completion Rate", "(b) Completion Rate (converged & safe) ↑",
         ylim=(0, 1.05), pct=True)

    m, s = _mv("expected_cost")
    _bar(axes[1, 0], m, s, labels, colors,
         "Expected Cost  J(u*)", "(c) Expected Cost ↓")

    m, s = _mv("runtime_s")
    _bar(axes[1, 1], m, s, labels, colors,
         "Runtime [s]", "(d) Optimisation Runtime ↓")

    plt.tight_layout()
    _save(fig, out_dir, "fig1_main_comparison")


# ---------------------------------------------------------------------------
# Figure 2: Safety–efficiency scatter (collision rate vs. expected cost)
# ---------------------------------------------------------------------------

def fig_safety_efficiency(rows, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))

    handles = []
    for planner in PLANNER_ORDER:
        cr   = planner_values(rows, planner, "collision_rate")
        cost = planner_values(rows, planner, "expected_cost")
        if not cr:
            continue
        sc = ax.scatter(cost, cr,
                        color=PLANNER_COLORS[planner],
                        s=45, alpha=0.75, zorder=3,
                        label=PLANNER_DISPLAY[planner].replace("\n", " "))

        # show mean as a large filled marker with black edge
        ax.scatter(np.mean(cost), np.mean(cr),
                   color=PLANNER_COLORS[planner],
                   s=180, marker="D", edgecolors="black", linewidths=1.0,
                   zorder=4)
        handles.append(mpatches.Patch(color=PLANNER_COLORS[planner],
                                       label=PLANNER_DISPLAY[planner].replace("\n", " ")))

    ax.set_xlabel("Expected Cost  J(u*)")
    ax.set_ylabel("Post-hoc Collision Rate")
    ax.set_title("Safety–Efficiency Trade-off\n(diamonds = per-planner mean)", pad=8)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)
    plt.tight_layout()
    _save(fig, out_dir, "fig2_safety_efficiency")


# ---------------------------------------------------------------------------
# Figure 3: Collision-rate heatmap (planners × scenarios)
# ---------------------------------------------------------------------------

def fig_heatmap(rows, out_dir):
    scen_ids = scenario_ids(rows)
    nS = len(scen_ids)
    nP = len(PLANNER_ORDER)

    mat = np.full((nP, nS), np.nan)
    for i, planner in enumerate(PLANNER_ORDER):
        for j, sid in enumerate(scen_ids):
            vals = [r["collision_rate"] for r in rows
                    if r["planner"] == planner
                    and r["scenario_id"] == sid
                    and r["collision_rate"] is not None]
            if vals:
                mat[i, j] = vals[0]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1,
                   cmap="RdYlGn_r", interpolation="nearest")

    ax.set_xticks(range(nS))
    ax.set_xticklabels([_short_id(s) for s in scen_ids],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(nP))
    ax.set_yticklabels([PLANNER_DISPLAY[p].replace("\n", " ")
                        for p in PLANNER_ORDER], fontsize=9)

    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label("Collision Rate", fontsize=9)
    cb.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # annotate each cell
    for i in range(nP):
        for j in range(nS):
            if not np.isnan(mat[i, j]):
                val = mat[i, j]
                txt_color = "white" if val > 0.6 or val < 0.15 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=6.5, color=txt_color)

    ax.set_title("Post-hoc Collision Rate per Scenario × Planner", pad=8)
    plt.tight_layout()
    _save(fig, out_dir, "fig3_collision_heatmap")


# ---------------------------------------------------------------------------
# Figure 4: Min obstacle distance box plots
# ---------------------------------------------------------------------------

def fig_min_distance(rows, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    data    = [planner_values(rows, p, "min_obstacle_dist_m") for p in PLANNER_ORDER]
    labels  = [PLANNER_DISPLAY[p].replace("\n", " ") for p in PLANNER_ORDER]
    colors  = [PLANNER_COLORS[p] for p in PLANNER_ORDER]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.axhline(3.0, color="red", linestyle="--", linewidth=1.0,
               label="Safe radius (3 m)")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Min. Distance to Obstacle [m]")
    ax.set_title("Minimum Obstacle Clearance across 20 Scenarios ↑", pad=8)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, out_dir, "fig4_min_distance")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
    print(f"  saved: {name}.pdf / .png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="evaluation/results/results.json")
    parser.add_argument("--out",     default="evaluation/results/figures")
    args = parser.parse_args()

    print(f"Loading results from {args.results}")
    rows, summary = load_data(args.results)
    print(f"  {len(rows)} rows  ({len({r['scenario_id'] for r in rows})} scenarios, "
          f"{len({r['planner'] for r in rows})} planners)\n")

    print("Generating figures …")
    fig_main_comparison(rows, summary, args.out)
    fig_safety_efficiency(rows, args.out)
    fig_heatmap(rows, args.out)
    fig_min_distance(rows, args.out)
    print(f"\nAll figures saved to: {os.path.abspath(args.out)}")
