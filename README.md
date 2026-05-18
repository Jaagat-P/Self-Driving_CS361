# Self-Driving CS361 – Environment Setup

## Prerequisites

- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- `conda` available on your `PATH`

## Create and activate the environment

Run this **once** from the repo root:

```bash
conda env create -f environment.yml
```

Then activate it every time you work on the project:

```bash
conda activate selfdriving-cs361
```

## Visualizations

`visualization/plot_scenario.py` loads a CommonRoad scenario XML and saves three PNG files to `scenarios/visualizations/<scenario_name>/` — no windows pop up.

| File | What it shows |
|---|---|
| `static.png` | Road network + obstacle starting positions at t=0, labeled with IDs |
| `trajectories.png` | Full time-horizon paths of each obstacle as colored lines over the road |
| `animation.png` | Grid of 8 snapshots at evenly-spaced time steps (poor man's animation) |

**How to run:**

```bash
conda activate selfdriving-cs361
cd visualization
python plot_scenario.py ../scenarios/USA_Lanker-1_1_T-1.xml
```

Replace the path with any other CommonRoad XML file in `scenarios/`.

---

## Updating the environment

If new packages are added to `environment.yml`, sync with:

```bash
conda env update -f environment.yml --prune
```
