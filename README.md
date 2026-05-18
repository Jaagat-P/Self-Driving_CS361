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

## Vehicle Dynamics – `rollout()`

`planners/vehicle_dynamics.py` implements the **kinematic bicycle model**: given a starting state and a sequence of controls, it simulates where the ego car ends up. Every planner in the project calls this.

**State:** `[x, y, ψ, v]` — position, heading angle, speed  
**Controls:** `[a, δ]` per time step — acceleration and front steering angle

**Main function:**

```python
from planners.vehicle_dynamics import VehicleState, rollout, states_to_array

s0 = VehicleState(x=10, y=5, psi=0.0, v=8.0)  # start state
controls = np.zeros((50, 2))                    # 50 steps × [accel, steer]

states = rollout(s0, controls, dt=0.1)          # returns list of 51 VehicleStates
arr    = states_to_array(states)                # (51, 4) numpy array: x, y, psi, v
```

`rollout` only moves the **ego vehicle** — obstacles are external. The optimizer proposes a `(T, 2)` control sequence, `rollout` turns it into a physical trajectory, and then the cost function checks that trajectory against the obstacle positions at each time step.

**How to test (no scenario file needed):**

```bash
cd planners
python vehicle_dynamics.py
```

Runs three built-in tests (straight line, left turn, acceleration from rest) and prints expected vs. actual values. -- All tests passed.

---

## Updating the environment

If new packages are added to `environment.yml`, sync with:

```bash
conda env update -f environment.yml --prune
```
