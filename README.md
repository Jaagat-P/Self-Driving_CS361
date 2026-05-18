# Self-Driving CS361 – Chance-Constrained Stochastic Trajectory Optimization

## System Architecture

Here is how all components interact at runtime:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          dataloader.py                              │
│                                                                     │
│  load_scenarios("scenarios/")                                       │
│    ├── reads CommonRoad XML files                                   │
│    ├── extracts ego_initial_state  ──────────────────────────────┐  │
│    ├── extracts lanelet_network  (for lane-deviation cost d_t)   │  │
│    └── calls predictor.py ──────────────────────────────────┐    │  │
│                                                             │    │  │
│         returns ScenarioData { obstacle_predictions,        │    │  │
│                                ego_initial_state,           │    │  │
│                                lanelet_network, dt, T }     │    │  │
└─────────────────────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────┐
│        predictor.py          │
│                              │
│  For each obstacle, produce  │
│  K predicted futures:        │
│   k=0  constant velocity     │
│   k=1  braking               │
│   k=2  lane change left      │
│   k=3  lane change right     │
│                              │
│  output: (K, T, 2) per obs   │
└──────────────────────────────┘
                   │
                   │  ScenarioData flows into optimizer
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         optimizer  (coming soon)                    │
│                                                                     │
│  decision variable:  u  of shape (T, 2)  = [accel, steer] per step │
│                                                                     │
│  each iteration:                                                    │
│    1. vehicle_dynamics.rollout(ego_initial_state, u)                │
│          → ego trajectory  (T+1, 4)   [x, y, ψ, v]                 │
│                                                                     │
│    2. cost_functions.compute(ego_trajectory,                        │
│                              obstacle_predictions,                  │
│                              lanelet_network)                       │
│          → J(u) = (1/K) Σ_k Σ_t ( a_t² + κ·j_t² + ρ·d_t² )       │
│                                                                     │
│    3. collision_check(ego_trajectory, obstacle_predictions)         │
│          → chance constraint: (1/K) Σ_k 1(collision_k) ≤ ε        │
│                                                                     │
│    4. update u  →  repeat until converged                           │
│                                                                     │
│  Three planner variants run the same loop with different inputs:    │
│    plan-on-mean   →  K=1, use mean obstacle position per step       │
│    worst-case     →  K=1, use most dangerous obstacle position      │
│    stochastic     →  K=4, all predicted futures (ours)              │
│                                                                     │
│  Two optimizer backends:                                            │
│    SQP    (scipy.optimize, trust-region, needs smooth gradients)    │
│    CMA-ES (gradient-free, handles discontinuous collision cost)     │
└─────────────────────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────┐
│        evaluation            │
│                              │
│  per scenario, per planner:  │
│  - collision rate            │
│  - completion rate           │
│  - expected cost J(u*)       │
│  - runtime                   │
│  - min obstacle distance     │
└──────────────────────────────┘
```

---

## Environment Setup

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

**How to test (no scenario file needed, run from repo root):**

```bash
conda activate selfdriving-cs361
python -m planners.vehicle_dynamics
```

Runs three built-in tests (straight line, left turn, acceleration from rest) and prints expected vs. actual values.

---

## Updating the environment

If new packages are added to `environment.yml`, sync with:

```bash
conda env update -f environment.yml --prune
```
