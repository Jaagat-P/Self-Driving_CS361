"""
Multi-modal obstacle predictor.

Given a dynamic obstacle's current state, generates K predicted future
position trajectories using simple behavioral models.  These K futures
are the "scenarios" the ego planner must hedge against.

Prediction models
-----------------
0  constant_velocity   Keep current speed and heading.
1  braking             Decelerate at a fixed rate until stopped.
2  lane_change_left    Constant velocity + smooth lateral shift left.
3  lane_change_right   Constant velocity + smooth lateral shift right.

Output shape per obstacle:  (K, T, 2)
  K  – number of modes
  T  – prediction horizon steps
  2  – (x, y) position at each step

Usage
-----
    from planners.predictor import predict_obstacle, DEFAULT_MODES

    # obs_state: dict with keys "x", "y", "heading" [rad], "speed" [m/s]
    predictions = predict_obstacle(obs_state, T=50, dt=0.1)
    # predictions.shape == (4, 50, 2)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Prediction modes
# ---------------------------------------------------------------------------

@dataclass
class PredictionMode:
    name: str
    accel: float          # longitudinal acceleration [m/s²]
    lateral_speed: float  # peak lateral speed for lane-change modes [m/s]
    lateral_duration: float  # fraction of horizon over which to ramp lateral [0-1]


DEFAULT_MODES: list[PredictionMode] = [
    PredictionMode("constant_velocity",  accel=0.0,   lateral_speed=0.0, lateral_duration=0.0),
    PredictionMode("braking",            accel=-3.0,  lateral_speed=0.0, lateral_duration=0.0),
    PredictionMode("lane_change_left",   accel=0.0,   lateral_speed=1.5, lateral_duration=0.4),
    PredictionMode("lane_change_right",  accel=0.0,   lateral_speed=-1.5, lateral_duration=0.4),
]


# ---------------------------------------------------------------------------
# Single-mode prediction
# ---------------------------------------------------------------------------

def _predict_one_mode(
    x0: float,
    y0: float,
    heading: float,
    speed: float,
    mode: PredictionMode,
    T: int,
    dt: float,
) -> np.ndarray:
    """
    Predict T future (x, y) positions for one behavioral mode.

    Uses a point-mass model in the obstacle's local frame, then rotates
    back to world frame.  Heading is assumed constant (obstacles don't
    steer aggressively in short horizons).

    Returns
    -------
    np.ndarray of shape (T, 2)
    """
    positions = np.zeros((T, 2))

    # unit vectors along and perpendicular to heading
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    # lateral profile: half-sine ramp for lane-change modes
    # ramps up over lateral_duration * T steps, then stays constant
    lateral_profile = np.zeros(T)
    if mode.lateral_speed != 0.0:
        ramp_steps = max(1, int(mode.lateral_duration * T))
        # smooth sinusoidal ramp
        for t in range(ramp_steps):
            lateral_profile[t] = mode.lateral_speed * np.sin(np.pi * t / ramp_steps)
        lateral_profile[ramp_steps:] = 0.0  # completed lane change, moving parallel

    x, y = x0, y0
    v = speed

    for t in range(T):
        # update speed (clamp to zero – no reversing)
        v = max(0.0, v + mode.accel * dt)

        # longitudinal displacement in world frame
        dx_lon = v * cos_h * dt
        dy_lon = v * sin_h * dt

        # lateral displacement (perpendicular to heading)
        lat = lateral_profile[t] * dt
        dx_lat = -sin_h * lat   # rotate 90° CCW
        dy_lat =  cos_h * lat

        x += dx_lon + dx_lat
        y += dy_lon + dy_lat

        positions[t] = [x, y]

    return positions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_obstacle(
    obs_state: dict,
    T: int = 50,
    dt: float = 0.1,
    modes: list[PredictionMode] | None = None,
) -> np.ndarray:
    """
    Generate K predicted trajectories for a single obstacle.

    Parameters
    ----------
    obs_state : dict
        Must contain keys:
          "x"       – current x position [m]
          "y"       – current y position [m]
          "heading" – current heading angle [rad]
          "speed"   – current forward speed [m/s]
    T : int
        Number of future time steps to predict.
    dt : float
        Time step [s].
    modes : list[PredictionMode] or None
        Behavioral modes to use. Defaults to DEFAULT_MODES (K=4).

    Returns
    -------
    np.ndarray of shape (K, T, 2)
        K predicted position sequences, each of length T.
    """
    if modes is None:
        modes = DEFAULT_MODES

    K = len(modes)
    predictions = np.zeros((K, T, 2))

    for k, mode in enumerate(modes):
        predictions[k] = _predict_one_mode(
            x0=obs_state["x"],
            y0=obs_state["y"],
            heading=obs_state["heading"],
            speed=obs_state["speed"],
            mode=mode,
            T=T,
            dt=dt,
        )

    return predictions


def predict_all_obstacles(
    obstacles: list[dict],
    T: int = 50,
    dt: float = 0.1,
    modes: list[PredictionMode] | None = None,
) -> dict[int, np.ndarray]:
    """
    Generate K predicted trajectories for every obstacle in the scenario.

    Parameters
    ----------
    obstacles : list of dicts
        Each dict has keys: "id", "x", "y", "heading", "speed".
    T : int
        Prediction horizon steps.
    dt : float
        Time step [s].
    modes : list[PredictionMode] or None
        Behavioral modes; defaults to DEFAULT_MODES.

    Returns
    -------
    dict mapping obstacle_id (int) → np.ndarray of shape (K, T, 2)
    """
    return {
        obs["id"]: predict_obstacle(obs, T=T, dt=dt, modes=modes)
        for obs in obstacles
    }


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    obs = {"id": 1, "x": 0.0, "y": 0.0, "heading": 0.0, "speed": 8.0}
    preds = predict_obstacle(obs, T=50, dt=0.1)

    print(f"Predictions shape: {preds.shape}  (K={preds.shape[0]}, T={preds.shape[1]}, xy=2)")
    for i, mode in enumerate(DEFAULT_MODES):
        print(f"  Mode {i} ({mode.name:25s}): "
              f"final pos = ({preds[i, -1, 0]:.2f}, {preds[i, -1, 1]:.2f})")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["steelblue", "crimson", "darkorange", "green"]
    for i, mode in enumerate(DEFAULT_MODES):
        ax.plot(preds[i, :, 0], preds[i, :, 1], "-o", markersize=2,
                color=colors[i], label=mode.name)
        ax.plot(preds[i, 0, 0], preds[i, 0, 1], "o", markersize=8, color=colors[i])

    ax.plot(obs["x"], obs["y"], "k*", markersize=14, label="obstacle start", zorder=5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("K=4 Predicted Futures for One Obstacle")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
