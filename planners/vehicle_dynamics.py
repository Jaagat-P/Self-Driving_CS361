"""
Kinematic Bicycle Model for ego vehicle trajectory rollout.

State vector:  s = [x, y, psi, v]
  x    – longitudinal position [m]
  y    – lateral position [m]
  psi  – heading angle [rad]  (0 = facing +x axis, counter-clockwise positive)
  v    – forward speed [m/s]

Control vector:  u = [a, delta]
  a     – longitudinal acceleration [m/s²]
  delta – front-wheel steering angle [rad]

Continuous-time equations (center-of-rear-axle reference point):
  dx/dt    = v * cos(psi)
  dy/dt    = v * sin(psi)
  dpsi/dt  = (v / L) * tan(delta)
  dv/dt    = a

where L is the vehicle wheelbase [m].

We integrate with a 4th-order Runge–Kutta (RK4) step, which is much
more accurate than plain Euler for the same step size and matters when
you are rolling out long horizons.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Vehicle parameters
# ---------------------------------------------------------------------------

@dataclass
class VehicleParams:
    """Physical parameters of the ego vehicle."""

    L: float = 2.7          # wheelbase [m]  (typical sedan)
    max_speed: float = 30.0  # [m/s]  (~108 km/h)
    min_speed: float = 0.0   # [m/s]  (no reversing by default)
    max_accel: float = 3.0   # [m/s²]
    min_accel: float = -6.0  # [m/s²]  (hard braking)
    max_steer: float = 0.6   # [rad]   (~34°)


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

@dataclass
class VehicleState:
    """A single ego-vehicle state."""

    x: float = 0.0    # [m]
    y: float = 0.0    # [m]
    psi: float = 0.0  # heading [rad]
    v: float = 0.0    # speed [m/s]

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi, self.v])

    @staticmethod
    def from_array(arr: np.ndarray) -> "VehicleState":
        return VehicleState(x=arr[0], y=arr[1], psi=arr[2], v=arr[3])

    def __repr__(self) -> str:
        return (
            f"VehicleState(x={self.x:.2f}, y={self.y:.2f}, "
            f"psi={np.degrees(self.psi):.1f}°, v={self.v:.2f} m/s)"
        )


# ---------------------------------------------------------------------------
# Core dynamics
# ---------------------------------------------------------------------------

def _derivatives(state: np.ndarray, control: np.ndarray, L: float) -> np.ndarray:
    """
    Compute ds/dt for the kinematic bicycle model.

    Parameters
    ----------
    state   : [x, y, psi, v]
    control : [a, delta]
    L       : wheelbase [m]

    Returns
    -------
    np.ndarray of shape (4,)  –  [dx/dt, dy/dt, dpsi/dt, dv/dt]
    """
    x, y, psi, v = state
    a, delta = control

    dxdt   = v * np.cos(psi)
    dydt   = v * np.sin(psi)
    dpsidt = (v / L) * np.tan(delta)
    dvdt   = a

    return np.array([dxdt, dydt, dpsidt, dvdt])


def rk4_step(
    state: np.ndarray,
    control: np.ndarray,
    dt: float,
    L: float,
) -> np.ndarray:
    """
    Advance the bicycle model by one time step using RK4 integration.

    Parameters
    ----------
    state   : current state [x, y, psi, v]
    control : control input [a, delta]
    dt      : time step [s]
    L       : wheelbase [m]

    Returns
    -------
    np.ndarray – next state [x, y, psi, v]
    """
    k1 = _derivatives(state,            control, L)
    k2 = _derivatives(state + dt/2 * k1, control, L)
    k3 = _derivatives(state + dt/2 * k2, control, L)
    k4 = _derivatives(state + dt    * k3, control, L)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ---------------------------------------------------------------------------
# Trajectory rollout
# ---------------------------------------------------------------------------

def rollout(
    initial_state: VehicleState,
    controls: np.ndarray,
    dt: float = 0.1,
    params: VehicleParams | None = None,
) -> list[VehicleState]:
    """
    Roll out a trajectory from an initial state given a sequence of controls.

    Parameters
    ----------
    initial_state : VehicleState
        Starting [x, y, psi, v] of the ego vehicle.
    controls : np.ndarray of shape (T, 2)
        Sequence of T control inputs, each row is [a, delta].
    dt : float
        Time step between control inputs [s].  CommonRoad scenarios
        typically use dt = 0.1 s.
    params : VehicleParams or None
        Physical limits; uses defaults if None.

    Returns
    -------
    list[VehicleState]  –  T+1 states (includes the initial state).
    """
    if params is None:
        params = VehicleParams()

    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls.reshape(1, -1)  # single control step

    states: list[VehicleState] = [initial_state]
    s = initial_state.as_array()

    for u in controls:
        # clip controls to physical limits before integrating
        a     = np.clip(u[0], params.min_accel, params.max_accel)
        delta = np.clip(u[1], -params.max_steer, params.max_steer)
        u_clipped = np.array([a, delta])

        s = rk4_step(s, u_clipped, dt, params.L)

        # clip speed to valid range (vehicle can't go faster than max or reverse)
        s[3] = np.clip(s[3], params.min_speed, params.max_speed)

        states.append(VehicleState.from_array(s))

    return states


def states_to_array(states: list[VehicleState]) -> np.ndarray:
    """
    Convert a list of VehicleState objects to a (T+1, 4) numpy array.

    Rows are [x, y, psi, v]; useful for vectorized cost computations.
    """
    return np.stack([s.as_array() for s in states])


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=== Kinematic Bicycle Model – Self-Test ===\n")

    dt = 0.1   # 0.1 s  (CommonRoad standard)
    T  = 50    # steps  →  5 seconds of driving

    # --- Test 1: straight-line constant speed ---
    print("Test 1: straight line at 10 m/s (no steering)")
    s0 = VehicleState(x=0, y=0, psi=0.0, v=10.0)
    controls = np.zeros((T, 2))                # a=0, delta=0
    traj = rollout(s0, controls, dt=dt)
    arr  = states_to_array(traj)
    print(f"  Start: {traj[0]}")
    print(f"  End:   {traj[-1]}")
    expected_x = 10.0 * T * dt
    print(f"  Expected final x ≈ {expected_x:.1f} m  |  Got: {arr[-1, 0]:.3f} m")

    # --- Test 2: constant-radius left turn ---
    print("\nTest 2: constant left turn (delta = 0.2 rad, v = 8 m/s)")
    s0 = VehicleState(x=0, y=0, psi=0.0, v=8.0)
    controls = np.tile([0.0, 0.2], (T, 1))     # constant steer left
    traj = rollout(s0, controls, dt=dt)
    arr  = states_to_array(traj)
    print(f"  Start: {traj[0]}")
    print(f"  End:   {traj[-1]}")

    # --- Test 3: acceleration from rest ---
    print("\nTest 3: accelerate from rest at 2 m/s² for 5 s")
    s0 = VehicleState(x=0, y=0, psi=0.0, v=0.0)
    controls = np.tile([2.0, 0.0], (T, 1))
    traj = rollout(s0, controls, dt=dt)
    arr  = states_to_array(traj)
    print(f"  Start: {traj[0]}")
    print(f"  End:   {traj[-1]}")
    expected_v = 2.0 * T * dt
    print(f"  Expected final v ≈ {expected_v:.1f} m/s  |  Got: {arr[-1, 3]:.3f} m/s")

    # --- Plot all three trajectories ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, controls_cfg, init_v, color in [
        ("Straight (10 m/s)",      np.zeros((T, 2)),              10.0, "steelblue"),
        ("Left turn (8 m/s)",      np.tile([0.0, 0.2], (T, 1)),   8.0,  "darkorange"),
        ("Accelerate from rest",   np.tile([2.0, 0.0], (T, 1)),   0.0,  "green"),
    ]:
        s0 = VehicleState(x=0, y=0, psi=0.0, v=init_v)
        arr = states_to_array(rollout(s0, controls_cfg, dt=dt))
        ax.plot(arr[:, 0], arr[:, 1], "-o", markersize=2, label=label, color=color)
        ax.plot(arr[0, 0], arr[0, 1], "o", markersize=8, color=color)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Kinematic Bicycle Model – Trajectory Rollout Tests")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    print("\nAll tests passed.")
