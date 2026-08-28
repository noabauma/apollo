#!/usr/bin/env python3
"""
Playable Earth-Moon spacecraft simulator (2D, restricted two-fixed-body problem).

Earth and Moon are held stationary at their real mean separation; the
spacecraft is integrated in their combined gravity field with a selectable
fixed-step integrator (forward Euler, leapfrog, RK4).

Everything you are meant to play with lives in the PLAYABLE SETTINGS block.
Run with:  python3 orbit_sim.py
"""

import time
from collections import deque

import numpy as np
# matplotlib is imported inside run_visual() so the headless mode
# (SAVE_TO = None) never touches the plotting machinery at all.

# ============================ PHYSICAL CONSTANTS (SI) ========================
G       = 6.67430e-11    # gravitational constant [m^3 kg^-1 s^-2]
M_EARTH = 5.9722e24      # Earth mass  [kg]
M_MOON  = 7.342e22       # Moon mass   [kg]
R_EARTH = 6.371e6        # Earth mean radius [m]  (for drawing + crash check)
R_MOON  = 1.7374e6       # Moon mean radius  [m]
D_MOON  = 3.844e8        # mean Earth-Moon center-to-center distance [m]

EARTH_POS = np.array([0.0, 0.0])       # Earth pinned at the origin
MOON_POS  = np.array([D_MOON, 0.0])    # Moon pinned on the +x axis

# ============================ PLAYABLE SETTINGS ==============================
# Apollo 8 flew with no lunar lander (the LM wasn't ready) — just the Command
# Module + Service Module, the "return vehicle". Total spacecraft: 28,817 kg.
# NOTE: in pure gravity the trajectory is independent of the craft's mass
# (it cancels out of F = m*a). It's kept here so you can play with it, and it
# will matter the moment you add thrust or drag.
M_CRAFT = 28_817.0                     # [kg]

# TMCMC winner (run 5, population 2000, best logL = -9.78): burn at angle
# THETA on the parking orbit, prograde/CCW — same parameterization as
# apollo_model.py. Flies a 114 km lunar flyby at t = 146.7 h, arriving only
# 610 m/s above local lunar circular speed (the no-burn optimum the sampler
# found), and is still flying at 200 h (no free return within the window).
ALTITUDE    = 433.483e3                          # [m] above Earth's surface
THETA       = 3.262417                           # [rad] TLI burn position angle
EXTRA_SPEED = 3061.588                           # [m/s] on top of circular speed
r0          = R_EARTH + ALTITUDE                 # orbit radius from Earth center
v_circular  = np.sqrt(G * M_EARTH / r0)          # ~7654 m/s here

START_POS = r0 * np.array([np.cos(THETA), np.sin(THETA)])
START_VEL = (v_circular + EXTRA_SPEED) * np.array([-np.sin(THETA), np.cos(THETA)])

# The historical Apollo 8-style free return (185.2 km parking orbit):
#   ALTITUDE = 185.2e3; THETA = np.pi; EXTRA_SPEED = 10_920.0 - v_circular
# Circular parking orbit: EXTRA_SPEED = 0.0

INTEGRATOR = "leapfrog"    # "euler" | "leapfrog" | "rk4"
DT              = 10.0     # integration time step [s] (accuracy knob)
STEPS_PER_FRAME = 60       # physics steps per animation frame (speed knob)
VIEW            = "full"   # "earth" | "moon" | "full"
TRAIL_POINTS    = 80_000   # how many past positions to keep drawn

# How the simulation runs:
#   SAVE_TO = None        -> HEADLESS: no plotting at all, fastest. Runs
#                            SIM_HOURS of physics, records every position and
#                            velocity, and writes them to trajectory.npz.
#   SAVE_TO = "window"    -> live interactive window (runs until closed)
#   SAVE_TO = "file.mp4"  -> render exactly SIM_HOURS of simulated time to video
SAVE_TO   = "apollo_tmcmc_winner.mp4"
SIM_HOURS = 200.0          # simulated time for headless + video modes [hours]

# ============================ PHYSICS ========================================

def acceleration(pos):
    """Gravitational acceleration [m/s^2] at position `pos` from Earth + Moon."""
    a = np.zeros(2)
    for body_pos, mass in ((EARTH_POS, M_EARTH), (MOON_POS, M_MOON)):
        d = body_pos - pos
        r = np.linalg.norm(d)
        a += G * mass * d / r**3
    return a


def euler_step(pos, vel, dt):
    """Forward (explicit) Euler. Simple but gains energy -> orbits spiral out."""
    a = acceleration(pos)
    return pos + vel * dt, vel + a * dt


def leapfrog_step(pos, vel, dt):
    """Kick-drift-kick leapfrog (velocity Verlet). Symplectic: energy stays
    bounded, so closed orbits stay closed. Same cost class as Euler."""
    vel_half = vel + 0.5 * dt * acceleration(pos)
    pos_new  = pos + dt * vel_half
    vel_new  = vel_half + 0.5 * dt * acceleration(pos_new)
    return pos_new, vel_new


def rk4_step(pos, vel, dt):
    """Classic 4th-order Runge-Kutta. Very accurate per step, not symplectic."""
    def deriv(p, v):
        return v, acceleration(p)

    k1p, k1v = deriv(pos,                 vel)
    k2p, k2v = deriv(pos + 0.5*dt*k1p,    vel + 0.5*dt*k1v)
    k3p, k3v = deriv(pos + 0.5*dt*k2p,    vel + 0.5*dt*k2v)
    k4p, k4v = deriv(pos + dt*k3p,        vel + dt*k3v)
    pos_new = pos + (dt/6.0) * (k1p + 2*k2p + 2*k3p + k4p)
    vel_new = vel + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    return pos_new, vel_new


INTEGRATORS = {
    "euler":    euler_step,
    "leapfrog": leapfrog_step,
    "rk4":      rk4_step,
}


def specific_energy(pos, vel):
    """Orbital energy per unit craft mass [J/kg] — conserved in the real system,
    so its drift is a direct readout of the integrator's error."""
    r_e = np.linalg.norm(pos - EARTH_POS)
    r_m = np.linalg.norm(pos - MOON_POS)
    return 0.5 * vel @ vel - G * M_EARTH / r_e - G * M_MOON / r_m


def crashed(pos):
    """Return the body name if the craft is below a surface, else None."""
    if np.linalg.norm(pos - EARTH_POS) < R_EARTH:
        return "Earth"
    if np.linalg.norm(pos - MOON_POS) < R_MOON:
        return "Moon"
    return None

# ============================ HEADLESS SIMULATION ============================

def simulate(hours=None):
    """Run the physics with no graphics, recording the full trajectory.

    Returns (t, pos, vel, fate): times [s] with shape (n,), positions [m] and
    velocities [m/s] with shape (n, 2) — sample 0 is the starting state — and
    a string describing how the run ended. Arrays are preallocated up front,
    so every single integration step is kept.
    """
    if hours is None:
        hours = SIM_HOURS
    step = INTEGRATORS[INTEGRATOR]
    n = int(hours * 3600.0 / DT)

    t   = np.empty(n + 1)
    pos = np.empty((n + 1, 2))
    vel = np.empty((n + 1, 2))
    t[0], pos[0], vel[0] = 0.0, START_POS, START_VEL

    p = START_POS.astype(float).copy()
    v = START_VEL.astype(float).copy()
    fate = f"completed {hours:g} h of flight"
    k = 0
    for k in range(1, n + 1):
        p, v = step(p, v, DT)
        t[k], pos[k], vel[k] = k * DT, p, v
        hit = crashed(p)
        if hit:
            fate = f"crashed into the {hit} after {k * DT / 3600.0:.2f} h"
            break
    return t[:k + 1], pos[:k + 1], vel[:k + 1], fate


def run_headless():
    """Simulate without any plotting, print a mission report, save the data."""
    wall_start = time.perf_counter()
    t, pos, vel, fate = simulate()
    wall = time.perf_counter() - wall_start

    speed  = np.linalg.norm(vel, axis=1)
    r_e    = np.linalg.norm(pos - EARTH_POS, axis=1)
    d_moon = np.linalg.norm(pos - MOON_POS, axis=1)
    E = 0.5 * speed**2 - G * M_EARTH / r_e - G * M_MOON / d_moon
    i_close = int(np.argmin(d_moon))

    np.savez("trajectory.npz", t=t, pos=pos, vel=vel)

    n_steps = len(t) - 1
    print(f"Fate            : {fate}")
    print(f"Steps           : {n_steps:,} x {DT:g} s "
          f"in {wall:.2f} s wall time ({n_steps / wall:,.0f} steps/s)")
    print(f"Closest to Moon : {(d_moon[i_close] - R_MOON) / 1e3:,.0f} km above "
          f"the surface at t={t[i_close] / 3600.0:.1f} h")
    print(f"Speed range     : {speed.min() / 1e3:.3f} - {speed.max() / 1e3:.3f} km/s")
    print(f"Energy drift    : {abs((E[-1] - E[0]) / E[0]) * 100.0:.4f} %")
    print(f"Saved           : trajectory.npz "
          f"(load with: d = np.load('trajectory.npz'); d['t'], d['pos'], d['vel'])")
    return t, pos, vel

# ============================ VISUALIZATION ==================================

def main():
    if SAVE_TO is None:
        return run_headless()
    return run_visual()


def run_visual():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle

    step = INTEGRATORS[INTEGRATOR]

    # Simulation state (kept in a dict so the animation callback can mutate it)
    state = {
        "pos": START_POS.astype(float).copy(),
        "vel": START_VEL.astype(float).copy(),
        "t": 0.0,
        "done": False,
    }
    E0 = specific_energy(state["pos"], state["vel"])
    trail = deque(maxlen=TRAIL_POINTS)

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.canvas.manager.set_window_title("Earth-Moon spacecraft simulator")
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_color("gray")
    ax.set_xlabel("x [m]", color="gray")
    ax.set_ylabel("y [m]", color="gray")

    ax.add_patch(Circle(EARTH_POS, R_EARTH, color="deepskyblue", zorder=3))
    ax.add_patch(Circle(MOON_POS, R_MOON, color="lightgray", zorder=3))

    views = {
        "earth": (EARTH_POS, 4.0 * r0),
        "moon":  (MOON_POS,  25.0 * R_MOON),
        "full":  (np.array([D_MOON / 2.0, 0.0]), 0.62 * D_MOON),
    }
    center, half_width = views[VIEW]
    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_width, center[1] + half_width)

    trail_line, = ax.plot([], [], color="orange", lw=1.0, zorder=4)
    craft_dot,  = ax.plot([], [], "o", color="white", ms=5, zorder=5)
    status = ax.set_title("", color="white", fontfamily="monospace")

    def update(_frame):
        if not state["done"]:
            for _ in range(STEPS_PER_FRAME):
                state["pos"], state["vel"] = step(state["pos"], state["vel"], DT)
                state["t"] += DT
                trail.append(state["pos"].copy())
                hit = crashed(state["pos"])
                if hit:
                    state["done"] = True
                    status.set_text(f"CRASHED into the {hit} "
                                    f"after {state['t']/3600.0:.2f} h")
                    break

        pts = np.array(trail)
        trail_line.set_data(pts[:, 0], pts[:, 1])
        craft_dot.set_data([state["pos"][0]], [state["pos"][1]])

        if not state["done"]:
            E = specific_energy(state["pos"], state["vel"])
            alt = np.linalg.norm(state["pos"] - EARTH_POS) - R_EARTH
            status.set_text(
                f"{INTEGRATOR:8s}  dt={DT:g}s   t={state['t']/3600.0:7.2f} h   "
                f"alt={alt/1e3:9.1f} km   "
                f"|v|={np.linalg.norm(state['vel'])/1e3:6.3f} km/s   "
                f"energy drift={abs((E - E0)/E0)*100.0:7.4f} %"
            )
        return trail_line, craft_dot, status

    if SAVE_TO == "window":
        anim = FuncAnimation(fig, update, interval=20,
                             blit=False, cache_frame_data=False)
        plt.show()
    else:
        # Without frames=..., anim.save renders only 100 frames by default —
        # compute how many frames cover SIM_HOURS of simulated time instead.
        n_frames = max(1, int(SIM_HOURS * 3600.0 / (DT * STEPS_PER_FRAME)))
        anim = FuncAnimation(fig, update, frames=n_frames, interval=20,
                             blit=False, cache_frame_data=False)
        anim.save(SAVE_TO, fps=30, progress_callback=lambda i, n:
                  print(f"\rrendering frame {i+1}/{n}", end="", flush=True))
        print(f"\nsaved {SAVE_TO}: {SIM_HOURS:g} h of simulation, "
              f"{n_frames/30.0:.0f} s of video")
    return anim  # keep a reference so the animation isn't garbage-collected


if __name__ == "__main__":
    print(f"Integrator      : {INTEGRATOR}")
    print(f"Craft mass      : {M_CRAFT:,.0f} kg (Apollo 8 CSM)")
    print(f"Start altitude  : {ALTITUDE/1e3:.1f} km")
    print(f"Start speed     : {np.linalg.norm(START_VEL):,.1f} m/s "
          f"(circular speed here: {v_circular:,.1f} m/s)")
    print(f"Orbital period  : {2*np.pi*np.sqrt(r0**3/(G*M_EARTH))/60.0:.1f} min")
    main()
