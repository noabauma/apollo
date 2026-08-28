#!/usr/bin/env python3
"""Apollo 8 translunar simulator, refactored as a computational model for
Korali (TMCMC). Physics identical to orbit_sim.py (restricted two-fixed-body
problem, kick-drift-kick leapfrog), but:

  * parameterized by the three sampled quantities:
      - start position angle [rad] on the parking orbit where the TLI burn
        happens (position r0*(cos, sin), burn tangential/prograde, CCW;
        pi reproduces the original [-r0, 0] configuration)
      - parking-orbit altitude above Earth's surface [km]
        (orbit radius r0 = R_EARTH + altitude, circular speed
        v0 = sqrt(G*M_EARTH/r0))
      - extra speed v1 [m/s] added on top of v0 at translunar injection
        (total burn speed v0+v1)
  * headless and importable: no globals mutated, no plotting, no file output
  * inner loop in scalar Python floats (~10x faster than per-step numpy,
    which matters when a sampler calls this thousands of times)

simulate() returns the full trajectory dataset — every recorded position and
velocity — from which any score function can be computed.
"""

import numpy as np

# ---- physical constants (SI), same values as orbit_sim.py -------------------
G       = 6.67430e-11    # gravitational constant [m^3 kg^-1 s^-2]
M_EARTH = 5.9722e24      # Earth mass  [kg]
M_MOON  = 7.342e22       # Moon mass   [kg]
R_EARTH = 6.371e6        # Earth mean radius [m]  (for drawing + crash check)
R_MOON  = 1.7374e6       # Moon mean radius  [m]
D_MOON  = 3.844e8        # mean Earth-Moon center-to-center distance [m]

EARTH_POS = np.array([0.0, 0.0])
MOON_POS  = np.array([D_MOON, 0.0])

_GME = G * M_EARTH
_GMM = G * M_MOON


def parking_speed(r0):
    """Circular orbit speed [m/s] at radius r0 [m] from Earth's center."""
    return np.sqrt(_GME / r0)


def simulate(start_pos, altitude_km, extra_speed, hours=130.0, dt=10.0):
    """Fly the mission for one parameter pair and record everything.

    Parameters
    ----------
    start_pos   : starting position of the craft in parking orbit [0, 2*pi)
    altitude_km : parking-orbit altitude above Earth's surface [km]
    extra_speed : v1, speed added on top of circular speed v0 at TLI [m/s]
    hours       : simulated time budget [h] (130 h covers the outbound leg
                  and the flyby of the historical trajectory)
    dt          : integrator time step [s]

    Returns a dict — the dataset available to the score function:
      't'         (n,)  : time of each sample [s], t[0] = 0
      'pos'       (n,2) : craft position [m]; Earth at (0,0), Moon at (D_MOON,0)
      'vel'       (n,2) : craft velocity [m/s]
      'fate'      str   : 'completed' | 'crashed Earth' | 'crashed Moon'
                          | 'escaped'

    Trajectories that crash (below either surface) or escape (beyond
    2*D_MOON from Earth) stop early; their arrays are simply shorter.
    """
    r0 = R_EARTH + altitude_km * 1e3
    v0 = float(parking_speed(r0))
    v1 = float(extra_speed)

    n = int(hours * 3600.0 / dt)
    t   = np.empty(n + 1)
    pos = np.empty((n + 1, 2))
    vel = np.empty((n + 1, 2))

    # TLI burn in parking orbit of the earth
    px, py = r0*np.cos(start_pos), r0*np.sin(start_pos)
    vx, vy = -(v0 + v1)*np.sin(start_pos), (v0 + v1)*np.cos(start_pos)
    t[0] = 0.0
    pos[0] = px, py
    vel[0] = vx, vy

    # scalar-math kick-drift-kick leapfrog (identical math to orbit_sim.py)
    dmoon, gme, gmm = D_MOON, _GME, _GMM
    r_earth2 = R_EARTH * R_EARTH
    r_moon2  = R_MOON * R_MOON
    esc2     = (2.0 * D_MOON) ** 2
    half_dt  = 0.5 * dt

    fate = "completed"
    k = 0
    for k in range(1, n + 1):
        re3 = (px * px + py * py) ** 1.5
        mx = px - dmoon
        rm3 = (mx * mx + py * py) ** 1.5
        ax = -gme * px / re3 - gmm * mx / rm3
        ay = -gme * py / re3 - gmm * py / rm3
        vx += half_dt * ax
        vy += half_dt * ay
        px += dt * vx
        py += dt * vy
        re3 = (px * px + py * py) ** 1.5
        mx = px - dmoon
        rm3 = (mx * mx + py * py) ** 1.5
        ax = -gme * px / re3 - gmm * mx / rm3
        ay = -gme * py / re3 - gmm * py / rm3
        vx += half_dt * ax
        vy += half_dt * ay

        t[k] = k * dt
        pos[k] = px, py
        vel[k] = vx, vy

        re2 = px * px + py * py
        rm2 = mx * mx + py * py
        if re2 < r_earth2:
            fate = "crashed Earth"
            break
        if rm2 < r_moon2:
            fate = "crashed Moon"
            break
        if re2 > esc2:
            fate = "escaped"
            break

    t, pos, vel = t[:k + 1], pos[:k + 1], vel[:k + 1]
    return {
        "t": t,
        "pos": pos,
        "vel": vel,
        "fate": fate,
        "r0": r0,
        "v0": v0,
        "v1": v1,
    }


if __name__ == "__main__":
    # Smoke test: the historical Apollo 8 free-return numbers from
    # orbit_sim.py (185.2 km parking orbit, 10,920 m/s total at TLI ->
    # ~4,700 km lunar flyby at ~113 h).
    import time
    alt = 185.2
    v0 = parking_speed(R_EARTH + alt * 1e3)
    start = time.perf_counter()
    d = simulate(np.pi, alt, 10_920.0 - v0, hours=130.0, dt=10.0)
    wall = time.perf_counter() - start
    d_moon = np.linalg.norm(d["pos"] - MOON_POS, axis=1)
    i = int(np.argmin(d_moon))
    print(f"fate: {d['fate']}, steps: {len(d['t']) - 1:,}, "
          f"wall: {wall:.2f} s")
    print(f"v0 = {d['v0']:,.1f} m/s, v1 = {d['v1']:,.1f} m/s")
    print(f"closest approach: {(d_moon[i] - R_MOON) / 1e3:,.0f} km above "
          f"the surface at t = {d['t'][i] / 3600.0:.1f} h")
