#!/usr/bin/env python3
"""TMCMC over the Apollo 8 translunar injection, using Korali.

Sampled parameters (uniform priors):
  1. "Start of Thrust" [rad]  — angle on the parking orbit where the TLI burn
     happens (position r0*(cos, sin), burn tangential/prograde, CCW).
     The historical optimum is opposite the Moon: ~pi.
  2. "Parking Altitude" [km]  — parking-orbit altitude above Earth's surface;
     the model derives r0 = R_EARTH + altitude and v0 = sqrt(G*M_E/r0).
  3. "Extra Speed" v1 [m/s]   — added on top of v0 at translunar injection.

The likelihood is fully custom: each sample runs the simulator and hands the
complete trajectory dataset to score() below. Implement your score there —
it must return a (higher = better) log-likelihood-like scalar.

Run with:
  LD_LIBRARY_PATH=$HOME/src/korali/build/source:$HOME/src/korali/build/subprojects/gsl-2.8/build/.libs \
  PYTHONPATH=$HOME/src/korali/python:$HOME/src/korali/build/source \
      $HOME/venvs/korali-lander/bin/python run_tmcmc_apollo.py
"""

import numpy as np

from apollo_model import G, M_MOON, MOON_POS, R_MOON, simulate, parking_speed

SIM_HOURS = 200.0
SIM_DT = 10.0

POPULATION_SIZE = 2000
CONCURRENT_JOBS = 8
CRASH_PENALTY = -1.0e6   # score for crashed/escaped/degenerate trajectories;
                         # below the worst non-crash score (~-2.5e5) so the
                         # ordering "crash < any flyby" holds, but small enough
                         # that TMCMC's annealing step search can resolve it

# ---------------------------- score hyperparameters --------------------------
# The log-likelihood is a weighted sum of squared-error terms,
#
#     logL = - sum_i  W_i * (error_i / SIGMA_i)**2
#
# so each W is an importance knob for one mission objective: W = 0 disables a
# term, and doubling W is the same as shrinking its SIGMA by sqrt(2). The
# defaults reproduce the original two-term score. Examples:
#   fuel efficiency above all:  W_CAPTURE = 2.0, W_TLI = 1.0
#   getting there fast matters: W_TIME = 1.0 (and maybe W_CAPTURE = 0.5)
# Keep each term's worst case at most ~1e5 (see the annealing note in main()).
TARGET_PERIAPSIS = R_MOON + 100e3  # [m] ideal closest approach (100 km alt)

W_PERIAPSIS = 1.0    # hit TARGET_PERIAPSIS at the Moon
W_CAPTURE   = 1.0    # arrive slow: capture burn = arrival speed - lunar circular
W_TLI       = 0.0    # fuel at Earth: size of the injection burn v1
W_TIME      = 0.0    # reach the Moon fast: time of the flyby

SIGMA_D   = 20e3     # [m]   tolerance on the periapsis target
SIGMA_V   = 200.0    # [m/s] tolerance on the capture burn
SIGMA_TLI = 500.0    # [m/s] scale of the injection-burn penalty
SIGMA_T   = 24.0     # [h]   scale of the transfer-time penalty

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# ============================================================================
# My Score function [0,1) zero means bad and 1 means perfect. Also CRASH_PENALTY
#
# `data` is the full dataset of one simulated mission (see apollo_model.py):
#   data["t"]         (n,)   time [s]
#   data["pos"]       (n,2)  position [m], Earth at (0,0), Moon at (D_MOON,0)
#   data["vel"]       (n,2)  velocity [m/s]
#   data["fate"]      str    "completed" | "crashed Earth" | "crashed Moon"
#                            | "escaped"
#
# Return value: scalar log-likelihood (higher = better trajectory).
# ============================================================================
def score(data):

    # First step, find the sample where the craft came closest to the Moon
    d_moon = np.linalg.norm(data["pos"] - MOON_POS, axis=1)
    i_closest = int(np.argmin(d_moon))

    # closest approach must be interior: index 0 means the craft never left,
    # the last index means the flight was truncated mid-approach
    if i_closest == 0 or i_closest == len(d_moon) - 1:
        return CRASH_PENALTY

    # refine the periapsis distance: at ~2.4 km/s a 10 s step moves ~24 km,
    # so take the minimum of the parabola through the three closest samples
    dm1, dm2, dm3 = d_moon[i_closest - 1], d_moon[i_closest], d_moon[i_closest + 1]
    denom = dm1 - 2.0 * dm2 + dm3
    if denom > 0.0:
        r_craft = dm2 - 0.125 * (dm1 - dm3) ** 2 / denom
    else:
        r_craft = dm2

    # Closer than 60km to the moon's surface is not good (i.e. potential crash)
    if r_craft < R_MOON + 60e3:
        return CRASH_PENALTY

    # 1. Periapsis targeting (ideal spot: TARGET_PERIAPSIS above the surface).
    # Gaussian in ln(r/r_target) rather than in r: near the target the two are
    # identical (ln(r/rt) ~ (r-rt)/rt, so sigma matches SIGMA_D), but a
    # prior-typical miss of ~1e8 m scores ~-2e5 instead of ~-2.5e8. TMCMC
    # sizes its annealing steps from the spread of the log-likelihoods, so
    # compressing this range is what lets the annealing exponent move at all.
    sigma_log = SIGMA_D / TARGET_PERIAPSIS
    log_periapsis = -(np.log(r_craft / TARGET_PERIAPSIS) / sigma_log) ** 2

    # 2. Capture burn: arrival speed vs circular speed AROUND THE MOON at the
    # flyby periapsis — sqrt(G*M_MOON/r), NOT parking_speed(r) which uses
    # Earth's GM. Unreachable by ~700 m/s without a capture burn: with the
    # Moon fixed, energy conservation floors the arrival speed at ~2.3 km/s
    # while lunar circular speed at 100 km is ~1.6 km/s -- kept on purpose to
    # see how the sampler copes.
    v_craft = np.linalg.norm(data["vel"][i_closest])
    v_circ_moon = np.sqrt(G * M_MOON / r_craft)
    log_capture = -((v_craft - v_circ_moon) / SIGMA_V) ** 2

    # 3. Fuel at Earth: the injection burn itself (extra speed v1 above the
    # parking-orbit circular speed).
    log_tli = -(data["v1"] / SIGMA_TLI) ** 2

    # 4. Transfer time: when the flyby happens.
    log_time = -(data["t"][i_closest] / 3600.0 / SIGMA_T) ** 2

    return (W_PERIAPSIS * log_periapsis + W_CAPTURE * log_capture
            + W_TLI * log_tli + W_TIME * log_time)


def likelihood(sample):
    start_pos, altitude_km, extra_speed = sample["Parameters"]
    data = simulate(start_pos, altitude_km, extra_speed, hours=SIM_HOURS, dt=SIM_DT)
    if data["fate"] != "completed":
        sample["logLikelihood"] = CRASH_PENALTY
        return

    value = float(score(data))
    sample["logLikelihood"] = value if np.isfinite(value) else CRASH_PENALTY


def main():
    import korali

    k = korali.Engine()
    e = korali.Experiment()

    e["Random Seed"] = 0xC0FFEE
    e["Problem"]["Type"] = "Bayesian/Custom"
    e["Problem"]["Likelihood Model"] = likelihood

    e["Distributions"][0]["Name"] = "Uniform Start of Thrust"
    e["Distributions"][0]["Type"] = "Univariate/Uniform"
    e["Distributions"][0]["Minimum"] = 0.0       # [rad]
    e["Distributions"][0]["Maximum"] = 2*np.pi
    
    e["Distributions"][1]["Name"] = "Uniform Altitude"
    e["Distributions"][1]["Type"] = "Univariate/Uniform"
    e["Distributions"][1]["Minimum"] = 100.0     # [km]
    e["Distributions"][1]["Maximum"] = 500.0

    e["Distributions"][2]["Name"] = "Uniform Extra Speed"
    e["Distributions"][2]["Type"] = "Univariate/Uniform"
    e["Distributions"][2]["Minimum"] = 1000.0    # [m/s]
    e["Distributions"][2]["Maximum"] = 6000.0
    
    e["Variables"][0]["Name"] = "Start of Thrust"
    e["Variables"][0]["Prior Distribution"] = "Uniform Start of Thrust"

    e["Variables"][1]["Name"] = "Parking Altitude"
    e["Variables"][1]["Prior Distribution"] = "Uniform Altitude"

    e["Variables"][2]["Name"] = "Extra Speed"
    e["Variables"][2]["Prior Distribution"] = "Uniform Extra Speed"

    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = POPULATION_SIZE
    # The log-likelihood range (perfect flyby ~ -10, crash penalty -1e6) sets
    # how fast the annealing exponent can move: TMCMC picks each beta step so
    # the plausibility-weight spread stays near its target, so steps start
    # around 1/|logL_worst| ~ 1e-6 and grow as the population concentrates.
    # Keep a small minimum step so early generations are not clamped, and
    # extra MH mixing so clones re-diversify.
    e["Solver"]["Min Annealing Exponent Update"] = 1e-9
    e["Solver"]["Burn In"] = 1
    e["Solver"]["Max Chain Length"] = 5
    # Wall-clock guard: ~28 s/generation at population 500, so ~115 s at
    # 2000 -> 60 generations stays under two hours even if the annealing
    # has not reached beta=1 by then (it converged in 13 at pop 500).
    e["Solver"]["Termination Criteria"]["Max Generations"] = 60

    e["File Output"]["Enabled"] = True
    e["File Output"]["Path"] = "results/_korali_result_apollo"
    e["Console Output"]["Frequency"] = 1

    k["Conduit"]["Type"] = "Concurrent"
    k["Conduit"]["Concurrent Jobs"] = CONCURRENT_JOBS

    k.run(e)

    samples = np.array(e["Results"]["Sample Database"])
    np.save("results/apollo_posterior.npy", samples)
    print(f"\nPosterior samples ({len(samples)}) saved to results/apollo_posterior.npy")
    print("Posterior mean +- std:")
    for i, name in enumerate(["Start of Thrust [rad]", "Parking Altitude [km]",
                              "Extra Speed [m/s]"]):
        print(f"  {name:22s} {samples[:, i].mean():9.2f} "
              f"+- {samples[:, i].std():.2f}")


if __name__ == "__main__":
    main()
