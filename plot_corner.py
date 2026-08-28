#!/usr/bin/env python3
"""Corner plot of the TMCMC posterior over (start of thrust, parking altitude,
extra speed). Diagonal: 1D marginals; lower triangle: 2D density. The orange
dashed cross marks the historical Apollo 8 configuration for reference.

Usage:  python3 plot_corner.py [samples.npy] [out.png]
        (defaults: apollo_posterior.npy -> apollo_corner.png)
"""

import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from apollo_model import parking_speed, R_EARTH

LABELS = ["Start of thrust [rad]", "Parking altitude [km]", "Extra speed [m/s]"]

# Historical Apollo 8 configuration (the free-return of orbit_sim.py):
# burn opposite the Moon from the 185.2 km parking orbit, 10,920 m/s total.
HIST = np.array([np.pi, 185.2,
                 10_920.0 - float(parking_speed(R_EARTH + 185.2e3))])

# one-hue sequential ramp (white surface -> dark blue) for the 2D densities
CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"])
BAR    = "#3987e5"   # marginal histogram fill
ACCENT = "#eb6834"   # historical-reference cross
INK    = "#444444"   # axis/label ink


def load_samples(path):
    """Load (n, 3) samples from a .npy file, or fall back to the newest
    Korali generation file if the .npy is missing."""
    try:
        return np.load(path)
    except FileNotFoundError:
        with open("results/_korali_result_apollo/latest") as f:
            gen = json.load(f)
        db = gen["Solver"]["Sample Database"]
        print(f"note: {path} not found, using results/_korali_result_apollo/latest")
        return np.array(db)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "results/apollo_posterior.npy"
    out = sys.argv[2] if len(sys.argv) > 2 else "media/apollo_corner.png"
    samples = load_samples(src)
    n, ndim = samples.shape

    # robust axis limits: central 99% of the samples, padded a little
    lims = []
    for i in range(ndim):
        lo, hi = np.percentile(samples[:, i], [0.5, 99.5])
        pad = 0.08 * (hi - lo) or 1.0
        lims.append((lo - pad, hi + pad))

    fig, axes = plt.subplots(ndim, ndim, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.06, wspace=0.06,
                        left=0.10, bottom=0.09, right=0.97, top=0.93)

    for i in range(ndim):          # row
        for j in range(ndim):      # column
            ax = axes[i, j]
            if j > i:              # upper triangle: unused
                ax.axis("off")
                continue

            if i == j:             # diagonal: 1D marginal
                ax.hist(samples[:, i], bins=40, range=lims[i],
                        color=BAR, edgecolor="white", linewidth=0.4)
                ax.axvline(HIST[i], color=ACCENT, ls="--", lw=1.4)
                m, s = samples[:, i].mean(), samples[:, i].std()
                ax.set_title(f"{m:,.4g} ± {s:,.2g}", fontsize=9, color=INK)
                ax.set_yticks([])
            else:                  # lower triangle: 2D density
                ax.hist2d(samples[:, j], samples[:, i], bins=45,
                          range=[lims[j], lims[i]], cmap=CMAP)
                ax.axvline(HIST[j], color=ACCENT, ls="--", lw=1.0)
                ax.axhline(HIST[i], color=ACCENT, ls="--", lw=1.0)
                ax.set_ylim(lims[i])

            ax.set_xlim(lims[j])
            ax.tick_params(labelsize=8, colors=INK)
            for spine in ax.spines.values():
                spine.set_color("#bbbbbb")

            if i < ndim - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(LABELS[j], fontsize=9, color=INK)
            if j > 0 or i == 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(LABELS[i], fontsize=9, color=INK)

    fig.suptitle(f"TMCMC posterior — {n:,} samples", fontsize=12, color=INK)
    fig.text(0.98, 0.965, "orange dashes: historical Apollo 8",
             ha="right", fontsize=9, color=ACCENT)
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"saved {out}  ({n:,} samples)")
    print("posterior mean ± std:")
    for i, name in enumerate(LABELS):
        print(f"  {name:24s} {samples[:, i].mean():10.3f} "
              f"± {samples[:, i].std():.3f}")


if __name__ == "__main__":
    main()
