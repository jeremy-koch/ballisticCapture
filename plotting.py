"""The three figures: (a) trajectories, (b) capture map, (c) Δv vs time of flight.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from physics import A_M, DAY, N_S, moon_state, sun_state
from search import successful
from simulate import trajectory


# Reference palette (dataviz skill), light mode
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#c9c8c2"
HOHMANN = "#2a78d6"      # categorical slot 1
BALLISTIC = "#eb6834"    # categorical slot 2
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "text.color": INK, "axes.grid": True, "grid.color": "#ecebe7",
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "legend.frameon": False,
})

KM6 = 1e6


def _to_sun_earth_rotating(t, y, sun_angle):
    """Rotate inertial positions so the Sun stays on the -x axis (Earth at origin)."""
    th = sun_angle + N_S * t + math.pi          # angle of the anti-Sun direction
    c, s = np.cos(th), np.sin(th)
    x, yy = y[0], y[1]
    return c * x + s * yy, -s * x + c * yy


def plot_trajectories(hoh, bal, path):
    """hoh, bal: dicts with dv, phase, sun, tof (s) -- each run ends at its periselene."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    ang = np.linspace(0, 2 * np.pi, 400)

    # (left) Hohmann-style, inertial
    ax = axes[0]
    t, y = trajectory(hoh["dv"], hoh["phase"], hoh["sun"], hoh["tof"])
    rm = np.array([moon_state(tt, hoh["phase"])[0] for tt in t]).T
    ax.plot(A_M / 1e3 * np.cos(ang), A_M / 1e3 * np.sin(ang), color=MUTED, lw=1, ls="--")
    ax.plot(rm[0] / 1e3, rm[1] / 1e3, color=INK_2, lw=1.5, label="Moon")
    ax.plot(y[0] / 1e3, y[1] / 1e3, color=HOHMANN, lw=2, label="Spacecraft")
    ax.plot(0, 0, "o", color=INK, ms=6)
    ax.plot(rm[0, -1] / 1e3, rm[1, -1] / 1e3, "o", color=INK_2, ms=8)
    ax.set_title(f"Hohmann-style  ·  {hoh['tof'] / DAY:.1f} d", loc="left")
    ax.set_xlabel("x [10³ km]"); ax.set_ylabel("y [10³ km]")
    ax.set_aspect("equal"); ax.legend(loc="lower left")

    # (middle) ballistic, inertial
    ax = axes[1]
    t, y = trajectory(bal["dv"], bal["phase"], bal["sun"], bal["tof"])
    ax.plot(A_M / KM6 * np.cos(ang), A_M / KM6 * np.sin(ang), color=MUTED, lw=1, ls="--",
            label="Moon's orbit")
    ax.plot(y[0] / KM6, y[1] / KM6, color=BALLISTIC, lw=1.5, label="Spacecraft")
    rs = sun_state(0.0, bal["sun"])[0]
    u = rs / np.linalg.norm(rs)
    ax.annotate("to Sun at t=0", xy=(1.2 * u[0], 1.2 * u[1]), xytext=(0.7 * u[0], 0.7 * u[1]),
                color=INK_2, arrowprops=dict(arrowstyle="->", color=INK_2))
    ax.plot(0, 0, "o", color=INK, ms=5)
    ax.set_title(f"Ballistic  ·  inertial  ·  {bal['tof'] / DAY:.0f} d", loc="left")
    ax.set_xlabel("x [10⁶ km]"); ax.set_aspect("equal"); ax.legend(loc="lower left")

    # (right) ballistic, Sun-Earth rotating frame -- where the trajectory makes sense
    ax = axes[2]
    xr, yr = _to_sun_earth_rotating(t, y, bal["sun"])
    ax.plot(xr / KM6, yr / KM6, color=BALLISTIC, lw=1.5)
    ax.plot(A_M / KM6 * np.cos(ang), A_M / KM6 * np.sin(ang), color=MUTED, lw=1, ls="--")
    ax.plot(0, 0, "o", color=INK, ms=5)
    ax.annotate("Sun ←", xy=(0.02, 0.95), xycoords="axes fraction", color=INK_2)
    ax.set_title("Ballistic  ·  Sun–Earth rotating frame", loc="left")
    ax.set_xlabel("x [10⁶ km]"); ax.set_aspect("equal")

    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_capture_map(runs, sun_deg, path, vmin=None, vmax=None, phase_range=None, note=""):
    """Departure Δv vs phase for one Sun angle, colored by projected capture Δv."""
    m = np.isclose(runs["sun_deg"], sun_deg)
    if phase_range is not None:
        m &= (runs["phase_deg"] >= phase_range[0]) & (runs["phase_deg"] < phase_range[1])
    dv, ph = runs["dv_dep"][m], runs["phase_deg"][m]
    ok = successful(runs)[m]
    cap = runs["captured"][m]
    val = runs["dv_cap_proj"][m]
    dvs, phs = np.unique(dv), np.unique(ph)
    grid = np.full((len(dvs), len(phs)), np.nan)
    i = np.searchsorted(dvs, dv); j = np.searchsorted(phs, ph)
    grid[i[ok], j[ok]] = val[ok]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.grid(False)
    dph = phs[1] - phs[0] if len(phs) > 1 else 1
    ddv = dvs[1] - dvs[0] if len(dvs) > 1 else 1e-3
    ext = [phs[0] - dph / 2, phs[-1] + dph / 2, dvs[0] - ddv / 2, dvs[-1] + ddv / 2]
    ax.imshow(np.zeros_like(grid), extent=ext, origin="lower", aspect="auto",
              cmap=LinearSegmentedColormap.from_list("bg", [MUTED, MUTED]))
    im = ax.imshow(grid, extent=ext, origin="lower", aspect="auto", cmap=SEQ_CMAP,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.plot(ph[cap & ok], dv[cap & ok], "o", ms=3.5, mfc="none", mec=BALLISTIC, mew=1.2,
            label="ballistically captured (≥ 1 lunar month)")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("capture Δv, projected to 100 km [km/s]")
    ax.set_xlabel("phase: Moon angle ahead of departure point at t=0 [deg]")
    ax.set_ylabel("departure Δv [m/s above 3 km/s]")
    ax.yaxis.set_major_formatter(lambda v, _: f"{(v - 3) * 1e3:.1f}")
    ax.set_title(f"Capture map{note}  ·  Sun at {sun_deg:.0f}° from departure point  ·  "
                 "gray = never within 10,000 km of the Moon", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_dv_vs_tof(runs, hoh, bal, path):
    ok = successful(runs)
    cap = runs["captured"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mask, color, label, ms in [(ok & ~cap, MUTED, "flyby (not captured)", 9),
                                   (ok & cap, BALLISTIC, "ballistically captured", 14)]:
        ax.scatter(runs["tof_days"][mask], runs["dv_tot_proj"][mask], s=ms, color=color,
                   edgecolors=SURFACE, linewidths=0.4, label=label, zorder=2)
    for d, color, name in [(hoh, HOHMANN, "Hohmann-style"), (bal, BALLISTIC, "best ballistic")]:
        ax.scatter([d["tof"] / DAY], [d["dv_tot"]], s=120, marker="*", color=color,
                   edgecolors=INK, linewidths=0.8, zorder=3)
        ax.annotate(f"{name}\n{d['dv_tot']:.3f} km/s, {d['tof'] / DAY:.1f} d",
                    (d["tof"] / DAY, d["dv_tot"]), xytext=(8, -18), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("time of flight to closest approach [days]")
    ax.set_ylabel("total Δv [km/s]")
    ax.set_title("Total Δv vs time of flight  ·  scan points use capture Δv projected to 100 km",
                 loc="left")
    lo = min(hoh["dv_tot"], bal["dv_tot"]) - 0.05
    ax.set_ylim(lo, lo + 1.0)
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
