"""The figures: (a) trajectories, (b) outcome map, (c) extra Δv vs time of flight, (d) Δv budget.

Figure sizes are close to their printed size (a 6.5 in text width), so the fonts
come out at about the size of the report's text.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from physics import A_M, DAY, N_S, moon_state
from search import successful
from simulate import trajectory


# Reference palette (dataviz skill), light mode, on a white page
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#c9c8c2"
HOHMANN = "#2a78d6"      # categorical slot 1
BALLISTIC = "#eb6834"    # categorical slot 2
FAST = "#1baa7d"         # categorical slot 3: the ~100-day ballistic transfer

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "text.color": INK, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})

KM6 = 1e6
ANG = np.linspace(0, 2 * np.pi, 400)


# ============================================================================
# (a) Trajectories
# ============================================================================

def _to_sun_earth_rotating(t, y, sun_angle):
    """Rotate inertial positions so the Sun stays on the -x axis (Earth at origin)."""
    th = sun_angle + N_S * t + math.pi          # angle of the anti-Sun direction
    c, s = np.cos(th), np.sin(th)
    x, yy = y[0], y[1]
    return c * x + s * yy, -s * x + c * yy


def _panel_hohmann(ax, hoh):
    t, y = trajectory(hoh["dv"], hoh["phase"], hoh["sun"], hoh["tof"])
    rm = np.array([moon_state(tt, hoh["phase"])[0] for tt in t]).T
    ax.plot(A_M / 1e3 * np.cos(ANG), A_M / 1e3 * np.sin(ANG), color=MUTED, lw=1, ls="--")
    ax.plot(rm[0] / 1e3, rm[1] / 1e3, color=INK_2, lw=1.5, label="Moon")
    ax.plot(y[0] / 1e3, y[1] / 1e3, color=HOHMANN, lw=2, label="Spacecraft")
    ax.plot(0, 0, "o", color=INK, ms=6)
    ax.plot(rm[0, -1] / 1e3, rm[1, -1] / 1e3, "o", color=INK_2, ms=8)
    ax.set_title(f"Hohmann-style, {hoh['tof'] / DAY:.1f} days", loc="left")
    ax.set_xlabel("x [10³ km]"); ax.set_ylabel("y [10³ km]")
    ax.set_aspect("equal"); ax.legend(loc="lower left")


def _panel_inertial(ax, bal, t, y, color):
    ax.plot(A_M / KM6 * np.cos(ANG), A_M / KM6 * np.sin(ANG), color=MUTED, lw=1, ls="--",
            label="Moon's orbit")
    ax.plot(y[0] / KM6, y[1] / KM6, color=color, lw=1.2, label="Spacecraft")
    ax.plot(0, 0, "o", color=INK, ms=5)
    ax.set_title(f"Ballistic, {bal['tof'] / DAY:.1f} days, Earth-centered", loc="left")
    ax.set_xlabel("x [10⁶ km]"); ax.set_ylabel("y [10⁶ km]")
    ax.set_aspect("equal"); ax.legend(loc="lower left")


def _panel_rotating(ax, bal, t, y, color):
    xr, yr = _to_sun_earth_rotating(t, y, bal["sun"])
    ax.plot(xr / KM6, yr / KM6, color=color, lw=1.2)
    ax.plot(A_M / KM6 * np.cos(ANG), A_M / KM6 * np.sin(ANG), color=MUTED, lw=1, ls="--")
    ax.plot(0, 0, "o", color=INK, ms=5)
    ax.annotate("← Sun", xy=(0.02, 0.95), xycoords="axes fraction", color=INK_2)
    ax.set_title("Same, Sun–Earth rotating frame", loc="left")
    ax.set_xlabel("x [10⁶ km]"); ax.set_ylabel("y [10⁶ km]")
    ax.set_aspect("equal")


def plot_trajectories(bal, path, hoh=None, color=BALLISTIC):
    """Panels stacked vertically: Hohmann-style (if given), ballistic inertial, ballistic rotating.

    bal, hoh: dicts with dv, phase, sun, tof (s) -- each run ends at its periselene.
    """
    n = 3 if hoh is not None else 2
    fig, axes = plt.subplots(n, 1, figsize=(4.6, 3.0 * n + 0.4))
    axes = list(axes)
    if hoh is not None:
        _panel_hohmann(axes.pop(0), hoh)
    t, y = trajectory(bal["dv"], bal["phase"], bal["sun"], bal["tof"])
    _panel_inertial(axes[0], bal, t, y, color)
    _panel_rotating(axes[1], bal, t, y, color)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ============================================================================
# (b) Outcome map
# ============================================================================
#
# What happened to each departure, as a categorical map over (phase, departure Δv).

OUTCOMES = [  # (label, color), in the order of the integer codes below
    ("escaped the Earth–Moon system", "#e9eef5"),
    ("fell back into the Earth", "#f3e3c7"),
    ("never within 10,000 km of the Moon", "#ffffff"),
    ("passed within 10,000 km of the Moon", "#8fb8ea"),
    ("hit the Moon", INK_2),
]


def outcome_codes(runs):
    """Integer code per run (index into OUTCOMES); captured runs are marked separately."""
    code = np.full(len(runs["outcome"]), 2)
    code[runs["outcome"] == "escape"] = 0
    code[runs["outcome"] == "impact_earth"] = 1
    code[successful(runs)] = 3
    code[runs["outcome"] == "impact_moon"] = 4
    return code


def _outcome_panel(ax, runs, sun_deg, title, phase_range=None, stars=()):
    m = np.isclose(runs["sun_deg"], sun_deg)
    if phase_range is not None:
        m &= (runs["phase_deg"] >= phase_range[0]) & (runs["phase_deg"] < phase_range[1])
    dv, ph = runs["dv_dep"][m], runs["phase_deg"][m]
    code = outcome_codes(runs)[m]
    cap = runs["captured"][m]
    dvs, phs = np.unique(dv), np.unique(ph)
    grid = np.full((len(dvs), len(phs)), np.nan)
    grid[np.searchsorted(dvs, dv), np.searchsorted(phs, ph)] = code

    dph = phs[1] - phs[0] if len(phs) > 1 else 1
    ddv = dvs[1] - dvs[0] if len(dvs) > 1 else 1e-3
    ext = [phs[0] - dph / 2, phs[-1] + dph / 2, dvs[0] - ddv / 2, dvs[-1] + ddv / 2]
    cmap = ListedColormap([c for _, c in OUTCOMES])
    ax.grid(False)
    ax.imshow(grid, extent=ext, origin="lower", aspect="auto", cmap=cmap,
              vmin=-0.5, vmax=len(OUTCOMES) - 0.5, interpolation="nearest")
    ax.plot(ph[cap], dv[cap], "o", ms=5, color=BALLISTIC, mec=INK, mew=0.5, ls="none")
    for d, color in stars:
        ax.plot(math.degrees(d["phase"]) % 360, d["dv"], "*", ms=14, color=color, mec=INK, mew=0.8)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("Moon's angle ahead of the departure point at t = 0 [deg]")
    ax.set_ylabel("departure Δv − 3 km/s [m/s]")
    ax.yaxis.set_major_formatter(lambda v, _: f"{(v - 3) * 1e3:.1f}")
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    ax.set_title(title, loc="left")


def plot_capture_map(runs, sun_deg, path, zoom=None, stars=()):
    """The full scan on top and, if given, the zoomed scan below.

    stars: sequence of (solution dict with dv and phase [rad], color), drawn on the zoom.
    """
    n = 2 if zoom is not None else 1
    fig, axes = plt.subplots(n, 1, figsize=(6.5, 3.4 * n + 0.9))
    axes = np.atleast_1d(axes)
    _outcome_panel(axes[0], runs, sun_deg, f"Full scan, Sun at {sun_deg:.0f}°")
    if zoom is not None:
        lo, hi = zoom["phase_deg"].min(), zoom["phase_deg"].max() + 1e-9
        _outcome_panel(axes[1], zoom, sun_deg, "Zoom", phase_range=(lo, hi), stars=stars)
        axes[0].axvspan(lo, hi, fill=False, ec=INK, lw=1, ls="--")
    handles = [Patch(fc=c, ec=INK_2, lw=0.5, label=l) for l, c in OUTCOMES]
    handles.append(plt.Line2D([], [], marker="o", ls="none", color=BALLISTIC, mec=INK, mew=0.5,
                              label="captured for ≥ 1 lunar month"))
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(path, dpi=200); plt.close(fig)


# ============================================================================
# (c) Extra Δv vs time of flight
# ============================================================================

def plot_dv_vs_tof(runs, best, path, marked=(), ymax=250.0):
    """Total Δv above the best ballistic solution [m/s], against time of flight.

    best: the reference solution (its own point sits at zero).
    marked: sequence of (solution dict, color, name), drawn as labeled stars.
    Runs above ymax are left off the plot; the count is printed in the corner.
    """
    ok = successful(runs)
    cap = runs["captured"]
    extra = (runs["dv_tot_proj"] - best["dv_tot"]) * 1e3
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.axhline(0, color=INK, lw=1)
    for mask, color, label, ms in [(ok & ~cap, MUTED, "passed near the Moon, not captured", 10),
                                   (ok & cap, BALLISTIC, "captured for ≥ 1 lunar month", 16)]:
        ax.scatter(runs["tof_days"][mask], extra[mask], s=ms, color=color, label=label, zorder=2)
    for d, color, name in marked:
        e = (d["dv_tot"] - best["dv_tot"]) * 1e3
        ax.scatter([d["tof"] / DAY], [e], s=180, marker="*", color=color, edgecolors=INK,
                   linewidths=0.8, zorder=3)
        ax.annotate(f"{name}\n+{e:.0f} m/s, {d['tof'] / DAY:.1f} days", (d["tof"] / DAY, e),
                    xytext=(8, 8), textcoords="offset points", fontsize=9)
    off = int((ok & (extra > ymax)).sum())
    ax.text(0.01, 0.99, f"({off} runs above {ymax:.0f} m/s not shown)", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=INK_2)
    ax.set_xlim(0, 210); ax.set_ylim(-10, ymax)
    ax.set_xlabel("time of flight [days]")
    ax.set_ylabel("extra Δv compared to the best ballistic [m/s]")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ============================================================================
# (d) Δv budget
# ============================================================================

def plot_dv_budget(cases, path):
    """Horizontal stacked bars from zero: departure + capture Δv.

    cases: sequence of (name, solution dict with dv, dv_cap, dv_tot, tof).
    """
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    names = [f"{n}\n{d['tof'] / DAY:.1f} days" for n, d in cases]
    ypos = np.arange(len(cases))[::-1]
    dep = np.array([d["dv"] for _, d in cases])
    capv = np.array([d["dv_cap"] for _, d in cases])
    ax.barh(ypos, dep, color="#9ec5f4", label="departure burn")
    ax.barh(ypos, capv, left=dep, color=HOHMANN, label="capture burn")
    for yp, (_, d) in zip(ypos, cases):
        ax.text(d["dv_tot"] + 0.04, yp, f"{d['dv_tot']:.3f} km/s", va="center", fontsize=9)
    ax.set_yticks(ypos, names)
    ax.set_xlim(0, 4.6)
    ax.set_xlabel("Δv [km/s]")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
