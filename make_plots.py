"""Pick and refine the best ballistic transfers, then make the figures.

Reads results/runs.csv (+ runs_zoom.csv if present) and results/hohmann.json; writes
results/ballistic.json (cheapest), results/ballistic_fast.json (cheapest under
FAST_TOF days) and results/figures/*.png.

    python make_plots.py            # refine (a couple of minutes), then plot
    python make_plots.py --replot   # reuse the saved ballistic*.json, just plot
"""
import argparse
import json
import math
import os

import numpy as np

from plotting import (FAST, BALLISTIC, HOHMANN, plot_capture_map, plot_dv_budget,
                      plot_dv_vs_tof, plot_trajectories)
from search import load_runs, refine, successful

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)

FAST_TOF = 120.0    # [days] "fast" ballistic transfer: the cheapest one quicker than this


def exact_ballistic(runs, n_candidates=15):
    """Refine the n most promising captured runs; return every exact 100 km solution.

    Note: refining can move the burn to a different periselene, so a solution's
    time of flight need not match its scan point's (a 188-day scan point can
    refine to a 102-day solution).
    """
    ok = runs["captured"] & np.isfinite(runs["dv_tot_proj"])
    idx = np.where(ok)[0]
    idx = idx[np.argsort(runs["dv_tot_proj"][idx])][:n_candidates]
    out = []
    for i in idx:
        dv, ph, sun = (runs["dv_dep"][i], math.radians(runs["phase_deg"][i]),
                       math.radians(runs["sun_deg"][i]))
        print(f"  candidate dv {dv:.4f} phase {runs['phase_deg'][i]:.2f} sun {runs['sun_deg'][i]:.0f}"
              f"  projected total {runs['dv_tot_proj'][i]:.4f}  alt {runs['alt_ca'][i]:.0f} km")
        for knob in ("phase", "dv"):
            sols = refine(dv, ph, sun, knob=knob)
            if sols:
                break
        for s in sols:
            print(f"     -> exact: total {s.dv_tot:.4f} (dep {s.dv_dep:.4f}, cap {s.dv_cap:.4f}) "
                  f"TOF {s.tof_days:.1f} d")
        out += sols
    return out


def _row(res):
    return dict(res.as_row(), phase=math.radians(res.phase_deg), sun=math.radians(res.sun_deg))


def _plot_dict(row):
    """The fields the plotting functions use, from a saved JSON row."""
    return dict(dv=row["dv_dep"], dv_cap=row["dv_cap"], dv_tot=row["dv_tot"],
                phase=row["phase"], sun=row["sun"], tof=row["t_ca"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replot", action="store_true",
                   help="skip the refinement and reuse results/ballistic*.json")
    a = p.parse_args()

    coarse_path = os.path.join(RESULTS, "runs.csv")
    zoom_path = os.path.join(RESULTS, "runs_zoom.csv")
    paths = [coarse_path] + ([zoom_path] if os.path.exists(zoom_path) else [])
    runs = load_runs(*paths)
    coarse = load_runs(coarse_path)
    zoom = load_runs(zoom_path) if len(paths) > 1 else None
    with open(os.path.join(RESULTS, "hohmann.json")) as fh:
        h = json.load(fh)
    print(f"{len(runs['run_id'])} runs, {successful(runs).sum()} successful, "
          f"{runs['captured'].sum()} captured")

    bal_path = os.path.join(RESULTS, "ballistic.json")
    fast_path = os.path.join(RESULTS, "ballistic_fast.json")
    if a.replot:
        with open(bal_path) as fh:
            b = json.load(fh)
        with open(fast_path) as fh:
            f = json.load(fh)
    else:
        sols = exact_ballistic(runs)
        if not sols:
            raise SystemExit("no exact 100 km ballistic solution found near the candidates")
        b = _row(min(sols, key=lambda s: s.dv_tot))
        quick = [s for s in sols if s.tof_days < FAST_TOF]
        if not quick:
            raise SystemExit(f"no exact solution under {FAST_TOF:.0f} days")
        f = _row(min(quick, key=lambda s: s.dv_tot))
        for path, row in [(bal_path, b), (fast_path, f)]:
            with open(path, "w") as fh:
                json.dump(row, fh, indent=2)

    print("\n                 dv_dep   dv_cap   total    TOF")
    for name, r in [("Hohmann-style", h), ("best ballistic", b), ("fast ballistic", f)]:
        print(f"{name:<16} {r['dv_dep']:.4f}   {r['dv_cap']:.4f}   {r['dv_tot']:.4f}   "
              f"{r['tof_days']:.2f} d")

    hoh = _plot_dict(dict(h, phase=h["phase"], sun=h["sun"]))
    bal, fast = _plot_dict(b), _plot_dict(f)

    plot_trajectories(bal, os.path.join(FIGURES, "a_trajectories.png"), hoh=hoh)
    plot_trajectories(fast, os.path.join(FIGURES, "a_trajectories_fast.png"), color=FAST)
    for s in np.unique(coarse["sun_deg"]):
        plot_capture_map(coarse, s, os.path.join(FIGURES, f"b_capture_map_sun{s:03.0f}.png"),
                         zoom=zoom, stars=[(bal, BALLISTIC), (fast, FAST)])
    plot_dv_vs_tof(runs, bal, os.path.join(FIGURES, "c_dv_vs_tof.png"),
                   marked=[(hoh, HOHMANN, "Hohmann-style"), (fast, FAST, "fast ballistic"),
                           (bal, BALLISTIC, "best ballistic")])
    plot_dv_budget([("Hohmann-style", hoh), ("Fast ballistic", fast), ("Best ballistic", bal)],
                   os.path.join(FIGURES, "d_dv_budget.png"))
    print(f"figures in {FIGURES}")


if __name__ == "__main__":
    main()
