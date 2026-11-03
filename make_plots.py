"""Pick and refine the best ballistic transfer, then make figures (a), (b), (c).

Reads results/runs.csv and results/hohmann.json; writes results/ballistic.json
and results/figures/*.png.
"""
import json
import math
import os

import numpy as np

from plotting import plot_capture_map, plot_dv_vs_tof, plot_trajectories
from search import load_runs, refine, successful

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)


def best_ballistic(runs, n_candidates=15):
    """Refine the n most promising captured runs; keep the cheapest exact solution."""
    ok = runs["captured"] & np.isfinite(runs["dv_tot_proj"])
    idx = np.where(ok)[0]
    idx = idx[np.argsort(runs["dv_tot_proj"][idx])][:n_candidates]
    best = None
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
            if best is None or s.dv_tot < best.dv_tot:
                best = s
    return best


def main():
    coarse_path = os.path.join(RESULTS, "runs.csv")
    zoom_path = os.path.join(RESULTS, "runs_zoom.csv")
    paths = [coarse_path] + ([zoom_path] if os.path.exists(zoom_path) else [])
    runs = load_runs(*paths)
    coarse = load_runs(coarse_path)
    with open(os.path.join(RESULTS, "hohmann.json")) as fh:
        h = json.load(fh)
    print(f"{len(runs['run_id'])} runs, {successful(runs).sum()} successful, "
          f"{runs['captured'].sum()} captured")

    b = best_ballistic(runs)
    if b is None:
        raise SystemExit("no exact 100 km ballistic solution found near the candidates")
    bal_row = dict(b.as_row(), phase=math.radians(b.phase_deg), sun=math.radians(b.sun_deg))
    with open(os.path.join(RESULTS, "ballistic.json"), "w") as fh:
        json.dump(bal_row, fh, indent=2)

    hoh = dict(dv=h["dv_dep"], phase=h["phase"], sun=h["sun"], tof=h["t_ca"], dv_tot=h["dv_tot"])
    bal = dict(dv=b.dv_dep, phase=bal_row["phase"], sun=bal_row["sun"], tof=b.t_ca,
               dv_tot=b.dv_tot)

    print("\n                 dv_dep   dv_cap   total    TOF")
    print(f"Hohmann-style    {h['dv_dep']:.4f}   {h['dv_cap']:.4f}   {h['dv_tot']:.4f}   "
          f"{h['tof_days']:.2f} d")
    print(f"best ballistic   {b.dv_dep:.4f}   {b.dv_cap:.4f}   {b.dv_tot:.4f}   {b.tof_days:.2f} d")

    plot_trajectories(hoh, bal, os.path.join(FIGURES, "a_trajectories.png"))
    ok = successful(runs)
    vmin, vmax = np.nanpercentile(runs["dv_cap_proj"][ok], [1, 99])
    for s in np.unique(coarse["sun_deg"]):
        plot_capture_map(coarse, s, os.path.join(FIGURES, f"b_capture_map_sun{s:03.0f}.png"),
                         vmin=vmin, vmax=vmax)
    if len(paths) > 1:
        zoom = load_runs(zoom_path)
        lo, hi = zoom["phase_deg"].min(), zoom["phase_deg"].max() + 1e-9
        for s in np.unique(zoom["sun_deg"]):
            plot_capture_map(zoom, s, os.path.join(FIGURES, f"b_capture_map_sun{s:03.0f}_zoom.png"),
                             vmin=vmin, vmax=vmax, phase_range=(lo, hi), note=" (zoom)")
    plot_dv_vs_tof(runs, hoh, bal, os.path.join(FIGURES, "c_dv_vs_tof.png"))
    print(f"figures in {FIGURES}")


if __name__ == "__main__":
    main()
