"""Searching for ballistic captures, in three steps.

  1. Moon off: find where the Sun raises perigee back up to lunar distance.
  2. Moon on: scan departure Δv and Moon phase, logging every run to CSV.
  3. Refine the most promising captured runs to an exact 100 km periselene.
"""
import csv
import math
import os
from dataclasses import fields
from multiprocessing import Pool

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from physics import DAY, R_E, R_LLO, R_M, Model, leo_departure_state
from simulate import RunResult, simulate


# ============================================================================
# Step 1: Moon-off pre-search over (Δv, Sun angle)
# ============================================================================
#
# Moon-off pre-search: where does the Sun raise perigee to lunar distance?
#
# A low-energy transfer goes out to ~10^6 km, where the solar tide raises the
# perigee from LEO up to the Moon's orbit. If perigee comes back at ~A_M moving
# at about the Moon's speed, the spacecraft can be captured almost for free.
# That depends very sharply on the Sun angle, so we find the right (Δv, Sun
# angle) pairs first -- with the Moon switched off, which is cheap -- and only
# then scan the Moon's phase.

def returning_perigee(dv, sun_angle, t_max=200 * DAY):
    """(t, r_p, v_p) of the first perigee after departure, Sun on, Moon off. NaNs if none."""
    model = Model(0.0, sun_angle, moon=False)

    def perigee(t, y):
        return y[0] * y[2] + y[1] * y[3]
    perigee.direction = 1

    def escape(t, y):
        return math.hypot(y[0], y[1]) - 3.0e6
    escape.terminal = True

    def impact(t, y):
        return math.hypot(y[0], y[1]) - R_E
    impact.terminal = True
    impact.direction = -1

    sol = solve_ivp(model.eom, (0.0, t_max), leo_departure_state(dv), method="DOP853",
                    rtol=1e-11, atol=1e-9, events=[perigee, escape, impact])
    te, ye = sol.t_events[0], sol.y_events[0]
    keep = te > DAY
    if not keep.any():
        if len(sol.t_events[2]):           # crashed: report perigee below the surface as 0
            return sol.t_events[2][0], 0.0, float("nan")
        return (float("nan"),) * 3
    y = ye[keep][0]
    return te[keep][0], math.hypot(y[0], y[1]), math.hypot(y[2], y[3])


def _job(args):
    dv, sun_deg = args
    return (dv, sun_deg) + returning_perigee(dv, math.radians(sun_deg))


def perigee_map(dv_values, sun_deg_values, workers=None):
    jobs = [(float(d), float(s)) for s in sun_deg_values for d in dv_values]
    with Pool(workers) as pool:
        rows = pool.map(_job, jobs, chunksize=16)
    return np.array(rows)       # columns: dv, sun_deg, t_p, r_p, v_p


# ============================================================================
# Step 2: grid scan over (Δv, phase, Sun angle)
# ============================================================================
#
# Grid scan over (departure Δv, Moon phase, Sun angle), logging every run to CSV.

COLUMNS = ["run_id"] + [f.name for f in fields(RunResult)]
ALT_CEILING = 10_000.0     # closest approach must be below this to count as reachable [km]


def make_grid(dv_values, phase_deg_values, sun_deg_values):
    """List of (run_id, dv, phase_deg, sun_deg)."""
    out = []
    for s in sun_deg_values:
        for p in phase_deg_values:
            for d in dv_values:
                out.append((f"{d:.6f}_{p:.4f}_{s:.4f}", float(d), float(p), float(s)))
    return out


def run_one(job):
    run_id, dv, phase_deg, sun_deg = job
    try:
        res = simulate(dv, math.radians(phase_deg), math.radians(sun_deg))
        row = res.as_row()
    except Exception as exc:              # log failures rather than kill the scan
        row = {c: float("nan") for c in COLUMNS}
        row.update(dv_dep=dv, phase_deg=phase_deg, sun_deg=sun_deg,
                   outcome=f"error:{type(exc).__name__}", captured=False)
    row["run_id"] = run_id
    return row


def run_scan(jobs, out_csv, workers=None, chunksize=8, progress=True):
    """Run every job not already in out_csv (so an interrupted scan can resume)."""
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as fh:
            done = {r["run_id"] for r in csv.DictReader(fh)}
    todo = [j for j in jobs if j[0] not in done]
    new_file = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as fh, Pool(workers) as pool:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_one, todo, chunksize=chunksize)):
            w.writerow(row)
            if progress and (i + 1) % 500 == 0:
                fh.flush()
                print(f"  {i + 1}/{len(todo)} runs", flush=True)
    return len(todo)


def load_runs(*paths):
    """CSV file(s) -> dict of numpy arrays (strings for outcome, bool for captured).

    Rows with a run_id already seen in an earlier file are skipped.
    """
    rows, seen = [], set()
    for path in paths:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r["run_id"] not in seen:
                    seen.add(r["run_id"])
                    rows.append(r)
    out = {}
    for c in COLUMNS:
        vals = [r[c] for r in rows]
        if c in ("run_id", "outcome"):
            out[c] = np.array(vals)
        elif c == "captured":
            out[c] = np.array([v == "True" for v in vals])
        else:
            out[c] = np.array([float(v) if v not in ("", "nan") else np.nan for v in vals])
    return out


def successful(runs, ceiling=ALT_CEILING):
    """Reached the Moon's vicinity without hitting anything."""
    ok_outcome = np.isin(runs["outcome"], ["captured", "flyby"])
    return ok_outcome & (runs["alt_ca"] < ceiling) & np.isfinite(runs["dv_cap_proj"])


# ============================================================================
# Step 3: refine the best captured runs to exactly 100 km
# ============================================================================
#
# Turn a promising scan point into an exact 100 km ballistic capture.
#
# The scan scores runs with the capture Δv *projected* to 100 km. Here we nudge
# one knob (phase by default) until the lowest periselene of the capture window is
# really at 100 km altitude, then the capture Δv is the literal one.
#
# Ballistic capture is chaotic, so the signed periselene radius is only piecewise
# continuous in the knob. We scan a small neighborhood, keep only brackets whose
# endpoints are both captured and not wildly different, and root-find in those.

def signed_window_periselene(dv, phase, sun):
    """Signed radius of the scored periselene of a captured run (Moon as point mass)."""
    res = simulate(dv, phase, sun, moon_impact=False)
    if not res.captured:
        return float("nan"), res
    return math.copysign(res.alt_ca + R_M, res.hz_ca), res


def refine(dv, phase, sun, knob="phase", half_width=None, n=41, max_jump=20_000.0):
    """Return a list of exact-100-km solutions (RunResult) near (dv, phase, sun).

    Angles in radians. half_width defaults to 0.5 deg for phase, 2 m/s for dv.
    """
    if half_width is None:
        half_width = math.radians(0.5) if knob == "phase" else 0.002
    x0 = phase if knob == "phase" else dv

    def run(x):
        return (signed_window_periselene(dv, x, sun) if knob == "phase"
                else signed_window_periselene(x, phase, sun))

    xs = np.linspace(x0 - half_width, x0 + half_width, n)
    s = np.array([run(x)[0] for x in xs])
    sols = []
    for target in (+R_LLO, -R_LLO):
        f = s - target
        for a, b, fa, fb in zip(xs[:-1], xs[1:], f[:-1], f[1:]):
            if not (np.isfinite(fa) and np.isfinite(fb)) or fa * fb > 0 or abs(fa - fb) > max_jump:
                continue

            def g(x):
                v = run(x)[0] - target
                if not np.isfinite(v):
                    raise ValueError("left the captured family inside the bracket")
                return v
            try:
                x = brentq(g, a, b, xtol=1e-13)
            except ValueError:
                continue
            d, p = (dv, x) if knob == "phase" else (x, phase)
            res = simulate(d, p, sun)          # verify with Moon impacts switched back on
            if res.captured and abs(res.alt_ca - (R_LLO - R_M)) < 1.0:
                sols.append(res)
    return sols
