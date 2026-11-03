"""The Hohmann-style reference transfer, solved in the full bicircular model.

Starting guess: the two-body Hohmann from LEO to the lunar distance, with the
Moon placed so that it arrives at apogee at the same time as the spacecraft:
    phase = pi - n_M * TOF_hohmann  (~114 deg).

At apogee the spacecraft moves tangent to the Moon's orbit, so sliding the
phase mostly moves the Moon *along* the spacecraft's path, not across it. The
miss distance is set by the apogee radius, i.e. by the departure Δv. So we fix
the phase and root-find the departure Δv for a periselene of 100 km altitude.
"""
import math

import numpy as np
from scipy.optimize import brentq

from physics import A_M, MU_E, N_M, R_LEO, R_LLO, hohmann_dv, hohmann_tof
from simulate import signed_periselene_radius, simulate


def two_body_guess():
    dv1, dv2 = hohmann_dv(MU_E, R_LEO, A_M)
    tof = hohmann_tof(MU_E, R_LEO, A_M)
    phase = math.pi - N_M * tof
    return dv1, phase, tof


def solve_hohmann(sun_angle=0.0, sense=+1, half_width=0.03, n=31):
    """Departure Δv giving a 100 km periselene on the first pass.

    sense = +1 for a counter-clockwise (prograde) pass, -1 for retrograde.
    Returns (dv_dep, phase, RunResult of the actual run, with Moon impacts on).
    """
    dv0, phase, _ = two_body_guess()
    target = sense * R_LLO

    def f(dv):
        s, _ = signed_periselene_radius(dv, phase, sun_angle)
        return s - target

    dvs = np.linspace(dv0 - half_width, dv0 + half_width, n)
    vals = np.array([f(d) for d in dvs])
    for a, b, fa, fb in zip(dvs[:-1], dvs[1:], vals[:-1], vals[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0 and abs(fa - fb) < 50_000:
            dv = brentq(f, a, b, xtol=1e-12)
            res = simulate(dv, phase, sun_angle, stop_at_first_periselene=100_000.0,
                           t_search=20 * 86400.0)
            return dv, phase, res
    raise RuntimeError("no bracket for the Hohmann periselene target")
