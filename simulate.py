"""Propagation of one departure, with event detection and capture bookkeeping.

One run alternates between two stages:

  search  -- coast from departure until the spacecraft is near the Moon
             (inside R_NEAR) with negative Moon-relative energy. That moment
             starts a capture window.
  window  -- coast for one lunar month. If the Moon-relative energy stays
             negative for that long, the run counts as captured. If it goes
             positive first, we drop back into the search stage (the
             spacecraft might be captured on a later pass).

Every periselene (local minimum of the distance to the Moon) is recorded. The
scored closest approach is the lowest periselene inside the capture window for
captured runs (that's where we would burn), and the lowest periselene of the whole trajectory otherwise.
"""
import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy.integrate import solve_ivp

from physics import (DAY, MU_M, R_E, R_LLO, R_M, T_MOON, Model, capture_dv,
                     leo_departure_state, moon_energy, projected_capture_dv, relative_to_moon)


RTOL = 1e-12
ATOL = 1e-10
R_NEAR = 100_000.0        # "near the Moon" for starting a capture window [km]
R_ESCAPE = 3.0e6          # beyond this from Earth the run is abandoned [km]
T_SEARCH = 180 * DAY      # latest time a capture window may start


def _event(fun, terminal=False, direction=0):
    fun.terminal = terminal
    fun.direction = direction
    return fun


def make_events(model, moon_impact=True):
    """Event functions; each returns a scalar whose zero crossing is the event."""
    th = model.theta_m0

    def earth_impact(t, y):
        return math.hypot(y[0], y[1]) - R_E

    def moon_dist(t, y):
        r, _ = relative_to_moon(t, y, th)
        return math.hypot(r[0], r[1]) - R_M

    def escape(t, y):
        return math.hypot(y[0], y[1]) - R_ESCAPE

    def periselene(t, y):
        r, v = relative_to_moon(t, y, th)
        return r[0] * v[0] + r[1] * v[1]      # d(|r|^2)/dt / 2, rising through 0 at a minimum

    def window_start(t, y):
        # Positive unless we are near the Moon AND bound to it. Entering the
        # sphere while already bound is a jump from +1 to negative, which the
        # root finder also locates (at the sphere crossing) -- which is what we want.
        r, _ = relative_to_moon(t, y, th)
        if math.hypot(r[0], r[1]) > R_NEAR:
            return 1.0
        return moon_energy(t, y, th)

    def window_end(t, y):
        return moon_energy(t, y, th)

    ev = {
        "impact_earth": _event(earth_impact, terminal=True, direction=-1),
        "escape": _event(escape, terminal=True, direction=1),
        "periselene": _event(periselene, direction=1),
        "window_start": _event(window_start, terminal=True, direction=-1),
        "window_end": _event(window_end, terminal=True, direction=1),
    }
    if moon_impact:
        ev["impact_moon"] = _event(moon_dist, terminal=True, direction=-1)
    return ev


def integrate(model, y0, t0, t1, events):
    """Thin wrapper around solve_ivp. `events` is a dict name -> event function."""
    names = list(events)
    sol = solve_ivp(model.eom, (t0, t1), y0, method="DOP853", rtol=RTOL, atol=ATOL,
                    events=[events[n] for n in names])
    hits = {n: (sol.t_events[i], sol.y_events[i]) for i, n in enumerate(names)}
    return sol, hits


@dataclass
class RunResult:
    dv_dep: float          # departure burn [km/s]
    phase_deg: float       # Moon angle ahead of the departure point at t=0
    sun_deg: float         # Sun angle from the departure point at t=0
    outcome: str           # captured / flyby / impact_moon / impact_earth / escape / timeout
    captured: bool
    t_ca: float            # time of scored closest approach [s] (= time of flight)
    alt_ca: float          # its altitude above the Moon [km]
    v_rel_ca: float        # Moon-relative speed there [km/s]
    energy_ca: float       # Moon-relative two-body energy there [km^2/s^2]
    dv_cap: float          # literal capture Δv: v_rel_ca - v_circ(100 km)
    dv_cap_proj: float     # capture Δv with the same energy carried down to 100 km
    dv_tot: float          # dv_dep + dv_cap
    dv_tot_proj: float     # dv_dep + dv_cap_proj
    tof_days: float
    t_window: float        # start of the (last) capture window, NaN if none
    hz_ca: float           # Moon-relative angular momentum at closest approach (sign = sense of pass)

    def as_row(self):
        return asdict(self)


def _score(t, y, theta_m0):
    r, v = relative_to_moon(t, y, theta_m0)
    d = math.hypot(*r)
    vrel = math.hypot(*v)
    eps = 0.5 * vrel**2 - MU_M / d
    proj = projected_capture_dv(eps) if eps > -MU_M / R_LLO else float("nan")
    hz = r[0] * v[1] - r[1] * v[0]
    return d - R_M, vrel, eps, capture_dv(vrel), proj, hz


def simulate(dv_dep, phase, sun_angle=0.0, moon_impact=True, sun=True,
             t_search=T_SEARCH, stop_at_first_periselene=None):
    """Run one departure. Angles in radians.

    stop_at_first_periselene: if a radius [km] is given, integration stops at the
    first periselene closer than that (used for short Hohmann-style transfers).
    """
    model = Model(theta_m0=phase, theta_s0=sun_angle, sun=sun)
    ev = make_events(model, moon_impact=moon_impact)
    y = leo_departure_state(dv_dep)
    t = 0.0
    peris = []                 # (t, y) of every periselene
    window_peri = None         # first periselene of a successful window
    outcome = "timeout"
    t_window = float("nan")
    term_names = ["impact_earth", "escape"] + (["impact_moon"] if moon_impact else [])

    if stop_at_first_periselene is not None:
        rmax = stop_at_first_periselene

        def near_peri(tt, yy):
            r, v = relative_to_moon(tt, yy, phase)
            if math.hypot(*r) > rmax:
                return -1.0
            return r[0] * v[0] + r[1] * v[1]
        ev = {k: ev[k] for k in term_names}
        ev["near_peri"] = _event(near_peri, terminal=True, direction=1)
        sol, hits = integrate(model, y, 0.0, t_search, ev)
        outcome = _first_terminal(hits, term_names + ["near_peri"]) or "timeout"
        if outcome == "near_peri":
            outcome = "flyby"
            peris.append((hits["near_peri"][0][0], hits["near_peri"][1][0]))
        elif outcome == "impact_moon":
            peris.append((sol.t[-1], sol.y[:, -1]))
        return _finish(dv_dep, phase, sun_angle, outcome, False, peris, None, t_window, model)

    while t < t_search:
        stage_ev = {k: ev[k] for k in term_names + ["periselene", "window_start"]}
        sol, hits = integrate(model, y, t, t_search, stage_ev)
        peris += list(zip(*hits["periselene"])) if len(hits["periselene"][0]) else []
        t, y = sol.t[-1], sol.y[:, -1]
        stop = _first_terminal(hits, term_names + ["window_start"])
        if stop != "window_start":
            outcome = stop or "timeout"
            if stop == "impact_moon":
                peris.append((t, y))
            break

        # Capture window: one lunar month of negative Moon-relative energy?
        t_window = t
        win_ev = {k: ev[k] for k in term_names + ["periselene", "window_end"]}
        sol, hits = integrate(model, y, t, t + T_MOON, win_ev)
        new_peris = list(zip(*hits["periselene"])) if len(hits["periselene"][0]) else []
        peris += new_peris
        t, y = sol.t[-1], sol.y[:, -1]
        stop = _first_terminal(hits, term_names + ["window_end"])
        if stop is None:                       # survived the whole window
            outcome = "captured"
            if new_peris:           # burn at the lowest pass inside the window
                d = [math.hypot(*relative_to_moon(tp, yp, model.theta_m0)[0]) for tp, yp in new_peris]
                window_peri = new_peris[int(np.argmin(d))]
            break
        if stop != "window_end":               # crashed or escaped inside the window
            outcome = stop
            if stop == "impact_moon":
                peris.append((t, y))
            break
        # energy went positive: temporary capture only; keep searching
        outcome = "flyby"

    if outcome == "timeout" and peris:
        outcome = "flyby"
    return _finish(dv_dep, phase, sun_angle, outcome, outcome == "captured", peris,
                   window_peri, t_window, model)


def _first_terminal(hits, names):
    for n in names:
        if n in hits and len(hits[n][0]):
            return n
    return None


def _finish(dv_dep, phase, sun_angle, outcome, captured, peris, window_peri, t_window, model):
    if window_peri is not None:
        tc, yc = window_peri
    elif outcome == "impact_moon":
        tc, yc = peris[-1]
    elif peris:
        dists = [math.hypot(*relative_to_moon(tp, yp, model.theta_m0)[0]) for tp, yp in peris]
        tc, yc = peris[int(np.argmin(dists))]
    else:
        tc, yc = float("nan"), None

    if yc is None:
        alt = vrel = eps = dvc = proj = hz = float("nan")
    else:
        alt, vrel, eps, dvc, proj, hz = _score(tc, yc, model.theta_m0)
    return RunResult(
        dv_dep=dv_dep, phase_deg=math.degrees(phase) % 360.0,
        sun_deg=math.degrees(sun_angle) % 360.0, outcome=outcome, captured=captured,
        t_ca=tc, alt_ca=alt, v_rel_ca=vrel, energy_ca=eps, dv_cap=dvc, dv_cap_proj=proj,
        dv_tot=dv_dep + dvc, dv_tot_proj=dv_dep + proj, tof_days=tc / DAY, t_window=t_window,
        hz_ca=hz,
    )


def signed_periselene_radius(dv_dep, phase, sun_angle=0.0, rmax=100_000.0, t_max=20 * DAY):
    """Signed radius of the first close periselene, with Moon impacts disabled.

    Positive = passes the Moon counter-clockwise. With the Moon a point mass this
    is continuous through zero as the trajectory sweeps across the Moon's center,
    so we can root-find on it.
    """
    res = simulate(dv_dep, phase, sun_angle, moon_impact=False,
                   stop_at_first_periselene=rmax, t_search=t_max)
    if res.outcome != "flyby":
        return float("nan"), res
    return math.copysign(res.alt_ca + R_M, res.hz_ca), res


def trajectory(dv_dep, phase, sun_angle, t_end, n=20_000):
    """Densely sampled trajectory for plotting: returns (t, y) with y shape (4, n)."""
    model = Model(phase, sun_angle)
    sol = solve_ivp(model.eom, (0.0, t_end), leo_departure_state(dv_dep), method="DOP853",
                    rtol=RTOL, atol=ATOL, dense_output=True)
    # sample finely near the start (LEO spiral) and uniformly afterwards
    t = np.unique(np.concatenate([np.linspace(0, min(t_end, 2 * DAY), n // 2),
                                  np.linspace(0, t_end, n // 2)]))
    return t, sol.sol(t)
