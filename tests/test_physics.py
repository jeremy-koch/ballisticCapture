"""Hand-checkable tests of the physics functions.
"""
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from physics import (A_M, A_S, DAY, MU_E, MU_M, MU_S, N_M, R_LEO, R_LLO, T_MOON, V_LEO, V_LLO,
                     Model, accel_central, accel_third_body, capture_dv, circular_speed,
                     escape_speed, hohmann_dv, hohmann_tof, leo_departure_state, moon_energy,
                     moon_state, projected_capture_dv, signed_periapsis, sun_state,
                     two_body_energy)


def test_leo_circular_speed():
    assert circular_speed(MU_E, R_LEO) == pytest.approx(7.7843, abs=1e-4)


def test_llo_circular_and_escape_speed():
    assert circular_speed(MU_M, R_LLO) == pytest.approx(1.6335, abs=1e-4)
    assert escape_speed(MU_M, R_LLO) == pytest.approx(2.3101, abs=1e-4)


def test_hohmann_leo_to_lunar_distance():
    dv1, _ = hohmann_dv(MU_E, R_LEO, A_M)
    assert dv1 == pytest.approx(3.1308, abs=1e-3)
    assert hohmann_tof(MU_E, R_LEO, A_M) / DAY == pytest.approx(4.98, abs=0.01)


def test_energy_of_circular_orbit_is_minus_mu_over_2r():
    r, v = np.array([R_LEO, 0.0]), np.array([0.0, V_LEO])
    assert two_body_energy(r, v, MU_E) == pytest.approx(-MU_E / (2 * R_LEO))


def test_signed_periapsis_circular_both_senses():
    r = np.array([R_LEO, 0.0])
    assert signed_periapsis(r, np.array([0.0, V_LEO]), MU_E) == pytest.approx(R_LEO)
    assert signed_periapsis(r, np.array([0.0, -V_LEO]), MU_E) == pytest.approx(-R_LEO)


def test_departure_state_is_tangential():
    y = leo_departure_state(3.1)
    assert math.hypot(y[0], y[1]) == pytest.approx(R_LEO)
    assert np.dot(y[:2], y[2:]) == pytest.approx(0.0, abs=1e-9)
    assert math.hypot(y[2], y[3]) == pytest.approx(V_LEO + 3.1)


def test_moon_period():
    assert T_MOON / DAY == pytest.approx(27.28, abs=0.01)
    r0, v0 = moon_state(0.0, 0.3)
    r1, v1 = moon_state(T_MOON, 0.3)
    assert np.allclose(r0, r1, atol=1e-6) and np.allclose(v0, v1, atol=1e-12)


def test_ephemeris_speeds():
    assert np.linalg.norm(moon_state(0.0, 0.0)[1]) == pytest.approx(1.0245, abs=1e-3)
    assert np.linalg.norm(sun_state(0.0, 0.0)[1]) == pytest.approx(29.78, abs=0.01)


def test_central_accel_magnitude():
    a = accel_central(np.array([R_LEO, 0.0]))
    assert a[0] == pytest.approx(-MU_E / R_LEO**2) and a[1] == 0.0


def test_indirect_term_cancels_at_earth_center():
    for rb, mu in [(np.array([A_M, 0.0]), MU_M), (np.array([0.0, A_S]), MU_S)]:
        assert np.all(accel_third_body(np.zeros(2), rb, mu) == 0.0)


def test_solar_tide_near_earth():
    # For |r| << |r_body| the perturbation is the tidal acceleration
    # (mu/d^3)(3 (r.rhat) rhat - r): stretching 2 mu x / d^3 along the Sun line,
    # squeezing -mu y / d^3 across it.
    rs = np.array([A_S, 0.0])
    k = MU_S / A_S**3
    assert accel_third_body(np.array([7000.0, 0.0]), rs, MU_S)[0] == pytest.approx(2 * k * 7000, rel=1e-3)
    assert accel_third_body(np.array([0.0, 7000.0]), rs, MU_S)[1] == pytest.approx(-k * 7000, rel=1e-3)


def _integrate(model, y0, tf):
    return solve_ivp(model.eom, (0, tf), y0, method="DOP853", rtol=1e-12, atol=1e-10,
                     dense_output=True)


def test_two_body_energy_conserved():
    model = Model(0.0, moon=False, sun=False)
    y0 = np.array([R_LEO, 0.0, 0.0, 9.5])            # an eccentric ellipse
    eps0 = two_body_energy(y0[:2], y0[2:], MU_E)
    sol = _integrate(model, y0, 10 * 86400.0)
    eps = [two_body_energy(sol.y[:2, i], sol.y[2:, i], MU_E) for i in range(sol.y.shape[1])]
    # ~50 eccentric orbits at rtol=1e-12: per-step errors accumulate to ~2e-9.
    assert np.max(np.abs(np.array(eps) / eps0 - 1)) < 1e-8


def jacobi(t, y, theta_m0):
    """CR3BP Jacobi constant computed from the Earth-centered inertial state."""
    rm, vm = moon_state(t, theta_m0)
    f = MU_M / (MU_E + MU_M)
    rho = y[:2] - f * rm                  # barycentric position
    rhodot = y[2:] - f * vm
    v_rot = rhodot - N_M * np.array([-rho[1], rho[0]])
    r1 = np.linalg.norm(y[:2])
    r2 = np.linalg.norm(y[:2] - rm)
    return N_M**2 * rho @ rho + 2 * (MU_E / r1 + MU_M / r2) - v_rot @ v_rot


def test_jacobi_constant_without_sun():
    # The Earth-centered model with the Moon's indirect term and the Sun off is
    # the CR3BP in disguise, so the Jacobi constant must be conserved.
    model = Model(theta_m0=2.0, sun=False)
    y0 = np.array([R_LEO, 0.0, 0.0, 10.9])      # a lunar-distance-ish orbit
    sol = _integrate(model, y0, 30 * DAY)
    C = [jacobi(sol.t[i], sol.y[:, i], 2.0) for i in range(len(sol.t))]
    assert np.ptp(C) / abs(C[0]) < 1e-9


def test_jacobi_changes_with_sun():
    # ...and the Sun should break it, a little (otherwise the Sun isn't doing anything).
    model = Model(theta_m0=2.0, sun=True)
    y0 = np.array([R_LEO, 0.0, 0.0, 10.9])
    sol = _integrate(model, y0, 30 * DAY)
    C = [jacobi(sol.t[i], sol.y[:, i], 2.0) for i in range(len(sol.t))]
    assert np.ptp(C) / abs(C[0]) > 1e-6


def test_capture_dv_zero_when_already_circular():
    assert capture_dv(V_LLO) == pytest.approx(0.0)


def test_projected_capture_dv_parabolic_arrival():
    # zero energy -> arrive at escape speed -> dv = (sqrt(2) - 1) v_circ
    assert projected_capture_dv(0.0) == pytest.approx((math.sqrt(2) - 1) * V_LLO)
    assert projected_capture_dv(0.0) == pytest.approx(0.6766, abs=1e-4)


def test_projected_matches_literal_at_100km():
    v = 2.4
    eps = 0.5 * v**2 - MU_M / R_LLO
    assert projected_capture_dv(eps) == pytest.approx(capture_dv(v))


def test_moon_energy_comoving_is_negative():
    # A spacecraft sitting on the Moon's orbit 10,000 km behind it, moving with it
    rm, vm = moon_state(0.0, 0.0)
    y = np.concatenate([rm + np.array([0.0, -10_000.0]), vm])
    assert moon_energy(0.0, y, 0.0) == pytest.approx(-MU_M / 10_000.0)
