"""Checks we can do by hand. Run with:  python -m pytest"""
import numpy as np
import pytest

from moon import (D_MOON, D_SUN, MU_EARTH, MU_MOON, MU_SUN, R_LEO, V_LEO, V_LLO, W_MOON,
                  capture_dv, pull, rk4_step)


def test_rk4_one_circular_orbit():
    # Earth's gravity only: after one period we should be back where we started.
    def two_body(t, s):
        x, y, vx, vy = s
        r3 = (x**2 + y**2) ** 1.5
        return np.array([vx, vy, -MU_EARTH * x / r3, -MU_EARTH * y / r3])

    period = 2 * np.pi * np.sqrt(R_LEO**3 / MU_EARTH)
    n = 1000
    s = np.array([R_LEO, 0.0, 0.0, V_LEO])
    for i in range(n):
        s = rk4_step(two_body, i * period / n, s, period / n)
    assert np.hypot(s[0] - R_LEO, s[1]) < 0.001        # within 1 m


def test_leo_and_moon_speeds():
    assert V_LEO == pytest.approx(7.784, abs=1e-3)
    assert W_MOON * D_MOON == pytest.approx(1.0245, abs=1e-3)   # Moon's orbital speed [km/s]


def test_no_pull_at_earths_center():
    # At the Earth's center we move with the Earth, so we feel nothing extra.
    # (approx, because (d^2)^1.5 and d^3 can differ in the last digit)
    assert pull(0.0, 0.0, D_MOON, 0.0, MU_MOON, D_MOON) == pytest.approx((0.0, 0.0), abs=1e-15)
    assert pull(0.0, 0.0, 0.0, D_SUN, MU_SUN, D_SUN) == pytest.approx((0.0, 0.0), abs=1e-15)


def test_capture_dv():
    assert capture_dv(-MU_MOON / (2 * (1737.4 + 100))) == pytest.approx(0.0)   # already circular
    # Arriving with zero energy means escape speed, sqrt(2) times circular speed
    assert capture_dv(0.0) == pytest.approx((np.sqrt(2) - 1) * V_LLO)
