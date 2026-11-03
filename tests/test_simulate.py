"""End-to-end checks of single simulated runs.
"""
import math

import pytest

from hohmann import solve_hohmann, two_body_guess
from simulate import simulate


def test_hohmann_in_full_model_close_to_two_body():
    dv, phase, res = solve_hohmann(sense=+1)
    dv0, _, _ = two_body_guess()
    assert res.alt_ca == pytest.approx(100.0, abs=0.01)
    assert abs(dv - dv0) < 0.01                   # within 10 m/s of the two-body value
    assert 4.5 < res.tof_days < 5.0               # Moon's pull shortens the 4.98 d slightly
    assert res.energy_ca > 0 and not res.captured  # Hohmann arrives hyperbolic
    assert res.dv_cap == pytest.approx(0.81, abs=0.02)


def test_low_departure_never_reaches_the_moon():
    # Well below the lunar Hohmann value: apogee ~203,000 km, nowhere near the Moon.
    # (With this phase the Moon's pull at apogee drops perigee ~340 km, below
    # Earth's surface, so an Earth impact on the first return is the right answer.)
    res = simulate(3.05, math.radians(114.0), t_search=8 * 86400.0)
    assert res.outcome == "impact_earth"
    assert res.alt_ca > 150_000 and not res.captured
