"""The Hohmann-style transfer, found with the shooting method.

1. Pencil-and-paper guess: the two-body Hohmann transfer from LEO out to the
   Moon's distance, with the Moon placed so it gets there when we do.
2. That guess ignores the Moon's own gravity, so in the full model it misses
   (or hits!) the Moon. We fix it by shooting: adjust the departure burn until
   the closest approach is exactly 100 km, using bisection.
"""
import numpy as np

from moon import D_MOON, DAY, MU_EARTH, R_LEO, R_LLO, V_LEO, W_MOON, capture_dv, fly

SUN_ANGLE = np.radians(25)


def two_body_guess():
    a = (R_LEO + D_MOON) / 2                              # transfer ellipse
    dv = np.sqrt(MU_EARTH * (2 / R_LEO - 1 / a)) - V_LEO  # vis-viva at perigee
    tof = np.pi * np.sqrt(a**3 / MU_EARTH)                # half an orbit
    phase = np.pi - W_MOON * tof      # the Moon must be at 180 deg after `tof`
    return dv, phase


def miss_distance(dv, phase):
    """Closest approach to the Moon minus our 100 km target [km].

    Negative means we passed too close (or crashed into the Moon).
    """
    result = fly(dv, phase, SUN_ANGLE, days=8)
    return result["closest"][0] - R_LLO


def shoot(phase, low, high, tol=1e-9):
    """Bisection: find dv in [low, high] where miss_distance changes sign."""
    assert miss_distance(low, phase) > 0 > miss_distance(high, phase), "bad bracket"
    while high - low > tol:
        mid = (low + high) / 2
        if miss_distance(mid, phase) > 0:
            low = mid
        else:
            high = mid
    return low


def solve():
    """Returns (departure dv, phase, result of fly) for the Hohmann-style transfer."""
    dv_guess, phase = two_body_guess()
    # The guess runs into the Moon; 10 m/s less misses it by ~38,000 km.
    dv = shoot(phase, dv_guess - 0.010, dv_guess)
    return dv, phase, fly(dv, phase, SUN_ANGLE, days=8)


if __name__ == "__main__":
    dv_guess, phase = two_body_guess()
    print(f"two-body guess:  dv = {dv_guess:.4f} km/s, phase = {np.degrees(phase):.1f} deg")
    if miss_distance(dv_guess, phase) < -100:
        print("  ...which crashes into the Moon (the Moon's gravity bends us in)")

    dv, phase, result = solve()
    dv_cap = capture_dv(result["energy"][0])
    print(f"after shooting:  dv = {dv:.4f} km/s, "
          f"miss = {result['closest'][0] - R_LLO:.3f} km")
    print(f"capture burn:    {dv_cap:.4f} km/s")
    print(f"total:           {dv + dv_cap:.4f} km/s in {result['t_closest'][0] / DAY:.2f} days")
