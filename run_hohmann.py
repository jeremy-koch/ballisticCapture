"""Solve the Hohmann-style transfer in the full model; writes results/hohmann.json.
"""
import json
import math
import os

from hohmann import solve_hohmann, two_body_guess

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)


def main(sun_deg=0.0):
    dv0, ph0, tof0 = two_body_guess()
    print(f"two-body guess: dv1 = {dv0:.4f} km/s, phase = {math.degrees(ph0):.2f} deg, "
          f"TOF = {tof0 / 86400:.3f} d")
    best = None
    for sense in (+1, -1):
        dv, ph, res = solve_hohmann(sun_angle=math.radians(sun_deg), sense=sense)
        print(f"  {'prograde ' if sense > 0 else 'retrograde'}: dv_dep {res.dv_dep:.4f}  "
              f"dv_cap {res.dv_cap:.4f}  total {res.dv_tot:.4f} km/s  TOF {res.tof_days:.3f} d  "
              f"alt {res.alt_ca:.3f} km  energy {res.energy_ca:+.4f}")
        if best is None or res.dv_tot < best.dv_tot:
            best = res
    out = dict(best.as_row(), phase=math.radians(best.phase_deg), sun=math.radians(sun_deg))
    with open(os.path.join(RESULTS, "hohmann.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
