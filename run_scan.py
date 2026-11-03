"""Grid scan; every run is appended to results/runs.csv (resumable).

python run_scan.py --dv 3.1938 3.1958 81 --phase 0 360 1 --sun 25
"""
import argparse
import os
import time

import numpy as np

from search import make_grid, run_scan

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dv", nargs=3, type=float, default=[3.1938, 3.1958, 81],
                   metavar=("MIN", "MAX", "N"), help="departure Δv range [km/s]")
    p.add_argument("--phase", nargs=3, type=float, default=[0.0, 360.0, 1.0],
                   metavar=("MIN", "MAX", "STEP"), help="Moon phase range [deg], MAX excluded")
    p.add_argument("--sun", nargs="+", type=float,
                   default=[25.0], help="Sun angles [deg]")
    p.add_argument("--out", default=os.path.join(RESULTS, "runs.csv"))
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args()
    jobs = make_grid(np.linspace(a.dv[0], a.dv[1], int(a.dv[2])),
                     np.round(np.arange(a.phase[0], a.phase[1], a.phase[2]), 6), a.sun)
    print(f"{len(jobs)} jobs -> {a.out}")
    t = time.time()
    n = run_scan(jobs, a.out, workers=a.workers)
    print(f"ran {n} new jobs in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
