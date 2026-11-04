"""The ballistic transfer, found by brute force: try lots of departures, keep the cheapest.

Each departure is set by two numbers, the burn dv and the Moon's phase. We fly
all ~29,000 combinations, note how close each one gets to the Moon, and score
it by the total dv (departure + capture burn at the closest approach).

Then we plot the results, with the Hohmann-style transfer for comparison.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import hohmann
from moon import D_MOON, DAY, MU_MOON, R_LLO, R_MOON, W_SUN, capture_dv, fly

SUN_ANGLE = hohmann.SUN_ANGLE

# The grid. The dv range is narrow on purpose: this is the window (found by an
# earlier, wider search) where the Sun's pull brings us back to the Moon's
# distance. Widen it and you'll mostly see escapes and Earth crashes.
DV_VALUES = np.linspace(3.1938, 3.1958, 81)              # [km/s]
PHASE_VALUES = np.radians(np.arange(0, 360, 1.0))        # [rad]

# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
dv, phase = np.meshgrid(DV_VALUES, PHASE_VALUES)         # every combination
dv, phase = dv.ravel(), phase.ravel()
print(f"flying {len(dv)} spacecraft for 200 days (about 2 minutes)...")
result = fly(dv, phase, SUN_ANGLE, days=200)

# Count a run if it got within 10,000 km of the Moon's surface without
# crashing, slowly enough that a burn at 100 km could circularize it
reached = (~result["crashed"]
           & (result["closest"] - R_MOON < 10_000)
           & (result["energy"] > -MU_MOON / R_LLO))
total = np.where(reached, dv + capture_dv(result["energy"]), np.nan)
tof = result["t_closest"] / DAY

best = np.nanargmin(total)
print(f"{reached.sum()} of {len(dv)} reached the Moon")
print(f"best: dv = {dv[best]:.5f} km/s, phase = {np.degrees(phase[best]):.0f} deg, "
      f"capture = {total[best] - dv[best]:.4f} km/s, total = {total[best]:.4f} km/s, "
      f"{tof[best]:.0f} days")

hoh_dv, hoh_phase, hoh = hohmann.solve()
hoh_total = hoh_dv + capture_dv(hoh["energy"][0])
hoh_tof = hoh["t_closest"][0] / DAY
print(f"Hohmann-style: total = {hoh_total:.4f} km/s, {hoh_tof:.1f} days")

# ---------------------------------------------------------------------------
# Figure 1: the two trajectories
# ---------------------------------------------------------------------------
# Fly the two winners again, this time keeping every step
hoh_path = fly(hoh_dv, hoh_phase, SUN_ANGLE, days=hoh_tof, keep_path=True)
bal_path = fly(dv[best], phase[best], SUN_ANGLE, days=tof[best], keep_path=True)
x, y = bal_path["path"][:, 0, 0], bal_path["path"][:, 1, 0]
t = bal_path["t"][:, 0]
print(f"(re-flown best: closest approach {bal_path['closest'][0] - R_MOON:.0f} km, "
      f"sweep said {result['closest'][best] - R_MOON:.0f} km)")

# In a frame that rotates with the Sun (Sun always on the -x axis), the
# ballistic trajectory is easier to read
angle = np.pi - (SUN_ANGLE + W_SUN * t)
x_rot = np.cos(angle) * x - np.sin(angle) * y
y_rot = np.sin(angle) * x + np.cos(angle) * y

circle = np.linspace(0, 2 * np.pi, 200)
os.makedirs("figures", exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
titles = [f"Hohmann-style, {hoh_tof:.1f} days",
          f"Ballistic, {tof[best]:.0f} days",
          "Ballistic, rotating with the Sun (Sun to the left)"]
paths = [hoh_path["path"][:, :2, 0].T, (x, y), (x_rot, y_rot)]
for ax, title, (px, py) in zip(axes, titles, paths):
    ax.plot(D_MOON * np.cos(circle) / 1e3, D_MOON * np.sin(circle) / 1e3,
            "--", color="gray", label="Moon's orbit")
    ax.plot(px / 1e3, py / 1e3, label="spacecraft")
    ax.plot(0, 0, "o", color="black")
    ax.set(title=title, xlabel="x [1000 km]", ylabel="y [1000 km]", aspect="equal")
axes[0].legend(loc="lower left")
fig.tight_layout()
fig.savefig("figures/trajectories.png", dpi=150)

# ---------------------------------------------------------------------------
# Figure 2: what the sweep found
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# (left) total dv over the whole grid; blank = never reached the Moon
grid = total.reshape(len(PHASE_VALUES), len(DV_VALUES)).T
image = ax1.imshow(grid, origin="lower", aspect="auto", vmin=3.82, vmax=3.95,
                   extent=[0, 360, DV_VALUES[0], DV_VALUES[-1]])
fig.colorbar(image, ax=ax1, label="total Δv [km/s]")
ax1.plot(np.degrees(phase[best]), dv[best], "r*", markersize=15, label="best")
ax1.set(xlabel="Moon's phase at departure [deg]", ylabel="departure Δv [km/s]",
        title="Which departures reach the Moon?")
ax1.legend()

# (right) the trade-off: cheaper, but much slower
ax2.plot(tof, total, ".", label="ballistic (sweep)")
ax2.plot(hoh_tof, hoh_total, "s", markersize=10, label="Hohmann-style")
ax2.set(xlabel="time of flight [days]", ylabel="total Δv [km/s]", ylim=(3.8, 4.1),
        title="Total Δv vs. time of flight")
ax2.legend()
ax2.grid()
fig.tight_layout()
fig.savefig("figures/sweep.png", dpi=150)
print("saved figures/trajectories.png and figures/sweep.png")
