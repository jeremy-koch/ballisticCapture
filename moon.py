"""The physics: Earth, Moon, and Sun gravity, and a hand-written RK4 time-stepper.

Everything happens in a plane, in a frame centered on the Earth that does not
rotate. Units are km, s, and radians throughout.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MU_EARTH = 398600.4418       # gravitational parameter G*M [km^3/s^2]
MU_MOON = 4902.800066
MU_SUN = 1.32712440018e11

R_EARTH = 6378.137           # radius [km]
R_MOON = 1737.4

D_MOON = 384400.0            # distance from Earth [km]
D_SUN = 1.495978707e8

W_MOON = np.sqrt((MU_EARTH + MU_MOON) / D_MOON**3)   # Moon's angular speed [rad/s]
W_SUN = 2 * np.pi / (365.256363 * 86400)             # Sun's (apparent) angular speed

DAY = 86400.0                # [s]

R_LEO = R_EARTH + 200        # start: circular orbit 200 km above the Earth
R_LLO = R_MOON + 100         # goal: circular orbit 100 km above the Moon
V_LEO = np.sqrt(MU_EARTH / R_LEO)   # circular orbit speeds [km/s]
V_LLO = np.sqrt(MU_MOON / R_LLO)


# ---------------------------------------------------------------------------
# Where the Moon and Sun are. Both move on circles, counter-clockwise.
#   phase     = the Moon's angle at t = 0
#   sun_angle = the Sun's angle at t = 0
# The spacecraft always starts on the +x axis.
# ---------------------------------------------------------------------------
def moon_position(t, phase):
    angle = phase + W_MOON * t
    return D_MOON * np.cos(angle), D_MOON * np.sin(angle)


def sun_position(t, sun_angle):
    angle = sun_angle + W_SUN * t
    return D_SUN * np.cos(angle), D_SUN * np.sin(angle)


# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------
def pull(x, y, bx, by, mu, d):
    """Acceleration on the spacecraft at (x, y) due to a body at (bx, by).

    Our frame is centered on the Earth, and the body pulls on the Earth too. So
    what we feel *relative to the Earth* is the body's pull on us minus its pull
    on the Earth. (d is the body's distance from the Earth.)
    """
    dx, dy = bx - x, by - y
    r3 = (dx**2 + dy**2) ** 1.5
    return mu * (dx / r3 - bx / d**3), mu * (dy / r3 - by / d**3)


def derivative(t, state, phase, sun_angle):
    """d(state)/dt, where state = [x, y, vx, vy]."""
    x, y, vx, vy = state
    r3 = (x**2 + y**2) ** 1.5
    ax, ay = -MU_EARTH * x / r3, -MU_EARTH * y / r3

    mx, my = moon_position(t, phase)
    amx, amy = pull(x, y, mx, my, MU_MOON, D_MOON)
    sx, sy = sun_position(t, sun_angle)
    asx, asy = pull(x, y, sx, sy, MU_SUN, D_SUN)

    return np.array([vx, vy, ax + amx + asx, ay + amy + asy])


# ---------------------------------------------------------------------------
# Time-stepping
# ---------------------------------------------------------------------------
def rk4_step(f, t, state, dt, *args):
    """One classic Runge-Kutta (RK4) step of d(state)/dt = f(t, state, *args)."""
    k1 = f(t, state, *args)
    k2 = f(t + dt / 2, state + dt / 2 * k1, *args)
    k3 = f(t + dt / 2, state + dt / 2 * k2, *args)
    k4 = f(t + dt, state + dt * k3, *args)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def fly(dv, phase, sun_angle, days=200, step=0.01, keep_path=False):
    """Burn dv [km/s] forward from the circular LEO, then coast for `days`.

    dv and phase can be arrays: then we fly one spacecraft per entry, all at
    the same time (numpy does the bookkeeping). Each spacecraft keeps its own
    clock, because each one gets its own step size.

    Returns a dict of arrays, one entry per spacecraft:
        closest   -- closest distance to the Moon's center [km]
        energy    -- energy relative to the Moon at that moment [km^2/s^2]
        t_closest -- when that happened [s]
        crashed   -- hit the Earth or the Moon
    and, if keep_path, also "t" and "path" (every step, for plotting).
    """
    dv = np.atleast_1d(np.asarray(dv, dtype=float))
    phase = np.broadcast_to(phase, dv.shape)
    n = len(dv)
    t_end = days * DAY

    # Start on the +x axis, moving in +y, with the burn added to the LEO speed
    state = np.array([np.full(n, R_LEO), np.zeros(n), np.zeros(n), V_LEO + dv])
    t = np.zeros(n)
    flying = np.ones(n, dtype=bool)
    crashed = np.zeros(n, dtype=bool)
    closest = np.full(n, np.inf)
    energy = np.full(n, np.nan)
    t_closest = np.full(n, np.nan)
    mx, my = moon_position(0.0, phase)
    r_moon = np.hypot(R_LEO - mx, my)
    r_earth = np.full(n, R_LEO)
    times, path = [t.copy()], [state.copy()]

    while flying.any():
        # Step size: a small fraction of the local orbital time scale
        # sqrt(r^3 / mu). That gives small steps near the Earth or the Moon
        # (where things change quickly) and big steps far from both.
        dt = step * np.minimum(np.sqrt(r_earth**3 / MU_EARTH), np.sqrt(r_moon**3 / MU_MOON))
        dt = np.minimum(dt, t_end - t)
        dt[~flying] = 0.0                    # finished spacecraft just stay put

        state = rk4_step(derivative, t, state, dt, phase, sun_angle)
        t = t + dt

        # Position and velocity relative to the Moon
        mx, my = moon_position(t, phase)
        rx, ry = state[0] - mx, state[1] - my
        vx, vy = state[2] + W_MOON * my, state[3] - W_MOON * mx   # Moon's velocity is W*(-my, mx)
        r_moon = np.hypot(rx, ry)
        r_earth = np.hypot(state[0], state[1])

        # Remember the closest approach to the Moon so far
        closer = flying & (r_moon < closest)
        closest[closer] = r_moon[closer]
        energy[closer] = 0.5 * (vx[closer]**2 + vy[closer]**2) - MU_MOON / r_moon[closer]
        t_closest[closer] = t[closer]

        # Stop spacecraft that crashed, escaped, or ran out of time
        crashed |= flying & ((r_earth < R_EARTH) | (r_moon < R_MOON))
        flying &= ~crashed & (r_earth < 3e6) & (t < t_end)

        if keep_path:
            times.append(t.copy())
            path.append(state.copy())

    out = dict(closest=closest, energy=energy, t_closest=t_closest, crashed=crashed)
    if keep_path:
        out["t"], out["path"] = np.array(times), np.array(path)
    return out


# ---------------------------------------------------------------------------
# The capture burn
# ---------------------------------------------------------------------------
def capture_dv(energy):
    """Burn needed at 100 km above the Moon to end up in a circular orbit there.

    We assume the energy relative to the Moon doesn't change on the way down to
    100 km (true if the Moon were the only thing pulling on us). Then
        v^2 / 2 - mu / r = energy   =>   v = sqrt(2 (energy + mu / r)),
    and the burn is the difference between that and circular speed.
    """
    return np.sqrt(2 * (energy + MU_MOON / R_LLO)) - V_LLO
