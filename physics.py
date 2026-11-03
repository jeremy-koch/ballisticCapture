"""All the physics for the Earth-Moon-Sun model, as small functions you can test by hand.

Units: km, s, rad. The frame is centered on Earth and does not rotate.
Sections, in order: constants, Moon/Sun positions, two-body formulas, equations
of motion, the departure burn, and Moon-relative quantities (capture Δv).
"""
import math

import numpy as np


# ============================================================================
# Constants
# ============================================================================
#
# Physical constants. Units: km, s, rad.

MU_E = 398600.4418          # Earth GM [km^3/s^2]
MU_M = 4902.800066          # Moon GM
MU_S = 1.32712440018e11     # Sun GM

R_E = 6378.137              # Earth equatorial radius [km]
R_M = 1737.4                # Moon mean radius [km]

A_M = 384400.0              # Earth-Moon distance (circular) [km]
A_S = 1.495978707e8         # Earth-Sun distance, 1 AU [km]

# The Moon circles Earth at the two-body rate of the Earth-Moon pair. With the
# Moon's indirect term included, this makes the Earth-Moon part of the model an
# exact solution (it reduces to the CR3BP when the Sun is switched off).
N_M = math.sqrt((MU_E + MU_M) / A_M**3)    # [rad/s]
T_MOON = 2.0 * math.pi / N_M               # ~27.28 d with A_M = 384400 km

# The Sun's apparent motion around Earth: the sidereal year.
T_YEAR = 365.256363 * 86400.0
N_S = 2.0 * math.pi / T_YEAR

DAY = 86400.0

# Mission
H_LEO = 200.0
H_LLO = 100.0
R_LEO = R_E + H_LEO
R_LLO = R_M + H_LLO


# ============================================================================
# Moon and Sun positions
# ============================================================================
#
# Prescribed circular motion of the Moon and Sun in the Earth-centered inertial frame.
#
# Both move counter-clockwise (the same sense as the spacecraft's LEO).

def circular_body_state(t, radius, rate, theta0):
    """Position and velocity of a body on a circle: angle = theta0 + rate*t."""
    th = theta0 + rate * t
    c, s = np.cos(th), np.sin(th)
    r = radius * np.array([c, s])
    v = radius * rate * np.array([-s, c])
    return r, v


def moon_state(t, theta_m0):
    return circular_body_state(t, A_M, N_M, theta_m0)


def sun_state(t, theta_s0):
    return circular_body_state(t, A_S, N_S, theta_s0)


# ============================================================================
# Two-body formulas
# ============================================================================
#
# Two-body (Keplerian) helpers. Every function takes plain floats or arrays.

def circular_speed(mu, r):
    return math.sqrt(mu / r)


def escape_speed(mu, r):
    return math.sqrt(2.0 * mu / r)


def two_body_energy(r_vec, v_vec, mu):
    """Specific orbital energy v^2/2 - mu/r."""
    r = np.linalg.norm(r_vec)
    v2 = float(np.dot(v_vec, v_vec))
    return 0.5 * v2 - mu / r


def vis_viva_speed(mu, r, a):
    return math.sqrt(mu * (2.0 / r - 1.0 / a))


def hohmann_dv(mu, r1, r2):
    """(dv1, dv2) for a Hohmann transfer between circular orbits r1 -> r2."""
    a = 0.5 * (r1 + r2)
    dv1 = vis_viva_speed(mu, r1, a) - circular_speed(mu, r1)
    dv2 = circular_speed(mu, r2) - vis_viva_speed(mu, r2, a)
    return dv1, dv2


def hohmann_tof(mu, r1, r2):
    """Half the transfer-ellipse period."""
    a = 0.5 * (r1 + r2)
    return math.pi * math.sqrt(a**3 / mu)


def signed_periapsis(r_vec, v_vec, mu):
    """Osculating periapsis radius, signed by the planar angular momentum h_z.

    Positive = prograde (counter-clockwise) pass, negative = retrograde. As a
    trajectory is swept across the attracting point the sign flips through zero,
    so the quantity is continuous -- handy for root finding.
    """
    x, y = r_vec
    vx, vy = v_vec
    hz = x * vy - y * vx
    eps = two_body_energy(r_vec, v_vec, mu)
    p = hz * hz / mu
    e2 = 1.0 + 2.0 * eps * hz * hz / (mu * mu)
    e = math.sqrt(max(e2, 0.0))
    rp = p / (1.0 + e)
    return math.copysign(rp, hz)


# ============================================================================
# Equations of motion
# ============================================================================
#
# Equations of motion for the planar bicircular model, Earth-centered inertial frame.

def accel_central(r, mu=MU_E):
    """Point-mass acceleration toward the origin: -mu r / |r|^3."""
    d = np.linalg.norm(r)
    return -mu * r / d**3


def accel_third_body(r, r_body, mu):
    """Third-body perturbation in a frame centered on the primary.

    direct term   : mu (r_body - r) / |r_body - r|^3   (body pulls the spacecraft)
    indirect term : -mu r_body / |r_body|^3            (body pulls Earth, i.e. the frame)
    At r = 0 the two cancel exactly.
    """
    d = r_body - r
    return mu * (d / np.linalg.norm(d)**3 - r_body / np.linalg.norm(r_body)**3)


class Model:
    """Holds the initial phase angles and which perturbers are on."""

    def __init__(self, theta_m0, theta_s0=0.0, moon=True, sun=True):
        self.theta_m0 = theta_m0
        self.theta_s0 = theta_s0
        self.moon = moon
        self.sun = sun

    def accel(self, t, r):
        a = accel_central(r, MU_E)
        if self.moon:
            rm, _ = moon_state(t, self.theta_m0)
            a = a + accel_third_body(r, rm, MU_M)
        if self.sun:
            rs, _ = sun_state(t, self.theta_s0)
            a = a + accel_third_body(r, rs, MU_S)
        return a

    def eom(self, t, y):
        r = y[:2]
        a = self.accel(t, r)
        return np.array([y[2], y[3], a[0], a[1]])


# ============================================================================
# Departure from LEO
# ============================================================================
#
# Departure from the 200 km circular LEO by a single tangential burn.
#
# By rotational symmetry we always depart from the +x axis; the Moon's and Sun's
# initial angles are measured from there.

V_LEO = circular_speed(MU_E, R_LEO)


def leo_departure_state(dv, angle=0.0, r0=R_LEO):
    """State [x, y, vx, vy] right after a prograde tangential burn of size dv."""
    v0 = circular_speed(MU_E, r0) + dv
    c, s = np.cos(angle), np.sin(angle)
    return np.array([r0 * c, r0 * s, -v0 * s, v0 * c])


# ============================================================================
# Moon-relative quantities and capture Δv
# ============================================================================
#
# Moon-relative quantities and the capture Δv.

V_LLO = circular_speed(MU_M, R_LLO)


def relative_to_moon(t, y, theta_m0):
    """Selenocentric position and velocity (both in the inertial orientation)."""
    rm, vm = moon_state(t, theta_m0)
    return y[:2] - rm, y[2:4] - vm


def moon_energy(t, y, theta_m0):
    r, v = relative_to_moon(t, y, theta_m0)
    return two_body_energy(r, v, MU_M)


def capture_dv(v_rel_peri, r_target=R_LLO):
    """The literal definition: speed at closest approach minus circular speed at 100 km."""
    return v_rel_peri - circular_speed(MU_M, r_target)


def projected_capture_dv(energy, r_target=R_LLO):
    """Capture Δv if the same selenocentric energy were carried down to r_target.

    v(r_target) = sqrt(2 (energy + mu/r_target)). This is only an estimate: it
    ignores the (small) cost of moving periselene to 100 km.
    """
    v = math.sqrt(2.0 * (energy + MU_M / r_target))
    return capture_dv(v, r_target)
