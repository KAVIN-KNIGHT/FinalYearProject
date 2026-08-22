from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

# Earth Constants
EARTH_RADIUS_KM = 6371.0
MU_EARTH_KM3_S2 = 398600.4418
EARTH_ROTATION_RAD_S = 7.2921159e-5
SPEED_OF_LIGHT_KM_S = 299792.458


@dataclass
class OrbitalState:
    sat_id: int
    epoch_s: float
    position_eci: np.ndarray  # shape (3,) in km
    velocity_eci: np.ndarray  # shape (3,) in km/s
    position_ecef: np.ndarray  # shape (3,) in km
    velocity_ecef: np.ndarray  # shape (3,) in km/s


@dataclass
class KeplerianElements:
    a: float  # semi-major axis (km)
    e: float  # eccentricity (0 <= e < 1)
    inc: float  # inclination (radians)
    raan: float  # right ascension of ascending node (radians)
    arg_perigee: float  # argument of perigee (radians)
    mean_anomaly_0: float  # initial mean anomaly at epoch 0 (radians)
    epoch_0: float = 0.0  # epoch reference time (seconds)

    @property
    def mean_motion(self) -> float:
        """Mean motion n = sqrt(mu / a^3) in rad/s."""
        return np.sqrt(MU_EARTH_KM3_S2 / (self.a**3))


class KeplerianPropagator:
    """
    Analytical Keplerian orbital propagator converting orbital elements to ECI & ECEF.

    ROTATION MATRIX CONVENTION (Active Euler Transformation):
    -------------------------------------------------------
    Computes position vector r_ECI = R_z(raan) @ R_x(inc) @ R_z(arg_perigee) @ r_orb
    where:
        R_z(theta) = [[cos(theta), -sin(theta), 0],
                      [sin(theta),  cos(theta), 0],
                      [0,           0,          1]]
        R_x(theta) = [[1, 0,           0          ],
                      [0, cos(theta), -sin(theta) ],
                      [0, sin(theta),  cos(theta) ]]

    This active right-handed rotation convention ensures positive inclination (inc) rotates the
    orbital plane around the X-axis (line of nodes), resulting in a peak z-coordinate latitude
    satisfying max(|arcsin(z / |r|)|) == inc.
    """

    def __init__(self, earth_rotation_rad_s: float = EARTH_ROTATION_RAD_S):
        self.omega_e = earth_rotation_rad_s

    def solve_kepler(self, M: float, e: float, tol: float = 1e-10) -> float:
        """Solves Kepler's equation M = E - e*sin(E) for Eccentric Anomaly E using Newton-Raphson."""
        if e < 1e-6:
            return M
        M = M % (2.0 * np.pi)
        E = M
        for _ in range(100):
            f = E - e * np.sin(E) - M
            f_prime = 1.0 - e * np.cos(E)
            delta = f / f_prime
            E -= delta
            if abs(delta) < tol:
                break
        return E

    def propagate_sat(
        self, elements: KeplerianElements, t_s: float, sat_id: int = 0
    ) -> OrbitalState:
        dt = t_s - elements.epoch_0
        n = elements.mean_motion
        M = (elements.mean_anomaly_0 + n * dt) % (2.0 * np.pi)

        # Actively solve Kepler's equation for Eccentric Anomaly E
        E = self.solve_kepler(M, elements.e)

        # True anomaly nu and radius r
        if elements.e < 1e-6:
            nu = M
            r = elements.a
        else:
            sin_nu = (np.sqrt(1.0 - elements.e**2) * np.sin(E)) / (1.0 - elements.e * np.cos(E))
            cos_nu = (np.cos(E) - elements.e) / (1.0 - elements.e * np.cos(E))
            nu = np.arctan2(sin_nu, cos_nu)
            r = elements.a * (1.0 - elements.e * np.cos(E))

        # Orbital plane coordinates & velocity
        u = elements.arg_perigee + nu
        x_orb = r * np.cos(u)
        y_orb = r * np.sin(u)
        z_orb = 0.0

        p = elements.a * (1.0 - elements.e**2)
        h = np.sqrt(MU_EARTH_KM3_S2 * p)
        vx_orb = (x_orb * h * elements.e / (r * p)) - (h / r) * np.sin(u)
        vy_orb = (y_orb * h * elements.e / (r * p)) + (h / r) * np.cos(u)
        vz_orb = 0.0

        r_orb = np.array([x_orb, y_orb, z_orb], dtype=np.float64)
        v_orb = np.array([vx_orb, vy_orb, vz_orb], dtype=np.float64)

        # Active rotation matrices
        raan = elements.raan
        inc = elements.inc

        R_raan = np.array(
            [
                [np.cos(raan), -np.sin(raan), 0.0],
                [np.sin(raan), np.cos(raan), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        R_inc = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(inc), -np.sin(inc)],
                [0.0, np.sin(inc), np.cos(inc)],
            ]
        )

        R = R_raan @ R_inc

        pos_eci = R @ r_orb
        vel_eci = R @ v_orb

        # ECI to ECEF transformation (Earth rotation)
        theta = self.omega_e * t_s
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        R_ecef = np.array(
            [
                [cos_t, sin_t, 0.0],
                [-sin_t, cos_t, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        pos_ecef = R_ecef @ pos_eci
        omega_vec = np.array([0.0, 0.0, self.omega_e])
        vel_ecef = (R_ecef @ vel_eci) - np.cross(omega_vec, pos_ecef)

        return OrbitalState(
            sat_id=sat_id,
            epoch_s=t_s,
            position_eci=pos_eci,
            velocity_eci=vel_eci,
            position_ecef=pos_ecef,
            velocity_ecef=vel_ecef,
        )


class SGP4Propagator:
    """Optional SGP4 propagator fallback interface."""

    def __init__(self):
        try:
            from sgp4.api import Satrec

            self.available = True
        except ImportError:
            self.available = False
        self.fallback = KeplerianPropagator()

    def propagate_sat(
        self, elements: KeplerianElements, t_s: float, sat_id: int = 0
    ) -> OrbitalState:
        return self.fallback.propagate_sat(elements, t_s, sat_id)
