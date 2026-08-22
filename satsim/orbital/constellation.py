from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from satsim.config import ConstellationConfig
from .propagation import (
    KeplerianElements,
    KeplerianPropagator,
    SGP4Propagator,
    OrbitalState,
    EARTH_RADIUS_KM,
)


class WalkerDeltaConstellation:
    """Models a Walker-Delta LEO constellation layout (e.g. 53°: 100/10/1)."""

    def __init__(
        self,
        config: Optional[ConstellationConfig] = None,
        num_satellites: int = 100,
        num_planes: int = 10,
        altitude_km: float = 550.0,
        inclination_deg: float = 53.0,
        propagation: str = "keplerian",
        phasing_f: int = 1,
    ):
        if config is not None:
            self.num_satellites = config.num_satellites
            self.num_planes = config.num_planes
            self.altitude_km = config.altitude_km
            self.inclination_deg = config.inclination_deg
            self.propagation_type = config.propagation
        else:
            self.num_satellites = num_satellites
            self.num_planes = num_planes
            self.altitude_km = altitude_km
            self.inclination_deg = inclination_deg
            self.propagation_type = propagation

        self.phasing_f = phasing_f
        self.sats_per_plane = self.num_satellites // self.num_planes

        if self.propagation_type == "sgp4":
            self.propagator = SGP4Propagator()
        else:
            self.propagator = KeplerianPropagator()

        self.elements: Dict[int, KeplerianElements] = {}
        self._build_constellation()

    def _build_constellation(self) -> None:
        """Generates initial orbital elements for each satellite in the Walker-Delta pattern."""
        a = EARTH_RADIUS_KM + self.altitude_km
        inc = np.radians(self.inclination_deg)
        delta_raan = 2.0 * np.pi / self.num_planes
        delta_mean_anomaly = 2.0 * np.pi / self.sats_per_plane
        delta_phase = self.phasing_f * (2.0 * np.pi / self.num_satellites)

        sat_id = 0
        for p in range(self.num_planes):
            raan = p * delta_raan
            for s in range(self.sats_per_plane):
                m_0 = (s * delta_mean_anomaly + p * delta_phase) % (2.0 * np.pi)
                self.elements[sat_id] = KeplerianElements(
                    a=a,
                    e=0.0,  # Circular orbit assumption
                    inc=inc,
                    raan=raan,
                    arg_perigee=0.0,
                    mean_anomaly_0=m_0,
                    epoch_0=0.0,
                )
                sat_id += 1

    def get_sat_plane_and_index(self, sat_id: int) -> Tuple[int, int]:
        """Returns (plane_index, index_within_plane) for a given satellite ID."""
        plane = sat_id // self.sats_per_plane
        idx_in_plane = sat_id % self.sats_per_plane
        return plane, idx_in_plane

    def get_states(self, t_s: float) -> Dict[int, OrbitalState]:
        """Propagates all satellites to time t_s and returns mapping sat_id -> OrbitalState."""
        states = {}
        for sat_id, elem in self.elements.items():
            states[sat_id] = self.propagator.propagate_sat(elem, t_s, sat_id=sat_id)
        return states

    def get_positions_eci(self, t_s: float) -> np.ndarray:
        """Returns array of shape (N, 3) with ECI positions in km."""
        states = self.get_states(t_s)
        return np.array([states[i].position_eci for i in range(self.num_satellites)])

    def get_positions_ecef(self, t_s: float) -> np.ndarray:
        """Returns array of shape (N, 3) with ECEF positions in km."""
        states = self.get_states(t_s)
        return np.array([states[i].position_ecef for i in range(self.num_satellites)])
