from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import networkx as nx
import numpy as np

from satsim.config import ISLConfig
from satsim.orbital.propagation import (
    OrbitalState,
    EARTH_RADIUS_KM,
    SPEED_OF_LIGHT_KM_S,
)


class ISLManager:
    """
    Manages dynamic Inter-Satellite Link (ISL) topology graph construction and updates.

    SEAM & POLAR ISL DISABLING:
    --------------------------
    Real Walker-Delta constellations disable inter-plane ISLs near polar regions (|lat| > 70 deg)
    and across the seam between counter-rotating adjacent planes (e.g. plane 0 <-> plane P-1),
    because relative Doppler velocities are too high for optical transceivers to maintain tracking.
    """

    def __init__(
        self,
        config: Optional[ISLConfig] = None,
        max_range_km: float = 5000.0,
        min_elevation_deg: float = 10.0,
        atm_clearance_km: float = 80.0,
        max_isls_per_sat: int = 4,
        disable_seam_and_polar_isls: bool = True,
        max_latitude_deg: float = 70.0,
    ):
        if config is not None:
            self.max_range_km = config.max_range_km
            self.min_elevation_deg = config.min_elevation_deg
        else:
            self.max_range_km = max_range_km
            self.min_elevation_deg = min_elevation_deg

        self.atm_clearance_km = atm_clearance_km
        self.earth_obstruction_radius = EARTH_RADIUS_KM + self.atm_clearance_km
        self.max_isls_per_sat = max_isls_per_sat
        self.disable_seam_and_polar_isls = disable_seam_and_polar_isls
        self.max_latitude_deg = max_latitude_deg

    def check_los(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """Determines if line-of-sight exists between two position vectors (no Earth occlusion)."""
        d_vec = pos2 - pos1
        dist = np.linalg.norm(d_vec)
        if dist < 1e-6:
            return True

        t = np.clip(-np.dot(pos1, d_vec) / (dist**2), 0.0, 1.0)
        closest_point = pos1 + t * d_vec
        dist_to_origin = np.linalg.norm(closest_point)

        return bool(dist_to_origin >= self.earth_obstruction_radius)

    def is_polar_region(self, pos_eci: np.ndarray) -> bool:
        """Checks if a satellite position is inside the high-latitude polar zone."""
        r = np.linalg.norm(pos_eci)
        if r < 1e-6:
            return False
        lat_rad = np.arcsin(np.clip(pos_eci[2] / r, -1.0, 1.0))
        return bool(np.abs(np.degrees(lat_rad)) > self.max_latitude_deg)

    def update_grid_topology(
        self,
        states: Dict[int, OrbitalState],
        num_planes: int = 10,
        sats_per_plane: int = 20,
    ) -> nx.Graph:
        """Constructs canonical Walker-Delta grid topology (4 links per satellite: 2 intra-plane, 2 inter-plane)."""
        G = nx.Graph()
        num_sats = len(states)

        for sat_id in range(num_sats):
            G.add_node(
                sat_id,
                pos_eci=states[sat_id].position_eci.tolist(),
                pos_ecef=states[sat_id].position_ecef.tolist(),
            )

        positions = {i: states[i].position_eci for i in range(num_sats)}

        for sat_id in range(num_sats):
            p = sat_id // sats_per_plane
            s = sat_id % sats_per_plane

            # 1. Intra-plane links (fore & aft)
            aft_s = (s + 1) % sats_per_plane
            aft_id = p * sats_per_plane + aft_s

            if not G.has_edge(sat_id, aft_id):
                pos1 = positions[sat_id]
                pos2 = positions[aft_id]
                dist = np.linalg.norm(pos1 - pos2)
                if dist <= self.max_range_km and self.check_los(pos1, pos2):
                    delay_ms = (dist / SPEED_OF_LIGHT_KM_S) * 1000.0
                    G.add_edge(
                        sat_id,
                        aft_id,
                        distance_km=float(dist),
                        delay_ms=float(delay_ms),
                        link_type="intra_plane",
                    )

            # 2. Inter-plane links (adjacent planes)
            # Skip inter-plane links if polar disabling is active
            if self.disable_seam_and_polar_isls and self.is_polar_region(positions[sat_id]):
                continue

            # Inter-plane link to next plane
            next_p = (p + 1) % num_planes
            # Skip seam link between last plane and first plane if seam disabling enabled
            if self.disable_seam_and_polar_isls and (p == num_planes - 1 and next_p == 0):
                continue

            inter_id = next_p * sats_per_plane + s
            if not G.has_edge(sat_id, inter_id):
                pos1 = positions[sat_id]
                pos2 = positions[inter_id]
                dist = np.linalg.norm(pos1 - pos2)
                if dist <= self.max_range_km and self.check_los(pos1, pos2):
                    delay_ms = (dist / SPEED_OF_LIGHT_KM_S) * 1000.0
                    G.add_edge(
                        sat_id,
                        inter_id,
                        distance_km=float(dist),
                        delay_ms=float(delay_ms),
                        link_type="inter_plane",
                    )

        return G

    def update_dynamic_topology(self, states: Dict[int, OrbitalState]) -> nx.Graph:
        """Constructs dynamic range-based ISL graph subject to max degree and LOS constraints."""
        G = nx.Graph()
        num_sats = len(states)

        for sat_id in range(num_sats):
            G.add_node(sat_id)

        positions = [states[i].position_eci for i in range(num_sats)]
        candidate_edges = []

        for i in range(num_sats):
            for j in range(i + 1, num_sats):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= self.max_range_km and self.check_los(positions[i], positions[j]):
                    candidate_edges.append((dist, i, j))

        candidate_edges.sort(key=lambda x: x[0])

        degrees = {i: 0 for i in range(num_sats)}
        for dist, i, j in candidate_edges:
            if degrees[i] < self.max_isls_per_sat and degrees[j] < self.max_isls_per_sat:
                delay_ms = (dist / SPEED_OF_LIGHT_KM_S) * 1000.0
                G.add_edge(
                    i,
                    j,
                    distance_km=float(dist),
                    delay_ms=float(delay_ms),
                    link_type="dynamic",
                )
                degrees[i] += 1
                degrees[j] += 1

        return G
