from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from satsim.config import GroundStationsConfig, GroundStationLocation
from satsim.orbital.propagation import OrbitalState, EARTH_RADIUS_KM, SPEED_OF_LIGHT_KM_S


@dataclass
class GroundStation:
    gs_id: int
    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float
    position_ecef: np.ndarray  # shape (3,) in km
    up_vector_ecef: np.ndarray  # shape (3,) unit vector


@dataclass
class GroundStationContact:
    gs_id: int
    gs_name: str
    sat_id: int
    elevation_deg: float
    distance_km: float
    delay_ms: float


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Converts spherical geodetic lat/lon/alt to ECEF position and local zenith unit vector."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    r = EARTH_RADIUS_KM + alt_km

    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)

    x = r * cos_lat * cos_lon
    y = r * cos_lat * sin_lon
    z = r * sin_lat

    pos_ecef = np.array([x, y, z], dtype=np.float64)
    up_vector = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=np.float64)

    return pos_ecef, up_vector


class GroundStationManager:
    """Manages ground station locations and dynamic satellite contact visibility windows."""

    def __init__(self, config: Optional[GroundStationsConfig] = None):
        self.stations: Dict[int, GroundStation] = {}
        if config is not None:
            self._load_from_config(config)
        else:
            self._load_default_stations()

    def _load_from_config(self, config: GroundStationsConfig) -> None:
        for idx, loc in enumerate(config.locations):
            pos_ecef, up_vec = geodetic_to_ecef(loc.lat, loc.lon, loc.alt_km)
            self.stations[idx] = GroundStation(
                gs_id=idx,
                name=loc.name,
                lat_deg=loc.lat,
                lon_deg=loc.lon,
                alt_km=loc.alt_km,
                position_ecef=pos_ecef,
                up_vector_ecef=up_vec,
            )

    def _load_default_stations(self) -> None:
        default_locs = [
            ("GS_London", 51.5074, -0.1278, 0.05),
            ("GS_NewYork", 40.7128, -74.0060, 0.01),
            ("GS_Tokyo", 35.6762, 139.6503, 0.04),
            ("GS_Sydney", -33.8688, 151.2093, 0.02),
            ("GS_SaoPaulo", -23.5505, -46.6333, 0.76),
            ("GS_Johannesburg", -26.2041, 28.0473, 1.75),
            ("GS_Frankfurt", 50.1109, 8.6821, 0.11),
            ("GS_Singapore", 1.3521, 103.8198, 0.01),
            ("GS_Mumbai", 19.0760, 72.8777, 0.01),
            ("GS_LosAngeles", 34.0522, -118.2437, 0.09),
            ("GS_Santiago", -33.4489, -70.6693, 0.57),
            ("GS_Cairo", 30.0444, 31.2357, 0.02),
        ]
        for idx, (name, lat, lon, alt) in enumerate(default_locs):
            pos_ecef, up_vec = geodetic_to_ecef(lat, lon, alt)
            self.stations[idx] = GroundStation(
                gs_id=idx,
                name=name,
                lat_deg=lat,
                lon_deg=lon,
                alt_km=alt,
                position_ecef=pos_ecef,
                up_vector_ecef=up_vec,
            )

    def get_elevation_deg(self, gs: GroundStation, sat_pos_ecef: np.ndarray) -> Tuple[float, float]:
        """Calculates elevation angle in degrees and slant range in km from GS to satellite."""
        rho = sat_pos_ecef - gs.position_ecef
        dist = float(np.linalg.norm(rho))
        if dist < 1e-6:
            return 90.0, 0.0

        sin_el = np.dot(rho, gs.up_vector_ecef) / dist
        sin_el = np.clip(sin_el, -1.0, 1.0)
        elevation_deg = float(np.degrees(np.arcsin(sin_el)))

        return elevation_deg, dist

    def get_active_contacts(
        self, states: Dict[int, OrbitalState], min_elevation_deg: float = 10.0
    ) -> List[GroundStationContact]:
        """Returns list of active GroundStationContact objects meeting minimum elevation threshold."""
        contacts = []
        for gs_id, gs in self.stations.items():
            for sat_id, state in states.items():
                elevation, dist = self.get_elevation_deg(gs, state.position_ecef)
                if elevation >= min_elevation_deg:
                    delay_ms = (dist / SPEED_OF_LIGHT_KM_S) * 1000.0
                    contacts.append(
                        GroundStationContact(
                            gs_id=gs_id,
                            gs_name=gs.name,
                            sat_id=sat_id,
                            elevation_deg=elevation,
                            distance_km=dist,
                            delay_ms=delay_ms,
                        )
                    )
        return contacts
