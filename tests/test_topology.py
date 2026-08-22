import networkx as nx
import numpy as np
import pytest

from satsim.orbital import WalkerDeltaConstellation, EARTH_RADIUS_KM
from satsim.topology import ISLManager, GroundStationManager, geodetic_to_ecef


def test_grid_isl_topology_creation_and_degree_bounds():
    constellation = WalkerDeltaConstellation(num_satellites=100, num_planes=10)
    states = constellation.get_states(0.0)

    isl_mgr = ISLManager(max_range_km=5000.0, max_isls_per_sat=4)
    G = isl_mgr.update_grid_topology(states, num_planes=10, sats_per_plane=10)

    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 100

    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    min_degree = min(degrees.values())

    # Critical Acceptance Criterion: satellite node degree MUST be physically plausible (<= 4)
    assert max_degree <= 4
    assert min_degree >= 2  # At least intra-plane links connected
    assert max_degree < 99  # Does NOT connect to all 99 satellites!


def test_dynamic_isl_topology_degree_enforcement():
    constellation = WalkerDeltaConstellation(num_satellites=100, num_planes=10)
    states = constellation.get_states(0.0)

    isl_mgr = ISLManager(max_range_km=10000.0, max_isls_per_sat=4)
    G = isl_mgr.update_dynamic_topology(states)

    degrees = dict(G.degree())
    assert max(degrees.values()) <= 4


def test_line_of_sight_earth_occlusion():
    isl_mgr = ISLManager(atm_clearance_km=80.0)

    # Two satellites on opposite sides of Earth
    pos_north = np.array([0.0, 0.0, EARTH_RADIUS_KM + 550.0])
    pos_south = np.array([0.0, 0.0, -(EARTH_RADIUS_KM + 550.0)])

    # Direct line between north and south passes directly through Earth center
    assert isl_mgr.check_los(pos_north, pos_south) is False
    assert not isl_mgr.check_los(pos_north, pos_south)

    # Two adjacent satellites with clear line of sight
    pos_sat1 = np.array([EARTH_RADIUS_KM + 550.0, 0.0, 0.0])
    pos_sat2 = np.array([EARTH_RADIUS_KM + 550.0, 500.0, 0.0])
    assert isl_mgr.check_los(pos_sat1, pos_sat2) is True


def test_ground_station_visibility():
    gs_mgr = GroundStationManager()
    assert len(gs_mgr.stations) == 12

    constellation = WalkerDeltaConstellation(num_satellites=100, num_planes=10)
    states = constellation.get_states(0.0)

    contacts = gs_mgr.get_active_contacts(states, min_elevation_deg=10.0)
    assert isinstance(contacts, list)
    for c in contacts:
        assert c.elevation_deg >= 10.0
        assert c.distance_km > 0.0
        assert c.delay_ms > 0.0
