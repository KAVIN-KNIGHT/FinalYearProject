import numpy as np
import pytest

from satsim.config import SimConfig
from satsim.orbital import (
    WalkerDeltaConstellation,
    KeplerianPropagator,
    KeplerianElements,
    EARTH_RADIUS_KM,
)


def test_constellation_layout_initialization():
    constellation = WalkerDeltaConstellation(
        num_satellites=100, num_planes=10, altitude_km=550.0, inclination_deg=53.0
    )
    assert len(constellation.elements) == 100
    assert constellation.sats_per_plane == 10

    states = constellation.get_states(0.0)
    assert len(states) == 100

    for sat_id, state in states.items():
        assert not np.isnan(state.position_eci).any()
        assert not np.isnan(state.position_ecef).any()
        r_norm = np.linalg.norm(state.position_eci)
        assert pytest.approx(r_norm, rel=1e-4) == (EARTH_RADIUS_KM + 550.0)


def test_one_hour_propagation_continuity():
    constellation = WalkerDeltaConstellation(num_satellites=100, num_planes=10)
    timestep_s = 5.0
    duration_s = 3600.0
    num_steps = int(duration_s // timestep_s)

    sat_0_positions = []
    for step in range(num_steps):
        t_s = step * timestep_s
        states = constellation.get_states(t_s)
        assert not np.isnan(states[0].position_eci).any()
        assert not np.isnan(states[0].velocity_eci).any()
        assert not np.isinf(states[0].position_eci).any()
        sat_0_positions.append(states[0].position_eci)

    sat_0_positions = np.array(sat_0_positions)

    norms = np.linalg.norm(sat_0_positions, axis=1)
    np.testing.assert_allclose(norms, EARTH_RADIUS_KM + 550.0, rtol=1e-4)

    diffs = np.linalg.norm(np.diff(sat_0_positions, axis=0), axis=1)
    assert (diffs < 50.0).all()
    assert (diffs > 30.0).all()


def test_eci_to_ecef_rotation():
    propagator = KeplerianPropagator()
    elements = KeplerianElements(
        a=EARTH_RADIUS_KM + 550.0,
        e=0.0,
        inc=np.radians(53.0),
        raan=0.0,
        arg_perigee=0.0,
        mean_anomaly_0=0.0,
    )
    state_0 = propagator.propagate_sat(elements, 0.0)
    state_100 = propagator.propagate_sat(elements, 100.0)

    assert pytest.approx(state_0.position_eci[2]) == state_0.position_ecef[2]
    assert pytest.approx(state_100.position_eci[2]) == state_100.position_ecef[2]


def test_inclination_max_latitude():
    """
    Directly tests the active rotation matrix Rx(inc) convention.
    Propagates a satellite over one complete orbital period (~5730s for 550km LEO orbit)
    and asserts max(|arcsin(z / |r|)|) matches inclination_deg (53.0 deg) within 0.5 deg.
    """
    inc_target_deg = 53.0
    constellation = WalkerDeltaConstellation(
        num_satellites=200, num_planes=10, altitude_km=550.0, inclination_deg=inc_target_deg
    )

    orbital_period_s = 5730.0
    num_samples = 200
    latitudes_deg = []

    for t_s in np.linspace(0.0, orbital_period_s, num_samples):
        state = constellation.propagator.propagate_sat(constellation.elements[0], t_s)
        r = np.linalg.norm(state.position_eci)
        z = state.position_eci[2]
        lat_deg = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
        latitudes_deg.append(lat_deg)

    max_lat = np.max(np.abs(latitudes_deg))

    # Assert max peak latitude matches target inclination directly
    assert pytest.approx(max_lat, abs=0.5) == inc_target_deg
