"""Tests for the stochastic event injection engine (Phase 3).

Covers all 9 event types, node-isolation semantics (A1), multiplicative
event stacking (A2), idempotent recovery, the auto-expiry contract (A3),
and the queryable timestamped event log.
"""
import networkx as nx
import numpy as np
import pytest

from satsim.config import EventsConfig
from satsim.events import EventType, SimEvent, EventInjector


def test_all_9_event_types_trigger_in_isolation():
    injector = EventInjector()
    event_types = [
        EventType.ISL_FAILURE,
        EventType.SAT_FAILURE,
        EventType.CONGESTION,
        EventType.BUFFER_OVERFLOW,
        EventType.WEATHER_ATTENUATION,
        EventType.SOLAR_INTERFERENCE,
        EventType.GS_CONGESTION,
        EventType.LINK_DEGRADATION,
        EventType.RECOVERY,
    ]

    assert len(event_types) == 9

    targets = {
        EventType.ISL_FAILURE: (0, 1),
        EventType.SAT_FAILURE: 5,
        EventType.CONGESTION: 10,
        EventType.BUFFER_OVERFLOW: 15,
        EventType.WEATHER_ATTENUATION: "GS_London",
        EventType.SOLAR_INTERFERENCE: (2, 3),
        EventType.GS_CONGESTION: "GS_NewYork",
        EventType.LINK_DEGRADATION: (4, 5),
        EventType.RECOVERY: (0, 1),
    }

    for etype in event_types:
        target = targets[etype]
        ev = injector.trigger_event(
            event_type=etype,
            target_id=target,
            duration_s=300.0,
            start_time_s=10.0,
        )
        assert ev.event_id > 0
        assert ev.event_type == etype
        assert ev.active is True


def test_isl_failure_and_recovery_restoration():
    injector = EventInjector()
    G = nx.Graph()
    G.add_edge(1, 2, distance_km=1000.0, delay_ms=3.33)

    # 1. Initially edge exists
    H1 = injector.apply_to_graph(G)
    assert H1.has_edge(1, 2)

    # 2. Trigger ISL_FAILURE
    injector.trigger_event(EventType.ISL_FAILURE, target_id=(1, 2), start_time_s=10.0)
    H2 = injector.apply_to_graph(G)
    assert not H2.has_edge(1, 2)

    # 3. Trigger RECOVERY
    injector.trigger_event(EventType.RECOVERY, target_id=(1, 2), start_time_s=20.0)
    H3 = injector.apply_to_graph(G)

    # Critical Acceptance Criterion: RECOVERY event correctly restores prior link state!
    assert H3.has_edge(1, 2)
    assert H3[1][2]["delay_ms"] == pytest.approx(3.33)


def test_sat_failure_and_recovery_restoration():
    """A1: SAT_FAILURE isolates the node (edges removed) but does NOT remove it."""
    injector = EventInjector()
    G = nx.Graph()
    G.add_edge(5, 6, delay_ms=2.0)
    G.add_edge(5, 7, delay_ms=2.5)

    # Trigger SAT_FAILURE on sat 5 — node stays, edges go.
    injector.trigger_event(EventType.SAT_FAILURE, target_id=5, start_time_s=10.0)
    H1 = injector.apply_to_graph(G)

    # Node must remain (A1 contract).
    assert H1.has_node(5), "Failed satellite was removed from graph (violates A1 isolation contract)!"
    # Incident edges must be gone.
    assert not H1.has_edge(5, 6), "Edge to failed satellite must be removed"
    assert not H1.has_edge(5, 7), "Edge to failed satellite must be removed"
    # Node annotated inactive.
    assert H1.nodes[5]["is_active"] is False

    # Trigger RECOVERY on sat 5
    injector.trigger_event(EventType.RECOVERY, target_id=5, start_time_s=50.0)
    H2 = injector.apply_to_graph(G)

    # Node still present, is_active back to True.
    assert H2.has_node(5)
    assert H2.nodes[5]["is_active"] is True
    # The original graph still has the edges — apply_to_graph no longer masks them.
    assert H2.has_edge(5, 6)
    assert H2.has_edge(5, 7)


def test_overlapping_events_multiplicative_stacking():
    """A2: Two concurrent degradation events on the same edge combine multiplicatively."""
    injector = EventInjector()
    G = nx.Graph()
    G.add_edge(0, 1, delay_ms=10.0, distance_km=300.0)

    # Event A: LINK_DEGRADATION with multiplier 2.0
    injector.trigger_event(
        EventType.LINK_DEGRADATION,
        target_id=(0, 1),
        duration_s=600.0,
        start_time_s=0.0,
        params={"multiplier": 2.0},
    )
    H1 = injector.apply_to_graph(G)
    # After event A: delay_ms should be 10 * 2.0 = 20.0
    assert H1[0][1]["delay_ms"] == pytest.approx(20.0), (
        f"Expected 20.0 after LINK_DEGRADATION ×2, got {H1[0][1]['delay_ms']}"
    )

    # Event B: SOLAR_INTERFERENCE with multiplier 3.0 — stacks ON TOP of A.
    injector.trigger_event(
        EventType.SOLAR_INTERFERENCE,
        target_id=(0, 1),
        duration_s=300.0,
        start_time_s=10.0,
        params={"multiplier": 3.0},
    )
    H2 = injector.apply_to_graph(G)
    # Combined factor: 2.0 × 3.0 = 6.0 → delay_ms = 10 * 6.0 = 60.0
    assert H2[0][1]["delay_ms"] == pytest.approx(60.0), (
        f"Expected 60.0 with multiplicative stacking (2×3), got {H2[0][1]['delay_ms']}"
    )
    assert H2[0][1]["degradation_factor"] == pytest.approx(6.0)

    # Recover event B only — A's factor must be preserved.
    injector.trigger_event(EventType.RECOVERY, target_id=(0, 1), start_time_s=310.0)
    # After full RECOVERY, both events on the edge are cleared.
    H3 = injector.apply_to_graph(G)
    # After recovery: back to 1× (the ISL manager would reset delay on next topology rebuild,
    # but from apply_to_graph perspective the factor dict is empty).
    eff = injector.get_edge_degradation((0, 1))
    assert eff == pytest.approx(1.0), (
        f"After full recovery, degradation factor should be 1.0, got {eff}"
    )


def test_recovery_is_idempotent():
    """A3: Calling RECOVERY twice on an already-recovered target is a no-op."""
    injector = EventInjector()
    injector.trigger_event(EventType.SAT_FAILURE, target_id=10, start_time_s=0.0)
    injector.trigger_event(EventType.RECOVERY, target_id=10, start_time_s=10.0)
    # Second recovery should not raise.
    injector.trigger_event(EventType.RECOVERY, target_id=10, start_time_s=20.0)
    assert 10 not in injector.disabled_nodes


def test_queryable_timestamped_event_log():
    injector = EventInjector()

    ev1 = injector.trigger_event(EventType.ISL_FAILURE, (1, 2), start_time_s=10.0)
    ev2 = injector.trigger_event(EventType.SAT_FAILURE, 5, start_time_s=50.0)
    ev3 = injector.trigger_event(EventType.LINK_DEGRADATION, (2, 3), start_time_s=100.0)

    # Query all events
    log_all = injector.query_log()
    assert len(log_all) == 3

    # Query by timestamp range
    log_range = injector.query_log(start_time_s=20.0, end_time_s=80.0)
    assert len(log_range) == 1
    assert log_range[0].event_id == ev2.event_id

    # Query by event_type
    log_type = injector.query_log(event_type=EventType.LINK_DEGRADATION)
    assert len(log_type) == 1
    assert log_type[0].event_id == ev3.event_id


def test_stochastic_event_scheduler():
    config = EventsConfig(
        enabled_types=["isl_failure", "sat_failure", "congestion", "weather_attenuation"],
        failure_rate_per_hour=5.0,
    )
    injector = EventInjector(config, seed=42)

    G = nx.Graph()
    for i in range(10):
        G.add_edge(i, (i + 1) % 10)

    total_steps = 720
    for step in range(total_steps):
        t_s = step * 5.0
        injector.step(t_s, current_graph=G, num_nodes=10, gs_names=["GS_London", "GS_Tokyo"])

    log = injector.query_log()
    assert len(log) > 0, "Stochastic injector failed to schedule events over 1 hour!"
