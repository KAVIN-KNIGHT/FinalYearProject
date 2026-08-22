"""Tests for the Dijkstra router and MetricsCollector (Phase 4).

Covers shortest-path correctness, all-19-global-field presence, per-satellite
metrics (B1), per-edge metrics (B1), and DataFrame export consistency.
"""
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from satsim.config import SimConfig, TrafficConfig, EventsConfig
from satsim.orbital import WalkerDeltaConstellation
from satsim.topology import ISLManager
from satsim.traffic import LowTrafficProfile
from satsim.events import EventInjector, EventType
from satsim.routing import DijkstraRouter
from satsim.metrics import MetricsCollector, METRIC_COLUMNS


def _make_graph_and_flows(num_sats: int = 50, num_planes: int = 5, duration_s: float = 50.0):
    constellation = WalkerDeltaConstellation(num_satellites=num_sats, num_planes=num_planes)
    states = constellation.get_states(0.0)
    isl_mgr = ISLManager(max_range_km=5000.0)
    G = isl_mgr.update_grid_topology(states, num_planes=num_planes, sats_per_plane=num_sats // num_planes)
    traffic_prof = LowTrafficProfile(TrafficConfig(), seed=42)
    flows = traffic_prof.generate_flows(0.0, duration_s, num_nodes=num_sats)
    return G, flows


def test_dijkstra_router_shortest_path():
    G = nx.Graph()
    G.add_edge(0, 1, delay_ms=10.0)
    G.add_edge(1, 2, delay_ms=15.0)
    G.add_edge(0, 2, delay_ms=50.0)

    router = DijkstraRouter()

    # Route next hop from 0 to 2 should be via node 1 (10 + 15 = 25 < 50)
    next_hop = router.get_next_hop(G, 0, 2)
    assert next_hop == 1

    routes = router.compute_all_pairs_routes(G)
    assert routes[(0, 2)] == [0, 1, 2]


def test_metrics_collector_all_19_fields_present_no_nans():
    G, flows = _make_graph_and_flows()
    router = DijkstraRouter()
    injector = EventInjector(EventsConfig(), seed=42)
    collector = MetricsCollector()

    for step in range(10):
        t_s = step * 5.0
        step_flows = [f for f in flows if t_s <= f.start_time_s < (t_s + 5.0)]
        result = collector.collect_step(
            t_s=t_s, step=step, graph=G,
            active_flows=step_flows, router=router, injector=injector, dt_s=5.0,
        )

        # collect_step now returns {"global": {...}, "per_satellite": {...}, "per_edge": {...}}
        assert "global" in result
        assert "per_satellite" in result
        assert "per_edge" in result

        record = result["global"]
        # Critical Acceptance Criterion: 19 global fields, no None/NaN/Inf
        assert len(record) == 19
        for key, val in record.items():
            assert val is not None, f"Field '{key}' is None!"
            if isinstance(val, float):
                assert not np.isnan(val), f"Field '{key}' is NaN!"
                assert not np.isinf(val), f"Field '{key}' is Inf!"


def test_per_satellite_metrics_present(tmp_path):
    """B1: per_satellite dict must have an entry for every node with all required fields."""
    G, flows = _make_graph_and_flows(num_sats=20, num_planes=4, duration_s=10.0)
    router = DijkstraRouter()
    injector = EventInjector(EventsConfig(), seed=42)
    collector = MetricsCollector()

    # Inject a SAT_FAILURE on node 3 to verify failure_indicator propagates.
    injector.trigger_event(EventType.SAT_FAILURE, target_id=3, start_time_s=0.0, duration_s=30.0)

    result = collector.collect_step(
        t_s=0.0, step=0, graph=injector.apply_to_graph(G),
        active_flows=flows[:2], router=router, injector=injector, dt_s=5.0,
    )
    per_sat = result["per_satellite"]

    required_fields = {
        "queue_length", "queue_occupancy", "congestion_score", "cpu_utilization",
        "memory_utilization", "neighbor_count", "node_degree", "routing_table_age",
        "routing_changes_in_window", "failure_indicator", "event_flags",
        "end_to_end_delay", "throughput",
    }

    for node in G.nodes():
        assert node in per_sat, f"Node {node} missing from per_satellite dict!"
        for field in required_fields:
            assert field in per_sat[node], f"Field '{field}' missing for node {node}!"
            val = per_sat[node][field]
            assert not np.isnan(float(val)), f"NaN in per_satellite[{node}]['{field}']"

    # Verify failure_indicator is 1.0 for node 3
    assert per_sat[3]["failure_indicator"] == pytest.approx(1.0), (
        "failure_indicator for failed satellite must be 1.0"
    )


def test_per_edge_metrics_present():
    """B1: per_edge dict must have an entry for every active ISL with all required fields."""
    # Use 50-sat/5-plane constellation: known to produce ISL edges at default max_range_km.
    # The 20-sat/4-plane mini-constellation places satellites too far apart for ISLs.
    G, flows = _make_graph_and_flows(num_sats=50, num_planes=5, duration_s=10.0)
    router = DijkstraRouter()
    injector = EventInjector(EventsConfig(), seed=42)
    collector = MetricsCollector()

    result = collector.collect_step(
        t_s=0.0, step=0, graph=G,
        active_flows=flows[:5], router=router, injector=injector, dt_s=5.0,
    )
    per_edge = result["per_edge"]

    required_fields = {
        "link_utilization", "available_bandwidth_kbps", "used_bandwidth_kbps",
        "ber", "snr", "link_stability", "link_lifetime",
        "link_failure_probability", "degradation_factor",
    }

    assert len(per_edge) > 0, "per_edge dict is empty — no ISLs in graph!"
    for edge_key, em in per_edge.items():
        for field in required_fields:
            assert field in em, f"Field '{field}' missing for edge '{edge_key}'!"
            val = em[field]
            assert not np.isnan(float(val)), f"NaN in per_edge['{edge_key}']['{field}']"

    # All link_utilization values must be in [0, 1]
    for edge_key, em in per_edge.items():
        assert 0.0 <= em["link_utilization"] <= 1.0, (
            f"link_utilization out of range for '{edge_key}': {em['link_utilization']}"
        )


def test_metrics_dataframe_export_and_consistency():
    G, flows = _make_graph_and_flows()
    router = DijkstraRouter()
    injector = EventInjector(EventsConfig(), seed=42)
    collector = MetricsCollector()

    for step in range(10):
        t_s = step * 5.0
        step_flows = [f for f in flows if t_s <= f.start_time_s < (t_s + 5.0)]
        collector.collect_step(
            t_s=t_s, step=step, graph=G,
            active_flows=step_flows, router=router, injector=injector, dt_s=5.0,
        )

    df = collector.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert list(df.columns) == METRIC_COLUMNS
    assert not df.isna().any().any(), "DataFrame contains NaN values!"

    # Internal Consistency Assertions:
    assert (df["packet_delivery_ratio"] >= 0.0).all()
    assert (df["packet_delivery_ratio"] <= 1.0).all()
    assert (df["average_buffer_utilization"] >= 0.0).all()
    assert (df["average_buffer_utilization"] <= 1.0).all()
    assert (df["average_link_utilization"] >= 0.0).all()
    assert (df["average_link_utilization"] <= 1.0).all()
    assert (df["average_delay_ms"] >= 0.0).all()
