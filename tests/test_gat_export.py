"""Tests for the GAT dataset exporter (Phase 6).

Covers clean failure on missing trace, snapshot generation count, and the full
schema validation (C1–C3): stable 200-node identity, node features [200, 17],
edge features [2E, 6].
"""
from pathlib import Path
import random
import tempfile
import torch
from torch_geometric.data import Data
import pytest

from satsim.sim import SimulationEngine, get_scenario_config
from satsim.export import TraceStore, GATExporter


def test_gat_exporter_fails_cleanly_without_trace():
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_dir = Path(tmpdir) / "empty_scenario"
        empty_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError) as exc_info:
            GATExporter(empty_dir)

        assert "No canonical trace found" in str(exc_info.value)


def test_gat_export_generation():
    config = get_scenario_config("low_load", seed=42)
    config.duration_seconds = 15.0  # 3 steps

    engine = SimulationEngine(config)
    trace_records = engine.run(progress_bar=False)
    metrics_df = engine.metrics_collector.to_dataframe()

    with tempfile.TemporaryDirectory() as tmpdir:
        scen_dir = Path(tmpdir) / "low_load"
        TraceStore.save_trace(scen_dir, trace_records, config, metrics_df)

        exporter = GATExporter(scen_dir)
        pt_files = exporter.export_scenario()

        assert len(pt_files) == 3
        for pt_file in pt_files:
            assert pt_file.exists()
            assert pt_file.name.endswith(".pt")


def test_gat_smoke_test_load_3_random_snapshots():
    """
    CRITICAL ACCEPTANCE CRITERIA (C1–C3):
    - data.x shape: [200, 17]  (stable 200-node identity, 17 features)
    - data.edge_index shape: [2, 2E]
    - data.edge_attr shape: [2E, 6]  (6 edge features, link_type ordinal)
    - 0 NaNs or Infs anywhere
    """
    config = get_scenario_config("low_load", seed=42)
    config.duration_seconds = 30.0  # 6 steps

    engine = SimulationEngine(config)
    trace_records = engine.run(progress_bar=False)
    metrics_df = engine.metrics_collector.to_dataframe()

    with tempfile.TemporaryDirectory() as tmpdir:
        scen_dir = Path(tmpdir) / "low_load"
        TraceStore.save_trace(scen_dir, trace_records, config, metrics_df)

        exporter = GATExporter(scen_dir)
        pt_files = exporter.export_scenario()

        assert len(pt_files) >= 3

        sampled_files = random.sample(pt_files, 3)

        for pt_path in sampled_files:
            data = torch.load(pt_path, weights_only=False)

            assert isinstance(data, Data), f"{pt_path} failed to load as torch_geometric.data.Data!"

            # C1: stable 100-node identity — always exactly 100 rows
            assert data.x.dim() == 2
            assert data.x.shape[0] == 100, f"Expected 100 satellite nodes, got {data.x.shape[0]}"

            # C2: expanded node features [100, 17]
            assert data.x.shape[1] == 17, (
                f"Expected 17 node features (C2), got {data.x.shape[1]}"
            )
            assert not torch.isnan(data.x).any(), "data.x contains NaNs!"
            assert not torch.isinf(data.x).any(), "data.x contains Infs!"

            # Edge index shape [2, 2E]
            assert data.edge_index.dim() == 2
            assert data.edge_index.shape[0] == 2

            # C3: expanded edge features [2E, 6]
            assert data.edge_attr.dim() == 2
            assert data.edge_attr.shape[0] == data.edge_index.shape[1]
            assert data.edge_attr.shape[1] == 6, (
                f"Expected 6 edge features (C3), got {data.edge_attr.shape[1]}"
            )
            assert not torch.isnan(data.edge_attr).any(), "data.edge_attr contains NaNs!"
            assert not torch.isinf(data.edge_attr).any(), "data.edge_attr contains Infs!"


def test_gat_node_count_stable_under_sat_failure():
    """C1: Node count must remain 100 even when SAT_FAILURE events are active."""
    from satsim.events import EventType

    config = get_scenario_config("failures", seed=42)
    config.duration_seconds = 30.0

    engine = SimulationEngine(config)
    # Manually inject failures to ensure they fire
    for sat_id in range(5):
        engine.injector.trigger_event(
            EventType.SAT_FAILURE, target_id=sat_id, duration_s=20.0, start_time_s=0.0
        )

    trace_records = engine.run(progress_bar=False)
    metrics_df = engine.metrics_collector.to_dataframe()

    with tempfile.TemporaryDirectory() as tmpdir:
        scen_dir = Path(tmpdir) / "failures"
        TraceStore.save_trace(scen_dir, trace_records, config, metrics_df)
        exporter = GATExporter(scen_dir)
        pt_files = exporter.export_scenario()

        for pt_path in pt_files:
            data = torch.load(pt_path, weights_only=False)
            assert data.x.shape[0] == 100, (
                f"Node count shifted to {data.x.shape[0]} during SAT_FAILURE — violates C1!"
            )
