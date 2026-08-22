import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from satsim.config import SimConfig
from satsim.sim import SimulationEngine, get_scenario_config
from satsim.export import TraceStore


def test_simulation_engine_run():
    config = get_scenario_config("low_load", seed=42)
    config.duration_seconds = 30.0  # 6 steps at 5s dt

    engine = SimulationEngine(config)
    trace_records = engine.run(progress_bar=False)

    assert len(trace_records) == 6
    for step, record in enumerate(trace_records):
        assert record["timestep"] == step
        assert record["simulation_time_s"] == step * 5.0
        assert len(record["satellite_states"]) == 100
        assert "metrics" in record
        assert record["metrics"]["timestep"] == step


def test_trace_store_save_and_load_roundtrip():
    config = get_scenario_config("low_load", seed=42)
    config.duration_seconds = 15.0

    engine = SimulationEngine(config)
    trace_records = engine.run(progress_bar=False)
    metrics_df = engine.metrics_collector.to_dataframe()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "low_load"
        TraceStore.save_trace(out_dir, trace_records, config, metrics_df)

        assert (out_dir / "config_used.yaml").exists()
        assert (out_dir / "trace.json").exists()
        assert (out_dir / "global_metrics" / "metrics.csv").exists()

        reloaded_records, reloaded_config = TraceStore.load_trace(out_dir)

        assert reloaded_config.model_dump() == config.model_dump()
        assert len(reloaded_records) == len(trace_records)


def test_reproducibility_gate_byte_identical_traces():
    """
    CRITICAL PHASE 5 REPRODUCIBILITY GATE:
    Running the CLI twice with identical seed MUST produce 100% numerically identical traces!
    """
    config1 = get_scenario_config("low_load", seed=42)
    config1.duration_seconds = 30.0

    config2 = get_scenario_config("low_load", seed=42)
    config2.duration_seconds = 30.0

    engine1 = SimulationEngine(config1)
    trace1 = engine1.run(progress_bar=False)
    df1 = engine1.metrics_collector.to_dataframe()

    engine2 = SimulationEngine(config2)
    trace2 = engine2.run(progress_bar=False)
    df2 = engine2.metrics_collector.to_dataframe()

    with tempfile.TemporaryDirectory() as tmpdir:
        dir1 = Path(tmpdir) / "run1"
        dir2 = Path(tmpdir) / "run2"

        TraceStore.save_trace(dir1, trace1, config1, df1)
        TraceStore.save_trace(dir2, trace2, config2, df2)

        # 1. Check config_used.yaml identity
        conf1_str = (dir1 / "config_used.yaml").read_text(encoding="utf-8")
        conf2_str = (dir2 / "config_used.yaml").read_text(encoding="utf-8")
        assert conf1_str == conf2_str, "config_used.yaml outputs are not byte-identical!"

        # 2. Check metrics DataFrame numerical identity
        np.testing.assert_allclose(df1.values, df2.values)

        # 3. Check per-timestep trace JSON identity
        json1_str = (dir1 / "trace.json").read_text(encoding="utf-8")
        json2_str = (dir2 / "trace.json").read_text(encoding="utf-8")
        assert json1_str == json2_str, "trace.json outputs are not byte-identical!"
