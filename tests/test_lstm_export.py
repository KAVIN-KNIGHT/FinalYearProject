"""Tests for the LSTM dataset exporter (Phase 7).

Covers clean failure on missing trace, window/stride config, failed-satellite
preservation (D1 critical contract), and all new D1/D2 column requirements.
"""
from pathlib import Path
import tempfile
import pandas as pd
import pytest

from satsim.config import SimConfig
from satsim.sim import SimulationEngine, get_scenario_config
from satsim.export import TraceStore, LSTMExporter
from satsim.events import EventInjector, EventType
from satsim.export.lstm_export import LSTM_FEATURE_COLUMNS, LSTM_ALL_COLUMNS


def test_lstm_exporter_fails_cleanly_without_trace():
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_dir = Path(tmpdir) / "empty_scenario"
        empty_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError) as exc_info:
            LSTMExporter(empty_dir)

        assert "No canonical trace found" in str(exc_info.value)


def test_lstm_export_generation_and_window_stride_config():
    config = get_scenario_config("low_load", seed=42)
    config.duration_seconds = 30.0  # 6 steps (step 0 to 5)

    engine = SimulationEngine(config)
    trace_records = engine.run(progress_bar=False)
    metrics_df = engine.metrics_collector.to_dataframe()

    with tempfile.TemporaryDirectory() as tmpdir:
        scen_dir = Path(tmpdir) / "low_load"
        TraceStore.save_trace(scen_dir, trace_records, config, metrics_df)

        exporter = LSTMExporter(scen_dir)

        # Export with window_size=3, stride=1 -> window starts: 0, 1, 2, 3 -> 4 windows
        out_dir = exporter.export_scenario(window_size=3, stride=1)

        csv_path = out_dir / "lstm_sequences.csv"
        parquet_path = out_dir / "lstm_sequences.parquet"
        meta_path = out_dir / "window_metadata.json"

        assert csv_path.exists()
        assert parquet_path.exists()
        assert meta_path.exists()

        df = pd.read_csv(csv_path)

        # 100 sats * 6 steps = 600 raw synchronized rows
        assert len(df) == 100 * 6
        assert set(df["satellite_id"]) == set(range(100))
        assert set(df["timestep"]) == set(range(6))
        assert "is_active" in df.columns
        assert "pos_eci_x" in df.columns


def test_failed_satellite_mid_window_is_preserved_not_dropped():
    """
    CRITICAL ACCEPTANCE CRITERION:
    No satellite is silently dropped if it had a failure event mid-window.
    The failure_indicator column must reflect it (1.0), keeping all satellite rows!
    """
    config = get_scenario_config("low_load", seed=42)
    config.duration_seconds = 30.0  # 6 steps

    engine = SimulationEngine(config)

    # Inject SAT_FAILURE on sat 5 at t=10s (step 2) for duration 30s
    engine.injector.trigger_event(
        event_type=EventType.SAT_FAILURE,
        target_id=5,
        duration_s=30.0,
        start_time_s=10.0,
    )

    trace_records = engine.run(progress_bar=False)
    metrics_df = engine.metrics_collector.to_dataframe()

    with tempfile.TemporaryDirectory() as tmpdir:
        scen_dir = Path(tmpdir) / "low_load"
        TraceStore.save_trace(scen_dir, trace_records, config, metrics_df)

        exporter = LSTMExporter(scen_dir)
        out_dir = exporter.export_scenario(window_size=4, stride=1)

        df = pd.read_csv(out_dir / "lstm_sequences.csv")

        # 1. Verify satellite 5 is present in the dataset (NOT dropped!)
        sat5_df = df[df["satellite_id"] == 5]
        assert not sat5_df.empty, "Failed satellite 5 was silently dropped from the LSTM dataset!"

        # 2. All satellites have identical total row counts
        sat_row_counts = df.groupby("satellite_id").size()
        assert (sat_row_counts == sat_row_counts.iloc[0]).all(), "Satellites have mismatched row counts!"

        # 3. is_active reflects failure (0.0) during failure timesteps
        failed_steps = sat5_df[sat5_df["timestep"] >= 2]
        assert (failed_steps["is_active"] == 0.0).all(), "Failed satellite is_active must be 0.0!"

        healthy_steps = sat5_df[sat5_df["timestep"] < 2]
        assert (healthy_steps["is_active"] == 1.0).all()

        # 4. failure_indicator is 1.0 during failure (new D1 column)
        assert "failure_indicator" in df.columns, "Missing failure_indicator column (D1)!"
        failed_fi = sat5_df[sat5_df["timestep"] >= 2]["failure_indicator"]
        assert (failed_fi == 1.0).all(), "failure_indicator must be 1.0 for failed sat!"


def test_lstm_all_d1_columns_present():
    """D1: All 14 Phase-4 B1 columns must be present in the LSTM output."""
    config = get_scenario_config("low_load", seed=42)
    config.duration_seconds = 15.0  # 3 steps

    engine = SimulationEngine(config)
    trace_records = engine.run(progress_bar=False)
    metrics_df = engine.metrics_collector.to_dataframe()

    with tempfile.TemporaryDirectory() as tmpdir:
        scen_dir = Path(tmpdir) / "low_load"
        TraceStore.save_trace(scen_dir, trace_records, config, metrics_df)

        exporter = LSTMExporter(scen_dir)
        out_dir = exporter.export_scenario(window_size=2, stride=1)
        df = pd.read_csv(out_dir / "lstm_sequences.csv")

        for col in LSTM_FEATURE_COLUMNS:
            assert col in df.columns, f"Missing D1 column: '{col}'"


def test_lstm_scenario_seed_columns():
    """D2: Every row must have a non-null 'scenario' matching the folder name and a 'seed'."""
    config = get_scenario_config("burst", seed=99)
    config.duration_seconds = 15.0

    engine = SimulationEngine(config)
    trace_records = engine.run(progress_bar=False)
    metrics_df = engine.metrics_collector.to_dataframe()

    with tempfile.TemporaryDirectory() as tmpdir:
        scen_dir = Path(tmpdir) / "burst"
        TraceStore.save_trace(scen_dir, trace_records, config, metrics_df)

        exporter = LSTMExporter(scen_dir)
        out_dir = exporter.export_scenario(window_size=2, stride=1)
        df = pd.read_csv(out_dir / "lstm_sequences.csv")

        assert "scenario" in df.columns, "Missing 'scenario' column (D2)!"
        assert "seed" in df.columns, "Missing 'seed' column (D2)!"
        assert (df["scenario"] == "burst").all(), "scenario column must match folder name!"
        assert (df["seed"] == 99).all(), "seed column must match config seed!"
