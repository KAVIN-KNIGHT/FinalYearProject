"""LSTM dataset exporter for the LEO mega-constellation simulator.

Converts a canonical per-timestep trace (``trace.json``) into synchronized
per-satellite multivariate sliding-window sequences suitable for LSTM / sequence
model training.

**Independence guarantee**: This exporter reads only from ``trace.json`` and
``config_used.yaml``.  It does not depend on the GAT exporter having run first.

**Failure handling**: Satellites that failed mid-window are **never silently
dropped**.  Their rows are preserved with ``is_active = 0.0`` (and
``failure_indicator = 1.0``).  This is the training signal for the
failures/weather/congestion_stress scenarios — excluding these rows would defeat
the purpose of fault-injection.

**Column layout note (E)**: Position and velocity columns (``pos_eci_*``,
``vel_eci_*``, ``pos_ecef_*``) are retained as *auxiliary context*.  They are
not the primary learning signal — a model can infer them exactly from
``(satellite_id, simulation_time_s)`` via deterministic orbital mechanics.  The
columns in ``LSTM_FEATURE_COLUMNS`` (network-state metrics from Phase 4 B1) are
the ones the LSTM is actually meant to learn temporal patterns from.
``pos_ecef_*`` is a deterministic rotation of ``pos_eci_*`` given
``simulation_time_s``, so both are kept only as convenience for visualisation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from satsim.export.trace_store import TraceStore
from satsim.logging import get_logger

logger = get_logger("satsim.export.lstm_export")

# ── Primary LSTM learning features (from Phase 4 B1) ──────────────────────────
LSTM_FEATURE_COLUMNS = [
    "queue_length",
    "queue_occupancy",
    "congestion_score",
    "end_to_end_delay",
    "throughput",
    "link_utilization",
    "traffic_load",
    "cpu_utilization",
    "memory_utilization",
    "neighbor_count",
    "routing_table_age",
    "routing_changes_in_window",
    "failure_indicator",
    "event_flags",
]

# ── Full raw column set (identification + orbital context + telemetry) ──────────
RAW_LSTM_COLUMNS = [
    # Identification
    "scenario", "seed", "satellite_id", "timestep", "simulation_time_s",
    # Auxiliary orbital context
    "pos_eci_x", "pos_eci_y", "pos_eci_z",
    "vel_eci_x", "vel_eci_y", "vel_eci_z",
    "pos_ecef_x", "pos_ecef_y", "pos_ecef_z",
    # Primary network & telemetry features
    "is_active", "buffer_utilization", "degree", "avg_isl_delay_ms",
    "queue_length", "queue_occupancy", "congestion_score",
    "end_to_end_delay", "throughput", "link_utilization", "traffic_load",
    "cpu_utilization", "memory_utilization", "neighbor_count",
    "node_degree", "routing_table_age", "routing_changes_in_window",
    "failure_indicator", "event_flags",
]

#: Backward-compatibility alias
LSTM_ALL_COLUMNS = RAW_LSTM_COLUMNS


class LSTMExporter:
    """Exports canonical simulation trace records into synchronized raw per-satellite
    time-series rows (1 row per scenario, satellite_id, timestep)."""

    def __init__(self, scenario_dir: Path) -> None:
        """Initialise and validate the scenario directory.

        Args:
            scenario_dir: Directory containing ``trace.json`` and
                ``config_used.yaml`` written by :class:`~satsim.export.trace_store.TraceStore`.

        Raises:
            FileNotFoundError: If ``trace.json`` or ``config_used.yaml`` are absent.
        """
        self.scenario_dir = Path(scenario_dir)
        self.trace_file = self.scenario_dir / "trace.json"
        self.config_file = self.scenario_dir / "config_used.yaml"

        if not self.trace_file.exists() or not self.config_file.exists():
            raise FileNotFoundError(
                f"No canonical trace found in '{self.scenario_dir}'! "
                f"Cannot export LSTM dataset. Run "
                f"'python -m satsim.cli.run_scenario --scenario {self.scenario_dir.name}' first."
            )

    def extract_raw_dataframe(self) -> pd.DataFrame:
        """Extract un-windowed raw synchronized time-series rows for this scenario.

        Returns:
            DataFrame with shape ``[num_satellites * timesteps, num_columns]``.
        """
        trace_records, config = TraceStore.load_trace(self.scenario_dir)
        num_steps = len(trace_records)
        num_sats = config.constellation.num_satellites
        scenario_name = self.scenario_dir.name
        seed = config.seed

        step_rows: List[Dict[str, Any]] = []

        for record in trace_records:
            step = record["timestep"]
            t_s = record["simulation_time_s"]
            sat_states = record["satellite_states"]
            active_events = record.get("active_events", [])
            sat_metrics_map: Dict[str, Any] = record.get("sat_metrics", {})

            failed_sats_legacy = {
                ev.get("target_id")
                for ev in active_events
                if ev.get("event_type") == "sat_failure" and ev.get("active", True)
            }

            isl_edges = record.get("isl_edges", [])
            node_degree: Dict[int, int] = {i: 0 for i in range(num_sats)}
            node_delays: Dict[int, List[float]] = {i: [] for i in range(num_sats)}
            for edge in isl_edges:
                u, v = edge["src"], edge["dst"]
                d = edge.get("delay_ms", 0.0)
                node_degree[u] = node_degree.get(u, 0) + 1
                node_degree[v] = node_degree.get(v, 0) + 1
                node_delays.setdefault(u, []).append(d)
                node_delays.setdefault(v, []).append(d)

            global_buf_util = record["metrics"].get("average_buffer_utilization", 0.0)
            global_link_util = record["metrics"].get("average_link_utilization", 0.0)

            for sat in sat_states:
                sat_id = sat["sat_id"]
                pos_eci = sat["pos_eci"]
                vel_eci = sat["vel_eci"]
                pos_ecef = sat["pos_ecef"]

                sm = sat_metrics_map.get(str(sat_id), {})

                fail_ind = sm.get("failure_indicator", None)
                if fail_ind is None:
                    fail_ind = 1.0 if sat_id in failed_sats_legacy else 0.0
                is_act = 1.0 - fail_ind

                deg = node_degree.get(sat_id, 0)
                delays = node_delays.get(sat_id, [])
                avg_delay = float(np.mean(delays)) if delays else 0.0

                queue_length = sm.get("queue_length", 0.0)
                queue_occupancy = sm.get("queue_occupancy", global_buf_util)
                buf_util = sm.get("buffer_utilization", queue_occupancy)
                congestion_score = sm.get("congestion_score", 0.0)
                end_to_end_delay = sm.get("end_to_end_delay", avg_delay)
                throughput = sm.get("throughput", 0.0)
                link_util = sm.get("link_utilization", global_link_util)
                traffic_load = sm.get("traffic_load", 0.0)
                cpu_util = sm.get("cpu_utilization", 0.0)
                mem_util = sm.get("memory_utilization", queue_occupancy)
                neighbor_count = sm.get("neighbor_count", float(deg))
                node_degree_val = sm.get("node_degree", float(deg))
                routing_table_age = sm.get("routing_table_age", 0.0)
                routing_changes = sm.get("routing_changes_in_window", 0.0)
                event_flags = sm.get("event_flags", 0.0)

                step_rows.append(
                    {
                        "scenario": scenario_name,
                        "seed": int(seed),
                        "satellite_id": int(sat_id),
                        "timestep": int(step),
                        "simulation_time_s": float(t_s),
                        "pos_eci_x": float(pos_eci[0]),
                        "pos_eci_y": float(pos_eci[1]),
                        "pos_eci_z": float(pos_eci[2]),
                        "vel_eci_x": float(vel_eci[0]),
                        "vel_eci_y": float(vel_eci[1]),
                        "vel_eci_z": float(vel_eci[2]),
                        "pos_ecef_x": float(pos_ecef[0]),
                        "pos_ecef_y": float(pos_ecef[1]),
                        "pos_ecef_z": float(pos_ecef[2]),
                        "is_active": float(is_act),
                        "buffer_utilization": float(buf_util),
                        "degree": int(deg),
                        "avg_isl_delay_ms": float(avg_delay),
                        "queue_length": float(queue_length),
                        "queue_occupancy": float(queue_occupancy),
                        "congestion_score": float(congestion_score),
                        "end_to_end_delay": float(end_to_end_delay),
                        "throughput": float(throughput),
                        "link_utilization": float(link_util),
                        "traffic_load": float(traffic_load),
                        "cpu_utilization": float(cpu_util),
                        "memory_utilization": float(mem_util),
                        "neighbor_count": float(neighbor_count),
                        "node_degree": float(node_degree_val),
                        "routing_table_age": float(routing_table_age),
                        "routing_changes_in_window": float(routing_changes),
                        "failure_indicator": float(fail_ind),
                        "event_flags": float(event_flags),
                    }
                )

        df_steps = pd.DataFrame(step_rows, columns=RAW_LSTM_COLUMNS)
        return df_steps

    def export_scenario(self, window_size: int = 12, stride: int = 1) -> Path:
        """Export scenario raw data to ``lstm/`` sub-folder for backward compat.

        Returns:
            Path to ``lstm/`` subfolder.
        """
        df_raw = self.extract_raw_dataframe()
        lstm_dir = self.scenario_dir / "lstm"
        lstm_dir.mkdir(parents=True, exist_ok=True)

        csv_path = lstm_dir / "lstm_sequences.csv"
        parquet_path = lstm_dir / "lstm_sequences.parquet"

        df_raw.to_csv(csv_path, index=False)
        try:
            df_raw.to_parquet(parquet_path, index=False)
        except Exception:
            pass

        meta = {
            "scenario": self.scenario_dir.name,
            "window_size": window_size,
            "stride": stride,
            "total_timesteps": len(df_raw["timestep"].unique()),
            "num_satellites": len(df_raw["satellite_id"].unique()),
            "total_rows": len(df_raw),
            "feature_columns": LSTM_FEATURE_COLUMNS,
        }
        with open(lstm_dir / "window_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return lstm_dir
