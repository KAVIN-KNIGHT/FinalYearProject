"""Canonical per-timestep trace persistence layer.

:class:`TraceStore` is the single source of truth for a completed simulation run.
Both the GAT and LSTM exporters read from the trace independently — neither
depends on the other having run first.

Directory layout written by :meth:`TraceStore.save_trace`::

    <scenario_dir>/
    ├── config_used.yaml   # Exact YAML config (reproducibility contract)
    ├── trace.json         # Per-timestep simulation records
    └── global_metrics/
        ├── metrics.csv
        └── metrics.parquet
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from satsim.config import SimConfig


class TraceStore:
    """Handles saving and loading of the canonical simulation trace and scenario deliverables."""

    @staticmethod
    def save_trace(
        scenario_dir: Path,
        trace_records: List[Dict[str, Any]],
        config: SimConfig,
        metrics_df: Optional[pd.DataFrame] = None,
    ) -> Path:
        """Persist a completed simulation run to *scenario_dir*.

        Writes ``config_used.yaml``, ``trace.json``, and (if provided)
        ``global_metrics/metrics.csv`` + ``metrics.parquet``.

        Args:
            scenario_dir: Root output directory for this scenario.
            trace_records: List of per-timestep trace dicts from
                :meth:`~satsim.sim.engine.SimulationEngine.run`.
            config: The config that produced the trace (embedded for reproducibility).
            metrics_df: Optional DataFrame of per-timestep telemetry to export.

        Returns:
            Path to the written ``trace.json`` file.
        """
        scenario_dir = Path(scenario_dir)
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save config_used.yaml (reproducibility contract)
        config_path = scenario_dir / "config_used.yaml"
        config.to_yaml(config_path)

        # 2. Save trace.json (canonical per-timestep simulation trace)
        trace_path = scenario_dir / "trace.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_records, f, indent=2)

        # 3. Save global metrics
        metrics_dir = scenario_dir / "global_metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        if metrics_df is not None:
            metrics_df.to_csv(metrics_dir / "metrics.csv", index=False)
            try:
                metrics_df.to_parquet(metrics_dir / "metrics.parquet", index=False)
            except Exception:
                pass

        return trace_path

    @staticmethod
    def load_trace(scenario_dir: Path) -> Tuple[List[Dict[str, Any]], SimConfig]:
        """Load a previously saved trace from *scenario_dir*.

        Args:
            scenario_dir: Directory containing ``config_used.yaml`` and ``trace.json``.

        Returns:
            A 2-tuple of ``(trace_records, config)``.

        Raises:
            FileNotFoundError: If ``config_used.yaml`` or ``trace.json`` are missing.
        """
        scenario_dir = Path(scenario_dir)
        config_path = scenario_dir / "config_used.yaml"
        trace_path = scenario_dir / "trace.json"

        if not config_path.exists() or not trace_path.exists():
            raise FileNotFoundError(
                f"Scenario directory '{scenario_dir}' missing config_used.yaml or trace.json!"
            )

        config = SimConfig.load_yaml(config_path)
        with open(trace_path, "r", encoding="utf-8") as f:
            trace_records = json.load(f)

        return trace_records, config
