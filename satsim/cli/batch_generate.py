from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional
import pandas as pd
from joblib import Parallel, delayed

from satsim.sim.scenario_registry import get_scenario_config, SCENARIO_MATRIX
from satsim.sim.engine import SimulationEngine
from satsim.export import TraceStore, GATExporter, LSTMExporter
from satsim.logging import setup_logging, get_logger

logger = get_logger("satsim.cli.batch_generate")


def process_single_scenario(
    scenario_name: str,
    seed: int,
    duration_s: Optional[float],
    output_root: Path,
    window_size: int = 12,
    stride: int = 1,
) -> Dict[str, Any]:
    scen_dir = output_root / scenario_name
    scen_dir.mkdir(parents=True, exist_ok=True)

    config = get_scenario_config(scenario_name, seed=seed)
    if duration_s is not None and duration_s > 0:
        config.duration_seconds = duration_s

    # 1. Run Discrete-Event Simulation
    engine = SimulationEngine(config)
    
    # Inject pre-seeded events if specified in the scenario matrix
    spec = SCENARIO_MATRIX.get(scenario_name.lower().strip(), {})
    for ev_spec in spec.get("pre_seeded_events", []):
        target = ev_spec["target"]
        if isinstance(target, list):
            target = tuple(target)
        engine.injector.trigger_event(
            event_type=ev_spec["type"],
            target_id=target,
            duration_s=ev_spec["duration_s"],
            start_time_s=ev_spec["start_s"],
            params=ev_spec.get("params", {}),
        )

    trace_records = engine.run(progress_bar=False)
    metrics_df = engine.metrics_collector.to_dataframe()

    # 2. Save Trace Store (trace.json, config_used.yaml, global_metrics/)
    TraceStore.save_trace(
        scenario_dir=scen_dir,
        trace_records=trace_records,
        config=config,
        metrics_df=metrics_df,
    )

    # 3. Create routing_history/
    routing_dir = scen_dir / "routing_history"
    routing_dir.mkdir(parents=True, exist_ok=True)
    routes_summary = {
        "scenario": scenario_name,
        "seed": seed,
        "duration_s": config.duration_seconds,
        "total_timesteps": len(trace_records),
        "total_packets_sent": int(metrics_df["total_packets_sent"].iloc[-1]) if not metrics_df.empty else 0,
        "total_packets_delivered": int(metrics_df["total_packets_delivered"].iloc[-1]) if not metrics_df.empty else 0,
        "pdr": float(metrics_df["packet_delivery_ratio"].iloc[-1]) if not metrics_df.empty else 0.0,
    }
    with open(routing_dir / "routes_summary.json", "w", encoding="utf-8") as f:
        json.dump(routes_summary, f, indent=2)

    # 4. Export GAT dataset
    gat_exporter = GATExporter(scen_dir)
    gat_files = gat_exporter.export_scenario()

    # 5. Export LSTM dataset — use window/stride from config (§ acceptance criteria)
    lstm_exporter = LSTMExporter(scen_dir)
    lstm_exporter.export_scenario(
        window_size=config.export.lstm.window_size,
        stride=config.export.lstm.stride,
    )

    lstm_meta_path = scen_dir / "lstm" / "window_metadata.json"
    lstm_rows = 0
    if lstm_meta_path.exists():
        with open(lstm_meta_path, "r", encoding="utf-8") as f:
            lstm_rows = json.load(f).get("total_rows", 0)

    summary = {
        "scenario": scenario_name,
        "seed": seed,
        "duration_s": config.duration_seconds,
        "timesteps": len(trace_records),
        "gat_snapshots": len(gat_files),
        "lstm_rows": lstm_rows,
        "total_sent": routes_summary["total_packets_sent"],
        "total_delivered": routes_summary["total_packets_delivered"],
        "pdr": routes_summary["pdr"],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch generate all simulation scenarios and export GAT, LSTM, routing, and metrics datasets."
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="all",
        help=f"Comma-separated scenario names or 'all'. Available: {list(SCENARIO_MATRIX.keys())}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for batch generation (default: 42).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional simulation duration override in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Output root directory for datasets.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="Number of parallel worker jobs (-1 for all CPUs).",
    )

    args = parser.parse_args()
    logger = setup_logging(level="INFO", structured=True)

    if args.scenarios.lower() == "all":
        target_scenarios = list(SCENARIO_MATRIX.keys())
    else:
        target_scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting batch scenario generation",
        scenarios=target_scenarios,
        seed=args.seed,
        duration_s=args.duration,
        output_dir=str(output_root),
        num_workers=args.num_workers,
    )

    summaries = Parallel(n_jobs=args.num_workers)(
        delayed(process_single_scenario)(
            scenario_name=scen,
            seed=args.seed,
            duration_s=args.duration,
            output_root=output_root,
        )
        for scen in target_scenarios
    )

    batch_log_json = output_root / "batch_run_log.json"
    batch_log_csv = output_root / "batch_run_log.csv"

    with open(batch_log_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    df_log = pd.DataFrame(summaries)
    df_log.to_csv(batch_log_csv, index=False)

    # ── Consolidate Master Datasets (Section E, I, Y) ──────────────────────
    logger.info("Consolidating master datasets across all scenarios...")
    lstm_dfs: List[pd.DataFrame] = []
    global_dfs: List[pd.DataFrame] = []
    routing_summaries: List[Dict[str, Any]] = []

    for scen in target_scenarios:
        scen_dir = output_root / scen
        # Raw LSTM
        lstm_csv = scen_dir / "lstm" / "lstm_sequences.csv"
        if lstm_csv.exists():
            lstm_dfs.append(pd.read_csv(lstm_csv))

        # Global metrics
        g_csv = scen_dir / "global_metrics" / "metrics.csv"
        if g_csv.exists():
            df_g = pd.read_csv(g_csv)
            df_g.insert(0, "scenario", scen)
            df_g.insert(1, "seed", args.seed)
            global_dfs.append(df_g)

        # Routing history summary
        r_json = scen_dir / "routing_history" / "routes_summary.json"
        if r_json.exists():
            with open(r_json, "r", encoding="utf-8") as f:
                routing_summaries.append(json.load(f))

    # 1. Master LSTM
    if lstm_dfs:
        df_master_lstm = pd.concat(lstm_dfs, ignore_index=True)
        master_csv = output_root / "lstm_all_scenarios.csv"
        master_pq = output_root / "lstm_all_scenarios.parquet"
        df_master_lstm.to_csv(master_csv, index=False)
        try:
            df_master_lstm.to_parquet(master_pq, index=False)
        except Exception:
            pass
        logger.info("Master LSTM dataset created", total_rows=len(df_master_lstm), path=str(master_pq))

    # 2. Master Global Metrics
    if global_dfs:
        df_master_g = pd.concat(global_dfs, ignore_index=True)
        df_master_g.to_csv(output_root / "global_metrics.csv", index=False)

    # 3. Master Routing History
    if routing_summaries:
        df_master_r = pd.DataFrame(routing_summaries)
        df_master_r.to_csv(output_root / "routing_history.csv", index=False)

    # 4. Master Metadata
    meta_dir = output_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    master_meta = {
        "num_satellites": 100,
        "num_planes": 10,
        "satellites_per_plane": 10,
        "total_scenarios": len(target_scenarios),
        "scenarios": target_scenarios,
        "seed": args.seed,
        "total_raw_lstm_rows": len(df_master_lstm) if lstm_dfs else 0,
        "timesteps_per_scenario": 720,
    }
    with open(meta_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(master_meta, f, indent=2)

    logger.info(
        "Batch scenario generation complete",
        total_scenarios=len(summaries),
        master_lstm=str(output_root / "lstm_all_scenarios.parquet"),
        batch_log_json=str(batch_log_json),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
