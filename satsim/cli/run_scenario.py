from __future__ import annotations
import argparse
from pathlib import Path
import sys

from satsim.sim.scenario_registry import get_scenario_config, SCENARIO_MATRIX
from satsim.sim.engine import SimulationEngine
from satsim.export.trace_store import TraceStore
from satsim.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a satsim scenario simulation and export canonical trace."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="low_load",
        help=f"Scenario preset name. Choices: {list(SCENARIO_MATRIX.keys())}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for exact numerical reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory path (default: datasets/<scenario_name>).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional simulation duration override in seconds.",
    )

    args = parser.parse_args()
    logger = setup_logging(level="INFO", structured=True)

    config = get_scenario_config(args.scenario, seed=args.seed)
    if args.duration is not None:
        config.duration_seconds = args.duration

    if args.output_dir is None:
        output_dir = Path("datasets") / args.scenario
    else:
        output_dir = Path(args.output_dir)

    logger.info(
        "Starting scenario simulation",
        scenario=args.scenario,
        seed=args.seed,
        duration_s=config.duration_seconds,
        output_dir=str(output_dir),
    )

    engine = SimulationEngine(config)

    # Inject pre-seeded events if specified in the scenario matrix
    spec = SCENARIO_MATRIX.get(args.scenario.lower().strip(), {})
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

    trace_records = engine.run(progress_bar=True)

    metrics_df = engine.metrics_collector.to_dataframe()
    trace_file = TraceStore.save_trace(
        scenario_dir=output_dir,
        trace_records=trace_records,
        config=config,
        metrics_df=metrics_df,
    )

    logger.info(
        "Scenario simulation complete",
        scenario=args.scenario,
        steps=len(trace_records),
        trace_file=str(trace_file),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
