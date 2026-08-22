from __future__ import annotations
import argparse
from pathlib import Path
import sys

from satsim.export.lstm_export import LSTMExporter
from satsim.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export canonical simulation trace records into per-satellite synchronized multivariate sliding-window time-series for LSTM models."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Target scenario name (or 'all' to export all available scenarios in datasets_dir).",
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="datasets",
        help="Root datasets directory containing scenario traces.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=12,
        help="Sliding window size in timesteps (default: 12).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding window stride in timesteps (default: 1).",
    )

    args = parser.parse_args()
    logger = setup_logging(level="INFO", structured=True)

    root_dir = Path(args.datasets_dir)
    if not root_dir.exists():
        logger.error(
            "Datasets directory does not exist",
            datasets_dir=str(root_dir),
        )
        sys.stderr.write(
            f"ERROR: Datasets directory '{root_dir}' does not exist! Run satsim.cli.run_scenario first.\n"
        )
        return 1

    if args.scenario.lower() == "all":
        scenario_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
        if not scenario_dirs:
            sys.stderr.write(
                f"ERROR: No scenario directories found in '{root_dir}'! Run satsim.cli.run_scenario first.\n"
            )
            return 1

        total_exported = 0
        for scen_dir in scenario_dirs:
            try:
                exporter = LSTMExporter(scen_dir)
                exporter.export_scenario(window_size=args.window_size, stride=args.stride)
                total_exported += 1
            except FileNotFoundError as e:
                logger.warning(
                    "Skipping directory without trace",
                    directory=str(scen_dir),
                    reason=str(e),
                )

        if total_exported == 0:
            sys.stderr.write("ERROR: No valid traces were found to export!\n")
            return 1

        logger.info("Batch LSTM export complete", total_scenarios=total_exported)
        return 0
    else:
        scen_dir = root_dir / args.scenario
        try:
            exporter = LSTMExporter(scen_dir)
            out_dir = exporter.export_scenario(window_size=args.window_size, stride=args.stride)
            logger.info("Single LSTM export complete", scenario=args.scenario, output_dir=str(out_dir))
            return 0
        except FileNotFoundError as e:
            sys.stderr.write(f"ERROR: {e}\n")
            return 1


if __name__ == "__main__":
    sys.exit(main())
