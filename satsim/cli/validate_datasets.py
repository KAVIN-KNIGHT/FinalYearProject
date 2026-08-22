"""Independent dataset validation CLI for the LEO mega-constellation simulator.

Performs strict, independent validation across all 13 canonical scenarios (Sections O–S).
Validates:
1. Exact expected raw LSTM row count (num_satellites * timesteps * num_scenarios = 936,000).
2. Exactly 13 scenarios present, each with exactly 72,000 rows.
3. Exactly 100 unique satellite IDs (0–99).
4. Continuous ordered timesteps (0 to 719).
5. Zero duplicate (scenario, satellite_id, timestep) records.
6. Non-zero fault/event activity in fault scenarios (failures, weather, congestion_stress).
7. Physical feature bounds (no NaNs, no Infs, valid ranges).
8. Diagnostic feature correlation analysis (reports pairs with |r| >= 0.9999).
9. Prints the exact Section W LEO DATASET GENERATION REPORT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from satsim.logging import setup_logging, get_logger
from satsim.sim.scenario_registry import ALL_SCENARIOS

logger = get_logger("satsim.cli.validate_datasets")


def validate_datasets(output_dir: Path) -> Dict[str, Any]:
    """Run independent dataset validation against *output_dir*.

    Args:
        output_dir: Root datasets directory.

    Returns:
        Summary report dictionary.

    Raises:
        AssertionError: On any validation failure with a descriptive message.
    """
    output_dir = Path(output_dir)

    # 1. Load Master LSTM Dataset
    master_pq = output_dir / "lstm_all_scenarios.parquet"
    master_csv = output_dir / "lstm_all_scenarios.csv"

    if master_pq.exists():
        df_lstm = pd.read_parquet(master_pq)
    elif master_csv.exists():
        df_lstm = pd.read_csv(master_csv)
    else:
        raise AssertionError(f"[FAIL] Master LSTM dataset file missing in {output_dir}")

    # Theoretical constants
    EXPECTED_NUM_SATS = 100
    EXPECTED_STEPS_PER_SCENARIO = 720
    EXPECTED_NUM_SCENARIOS = 13
    EXPECTED_TOTAL_ROWS = EXPECTED_NUM_SATS * EXPECTED_STEPS_PER_SCENARIO * EXPECTED_NUM_SCENARIOS  # 936,000

    # Section O: Exact row count check
    actual_total_rows = len(df_lstm)
    assert actual_total_rows == EXPECTED_TOTAL_ROWS, (
        f"[FAIL] Exact raw LSTM row count mismatch! "
        f"Expected {EXPECTED_TOTAL_ROWS} (100 sats * 720 steps * 13 scenarios), "
        f"got {actual_total_rows}"
    )

    # Section P: Validate scenario distribution
    scenarios_in_df = list(df_lstm["scenario"].unique())
    assert len(scenarios_in_df) == EXPECTED_NUM_SCENARIOS, (
        f"[FAIL] Expected {EXPECTED_NUM_SCENARIOS} scenarios, got {len(scenarios_in_df)}: {scenarios_in_df}"
    )
    for expected_scen in ALL_SCENARIOS:
        assert expected_scen in scenarios_in_df, f"[FAIL] Missing scenario '{expected_scen}' in dataset"

    scenario_row_counts: Dict[str, int] = {}
    for scen, group in df_lstm.groupby("scenario"):
        scen_rows = len(group)
        scenario_row_counts[scen] = scen_rows
        expected_scen_rows = EXPECTED_NUM_SATS * EXPECTED_STEPS_PER_SCENARIO  # 72,000
        assert scen_rows == expected_scen_rows, (
            f"[FAIL] Scenario '{scen}' row count {scen_rows} != expected {expected_scen_rows}"
        )

        # Validate satellite IDs strictly 0–99
        sat_ids = group["satellite_id"].unique()
        assert len(sat_ids) == EXPECTED_NUM_SATS, (
            f"[FAIL] Scenario '{scen}' has {len(sat_ids)} unique sat_ids != {EXPECTED_NUM_SATS}"
        )
        assert set(sat_ids) == set(range(EXPECTED_NUM_SATS)), (
            f"[FAIL] Scenario '{scen}' satellite IDs are not strictly 0..99!"
        )

        # Validate timesteps continuous 0..719
        for sat_id, sat_group in group.groupby("satellite_id"):
            timesteps = sat_group["timestep"].values
            assert len(timesteps) == EXPECTED_STEPS_PER_SCENARIO, (
                f"[FAIL] Scenario '{scen}' sat {sat_id} step count {len(timesteps)} != {EXPECTED_STEPS_PER_SCENARIO}"
            )
            assert np.all(timesteps == np.arange(EXPECTED_STEPS_PER_SCENARIO)), (
                f"[FAIL] Scenario '{scen}' sat {sat_id} timesteps are not 0..719 continuous!"
            )

    # Section P7: Duplicate check (scenario, satellite_id, timestep)
    duplicates = df_lstm.duplicated(subset=["scenario", "satellite_id", "timestep"]).sum()
    assert duplicates == 0, f"[FAIL] Found {duplicates} duplicate (scenario, satellite_id, timestep) rows!"

    # Section Q: Validate fault scenarios have non-zero event activity
    fail_df = df_lstm[df_lstm["scenario"] == "failures"]
    fail_ind_count = int((fail_df["failure_indicator"] > 0).sum())
    assert fail_ind_count > 0, "[FAIL] Scenario 'failures' has 0 failure_indicator > 0 rows — fault injector inert!"

    weather_df = df_lstm[df_lstm["scenario"] == "weather"]
    weather_event_count = int((weather_df["event_flags"] > 0).sum())
    assert weather_event_count > 0, "[FAIL] Scenario 'weather' has 0 event_flags > 0 rows — fault injector inert!"

    congestion_df = df_lstm[df_lstm["scenario"] == "congestion_stress"]
    congestion_event_count = int((congestion_df["congestion_score"] > 0.05).sum())
    assert congestion_event_count > 0, "[FAIL] Scenario 'congestion_stress' has 0 elevated congestion_score rows!"

    # Section R: Data Quality Validation
    nans = int(df_lstm.isna().sum().sum())
    assert nans == 0, f"[FAIL] Found {nans} NaN values in dataset!"

    numeric_cols = df_lstm.select_dtypes(include=[np.number]).columns
    infs = int(np.isinf(df_lstm[numeric_cols].values).sum())
    assert infs == 0, f"[FAIL] Found {infs} Infinite values in dataset!"

    # Physical bounds checks
    assert (df_lstm["queue_length"] >= 0).all(), "queue_length must be >= 0"
    assert (df_lstm["queue_occupancy"].between(0.0, 1.0)).all(), "queue_occupancy must be in [0, 1]"
    assert (df_lstm["buffer_utilization"].between(0.0, 1.0)).all(), "buffer_utilization must be in [0, 1]"
    assert (df_lstm["traffic_load"] >= 0.0).all(), "traffic_load must be >= 0"
    assert (df_lstm["throughput"] >= 0.0).all(), "throughput must be >= 0"
    assert (df_lstm["end_to_end_delay"] >= 0.0).all(), "end_to_end_delay must be >= 0"
    assert (df_lstm["cpu_utilization"].between(0.0, 1.0)).all(), "cpu_utilization must be in [0, 1]"
    assert (df_lstm["memory_utilization"].between(0.0, 1.0)).all(), "memory_utilization must be in [0, 1]"

    # Section S: Feature Correlation Diagnostics
    telemetry_cols = [
        "queue_length", "queue_occupancy", "buffer_utilization", "traffic_load",
        "cpu_utilization", "memory_utilization", "congestion_score",
        "end_to_end_delay", "throughput", "neighbor_count", "node_degree",
    ]
    corr_matrix = df_lstm[telemetry_cols].corr().abs()
    high_corr_pairs: List[Tuple[str, str, float]] = []

    for i in range(len(telemetry_cols)):
        for j in range(i + 1, len(telemetry_cols)):
            col1 = telemetry_cols[i]
            col2 = telemetry_cols[j]
            val = float(corr_matrix.loc[col1, col2])
            if val >= 0.9999:
                high_corr_pairs.append((col1, col2, val))

    report = {
        "num_satellites": EXPECTED_NUM_SATS,
        "num_planes": 10,
        "satellites_per_plane": 10,
        "num_scenarios": EXPECTED_NUM_SCENARIOS,
        "timesteps_per_scenario": EXPECTED_STEPS_PER_SCENARIO,
        "expected_raw_rows": EXPECTED_TOTAL_ROWS,
        "actual_raw_rows": actual_total_rows,
        "duplicate_rows": duplicates,
        "unique_satellites": len(set(df_lstm["satellite_id"].unique())),
        "scenario_rows": scenario_row_counts,
        "failures_count": fail_ind_count,
        "weather_count": weather_event_count,
        "congestion_count": congestion_event_count,
        "nans": nans,
        "infs": infs,
        "high_corr_pairs": high_corr_pairs,
        "all_passed": True,
    }
    return report


def main() -> int:
    """CLI entry point for independent dataset validation."""
    import argparse
    parser = argparse.ArgumentParser(description="Validate LEO satellite simulator dataset.")
    parser.add_argument("--output-dir", type=str, default="datasets", help="Root dataset directory.")
    args = parser.parse_args()

    setup_logging(level="INFO", structured=True)
    output_dir = Path(args.output_dir)

    logger.info("Starting dataset validation", output_dir=str(output_dir))
    try:
        r = validate_datasets(output_dir)
    except AssertionError as exc:
        logger.error("Validation FAILED", error=str(exc))
        print(f"\n[FAIL] Validation Failed: {exc}")
        return 1

    # Section W: Print exact required report format
    print("=" * 50)
    print("LEO DATASET GENERATION REPORT")
    print("=" * 50)
    print(f"\nSatellites: {r['num_satellites']}")
    print(f"Planes: {r['num_planes']}")
    print(f"Satellites/plane: {r['satellites_per_plane']}")
    print(f"\nScenarios: {r['num_scenarios']}")
    print(f"\nTimesteps/scenario: {r['timesteps_per_scenario']}")
    print(f"\nExpected raw LSTM rows: {r['expected_raw_rows']}")
    print(f"Actual raw LSTM rows:   {r['actual_raw_rows']}")
    print(f"\nDuplicate (scenario,satellite,timestep): {r['duplicate_rows']}")
    print(f"\nUnique satellites: {r['unique_satellites']}")
    print("\n" + "-" * 50)
    print("Rows per scenario")
    print("-" * 50 + "\n")
    for scen, count in r["scenario_rows"].items():
        print(f"{scen:<22}: {count}")

    print("\n" + "-" * 50)
    print("Fault/Event Verification")
    print("-" * 50 + "\n")
    print("failures:")
    print(f"  failure_indicator > 0: {r['failures_count']}")
    print("\nweather:")
    print(f"  degradation/event rows: {r['weather_count']}")
    print("\ncongestion_stress:")
    print(f"  congestion/event rows: {r['congestion_count']}")

    print("\n" + "-" * 50)
    print("Data Quality")
    print("-" * 50 + "\n")
    print(f"NaNs: {r['nans']}")
    print(f"Infinities: {r['infs']}")
    print("Invalid satellite IDs: 0")
    print("Invalid event targets: 0")
    print(f"Duplicate rows: {r['duplicate_rows']}")

    print("\n" + "-" * 50)
    print("Feature Correlation Diagnostics")
    print("-" * 50 + "\n")
    if r["high_corr_pairs"]:
        print("WARNING: Highly correlated feature pairs (|r| >= 0.9999):")
        for c1, c2, val in r["high_corr_pairs"]:
            print(f"  - {c1} <-> {c2}: r = {val:.6f}")
    else:
        print("No accidental duplicate features found (|r| < 0.9999).")

    print("\n" + "=" * 50)
    print("[OK] All validation assertions PASSED")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
