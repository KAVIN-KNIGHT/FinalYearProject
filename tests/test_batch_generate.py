"""Tests for the batch scenario generator CLI.

Validates that:
- All 13 canonical scenario names are present in the registry.
- A single-scenario run produces every required sub-folder and file.
- Re-running with the same seed produces byte-identical outputs (reproducibility gate).
"""
import json
from pathlib import Path
import tempfile

import pytest

from satsim.sim.scenario_registry import SCENARIO_MATRIX, ALL_SCENARIOS
from satsim.cli.batch_generate import process_single_scenario


# ---------------------------------------------------------------------------
# Scenario registry coverage
# ---------------------------------------------------------------------------

EXPECTED_SCENARIOS = {
    "low_load", "medium_load", "high_load", "peak_load",
    "burst", "flash_crowd", "hotspot", "random_traffic",
    "self_similar", "mixed",
    "failures", "weather", "congestion_stress",
}


def test_all_13_scenario_names_in_registry():
    """All 13 canonical (§6 reconciled) scenario names must be registered."""
    assert set(SCENARIO_MATRIX.keys()) == EXPECTED_SCENARIOS, (
        f"Missing scenarios: {EXPECTED_SCENARIOS - set(SCENARIO_MATRIX.keys())}\n"
        f"Extra scenarios:  {set(SCENARIO_MATRIX.keys()) - EXPECTED_SCENARIOS}"
    )


def test_all_scenarios_list_matches_matrix():
    """ALL_SCENARIOS convenience list must equal SCENARIO_MATRIX keys."""
    assert set(ALL_SCENARIOS) == set(SCENARIO_MATRIX.keys())


# ---------------------------------------------------------------------------
# Folder layout compliance
# ---------------------------------------------------------------------------

def test_process_single_scenario_generates_all_required_folders():
    """process_single_scenario must populate every required sub-folder and file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir) / "datasets"

        summary = process_single_scenario(
            scenario_name="low_load",
            seed=42,
            duration_s=15.0,   # 3 timesteps → fast
            output_root=output_root,
        )

        scen_dir = output_root / "low_load"
        assert scen_dir.exists()

        # Critical Acceptance Criteria: every required sub-folder populated
        assert (scen_dir / "config_used.yaml").exists(), "Missing config_used.yaml"
        assert (scen_dir / "trace.json").exists(), "Missing trace.json"
        assert (scen_dir / "global_metrics" / "metrics.csv").exists(), "Missing metrics.csv"
        assert (scen_dir / "gat").exists(), "Missing gat/"
        assert len(list((scen_dir / "gat").glob("*.pt"))) == 3, "Wrong number of .pt snapshots"
        assert (scen_dir / "lstm" / "lstm_sequences.csv").exists(), "Missing lstm_sequences.csv"
        assert (scen_dir / "routing_history" / "routes_summary.json").exists(), "Missing routes_summary.json"

        # Summary metadata assertions
        assert summary["scenario"] == "low_load"
        assert summary["seed"] == 42
        assert summary["gat_snapshots"] == 3
        assert summary["lstm_rows"] > 0


# ---------------------------------------------------------------------------
# Reproducibility gate
# ---------------------------------------------------------------------------

def test_batch_generate_reproducibility():
    """Re-running with the same seed must produce byte-identical trace, config, and LSTM outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root1 = Path(tmpdir) / "run1"
        root2 = Path(tmpdir) / "run2"

        sum1 = process_single_scenario("low_load", seed=42, duration_s=15.0, output_root=root1)
        sum2 = process_single_scenario("low_load", seed=42, duration_s=15.0, output_root=root2)

        assert sum1 == sum2, "Summary statistics differ between runs"

        c1 = (root1 / "low_load" / "config_used.yaml").read_text()
        c2 = (root2 / "low_load" / "config_used.yaml").read_text()
        assert c1 == c2, "config_used.yaml differs between identical-seed runs"

        t1 = (root1 / "low_load" / "trace.json").read_text()
        t2 = (root2 / "low_load" / "trace.json").read_text()
        assert t1 == t2, "trace.json differs between identical-seed runs"

        l1 = (root1 / "low_load" / "lstm" / "lstm_sequences.csv").read_text()
        l2 = (root2 / "low_load" / "lstm" / "lstm_sequences.csv").read_text()
        assert l1 == l2, "lstm_sequences.csv differs between identical-seed runs"
