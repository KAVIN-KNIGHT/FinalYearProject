"""Fresh-install acceptance gate (Phase 10).

This test validates the "fresh clone → pip install -e . → batch_generate" requirement:
a subprocess invocation of the batch_generate CLI for the smallest possible scenario
(low_load, 15 s) must succeed without any manual intervention and produce every required
file in the output directory.

The test deliberately uses ``subprocess`` rather than a direct Python call so it catches
any import-time errors, missing entry points, or PATH issues that an in-process call
would mask.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tempfile


def test_fresh_install_batch_generate_low_load():
    """Subprocess invocation of batch_generate for low_load must succeed end-to-end.

    This is the acceptance gate for the fresh-clone requirement:
    `pip install -e . && python -m satsim.cli.batch_generate --scenarios low_load
    --duration 15.0 --seed 42` must produce all required sub-folders with no
    manual intervention.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "datasets"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "satsim.cli.batch_generate",
                "--scenarios", "low_load",
                "--duration", "15.0",
                "--seed", "42",
                "--output-dir", str(output_dir),
                "--num-workers", "1",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, (
            f"batch_generate exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        scen_dir = output_dir / "low_load"

        # All required subfolders and files must be present
        assert (scen_dir / "config_used.yaml").exists(), "Missing config_used.yaml"
        assert (scen_dir / "trace.json").exists(), "Missing trace.json"
        assert (scen_dir / "global_metrics" / "metrics.csv").exists(), "Missing global_metrics/metrics.csv"
        assert len(list((scen_dir / "gat").glob("*.pt"))) > 0, "No GAT snapshots generated"
        assert (scen_dir / "lstm" / "lstm_sequences.csv").exists(), "Missing lstm_sequences.csv"
        assert (scen_dir / "routing_history" / "routes_summary.json").exists(), "Missing routes_summary.json"

        # Batch log must exist at the root
        assert (output_dir / "batch_run_log.json").exists(), "Missing batch_run_log.json"
        assert (output_dir / "batch_run_log.csv").exists(), "Missing batch_run_log.csv"
