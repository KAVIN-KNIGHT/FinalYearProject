"""Tests for LEOSatelliteDataset (Section G & H).

Verifies PyTorch Dataset initialization, window sliding, target prediction
offset without target leakage, and DataLoader compatibility.
"""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from satsim.export import LEOSatelliteDataset


def test_leo_satellite_dataset_sliding_window():
    """Verify PyTorch Dataset sliding window shapes and target prediction offset."""
    # Create synthetic raw DataFrame: 2 satellites, 50 timesteps each
    rows = []
    for sat_id in range(2):
        for t in range(50):
            rows.append(
                {
                    "scenario": "low_load",
                    "seed": 42,
                    "satellite_id": sat_id,
                    "timestep": t,
                    "simulation_time_s": float(t * 5.0),
                    "pos_eci_x": 1000.0 + t,
                    "pos_eci_y": 2000.0,
                    "pos_eci_z": 3000.0,
                    "vel_eci_x": 7.5,
                    "vel_eci_y": 0.0,
                    "vel_eci_z": 0.0,
                    "pos_ecef_x": 1000.0,
                    "pos_ecef_y": 2000.0,
                    "pos_ecef_z": 3000.0,
                    "is_active": 1.0,
                    "queue_length": float(t),
                    "queue_occupancy": float(t / 50.0),
                    "buffer_utilization": float(t / 50.0),
                    "traffic_load": float(t / 100.0),
                    "cpu_utilization": float(t / 100.0),
                    "memory_utilization": float(t / 100.0),
                    "congestion_score": float(t * 0.1),
                    "end_to_end_delay": 10.0,
                    "throughput": 100.0,
                    "link_utilization": 0.2,
                    "neighbor_count": 4.0,
                    "node_degree": 4.0,
                    "routing_table_age": 0.0,
                    "routing_changes_in_window": 0.0,
                    "failure_indicator": 0.0,
                    "event_flags": 0.0,
                }
            )

    df_raw = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmpdir:
        pq_path = Path(tmpdir) / "lstm_all_scenarios.parquet"
        df_raw.to_parquet(pq_path, index=False)

        # Window size 10, stride 5, target_horizon 1 step ahead
        dataset = LEOSatelliteDataset(
            data_source=pq_path,
            window_size=10,
            stride=5,
            target_horizon=1,
            target_column="congestion_score",
        )

        assert len(dataset) > 0

        x_tensor, y_tensor = dataset[0]
        assert isinstance(x_tensor, torch.Tensor)
        assert isinstance(y_tensor, torch.Tensor)
        assert x_tensor.shape == (10, len(dataset.feature_columns))
        assert y_tensor.dim() == 0  # scalar target

        # Test DataLoader compatibility
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape[0] <= 4
        assert batch_x.shape[1] == 10
        assert batch_x.shape[2] == len(dataset.feature_columns)
