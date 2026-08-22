"""PyTorch Dataset module for training-time sliding-window sequence generation.

Section G & H: Moves sliding window generation from dataset export time to training time.
Reads the consolidated raw synchronized dataset (``lstm_all_scenarios.parquet`` / ``.csv``)
and produces standardized window tensors ``(window_size, num_features)`` with explicit
future target prediction (e.g. future congestion_score at t + target_horizon) without
target leakage.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from satsim.logging import get_logger

logger = get_logger("satsim.export.lstm_dataset")

#: Default feature set used for sequence modeling (Phase 4 telemetry features)
DEFAULT_FEATURE_COLUMNS: List[str] = [
    "pos_eci_x", "pos_eci_y", "pos_eci_z",
    "vel_eci_x", "vel_eci_y", "vel_eci_z",
    "pos_ecef_x", "pos_ecef_y", "pos_ecef_z",
    "is_active",
    "queue_length",
    "queue_occupancy",
    "buffer_utilization",
    "traffic_load",
    "cpu_utilization",
    "memory_utilization",
    "congestion_score",
    "end_to_end_delay",
    "throughput",
    "link_utilization",
    "neighbor_count",
    "node_degree",
    "routing_table_age",
    "routing_changes_in_window",
    "failure_indicator",
    "event_flags",
]


class LEOSatelliteDataset(Dataset):
    """PyTorch Dataset for synchronized per-satellite sliding-window time-series modeling."""

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame],
        window_size: int = 30,
        stride: int = 5,
        target_horizon: int = 1,
        target_column: str = "congestion_score",
        feature_columns: Optional[List[str]] = None,
        scenario_filter: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Initialise the LEOSatelliteDataset.

        Args:
            data_source: Path to parquet/csv file or an in-memory DataFrame.
            window_size: Sequence window length in timesteps.
            stride: Timestep stride between consecutive windows.
            target_horizon: Future step offset for target prediction (default 1 step ahead).
            target_column: Target metric column name (default 'congestion_score').
            feature_columns: List of input feature column names.
            scenario_filter: Optional scenario name or list of names to filter by.
        """
        self.window_size = window_size
        self.stride = stride
        self.target_horizon = target_horizon
        self.target_column = target_column
        self.feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS

        # Load DataFrame
        if isinstance(data_source, (str, Path)):
            path = Path(data_source)
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")

        # Optional scenario filtering
        if scenario_filter is not None:
            if isinstance(scenario_filter, str):
                scenario_filter = [scenario_filter]
            df = df[df["scenario"].isin(scenario_filter)].copy()

        if df.empty:
            raise ValueError("DataFrame is empty after applying scenario filter!")

        # Verify required columns
        missing_feats = [col for col in self.feature_columns if col not in df.columns]
        if missing_feats:
            raise ValueError(f"Missing required feature columns in dataset: {missing_feats}")
        if self.target_column not in df.columns:
            raise ValueError(f"Missing target column '{self.target_column}' in dataset.")

        # Build window index: list of (group_df_slice, target_val)
        self.samples: List[Tuple[np.ndarray, float]] = []

        # Group by (scenario, satellite_id) and sort by timestep
        grouped = df.groupby(["scenario", "satellite_id"])
        for (scen, sat_id), group in grouped:
            group_sorted = group.sort_values("timestep").reset_index(drop=True)
            n_steps = len(group_sorted)

            # Check continuous timestep ordering
            timesteps = group_sorted["timestep"].values
            diffs = np.diff(timesteps)
            if len(diffs) > 0 and not np.all(diffs == 1):
                logger.warning(
                    "Non-continuous timesteps detected",
                    scenario=scen,
                    sat_id=sat_id,
                )

            feats = group_sorted[self.feature_columns].values.astype(np.float32)
            targets = group_sorted[self.target_column].values.astype(np.float32)

            # Slide windows: sample from start up to (n_steps - window_size - target_horizon + 1)
            max_start = n_steps - window_size - target_horizon + 1
            for start in range(0, max_start, stride):
                end = start + window_size
                target_idx = end - 1 + target_horizon
                x_window = feats[start:end]
                y_val = targets[target_idx]
                self.samples.append((x_window, y_val))

        logger.info(
            "LEOSatelliteDataset initialized",
            num_samples=len(self.samples),
            window_size=self.window_size,
            stride=self.stride,
            num_features=len(self.feature_columns),
            target=self.target_column,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_win, y_val = self.samples[idx]
        return torch.from_numpy(x_win), torch.tensor(y_val, dtype=torch.float32)
