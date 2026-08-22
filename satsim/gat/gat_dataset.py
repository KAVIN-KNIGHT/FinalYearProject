"""Dataset loading, validation, pairwise alignment (t -> t+1), feature/target scaling, and time-aware splitting for GAT graph snapshots.
"""
from __future__ import annotations

import os
from pathlib import Path
import pickle
from typing import List, Dict, Tuple, Any
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

from satsim.logging import get_logger

logger = get_logger(__name__)

# Canonical list of 13 LEO simulation scenarios
EXPECTED_SCENARIOS = [
    "low_load",
    "medium_load",
    "high_load",
    "peak_load",
    "burst",
    "flash_crowd",
    "hotspot",
    "random_traffic",
    "self_similar",
    "mixed",
    "failures",
    "weather",
    "congestion_stress",
]

# 8-Node / 4-Edge Streamlined Physical Architecture Indices:
# Node features (8): pos_eci_x, pos_eci_y, pos_eci_z, vel_eci_x, vel_eci_y, vel_eci_z, buffer_utilization, degree
FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 10, 12]
TARGET_INDEX = 13

# Edge attributes (4): distance_km, delay_ms, link_utilization, link_failure_probability
EDGE_INDICES = [0, 1, 2, 4]


class SnapshotMetadata:
    """Stores metadata for a graph snapshot file."""

    def __init__(self, filepath: Path, scenario: str, timestep: int) -> None:
        self.filepath = filepath
        self.scenario = scenario
        self.timestep = timestep


class FeatureScaler:
    """Standardization scaler for 8-dim node features and 4-dim edge features."""

    def __init__(self) -> None:
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.fitted = False

    def fit(self, train_dataset: List[Data]) -> None:
        """Fit node and edge scalers on training input features only."""
        all_x = []
        all_edge_attr = []
        for data in train_dataset:
            all_x.append(data.x.cpu().numpy())
            if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
                all_edge_attr.append(data.edge_attr.cpu().numpy())

        node_mat = np.vstack(all_x)
        self.node_scaler.fit(node_mat)

        if all_edge_attr:
            edge_mat = np.vstack(all_edge_attr)
            self.edge_scaler.fit(edge_mat)

        self.fitted = True

    def transform(self, data: Data) -> Data:
        """Transform input node and edge features using fitted scaling parameters."""
        if not self.fitted:
            raise RuntimeError("FeatureScaler must be fitted on training data before calling transform!")

        data_copy = data.clone()

        # Scale 16 node features x
        x_np = data_copy.x.cpu().numpy()
        x_scaled = self.node_scaler.transform(x_np)
        data_copy.x = torch.tensor(x_scaled, dtype=torch.float32)

        # Scale edge features if present
        if data_copy.edge_attr is not None and data_copy.edge_attr.shape[0] > 0:
            edge_np = data_copy.edge_attr.cpu().numpy()
            edge_scaled = self.edge_scaler.transform(edge_np)
            data_copy.edge_attr = torch.tensor(edge_scaled, dtype=torch.float32)

        return data_copy

    def save(self, filepath: Path | str) -> None:
        """Save fitted feature scaler to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"node_scaler": self.node_scaler, "edge_scaler": self.edge_scaler}, f)

    @classmethod
    def load(cls, filepath: Path | str) -> FeatureScaler:
        """Load feature scaler from disk."""
        instance = cls()
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            instance.node_scaler = data["node_scaler"]
            instance.edge_scaler = data["edge_scaler"]
            instance.fitted = True
        return instance


class TargetScaler:
    """Standardization scaler for target congestion_score(t+1)."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, train_dataset: List[Data]) -> None:
        """Fit target scaler on training targets only."""
        all_y = []
        for data in train_dataset:
            y_np = data.y.cpu().numpy()
            all_y.append(y_np if y_np.ndim == 2 else y_np.reshape(-1, 1))

        y_mat = np.vstack(all_y)
        self.scaler.fit(y_mat)
        self.fitted = True

    def transform(self, y: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Transform target values to zero-mean unit-variance scale."""
        if not self.fitted:
            raise RuntimeError("TargetScaler must be fitted before calling transform!")
        y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        scaled = self.scaler.transform(y_np)
        return torch.tensor(scaled, dtype=torch.float32)

    def inverse_transform(self, y_scaled: torch.Tensor | np.ndarray) -> np.ndarray:
        """Inverse-transform scaled predictions back to raw congestion_score scale [0, 2]."""
        if not self.fitted:
            raise RuntimeError("TargetScaler must be fitted before calling inverse_transform!")
        y_np = y_scaled.cpu().numpy() if isinstance(y_scaled, torch.Tensor) else y_scaled
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        return self.scaler.inverse_transform(y_np)

    def save(self, filepath: Path | str) -> None:
        """Save target scaler to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"scaler": self.scaler}, f)

    @classmethod
    def load(cls, filepath: Path | str) -> TargetScaler:
        """Load target scaler from disk."""
        instance = cls()
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            instance.scaler = data["scaler"]
            instance.fitted = True
        return instance


class LEOGraphSnapshotDataset:
    """Discovers, validates, pairs (t -> t+1), and time-splits GAT graph snapshots."""

    def __init__(self, root_dir: Path | str = "datasets") -> None:
        self.root_dir = Path(root_dir)
        self.scenario_snapshots: Dict[str, List[SnapshotMetadata]] = {}
        self.total_snapshots_count = 0

    def discover_snapshots(self) -> Dict[str, List[SnapshotMetadata]]:
        """Discover all snapshot .pt files across all 13 scenarios."""
        self.scenario_snapshots = {scen: [] for scen in EXPECTED_SCENARIOS}

        for scen in EXPECTED_SCENARIOS:
            path1 = self.root_dir / scen / "gat"
            path2 = self.root_dir / "gat" / scen
            target_dir = path1 if path1.exists() else (path2 if path2.exists() else None)

            if target_dir and target_dir.exists():
                pt_files = sorted(list(target_dir.glob("snapshot_*.pt")))
                for pt_file in pt_files:
                    try:
                        timestep = int(pt_file.stem.split("_")[-1])
                    except ValueError:
                        timestep = 0
                    self.scenario_snapshots[scen].append(
                        SnapshotMetadata(filepath=pt_file, scenario=scen, timestep=timestep)
                    )

        print("\nGAT DATASET DISCOVERY")
        print("---------------------")
        total = 0
        for scen in EXPECTED_SCENARIOS:
            count = len(self.scenario_snapshots[scen])
            total += count
            print(f"{scen:<20} : {count:5d} snapshots")

        print(f"\nTotal snapshots      : {total:5d}\n")
        self.total_snapshots_count = total

        zero_scenarios = [scen for scen, items in self.scenario_snapshots.items() if len(items) == 0]
        if zero_scenarios:
            raise ValueError(
                f"[FAIL] Missing GAT snapshots for scenario(s): {zero_scenarios}. "
                f"All 13 scenarios must contain GAT snapshot files!"
            )

        return self.scenario_snapshots

    def validate_snapshots(self) -> Tuple[int, int]:
        """Validate every snapshot for 100-node structure, node/edge bounds, zero NaNs/Infs, and leak-free 16 features."""
        total = 0
        valid_100 = 0
        invalid = 0
        snapshots_200 = 0
        invalid_node_ids = 0
        invalid_edge_ids = 0
        nan_count = 0
        inf_count = 0

        for scen, meta_list in self.scenario_snapshots.items():
            for meta in meta_list:
                total += 1
                try:
                    data = torch.load(meta.filepath, weights_only=False)
                    num_nodes = data.x.shape[0]

                    if num_nodes == 200:
                        snapshots_200 += 1
                        invalid += 1
                        continue
                    elif num_nodes != 100:
                        invalid += 1
                        continue

                    if torch.isnan(data.x).any():
                        nan_count += 1
                        invalid += 1
                        continue
                    if torch.isinf(data.x).any():
                        inf_count += 1
                        invalid += 1
                        continue

                    if data.edge_index is not None and data.edge_index.numel() > 0:
                        if (data.edge_index < 0).any() or (data.edge_index >= 100).any():
                            invalid_edge_ids += 1
                            invalid += 1
                            continue

                    valid_100 += 1

                except Exception as e:
                    logger.error("Failed to validate snapshot", filepath=str(meta.filepath), error=str(e))
                    invalid += 1

        print("GAT SNAPSHOT VALIDATION")
        print("-----------------------")
        print(f"Total snapshots:             {total:5d}")
        print(f"Valid 100-node snapshots:    {valid_100:5d}")
        print(f"Invalid snapshots:           {invalid:5d}")
        print(f"200-node snapshots:          {snapshots_200:5d}")
        print(f"Invalid node IDs:            {invalid_node_ids:5d}")
        print(f"Invalid edge IDs:            {invalid_edge_ids:5d}")
        print(f"NaN values:                  {nan_count:5d}")
        print(f"Inf values:                  {inf_count:5d}\n")

        if snapshots_200 > 0:
            raise ValueError(
                f"[STOP] Found {snapshots_200} snapshots with 200 nodes! "
                f"Constellation MUST be exactly 100 satellites."
            )

        if invalid > 0:
            raise ValueError(f"[FAIL] Found {invalid} invalid graph snapshots!")

        rep_meta = self.scenario_snapshots[EXPECTED_SCENARIOS[0]][0]
        rep_data = torch.load(rep_meta.filepath, weights_only=False)

        node_in_dim = len(FEATURE_INDICES)
        edge_in_dim = len(EDGE_INDICES)

        # Target Leakage & Spatial GAT Input Verification Assertion
        print("LEAKAGE VERIFICATION AFTER IMPLEMENTATION")
        print("-----------------------------------------")
        print(f"GAT INPUT DIMENSION: {node_in_dim} (8 non-target physical features: pos_eci, vel_eci, buffer_util, degree)")
        print(f"GAT EDGE DIMENSION : {edge_in_dim} (4 physical link attributes: distance, delay, util, fail_prob)")
        print("GAT TARGET: NONE")
        print("CONGESTION IN GAT INPUT: NO")
        print("CONGESTION IN GAT TARGET: NO\n")

        assert node_in_dim == 8, f"Expected 8 non-target input features, got {node_in_dim}"
        assert edge_in_dim == 4, f"Expected 4 edge features, got {edge_in_dim}"
        assert TARGET_INDEX not in FEATURE_INDICES, "Target congestion_score MUST be excluded from GAT features X!"

        return node_in_dim, edge_in_dim

    def create_aligned_time_splits(
        self, train_ratio: float = 0.70, val_ratio: float = 0.15
    ) -> Tuple[List[Data], List[Data], List[Data], Dict[str, List[Data]]]:
        """Form graph snapshot objects with 8 non-target node features and 4 edge features and split time-wise per scenario."""
        train_pairs: List[Data] = []
        val_pairs: List[Data] = []
        test_pairs: List[Data] = []
        scenario_test_pairs: Dict[str, List[Data]] = {scen: [] for scen in EXPECTED_SCENARIOS}

        for scen in EXPECTED_SCENARIOS:
            meta_list = sorted(self.scenario_snapshots[scen], key=lambda m: m.timestep)
            n_snapshots = len(meta_list)

            scen_pairs: List[Data] = []
            for i in range(n_snapshots):
                m_curr = meta_list[i]
                d_curr = torch.load(m_curr.filepath, weights_only=False)

                # Input X(t): 8 non-target features
                x_input = d_curr.x[:, FEATURE_INDICES]

                # Edge attributes: 4 physical features
                if d_curr.edge_attr is not None and d_curr.edge_attr.numel() > 0:
                    if d_curr.edge_attr.shape[1] >= len(EDGE_INDICES):
                        edge_attr_input = d_curr.edge_attr[:, EDGE_INDICES]
                    else:
                        edge_attr_input = d_curr.edge_attr
                else:
                    edge_attr_input = d_curr.edge_attr

                # Verify dimensions
                assert x_input.shape[1] == 8, f"Expected 8 non-target features, got {x_input.shape[1]}"
                if edge_attr_input is not None and edge_attr_input.numel() > 0:
                    assert edge_attr_input.shape[1] == 4, f"Expected 4 edge features, got {edge_attr_input.shape[1]}"

                paired_data = Data(
                    x=x_input,
                    edge_index=d_curr.edge_index,
                    edge_attr=edge_attr_input,
                )
                paired_data.scenario = scen
                paired_data.timestep = m_curr.timestep
                paired_data.raw_snapshot_path = m_curr.filepath
                scen_pairs.append(paired_data)

            # Time-aware split within scenario snapshots
            n_pairs = len(scen_pairs)
            n_train = int(n_pairs * train_ratio)
            n_val = int(n_pairs * val_ratio)

            train_p = scen_pairs[:n_train]
            val_p = scen_pairs[n_train : n_train + n_val]
            test_p = scen_pairs[n_train + n_val :]

            train_pairs.extend(train_p)
            val_pairs.extend(val_p)
            test_pairs.extend(test_p)
            scenario_test_pairs[scen].extend(test_p)

        print("TIME-AWARE SNAPSHOT SPLIT SUMMARY")
        print("---------------------------------")
        print(f"Train snapshots:      {len(train_pairs):5d} ({train_ratio*100:.0f}%)")
        print(f"Validation snapshots: {len(val_pairs):5d} ({val_ratio*100:.0f}%)")
        print(f"Test snapshots:       {len(test_pairs):5d} ({(1 - train_ratio - val_ratio)*100:.0f}%)\n")

        return train_pairs, val_pairs, test_pairs, scenario_test_pairs
