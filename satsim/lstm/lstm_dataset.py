"""Dataset discovery, validation, feature audit, time-aware split, sliding window sequence creation, and scaling for PyTorch LSTM training.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
import pickle
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from satsim.logging import get_logger

logger = get_logger(__name__)

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

EXCLUDE_COLUMNS = {
    "scenario",
    "seed",
    "satellite_id",
    "timestep",
    "window_id",
    "step_in_window",
    "Unnamed: 0",
    "index",
}

TARGET_COLUMN = "congestion_score"


class SequenceSample:
    """Dataclass storing a single sliding window sample."""

    def __init__(
        self,
        x: np.ndarray,  # shape [W, num_features]
        y: float,       # target congestion_score(t+1)
        y_curr: float,  # current congestion_score(t) for persistence baseline
        scenario: str,
        seed: int,
        satellite_id: int,
        input_start_t: int,
        input_end_t: int,
        target_t: int,
    ) -> None:
        self.x = x
        self.y = y
        self.y_curr = y_curr
        self.scenario = scenario
        self.seed = seed
        self.satellite_id = satellite_id
        self.input_start_t = input_start_t
        self.input_end_t = input_end_t
        self.target_t = target_t


class FeatureScaler:
    """Standardization scaler for sequence input features."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, samples: List[SequenceSample]) -> None:
        """Fit scaler ONLY on training sequence input features."""
        all_x = []
        for s in samples:
            all_x.append(s.x)  # shape [W, F]
        x_mat = np.vstack(all_x)  # shape [N * W, F]
        self.scaler.fit(x_mat)
        self.fitted = True

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform input feature array [W, F] or [B, W, F]."""
        if not self.fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform!")
        orig_shape = x.shape
        if x.ndim == 2:
            scaled = self.scaler.transform(x)
            return scaled.astype(np.float32)
        elif x.ndim == 3:
            B, W, F = x.shape
            x_flat = x.reshape(-1, F)
            scaled_flat = self.scaler.transform(x_flat)
            return scaled_flat.reshape(B, W, F).astype(np.float32)
        else:
            raise ValueError(f"Unsupported array dimension: {x.ndim}")

    def save(self, filepath: Path | str) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"scaler": self.scaler}, f)

    @classmethod
    def load(cls, filepath: Path | str) -> FeatureScaler:
        instance = cls()
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            instance.scaler = data["scaler"]
            instance.fitted = True
        return instance


class TargetScaler:
    """Standardization scaler for target congestion_score(t+1)."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, samples: List[SequenceSample]) -> None:
        """Fit scaler ONLY on training targets."""
        all_y = np.array([s.y for s in samples]).reshape(-1, 1)
        self.scaler.fit(all_y)
        self.fitted = True

    def transform(self, y: np.ndarray | float) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TargetScaler must be fitted before transform!")
        y_arr = np.array(y, dtype=np.float32).reshape(-1, 1)
        scaled = self.scaler.transform(y_arr)
        return scaled.astype(np.float32)

    def inverse_transform(self, y_scaled: np.ndarray | torch.Tensor) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TargetScaler must be fitted before inverse_transform!")
        y_np = y_scaled.cpu().numpy() if isinstance(y_scaled, torch.Tensor) else y_scaled
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        return self.scaler.inverse_transform(y_np)

    def save(self, filepath: Path | str) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"scaler": self.scaler}, f)

    @classmethod
    def load(cls, filepath: Path | str) -> TargetScaler:
        instance = cls()
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            instance.scaler = data["scaler"]
            instance.fitted = True
        return instance


class PyGSequenceDataset(Dataset):
    """PyTorch Dataset wrapping scaled window sequences."""

    def __init__(self, samples: List[SequenceSample], feature_scaler: FeatureScaler, target_scaler: TargetScaler) -> None:
        self.samples = samples
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        sample = self.samples[idx]
        x_scaled = self.feature_scaler.transform(sample.x)
        y_scaled = self.target_scaler.transform(sample.y).squeeze()

        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(-1)
        y_raw_tensor = torch.tensor(sample.y, dtype=torch.float32).unsqueeze(-1)
        y_curr = float(sample.y_curr)

        return x_tensor, y_tensor, y_raw_tensor, y_curr


class LEOLSTMDataset:
    """Manages raw dataset loading, validation, feature audit, time-aware splitting, and sequence generation."""

    def __init__(self, data_path: Path | str = "datasets/lstm_all_scenarios.csv") -> None:
        self.data_path = Path(data_path)
        self.df: pd.DataFrame = pd.DataFrame()
        self.feature_columns: List[str] = []
        self.target_column = TARGET_COLUMN

    def load_and_validate(self) -> pd.DataFrame:
        """Load raw CSV/Parquet and validate structure."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"[FAIL] Raw LSTM dataset file not found at: {self.data_path}")

        if self.data_path.suffix == ".parquet":
            self.df = pd.read_parquet(self.data_path)
        else:
            self.df = pd.read_csv(self.data_path)

        rows = len(self.df)
        scenarios = sorted(self.df["scenario"].unique().tolist())
        sats = sorted(self.df["satellite_id"].unique().tolist())
        timesteps = sorted(self.df["timestep"].unique().tolist())
        seeds = sorted(self.df["seed"].unique().tolist())

        print("\n====================================================")
        print("LSTM DATASET VALIDATION")
        print("====================================================")
        print(f"File:        {self.data_path}")
        print(f"Rows:        {rows:,}")
        print(f"Scenarios:   {len(scenarios)} {scenarios[:3]}...")
        print(f"Satellites:  {len(sats)} (IDs {sats[0]}..{sats[-1]})")
        print(f"Timesteps:   {len(timesteps)} (range {timesteps[0]}..{timesteps[-1]})")
        print(f"Seeds:       {len(seeds)} {seeds}")
        print("====================================================\n")

        # Programmatic Structural Assertions
        assert len(scenarios) == 13, f"Expected 13 scenarios, found {len(scenarios)}"
        assert len(sats) == 100, f"Expected 100 satellites, found {len(sats)}"
        assert sats[0] == 0 and sats[-1] == 99, "Satellite IDs must be strictly 0-99"

        # Check NaNs and Infs
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        nan_count = self.df[num_cols].isna().sum().sum()
        inf_count = np.isinf(self.df[num_cols].values).sum()

        if nan_count > 0 or inf_count > 0:
            raise ValueError(f"[FAIL] Dataset contains {nan_count} NaNs and {inf_count} Infs!")

        # Check duplicate rows
        dup_count = self.df.duplicated(subset=["scenario", "seed", "satellite_id", "timestep"]).sum()
        if dup_count > 0:
            raise ValueError(f"[FAIL] Found {dup_count} duplicate (scenario, seed, satellite_id, timestep) rows!")

        return self.df

    def audit_features(self, artifacts_dir: Path | str = "artifacts/lstm") -> List[str]:
        """Perform feature audit, correlation matrix calculation, exact duplicate check, and save audit CSV."""
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        all_cols = list(self.df.columns)
        candidate_cols = [c for c in all_cols if c not in EXCLUDE_COLUMNS and c != TARGET_COLUMN]
        # Only numeric columns
        candidate_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(self.df[c])]

        # Correlation analysis
        corr_matrix = self.df[candidate_cols].corr()

        exact_duplicates = []
        high_correlations = []

        for i in range(len(candidate_cols)):
            for j in range(i + 1, len(candidate_cols)):
                c1, c2 = candidate_cols[i], candidate_cols[j]
                r_val = corr_matrix.loc[c1, c2]
                if np.isnan(r_val):
                    continue
                if abs(r_val) > 0.99999:
                    exact_duplicates.append((c1, c2, r_val))
                elif abs(r_val) >= 0.95:
                    high_correlations.append((c1, c2, r_val))

        # Remove redundant exact duplicate columns from feature list
        redundant_to_remove = set()
        for c1, c2, _ in exact_duplicates:
            redundant_to_remove.add(c2)

        selected_features = [c for c in candidate_cols if c not in redundant_to_remove]
        self.feature_columns = selected_features

        # Save feature audit CSV
        audit_rows = []
        for c in candidate_cols:
            is_sel = c in selected_features
            audit_rows.append({
                "feature_name": c,
                "selected": is_sel,
                "mean": float(self.df[c].mean()),
                "std": float(self.df[c].std()),
                "min": float(self.df[c].min()),
                "max": float(self.df[c].max()),
            })

        audit_df = pd.DataFrame(audit_rows)
        audit_df.to_csv(artifacts_dir / "feature_audit.csv", index=False)

        print("\nLSTM INPUT FEATURES:")
        print("--------------------")
        for idx, feat in enumerate(selected_features, 1):
            print(f"{idx:2d}. {feat}")

        if exact_duplicates:
            print(f"\nExact duplicate feature columns detected & excluded: {redundant_to_remove}\n")

        return selected_features

    def build_time_aware_sequences(
        self, window_size: int = 30, stride: int = 1, train_ratio: float = 0.70, val_ratio: float = 0.15
    ) -> Tuple[List[SequenceSample], List[SequenceSample], List[SequenceSample], Dict[str, List[SequenceSample]]]:
        """Form sliding window sequences independently per (scenario, seed, satellite_id) strictly within time-aware splits."""
        train_samples: List[SequenceSample] = []
        val_samples: List[SequenceSample] = []
        test_samples: List[SequenceSample] = []
        scenario_test_samples: Dict[str, List[SequenceSample]] = {scen: [] for scen in EXPECTED_SCENARIOS}

        # Sort dataframe by group and timestep
        grouped = self.df.groupby(["scenario", "seed", "satellite_id"])

        for (scen, seed, sat_id), group_df in grouped:
            group_sorted = group_df.sort_values("timestep").reset_index(drop=True)
            n_t = len(group_sorted)

            # Determine split timestep boundaries on the raw timeline
            n_train_t = int(n_t * train_ratio)
            n_val_t = int(n_t * val_ratio)

            train_df = group_sorted.iloc[:n_train_t].reset_index(drop=True)
            val_df = group_sorted.iloc[n_train_t : n_train_t + n_val_t].reset_index(drop=True)
            test_df = group_sorted.iloc[n_train_t + n_val_t :].reset_index(drop=True)

            def create_windows_from_df(sub_df: pd.DataFrame) -> List[SequenceSample]:
                samples = []
                m = len(sub_df)
                if m <= window_size:
                    return samples
                x_feat_mat = sub_df[self.feature_columns].values.astype(np.float32)
                y_target_mat = sub_df[TARGET_COLUMN].values.astype(np.float32)
                t_mat = sub_df["timestep"].values.astype(int)

                for idx in range(0, m - window_size, stride):
                    # Input sequence: idx .. idx + window_size - 1
                    # Target: idx + window_size
                    x_win = x_feat_mat[idx : idx + window_size]
                    y_t_plus_1 = float(y_target_mat[idx + window_size])
                    y_curr = float(y_target_mat[idx + window_size - 1])

                    t_start = int(t_mat[idx])
                    t_end = int(t_mat[idx + window_size - 1])
                    t_target = int(t_mat[idx + window_size])

                    # Target Leakage Programmatic Assertion
                    assert t_end < t_target, f"Target leakage! input_end_t ({t_end}) >= target_t ({t_target})"
                    assert t_target == t_end + 1, f"Discontinuous timesteps! {t_end} -> {t_target}"

                    samples.append(
                        SequenceSample(
                            x=x_win,
                            y=y_t_plus_1,
                            y_curr=y_curr,
                            scenario=scen,
                            seed=int(seed),
                            satellite_id=int(sat_id),
                            input_start_t=t_start,
                            input_end_t=t_end,
                            target_t=t_target,
                        )
                    )
                return samples

            tr_s = create_windows_from_df(train_df)
            va_s = create_windows_from_df(val_df)
            te_s = create_windows_from_df(test_df)

            train_samples.extend(tr_s)
            val_samples.extend(va_s)
            test_samples.extend(te_s)
            scenario_test_samples[scen].extend(te_s)

        print("\nTIME-AWARE SLIDING WINDOW SPLIT SUMMARY")
        print("---------------------------------------")
        print(f"Window Size:            {window_size}")
        print(f"Train Sequences:        {len(train_samples):7,d} (70%)")
        print(f"Validation Sequences:   {len(val_samples):7,d} (15%)")
        print(f"Test Sequences:         {len(test_samples):7,d} (15%)\n")

        return train_samples, val_samples, test_samples, scenario_test_samples
