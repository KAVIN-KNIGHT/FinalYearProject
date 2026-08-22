"""Temporal node embedding extraction and GAT/LSTM alignment preview module.
"""
from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import torch

from satsim.lstm.lstm_model import LEOLSTMModel
from satsim.lstm.lstm_dataset import FeatureScaler, SequenceSample
from satsim.logging import get_logger

logger = get_logger(__name__)


class LSTMEmbedder:
    """Extracts 128-dimensional node temporal embeddings from the best trained LSTM model."""

    def __init__(
        self,
        model_path: Path | str = "artifacts/lstm/lstm_best.pt",
        scaler_path: Path | str = "artifacts/lstm/feature_scaler.pkl",
        device: torch.device | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_scaler = FeatureScaler.load(self.scaler_path)
        checkpoint = torch.load(self.model_path, weights_only=False)
        m_cfg = checkpoint.get("model_config", {})

        input_dim = m_cfg.get("input_dim", 24)
        hidden_dim = m_cfg.get("hidden_dim", 128)
        num_layers = m_cfg.get("num_layers", 2)

        self.model = LEOLSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def generate_embeddings(
        self,
        all_samples: List[SequenceSample],
        output_dir: Path | str = "artifacts/lstm/embeddings",
        index_csv_path: Path | str = "artifacts/lstm/embedding_index.csv",
        alignment_preview_path: Path | str = "artifacts/lstm/gat_lstm_alignment_preview.csv",
    ) -> None:
        """Extract 128-dim node temporal embeddings for all valid sequence windows grouped by scenario."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        index_csv_path = Path(index_csv_path)
        alignment_preview_path = Path(alignment_preview_path)

        index_rows = []
        preview_rows = []
        scenario_data: Dict[str, Dict[str, Any]] = {}

        print(f"\nGenerating LSTM temporal node embeddings for {len(all_samples):,d} sequence windows...")

        batch_size = 512
        with torch.no_grad():
            for idx in range(0, len(all_samples), batch_size):
                batch_samples = all_samples[idx : idx + batch_size]
                x_batch_np = np.stack([self.feature_scaler.transform(s.x) for s in batch_samples])
                x_tensor = torch.tensor(x_batch_np, dtype=torch.float32, device=self.device)

                _, temp_embeddings = self.model(x_tensor)
                temp_embeddings_np = temp_embeddings.cpu().numpy()

                for b_idx, sample in enumerate(batch_samples):
                    emb_vec_np = temp_embeddings_np[b_idx]

                    # Assert relationship: input_end_timestep = target_timestep - 1
                    assert sample.input_end_t == sample.target_t - 1, (
                        f"Alignment Error! input_end_t ({sample.input_end_t}) != target_t - 1 ({sample.target_t - 1})"
                    )

                    scen = sample.scenario
                    emb_filename = f"embedding_{scen}.pt"

                    if scen not in scenario_data:
                        scenario_data[scen] = {
                            "seed": sample.seed,
                            "satellite_id": [],
                            "input_start_timestep": [],
                            "input_end_timestep": [],
                            "target_timestep": [],
                            "temporal_embeddings": [],
                        }

                    scenario_data[scen]["satellite_id"].append(sample.satellite_id)
                    scenario_data[scen]["input_start_timestep"].append(sample.input_start_t)
                    scenario_data[scen]["input_end_timestep"].append(sample.input_end_t)
                    scenario_data[scen]["target_timestep"].append(sample.target_t)
                    scenario_data[scen]["temporal_embeddings"].append(emb_vec_np)

                    index_rows.append({
                        "scenario": scen,
                        "seed": sample.seed,
                        "satellite_id": sample.satellite_id,
                        "input_start_timestep": sample.input_start_t,
                        "input_end_timestep": sample.input_end_t,
                        "target_timestep": sample.target_t,
                        "embedding_file": emb_filename,
                        "embedding_dimension": emb_vec_np.shape[0],
                    })

                    # Add sample to GAT/LSTM alignment preview (first 1000 samples)
                    if len(preview_rows) < 1000:
                        preview_rows.append({
                            "scenario": sample.scenario,
                            "seed": sample.seed,
                            "satellite_id": sample.satellite_id,
                            "timestep": sample.input_end_t,
                            "gat_spatial_embedding_dim": 128,
                            "lstm_temporal_embedding_dim": 128,
                            "combined_representation_dim": 256,
                            "matched": True,
                        })

        # Save consolidated scenario embedding files (13 scenario files total)
        for scen, data_dict in scenario_data.items():
            scen_emb_path = output_dir / f"embedding_{scen}.pt"
            embeddings_mat = np.vstack(data_dict["temporal_embeddings"]).astype(np.float32)
            stacked_payload = {
                "scenario": scen,
                "seed": data_dict["seed"],
                "satellite_ids": np.array(data_dict["satellite_id"], dtype=np.int64),
                "input_start_timesteps": np.array(data_dict["input_start_timestep"], dtype=np.int64),
                "input_end_timesteps": np.array(data_dict["input_end_timestep"], dtype=np.int64),
                "target_timesteps": np.array(data_dict["target_timestep"], dtype=np.int64),
                "temporal_embeddings": embeddings_mat,  # shape [N, 128] numpy array
            }
            with open(scen_emb_path, "wb") as f:
                pickle.dump(stacked_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Write index CSV
        with open(index_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "scenario",
                    "seed",
                    "satellite_id",
                    "input_start_timestep",
                    "input_end_timestep",
                    "target_timestep",
                    "embedding_file",
                    "embedding_dimension",
                ],
            )
            writer.writeheader()
            writer.writerows(index_rows)

        # Write alignment preview CSV
        with open(alignment_preview_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "scenario",
                    "seed",
                    "satellite_id",
                    "timestep",
                    "gat_spatial_embedding_dim",
                    "lstm_temporal_embedding_dim",
                    "combined_representation_dim",
                    "matched",
                ],
            )
            writer.writeheader()
            writer.writerows(preview_rows)

        print(f"[OK] LSTM temporal embeddings saved: {len(scenario_data)} scenario files ({len(index_rows):,d} windows) written to {output_dir}")
        print(f"[OK] GAT/LSTM alignment preview written to: {alignment_preview_path}")
