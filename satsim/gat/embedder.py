"""Spatial node embedding extraction module for downstream LSTM/PPO integration.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch_geometric.data import Data

from satsim.gat.gat_model import LEOGATModel
from satsim.gat.gat_dataset import FeatureScaler
from satsim.logging import get_logger

logger = get_logger(__name__)


class GATEmbedder:
    """Extracts 100x128 spatial node embeddings from trained self-supervised GAT checkpoint."""

    def __init__(
        self,
        model_path: Path | str = "artifacts/gat/corrected/gat_best.pt",
        scaler_path: Path | str = "artifacts/gat/corrected/feature_scaler.pkl",
        device: torch.device | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = FeatureScaler.load(self.scaler_path)
        checkpoint = torch.load(self.model_path, weights_only=False)
        m_cfg = checkpoint.get("model_config", {})

        node_in_dim = m_cfg.get("node_in_dim", 8)
        edge_in_dim = m_cfg.get("edge_in_dim", 4)
        hidden_dim = m_cfg.get("hidden_dim", 128)
        embedding_dim = m_cfg.get("embedding_dim", 128)
        heads = m_cfg.get("heads", 4)

        self.model = LEOGATModel(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            heads=heads,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def generate_embeddings(
        self,
        all_snapshots: List[Data],
        output_dir: Path | str = "artifacts/gat/corrected/embeddings",
        index_csv_path: Path | str = "artifacts/gat/corrected/embedding_index.csv",
    ) -> None:
        """Run GAT inference on all graph snapshots and export spatial node embedding files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        index_csv_path = Path(index_csv_path)

        index_rows = []
        print(f"\nGenerating GAT spatial node embeddings for {len(all_snapshots)} snapshots...")

        with torch.no_grad():
            for idx, raw_data in enumerate(all_snapshots):
                scaled_data = self.scaler.transform(raw_data).to(self.device)

                batch_vec = torch.zeros(scaled_data.x.shape[0], dtype=torch.long, device=self.device)
                _, node_embeddings, _ = self.model(
                    x=scaled_data.x,
                    edge_index=scaled_data.edge_index,
                    edge_attr=scaled_data.edge_attr,
                    batch=batch_vec,
                )

                # Validation Assertions
                num_nodes, emb_dim = node_embeddings.shape
                assert num_nodes == 100, f"Expected exactly 100 satellites, got {num_nodes}"
                assert emb_dim == 128, f"Expected 128 embedding dimensions, got {emb_dim}"
                assert not torch.isnan(node_embeddings).any(), "Embedding tensor contains NaNs!"
                assert not torch.isinf(node_embeddings).any(), "Embedding tensor contains Infs!"

                scen = getattr(raw_data, "scenario", "unknown")
                t_step = getattr(raw_data, "timestep", idx)

                emb_filename = f"embedding_{idx:06d}.pt"
                emb_path = output_dir / emb_filename

                payload = {
                    "scenario": scen,
                    "seed": 42,
                    "timestep": t_step,
                    "satellite_ids": list(range(num_nodes)),
                    "node_embeddings": node_embeddings.cpu(),
                }
                torch.save(payload, emb_path)

                index_rows.append({
                    "scenario": scen,
                    "seed": 42,
                    "timestep": t_step,
                    "embedding_file": emb_filename,
                    "num_nodes": num_nodes,
                    "embedding_dimension": emb_dim,
                })

        with open(index_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "scenario",
                    "seed",
                    "timestep",
                    "embedding_file",
                    "num_nodes",
                    "embedding_dimension",
                ],
            )
            writer.writeheader()
            writer.writerows(index_rows)

        print(f"[OK] Spatial embeddings generated: {len(index_rows)} files saved to {output_dir}")
