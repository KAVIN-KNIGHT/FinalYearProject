"""Spatial representation diagnostic and visual evidence generator for the GAT model.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch_geometric.data import Data

from satsim.gat.gat_model import LEOGATModel
from satsim.gat.gat_dataset import FeatureScaler
from satsim.logging import get_logger

logger = get_logger(__name__)


class GATPlotter:
    """Generates quantitative loss curves and supporting spatial visual evidence for the GAT model."""

    def __init__(self, output_dir: Path | str = "artifacts/gat/corrected/plots") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    def plot_reconstruction_loss(self, history: List[Dict[str, Any]]) -> Path:
        """Plot 1: Quantitative GAT Spatial Representation Learning Loss Curve (Reconstruction MSE)."""
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        ax.plot(epochs, train_loss, label="Train Reconstruction Loss (MSE)", color="#1f77b4", linewidth=2.0)
        ax.plot(epochs, val_loss, label="Val Reconstruction Loss (MSE)", color="#ff7f0e", linewidth=2.0, linestyle="--")

        ax.set_title("GAT Spatial Representation Learning Loss (Non-Target Feature Reconstruction)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Reconstruction MSE Loss", fontsize=11)
        ax.legend(frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()

        out_path = self.output_dir / "training_validation_loss.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_topology_attention(
        self,
        model: LEOGATModel,
        scaler: FeatureScaler,
        snapshot_data: Data,
        device: torch.device,
        top_percentile: float = 75.0,
    ) -> Path:
        """Plot 2: GAT Attention Weights on LEO Satellite Network Topology (Top 25% Highest Attention Edges Filtered for Clarity)."""
        model.eval()
        d_scaled = scaler.transform(snapshot_data).to(device)

        edge_index_att, alpha = model.get_attention_weights(
            x=d_scaled.x,
            edge_index=d_scaled.edge_index,
            edge_attr=d_scaled.edge_attr,
        )

        edge_index_np = edge_index_att.cpu().numpy()
        # Mean attention weight across heads
        alpha_np = alpha.mean(dim=-1).cpu().numpy()

        num_nodes = d_scaled.x.shape[0]
        # Position nodes in a 2D circular/ring constellation layout
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
        pos_x = np.cos(angles)
        pos_y = np.sin(angles)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

        # Draw background full topology with light alpha
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            ax.plot([pos_x[src], pos_x[dst]], [pos_y[src], pos_y[dst]], color="#d3d3d3", alpha=0.25, linewidth=0.5, zorder=1)

        # Highlight top percentile highest-attention ISL edges
        threshold = np.percentile(alpha_np, top_percentile) if len(alpha_np) > 0 else 0.0
        max_alpha = float(np.max(alpha_np)) if len(alpha_np) > 0 else 1.0

        for i in range(edge_index_np.shape[1]):
            if alpha_np[i] >= threshold:
                src, dst = edge_index_np[0, i], edge_index_np[1, i]
                w = (alpha_np[i] - threshold) / max(1e-6, max_alpha - threshold)
                ax.plot(
                    [pos_x[src], pos_x[dst]],
                    [pos_y[src], pos_y[dst]],
                    color="#d62728",
                    alpha=float(0.4 + 0.6 * w),
                    linewidth=float(1.2 + 2.5 * w),
                    zorder=2,
                )

        # Draw satellite nodes
        ax.scatter(pos_x, pos_y, c="#1f77b4", s=60, edgecolors="black", linewidths=0.8, zorder=3, label="Satellite Node")

        ax.set_title("GAT Top 25% Spatial Attention Weights on Constellation ISLs", fontsize=13, fontweight="bold", pad=12)
        ax.set_axis_off()
        ax.legend(loc="upper right", frameon=True, facecolor="white")
        plt.tight_layout()

        out_path = self.output_dir / "gat_topology_attention.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


    def plot_spatial_embedding_pca(
        self,
        model: LEOGATModel,
        scaler: FeatureScaler,
        scenario_test_pairs: Dict[str, List[Data]],
        device: torch.device,
        samples_per_scenario: int = 5,
    ) -> Path:
        """Plot 3: Supporting Evidence — 2D PCA Visualization of GAT Spatial Node Embeddings."""
        model.eval()
        embeddings_list = []
        labels_list = []

        with torch.no_grad():
            for scen, pair_list in scenario_test_pairs.items():
                selected = pair_list[:min(samples_per_scenario, len(pair_list))]
                for raw_d in selected:
                    d_scaled = scaler.transform(raw_d).to(device)
                    b_vec = torch.zeros(d_scaled.x.shape[0], dtype=torch.long, device=device)
                    _, node_emb, _ = model(
                        x=d_scaled.x,
                        edge_index=d_scaled.edge_index,
                        edge_attr=d_scaled.edge_attr,
                        batch=b_vec,
                    )
                    embeddings_list.append(node_emb.cpu().numpy())
                    labels_list.extend([scen] * node_emb.shape[0])

        all_emb = np.vstack(embeddings_list)
        pca = PCA(n_components=2, random_state=42)
        emb_2d = pca.fit_transform(all_emb)

        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        unique_scens = list(scenario_test_pairs.keys())
        cmap = plt.get_cmap("tab20", len(unique_scens))

        for idx, scen in enumerate(unique_scens):
            mask = np.array(labels_list) == scen
            ax.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                label=scen,
                alpha=0.6,
                s=12,
                color=cmap(idx),
            )

        ax.set_title("Supporting Evidence: 2D PCA Projection of GAT Spatial Node Embeddings", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, frameon=True)
        plt.tight_layout()

        out_path = self.output_dir / "gat_embedding_visualization.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_embedding_similarity_heatmap(
        self,
        model: LEOGATModel,
        scaler: FeatureScaler,
        snapshot_data: Data,
        device: torch.device,
    ) -> Path:
        """Plot 4: Supporting Evidence — Pairwise Cosine Similarity Heatmap Across 100 Satellite Embeddings."""
        model.eval()
        with torch.no_grad():
            d_scaled = scaler.transform(snapshot_data).to(device)
            b_vec = torch.zeros(d_scaled.x.shape[0], dtype=torch.long, device=device)
            _, node_emb, _ = model(
                x=d_scaled.x,
                edge_index=d_scaled.edge_index,
                edge_attr=d_scaled.edge_attr,
                batch=b_vec,
            )

        emb_np = node_emb.cpu().numpy()  # [100, 128]
        norm = np.linalg.norm(emb_np, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)
        norm_emb = emb_np / norm
        sim_matrix = np.dot(norm_emb, norm_emb.T)  # [100, 100]

        fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
        im = ax.imshow(sim_matrix, cmap="viridis", vmin=-1.0, vmax=1.0)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Cosine Similarity", fontsize=11)

        ax.set_title("GAT Spatial Embedding Similarity Across Satellites", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Satellite ID", fontsize=11)
        ax.set_ylabel("Satellite ID", fontsize=11)
        plt.tight_layout()

        out_path = self.output_dir / "gat_embedding_similarity_heatmap.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_attention_distribution(
        self,
        model: LEOGATModel,
        scaler: FeatureScaler,
        snapshot_data: Data,
        device: torch.device,
    ) -> Path:
        """Plot 5: Supporting Evidence — Distribution of GAT Edge Attention Coefficients."""
        model.eval()
        d_scaled = scaler.transform(snapshot_data).to(device)

        _, alpha = model.get_attention_weights(
            x=d_scaled.x,
            edge_index=d_scaled.edge_index,
            edge_attr=d_scaled.edge_attr,
        )

        alpha_np = alpha.cpu().numpy().flatten()

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.hist(alpha_np, bins=40, color="#9467bd", edgecolor="black", alpha=0.75, density=True)

        ax.set_title("Distribution of Learned GAT Spatial Attention Coefficients", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Attention Coefficient (Alpha)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        plt.tight_layout()

        out_path = self.output_dir / "gat_attention_distribution.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
