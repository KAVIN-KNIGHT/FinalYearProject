"""Diagnostic plot generator for LSTM model training, evaluation, baselines, and temporal embeddings.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle
import torch

from satsim.lstm.lstm_dataset import SequenceSample
from satsim.logging import get_logger

logger = get_logger(__name__)


class LSTMPlotter:
    """Generates 8 diagnostic evaluation plots for the LSTM model."""

    def __init__(self, output_dir: Path | str = "artifacts/lstm/plots") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    def plot_training_validation_loss(self, history: List[Dict[str, Any]]) -> Path:
        """Plot 1: Epoch vs Training and Validation MSE Loss."""
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        ax.plot(epochs, train_loss, label="Training Loss (Standardized)", color="#1f77b4", linewidth=2.0)
        ax.plot(epochs, val_loss, label="Validation Loss (Standardized)", color="#ff7f0e", linewidth=2.0, linestyle="--")

        ax.set_title("LSTM Training vs Validation Loss Curve", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("MSE Loss", fontsize=11)
        ax.legend(frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()

        out_path = self.output_dir / "training_validation_loss.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_actual_vs_predicted(self, y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> Path:
        """Plot 2: Scatter plot of Actual vs Predicted congestion_score(t+1) on raw scale."""
        if len(y_true_raw) > 5000:
            idx = np.random.choice(len(y_true_raw), 5000, replace=False)
            y_t = y_true_raw[idx]
            y_p = y_pred_raw[idx]
        else:
            y_t, y_p = y_true_raw, y_pred_raw

        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        ax.scatter(y_t, y_p, alpha=0.3, color="#1f77b4", edgecolors="none", s=15, label="Sequence Predictions (t+1)")

        lim_min = min(y_t.min(), y_p.min())
        lim_max = max(y_t.max(), y_p.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#d62728", linestyle="--", linewidth=2.0, label="Ideal (y = x)")

        ax.set_title("LSTM: Predicted vs Actual Congestion Score (t+1)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Actual Congestion Score (t+1)", fontsize=11)
        ax.set_ylabel("Predicted Congestion Score (t+1)", fontsize=11)
        ax.legend(frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()

        out_path = self.output_dir / "actual_vs_predicted.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_baseline_comparison(
        self,
        mean_metrics: Dict[str, float],
        persistence_metrics: Dict[str, float],
        lstm_metrics: Dict[str, float],
    ) -> Path:
        """Plot 3: Comparison of Mean Baseline vs Persistence Baseline vs LSTM (RMSE & MAE)."""
        metrics = ["RMSE", "MAE"]
        mean_vals = [mean_metrics["rmse"], mean_metrics["mae"]]
        pers_vals = [persistence_metrics["rmse"], persistence_metrics["mae"]]
        lstm_vals = [lstm_metrics["rmse"], lstm_metrics["mae"]]

        x = np.arange(len(metrics))
        width = 0.25

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        bars1 = ax.bar(x - width, mean_vals, width, label="Mean Baseline", color="#7f7f7f", edgecolor="black")
        bars2 = ax.bar(x, pers_vals, width, label="Persistence Baseline", color="#ff7f0e", edgecolor="black")
        bars3 = ax.bar(x + width, lstm_vals, width, label="LSTM Model", color="#2ca02c", edgecolor="black")

        ax.set_title("Test Performance Comparison: Baselines vs LSTM", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Error (Raw Congestion Score Scale)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(frameon=True, facecolor="white", edgecolor="none")

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

        plt.tight_layout()

        out_path = self.output_dir / "baseline_comparison.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_prediction_error_distribution(self, y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> Path:
        """Plot 4: Histogram of residual errors (y_true - y_pred) on raw scale."""
        residuals = y_true_raw - y_pred_raw

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.hist(residuals, bins=50, color="#9467bd", edgecolor="black", alpha=0.75, density=True)
        ax.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.8, label="Zero Error")

        ax.set_title("LSTM Residual Error Distribution (t+1)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Residual Error (Actual - Predicted)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()

        out_path = self.output_dir / "prediction_error_distribution.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_scenario_performance(self, scenario_metrics: Dict[str, Dict[str, float]]) -> Path:
        """Plot 5: Test RMSE across all 13 scenarios on raw scale."""
        scenarios = list(scenario_metrics.keys())
        rmse_vals = [scenario_metrics[s]["rmse"] for s in scenarios]

        fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
        bars = ax.bar(scenarios, rmse_vals, color="#1f77b4", edgecolor="black", linewidth=0.8, alpha=0.85)

        for idx, s in enumerate(scenarios):
            if s in ["failures", "weather", "congestion_stress"]:
                bars[idx].set_color("#d62728")

        ax.set_title("LSTM Test RMSE (Raw Scale) Across All 13 Scenarios", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Scenario", fontsize=11)
        ax.set_ylabel("Test RMSE (Raw Scale)", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.tight_layout()

        out_path = self.output_dir / "scenario_performance.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_target_distribution(self, df: pd.DataFrame) -> Path:
        """Plot 6: Congestion score target distribution across all 13 scenarios."""
        fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
        scens = sorted(df["scenario"].unique().tolist())

        data_to_plot = [df[df["scenario"] == s]["congestion_score"].values for s in scens]
        ax.boxplot(data_to_plot, labels=scens, patch_artist=True, boxprops=dict(facecolor="#1f77b4", alpha=0.6))

        ax.set_title("Congestion Score Target Distribution Across 13 Scenarios", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Scenario", fontsize=11)
        ax.set_ylabel("Raw Congestion Score", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.tight_layout()

        out_path = self.output_dir / "target_distribution.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_temporal_prediction_example(
        self, test_samples: List[SequenceSample], y_true_all: np.ndarray, y_pred_all: np.ndarray
    ) -> Path:
        """Plot 7: Time series comparison of actual vs predicted congestion over time for a representative satellite."""
        # Pick first satellite from first scenario
        sample_scen = test_samples[0].scenario
        sample_sat = test_samples[0].satellite_id

        indices = [
            idx for idx, s in enumerate(test_samples)
            if s.scenario == sample_scen and s.satellite_id == sample_sat
        ]

        t_targets = [test_samples[idx].target_t for idx in indices]
        y_actual = y_true_all[indices]
        y_predict = y_pred_all[indices]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        ax.plot(t_targets, y_actual, label="Actual Congestion (t+1)", color="#1f77b4", linewidth=2.0)
        ax.plot(t_targets, y_predict, label="Predicted Congestion (t+1)", color="#d62728", linestyle="--", linewidth=2.0)

        ax.set_title(f"LSTM Temporal Congestion Prediction Over Time ({sample_scen}, Sat {sample_sat})", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Target Timestep (t+1)", fontsize=11)
        ax.set_ylabel("Raw Congestion Score", fontsize=11)
        ax.legend(frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()

        out_path = self.output_dir / "temporal_prediction_example.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_embedding_pca(
        self,
        embeddings_dir: Path | str = "artifacts/lstm/embeddings",
        index_csv_path: Path | str = "artifacts/lstm/embedding_index.csv",
        samples_per_scenario: int = 50,
    ) -> Path:
        """Plot 8: 2D PCA visualization of LSTM temporal node embeddings colored by scenario."""
        embeddings_dir = Path(embeddings_dir)
        index_df = pd.read_csv(index_csv_path)

        emb_list = []
        labels = []

        for scen in index_df["scenario"].unique():
            f_path = embeddings_dir / f"embedding_{scen}.pt"
            if f_path.exists():
                try:
                    with open(f_path, "rb") as f:
                        payload = pickle.load(f)
                    embs = payload["temporal_embeddings"][:samples_per_scenario]
                    emb_list.append(embs)
                    labels.extend([scen] * len(embs))
                except Exception as e:
                    logger.warning("Failed loading embedding file", file=str(f_path), error=str(e))

        if not emb_list:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No embeddings available", ha="center")
            out_path = self.output_dir / "lstm_embedding_pca.png"
            fig.savefig(out_path)
            plt.close(fig)
            return out_path

        all_emb = np.vstack(emb_list)
        pca = PCA(n_components=2, random_state=42)
        emb_2d = pca.fit_transform(all_emb)

        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        unique_scens = sorted(list(set(labels)))
        cmap = plt.get_cmap("tab20", len(unique_scens))

        for idx, scen in enumerate(unique_scens):
            mask = np.array(labels) == scen
            ax.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                label=scen,
                alpha=0.6,
                s=15,
                color=cmap(idx),
            )

        ax.set_title("2D PCA Visualization of LSTM Temporal Node Embeddings", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, frameon=True)
        plt.tight_layout()

        out_path = self.output_dir / "lstm_embedding_pca.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
