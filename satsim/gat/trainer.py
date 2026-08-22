"""GAT model self-supervised spatial representation training, evaluation, early stopping, and metric export module.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader as PyGDataLoader

from satsim.gat.gat_model import LEOGATModel
from satsim.gat.gat_dataset import FeatureScaler
from satsim.logging import get_logger

logger = get_logger(__name__)


FEATURE_NAMES_8 = [
    "pos_eci_x", "pos_eci_y", "pos_eci_z",
    "vel_eci_x", "vel_eci_y", "vel_eci_z",
    "buffer_utilization", "degree"
]


def compute_reconstruction_metrics(x_true: np.ndarray, x_pred: np.ndarray) -> Dict[str, float]:
    """Compute MSE, MAE, and per-feature MAE for 8 non-target node feature reconstruction.

    Args:
        x_true: True non-target node features [N, 8].
        x_pred: Reconstructed non-target node features [N, 8].

    Returns:
        Dict containing 'reconstruction_mse', 'reconstruction_mae', and per-feature metrics.
    """
    mse = float(np.mean((x_true - x_pred) ** 2))
    mae = float(np.mean(np.abs(x_true - x_pred)))
    per_feat_mae = np.mean(np.abs(x_true - x_pred), axis=0)

    res = {
        "reconstruction_mse": mse,
        "reconstruction_mae": mae,
    }
    for i, feat_name in enumerate(FEATURE_NAMES_8[: x_true.shape[1]]):
        res[f"mae_{feat_name}"] = float(per_feat_mae[i])

    return res


class GATTrainer:
    """Manages self-supervised GAT spatial representation training, evaluation, early stopping, and metric exports."""

    def __init__(
        self,
        model: LEOGATModel,
        device: torch.device,
        artifacts_dir: Path | str = "artifacts/gat/spatial",
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        early_stopping_patience: int = 7,
    ) -> None:

        self.model = model.to(device)
        self.device = device
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)
        self.criterion = nn.MSELoss()

        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

        self.history: List[Dict[str, float | int]] = []

    def train_epoch(self, train_loader: PyGDataLoader) -> float:
        """Run one training epoch predicting reconstructed non-target node features."""
        self.model.train()
        total_loss = 0.0
        total_nodes = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            reconstructed_x, _, _ = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
            )

            # Self-supervised spatial reconstruction loss against standardized non-target features batch.x
            loss = self.criterion(reconstructed_x, batch.x)
            loss.backward()
            self.optimizer.step()

            n_nodes = batch.x.shape[0]
            total_loss += float(loss.item()) * n_nodes
            total_nodes += n_nodes

        return total_loss / max(1, total_nodes)

    def evaluate_reconstruction(self, loader: PyGDataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluate self-supervised non-target node feature reconstruction MSE and MAE."""
        self.model.eval()
        total_loss = 0.0
        total_nodes = 0
        all_true = []
        all_pred = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                reconstructed_x, _, _ = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                )

                loss = self.criterion(reconstructed_x, batch.x)

                n_nodes = batch.x.shape[0]
                total_loss += float(loss.item()) * n_nodes
                total_nodes += n_nodes

                all_true.append(batch.x.cpu().numpy())
                all_pred.append(reconstructed_x.cpu().numpy())

        avg_loss = total_loss / max(1, total_nodes)
        x_true_mat = np.vstack(all_true)
        x_pred_mat = np.vstack(all_pred)

        metrics = compute_reconstruction_metrics(x_true_mat, x_pred_mat)
        return avg_loss, metrics

    def fit(
        self,
        train_loader: PyGDataLoader,
        val_loader: PyGDataLoader,
        epochs: int = 50,
        model_config: Dict[str, Any] | None = None,
        feature_config: Dict[str, Any] | None = None,
    ) -> None:
        """Run self-supervised training loop with early stopping and save best checkpoint."""
        print(f"Starting GAT Spatial Representation Learning training on device: {self.device}")
        print("-" * 75)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate_reconstruction(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)
            elapsed_s = time.time() - t0

            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_reconstruction_mse": val_metrics["reconstruction_mse"],
                "val_reconstruction_mae": val_metrics["reconstruction_mae"],
                "lr": current_lr,
                "duration_s": elapsed_s,
            })

            print(
                f"Epoch {epoch:2d}/{epochs:2d} | "
                f"Train Loss (MSE): {train_loss:.6f} | "
                f"Val Loss (MSE): {val_loss:.6f} | "
                f"Val MAE: {val_metrics['reconstruction_mae']:.6f} | "
                f"LR: {current_lr:.6f} | Time: {elapsed_s:.2f}s"
            )

            self.save_checkpoint(
                filepath=self.artifacts_dir / "gat_last.pt",
                epoch=epoch,
                val_loss=val_loss,
                model_config=model_config,
                feature_config=feature_config,
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0

                self.save_checkpoint(
                    filepath=self.artifacts_dir / "gat_best.pt",
                    epoch=epoch,
                    val_loss=val_loss,
                    model_config=model_config,
                    feature_config=feature_config,
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\n[Early Stopping] No validation improvement for {self.early_stopping_patience} epochs. Stopping at epoch {epoch}.")
                    break

        history_csv = self.artifacts_dir / "training_history.csv"
        with open(history_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)

    def save_checkpoint(
        self,
        filepath: Path,
        epoch: int,
        val_loss: float,
        model_config: Dict[str, Any] | None = None,
        feature_config: Dict[str, Any] | None = None,
    ) -> None:
        """Save checkpoint dictionary."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_metric": val_loss,
            "model_config": model_config or {},
            "feature_config": feature_config or {},
        }
        torch.save(checkpoint, filepath)

    def evaluate_and_export_metrics(
        self,
        train_loader: PyGDataLoader,
        val_loader: PyGDataLoader,
        test_loader: PyGDataLoader,
        scenario_test_loaders: Dict[str, PyGDataLoader],
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
        """Load best model checkpoint and evaluate quantitative spatial reconstruction metrics."""
        best_ckpt = torch.load(self.artifacts_dir / "gat_best.pt", weights_only=False)
        self.model.load_state_dict(best_ckpt["model_state_dict"])
        self.model.eval()

        val_loss, val_metrics = self.evaluate_reconstruction(val_loader)
        test_loss, test_metrics = self.evaluate_reconstruction(test_loader)

        # Save quantitative spatial reconstruction metrics JSON files
        with open(self.artifacts_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"loss": val_loss, **val_metrics}, f, indent=2)

        with open(self.artifacts_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"loss": test_loss, **test_metrics}, f, indent=2)

        # Export per-feature MAE CSV
        per_feat_rows = []
        for feat_name in FEATURE_NAMES_8:
            k = f"mae_{feat_name}"
            per_feat_rows.append({
                "feature_name": feat_name,
                "val_mae": val_metrics.get(k, 0.0),
                "test_mae": test_metrics.get(k, 0.0),
            })
        with open(self.artifacts_dir / "per_feature_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["feature_name", "val_mae", "test_mae"])
            writer.writeheader()
            writer.writerows(per_feat_rows)

        # Per-scenario evaluation
        scenario_results: Dict[str, Dict[str, float]] = {}
        scen_csv_rows = []

        for scen, loader in scenario_test_loaders.items():
            s_loss, s_metrics = self.evaluate_reconstruction(loader)
            scenario_results[scen] = {"loss": s_loss, **s_metrics}
            scen_csv_rows.append({
                "scenario": scen,
                "reconstruction_loss": s_loss,
                "reconstruction_MSE": s_metrics["reconstruction_mse"],
                "reconstruction_MAE": s_metrics["reconstruction_mae"],
            })

        scen_csv_path = self.artifacts_dir / "scenario_metrics.csv"
        with open(scen_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["scenario", "reconstruction_loss", "reconstruction_MSE", "reconstruction_MAE"])
            writer.writeheader()
            writer.writerows(scen_csv_rows)

        return val_metrics, test_metrics, scenario_results

    def diagnose_train_eval_loss_gap(self, train_loader: PyGDataLoader, val_loader: PyGDataLoader) -> Dict[str, Any]:
        """Empirically diagnose the train vs val loss gap by running train set under eval() mode vs train() mode."""
        self.model.eval()
        train_eval_loss, train_eval_metrics = self.evaluate_reconstruction(train_loader)
        val_eval_loss, val_eval_metrics = self.evaluate_reconstruction(val_loader)

        self.model.train()
        train_train_loss = self.train_epoch(train_loader)

        diagnosis = {
            "train_loss_train_mode": train_train_loss,
            "train_loss_eval_mode": train_eval_loss,
            "val_loss_eval_mode": val_eval_loss,
            "dropout_effect_ratio": train_train_loss / max(1e-8, train_eval_loss),
            "train_val_eval_gap": abs(train_eval_loss - val_eval_loss),
        }
        print("\n============================================================")
        print("EMPIRICAL TRAIN/VAL LOSS GAP DIAGNOSTIC REPORT")
        print("============================================================")
        print(f"Train Loss (train() mode, with dropout): {train_train_loss:.6f}")
        print(f"Train Loss (eval() mode, no dropout):   {train_eval_loss:.6f}")
        print(f"Val Loss   (eval() mode, no dropout):   {val_eval_loss:.6f}")
        print(f"Dropout Impact Ratio (train/eval):      {diagnosis['dropout_effect_ratio']:.2f}x")
        print(f"True Train/Val Gap in eval() mode:      {diagnosis['train_val_eval_gap']:.6f}")
        print("============================================================\n")

        return diagnosis

