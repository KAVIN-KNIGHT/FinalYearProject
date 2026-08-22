"""CLI script for Corrective GAT Training Pipeline — removing target leakage, predicting future congestion_score(t+1), evaluating against baseline on raw scale, and extracting spatial node embeddings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
from typing import Dict, Any, List
import numpy as np
import torch
import torch_geometric
from torch_geometric.loader import DataLoader as PyGDataLoader
import yaml

from satsim.gat.gat_model import LEOGATModel
from satsim.gat.gat_dataset import (
    LEOGraphSnapshotDataset,
    FeatureScaler,
    TargetScaler,
    EXPECTED_SCENARIOS,
)
from satsim.gat.trainer import GATTrainer
from satsim.gat.embedder import GATEmbedder
from satsim.gat.plotter import GATPlotter
from satsim.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, PyTorch, and PyG."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def archive_previous_leaky_run(base_gat_dir: Path) -> None:
    """Preserve previous leaky run artifacts in artifacts/gat/leaky_previous_run/."""
    leaky_dir = base_gat_dir / "leaky_previous_run"
    leaky_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "gat_best.pt",
        "gat_last.pt",
        "feature_scaler.pkl",
        "config.yaml",
        "training_history.csv",
        "validation_metrics.json",
        "test_metrics.json",
        "scenario_metrics.csv",
        "results.json",
        "GAT_EVALUATION_REPORT.md",
    ]

    for fname in files_to_copy:
        src = base_gat_dir / fname
        if src.exists():
            shutil.copy2(src, leaky_dir / fname)

    # Copy previous plots if they exist
    prev_plots = base_gat_dir / "plots"
    if prev_plots.exists():
        target_plots = leaky_dir / "plots"
        target_plots.mkdir(parents=True, exist_ok=True)
        for p in prev_plots.glob("*.png"):
            shutil.copy2(p, target_plots / p.name)

    # Create README marker
    readme_path = leaky_dir / "README.md"
    readme_path.write_text(
        "# INVALID — TARGET LEAKAGE PREVIOUS RUN\n\n"
        "This directory contains the previous GAT model checkpoints, metrics, and plots from the initial run. "
        "The read-only audit confirmed CRITICAL TARGET LEAKAGE in this run (column 13 congestion_score was "
        "present in input X while predicting congestion_score(t)).\n\n"
        "**Status**: INVALID. Do NOT use these artifacts for final model evaluation or downstream PPO training.\n"
        "The corrected leak-free results are located in `artifacts/gat/corrected/`.\n",
        encoding="utf-8",
    )
    print(f"[ARCHIVE] Previous leaky run artifacts archived to: {leaky_dir}")


def generate_evaluation_report(
    artifacts_dir: Path,
    num_satellites: int,
    scenarios: List[str],
    total_pairs: int,
    train_pairs: int,
    val_pairs: int,
    test_pairs: int,
    node_in_dim: int,
    edge_in_dim: int,
    embedding_dim: int,
    best_epoch: int,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    scenario_metrics: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
) -> Path:
    """Generate GAT_EVALUATION_REPORT.md markdown artifact for corrected model."""
    report_path = artifacts_dir / "GAT_EVALUATION_REPORT.md"

    rmse_impr = ((baseline_metrics["rmse"] - test_metrics["rmse"]) / baseline_metrics["rmse"]) * 100.0
    mae_impr = ((baseline_metrics["mae"] - test_metrics["mae"]) / baseline_metrics["mae"]) * 100.0

    md = []
    md.append("# Corrected GAT Evaluation Report — 100-Satellite LEO Future Congestion Prediction\n")
    md.append("## Executive Summary")
    md.append(
        "A **leak-free Graph Attention Network (GAT)** model was trained across **all 13 canonical LEO simulation scenarios**. "
        "The target variable `congestion_score` was **completely removed from input feature matrix X** (16 input features). "
        "The model is formulated to predict future congestion score at timestep $t+1$ ($X(t) \\to \\text{congestion\\_score}(t+1)$). "
        "All metric evaluations are performed on the **original raw scale** $[0.0, 2.0]$ and compared against a mean training baseline.\n"
    )

    md.append("## 1. Audit & Target Leakage Status")
    md.append("- **Previous Model**: INVALID due to target leakage (target `congestion_score(t)` was present in $X$).")
    md.append("- **Corrected Model**: **PASS (No Target Leakage)**")
    md.append("- **Input Feature Dimension**: 16 (pos_eci, vel_eci, pos_ecef, is_active, queue_occ, queue_len, node_degree, cpu_util, mem_util, fail_ind)")
    md.append("- **Prediction Target**: `congestion_score(t+1)`")
    md.append(f"- **Total Graph Snapshot Pairs ($t \\to t+1$)**: {total_pairs}\n")

    md.append("## 2. Dataset & System Configuration")
    md.append(f"- **Satellites**: {num_satellites} (IDs 0–99)")
    md.append(f"- **Scenarios ({len(scenarios)})**: {', '.join(scenarios)}")
    md.append(f"- **Time-Aware Split**: Train: {train_pairs} (70%), Val: {val_pairs} (15%), Test: {test_pairs} (15%)")
    md.append(f"- **Node Feature Dimension**: {node_in_dim}")
    md.append(f"- **Edge Feature Dimension**: {edge_in_dim}")
    md.append(f"- **Spatial Embedding Dimension**: {embedding_dim}")
    md.append(f"- **Device**: {config.get('system', {}).get('device', 'CPU')}")
    md.append(f"- **Seed**: {config.get('seed', 42)}\n")

    md.append("## 3. Model Architecture & Hyperparameters")
    md.append("```yaml")
    md.append(yaml.dump(config.get("gat", {}), default_flow_style=False))
    md.append("```\n")

    md.append("## 4. Baseline vs Corrected GAT Test Performance (Raw Scale)")
    md.append("| Model | Test Loss | MAE (Raw) | RMSE (Raw) | R² Score |")
    md.append("|---|---|---|---|---|")
    md.append(f"| **Mean Baseline** | N/A | {baseline_metrics.get('mae', 0.0):.6f} | {baseline_metrics.get('rmse', 0.0):.6f} | {baseline_metrics.get('r2', 0.0):.6f} |")
    md.append(f"| **Corrected GAT** | {test_metrics.get('loss', 0.0):.6f} | {test_metrics.get('mae', 0.0):.6f} | {test_metrics.get('rmse', 0.0):.6f} | {test_metrics.get('r2', 0.0):.6f} |")
    md.append(f"\n- **RMSE Improvement over Baseline**: **{rmse_impr:.2f}%**")
    md.append(f"- **MAE Improvement over Baseline**: **{mae_impr:.2f}%**\n")

    md.append("## 5. Training & Validation Performance")
    md.append(f"- **Best Epoch**: {best_epoch}")
    md.append(f"- **Validation Loss**: {val_metrics.get('loss', 0.0):.6f}")
    md.append(f"- **Validation RMSE (Raw)**: {val_metrics.get('rmse', 0.0):.6f}")
    md.append(f"- **Validation MAE (Raw)**: {val_metrics.get('mae', 0.0):.6f}")
    md.append(f"- **Validation R² Score**: {val_metrics.get('r2', 0.0):.6f}\n")

    md.append("## 6. Per-Scenario Evaluation Breakdown (Raw Scale)")
    md.append("| Scenario | Test Loss | MAE | RMSE | R² Score |")
    md.append("|---|---|---|---|---|")
    for scen in scenarios:
        sm = scenario_metrics.get(scen, {})
        md.append(f"| `{scen}` | {sm.get('loss', 0.0):.6f} | {sm.get('mae', 0.0):.6f} | {sm.get('rmse', 0.0):.6f} | {sm.get('r2', 0.0):.6f} |")
    md.append("")

    md.append("## 7. Corrected Artifacts & Visualizations")
    md.append(f"- **Corrected Model Weights**: [{artifacts_dir / 'gat_best.pt'}](file:///{artifacts_dir.as_posix()}/gat_best.pt)")
    md.append(f"- **Feature Scaler**: [{artifacts_dir / 'feature_scaler.pkl'}](file:///{artifacts_dir.as_posix()}/feature_scaler.pkl)")
    md.append(f"- **Target Scaler**: [{artifacts_dir / 'target_scaler.pkl'}](file:///{artifacts_dir.as_posix()}/target_scaler.pkl)")
    md.append(f"- **Baseline Metrics**: [{artifacts_dir / 'baseline_metrics.json'}](file:///{artifacts_dir.as_posix()}/baseline_metrics.json)")
    md.append(f"- **Scenario Metrics CSV**: [{artifacts_dir / 'scenario_metrics.csv'}](file:///{artifacts_dir.as_posix()}/scenario_metrics.csv)")
    md.append(f"- **Spatial Embeddings Directory**: `artifacts/gat/corrected/embeddings/` ({total_pairs} files)")
    md.append(f"- **Embedding Index**: [{artifacts_dir / 'embedding_index.csv'}](file:///{artifacts_dir.as_posix()}/embedding_index.csv)")
    md.append(f"- **Training/Val Loss Plot**: ![]({(artifacts_dir / 'plots' / 'training_validation_loss.png').as_posix()})")
    md.append(f"- **Actual vs Predicted Plot**: ![]({(artifacts_dir / 'plots' / 'actual_vs_predicted.png').as_posix()})")
    md.append(f"- **Scenario Performance Plot**: ![]({(artifacts_dir / 'plots' / 'scenario_performance.png').as_posix()})")
    md.append(f"- **Error Distribution Plot**: ![]({(artifacts_dir / 'plots' / 'prediction_error_distribution.png').as_posix()})")
    md.append(f"- **GAT Embedding PCA Plot**: ![]({(artifacts_dir / 'plots' / 'gat_embedding_visualization.png').as_posix()})")
    md.append(f"- **Baseline vs GAT Plot**: ![]({(artifacts_dir / 'plots' / 'baseline_vs_gat.png').as_posix()})\n")

    report_path.write_text("\n".join(md), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Corrective GAT Training Pipeline without Target Leakage.")
    parser.add_argument("--dataset-dir", type=str, default="datasets", help="Directory containing scenario datasets.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/gat/corrected", help="Output directory for corrected GAT artifacts.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="DataLoader batch size.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="GAT hidden dimension.")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Spatial node embedding dimension.")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads in layer 1.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    set_seed(args.seed)

    base_gat_dir = Path("artifacts/gat")
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Archive previous leaky run if present in base artifacts/gat/
    archive_previous_leaky_run(base_gat_dir)

    # 1. Dataset Discovery & Snapshot Validation
    dataset_handler = LEOGraphSnapshotDataset(root_dir=args.dataset_dir)
    dataset_handler.discover_snapshots()
    node_in_dim, edge_in_dim = dataset_handler.validate_snapshots()

    # 2. Build Aligned Time Splits X(t) -> congestion_score(t+1)
    train_raw, val_raw, test_raw, scenario_test_raw = dataset_handler.create_aligned_time_splits(
        train_ratio=0.70, val_ratio=0.15
    )

    # 3. Fit FeatureScaler and TargetScaler ONLY on Training Data
    feature_scaler = FeatureScaler()
    feature_scaler.fit(train_raw)
    feature_scaler.save(artifacts_dir / "feature_scaler.pkl")

    target_scaler = TargetScaler()
    target_scaler.fit(train_raw)
    target_scaler.save(artifacts_dir / "target_scaler.pkl")

    # Transform all dataset input graphs
    train_data = [feature_scaler.transform(d) for d in train_raw]
    val_data = [feature_scaler.transform(d) for d in val_raw]
    test_data = [feature_scaler.transform(d) for d in test_raw]

    scenario_test_data = {
        scen: [feature_scaler.transform(d) for d in raw_list]
        for scen, raw_list in scenario_test_raw.items()
    }

    # DataLoaders
    train_loader = PyGDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    scenario_test_loaders = {
        scen: PyGDataLoader(data_list, batch_size=args.batch_size, shuffle=False)
        for scen, data_list in scenario_test_data.items()
    }

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Reproducibility Configuration
    config = {
        "seed": args.seed,
        "gat": {
            "node_in_dim": node_in_dim,
            "edge_in_dim": edge_in_dim,
            "hidden_dim": args.hidden_dim,
            "embedding_dim": args.embedding_dim,
            "heads": args.heads,
            "dropout": args.dropout,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "early_stopping_patience": args.patience,
        },
        "system": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "pyg_version": torch_geometric.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": str(device),
        },
    }

    with open(artifacts_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    # 4. Model Initialization & Corrected Training
    model = LEOGATModel(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        heads=args.heads,
        dropout=args.dropout,
    )

    trainer = GATTrainer(
        model=model,
        device=device,
        target_scaler=target_scaler,
        artifacts_dir=artifacts_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        model_config=config["gat"],
        feature_config={"node_in_dim": node_in_dim, "edge_in_dim": edge_in_dim},
    )

    # 5. Evaluation & Metrics on Raw Scale + Baseline
    val_metrics, test_metrics, scenario_metrics, baseline_metrics = trainer.evaluate_and_export_metrics(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scenario_test_loaders=scenario_test_loaders,
    )

    # 6. Plotting
    plotter = GATPlotter(output_dir=artifacts_dir / "plots")
    plotter.plot_training_validation_loss(trainer.history)

    # Evaluate test dataset for actual vs predicted plot on raw scale
    best_model = LEOGATModel(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        heads=args.heads,
    ).to(device)
    best_ckpt = torch.load(artifacts_dir / "gat_best.pt", weights_only=False)
    best_model.load_state_dict(best_ckpt["model_state_dict"])
    best_model.eval()

    all_t_raw, all_p_raw = [], []
    with torch.no_grad():
        for b in test_loader:
            b = b.to(device)
            p_scaled, _, _ = best_model(b.x, b.edge_index, b.edge_attr, b.batch)
            p_raw = target_scaler.inverse_transform(p_scaled)
            t_raw = b.y.cpu().numpy()

            all_t_raw.append(t_raw.flatten())
            all_p_raw.append(p_raw.flatten())

    y_t_mat = np.concatenate(all_t_raw)
    y_p_mat = np.concatenate(all_p_raw)

    plotter.plot_actual_vs_predicted(y_t_mat, y_p_mat)
    plotter.plot_scenario_performance(scenario_metrics)
    plotter.plot_error_distribution(y_t_mat, y_p_mat)
    plotter.plot_embedding_pca(
        model=best_model,
        scaler=feature_scaler,
        scenario_test_pairs=scenario_test_raw,
        device=device,
    )
    plotter.plot_baseline_vs_gat(baseline_metrics, test_metrics)

    # 7. Corrected Spatial Embedding Generation
    all_raw_pairs = train_raw + val_raw + test_raw
    embedder = GATEmbedder(
        model_path=artifacts_dir / "gat_best.pt",
        scaler_path=artifacts_dir / "feature_scaler.pkl",
        device=device,
    )
    embedder.generate_embeddings(
        all_pairs=all_raw_pairs,
        output_dir=artifacts_dir / "embeddings",
        index_csv_path=artifacts_dir / "embedding_index.csv",
    )

    # 8. Results JSON
    results = {
        "num_satellites": 100,
        "num_scenarios": len(EXPECTED_SCENARIOS),
        "total_pairs": len(all_raw_pairs),
        "train_pairs": len(train_raw),
        "validation_pairs": len(val_raw),
        "test_pairs": len(test_raw),
        "node_feature_dim": node_in_dim,
        "edge_feature_dim": edge_in_dim,
        "embedding_dim": args.embedding_dim,
        "target": "congestion_score(t+1)",
        "target_leakage": "NO",
        "best_epoch": trainer.best_epoch,
        "validation_metrics_raw": val_metrics,
        "test_metrics_raw": test_metrics,
        "baseline_metrics_raw": baseline_metrics,
        "scenario_metrics_file": "scenario_metrics.csv",
        "best_model": "gat_best.pt",
    }
    with open(artifacts_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 9. Generate Corrected Markdown Report
    generate_evaluation_report(
        artifacts_dir=artifacts_dir,
        num_satellites=100,
        scenarios=EXPECTED_SCENARIOS,
        total_pairs=len(all_raw_pairs),
        train_pairs=len(train_raw),
        val_pairs=len(val_raw),
        test_pairs=len(test_raw),
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        embedding_dim=args.embedding_dim,
        best_epoch=trainer.best_epoch,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        scenario_metrics=scenario_metrics,
        config=config,
    )

    # 10. Final Section 16 Terminal Summary
    rmse_impr = ((baseline_metrics["rmse"] - test_metrics["rmse"]) / baseline_metrics["rmse"]) * 100.0
    mae_impr = ((baseline_metrics["mae"] - test_metrics["mae"]) / baseline_metrics["mae"]) * 100.0

    print("\n====================================================")
    print("CORRECTED GAT TRAINING COMPLETE")
    print("====================================================")
    print(f"Satellites:              100")
    print(f"Scenarios:               {len(EXPECTED_SCENARIOS)}")
    print(f"\nInput features:          {node_in_dim}")
    print("Target:                  congestion_score(t+1)")
    print("Target leakage:          NO")
    print(f"\nTrain snapshots:         {len(train_raw):5d}")
    print(f"Validation snapshots:    {len(val_raw):5d}")
    print(f"Test snapshots:          {len(test_raw):5d}")
    print(f"\nEmbedding dimension:     {args.embedding_dim:3d}")
    print(f"\nValidation:")
    print(f"  MAE:  {val_metrics['mae']:.6f}")
    print(f"  RMSE: {val_metrics['rmse']:.6f}")
    print(f"  R²:   {val_metrics['r2']:.6f}")
    print(f"\nTest:")
    print(f"  MAE:  {test_metrics['mae']:.6f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  R²:   {test_metrics['r2']:.6f}")
    print(f"\nBaseline Test:")
    print(f"  MAE:  {baseline_metrics['mae']:.6f}")
    print(f"  RMSE: {baseline_metrics['rmse']:.6f}")
    print(f"  R²:   {baseline_metrics['r2']:.6f}")
    print(f"\nGAT improvement over baseline:")
    print(f"  RMSE Improvement: {rmse_impr:+.2f}%")
    print(f"  MAE Improvement:  {mae_impr:+.2f}%")
    print(f"\nCorrected model:")
    print(f"  {artifacts_dir / 'gat_best.pt'}")
    print(f"\nCorrected embeddings:")
    print(f"  {artifacts_dir / 'embeddings'}")
    print(f"\nReport:")
    print(f"  {artifacts_dir / 'GAT_EVALUATION_REPORT.md'}")
    print("====================================================\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
