import os
import sys
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add workspace directory to path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from satsim.gat.gat_model import LEOGATModel
from satsim.gat.gat_dataset import LEOGraphSnapshotDataset, FeatureScaler
from satsim.gat.trainer import FEATURE_NAMES_8
from torch_geometric.loader import DataLoader as PyGDataLoader

def generate_and_print_report():
    artifacts_dir = Path(repo_root) / "artifacts" / "gat" / "spatial"
    checkpoint_path = artifacts_dir / "gat_best.pt"
    scaler_path = artifacts_dir / "feature_scaler.pkl"

    if not checkpoint_path.exists():
        print(f"❌ Error: Model checkpoint not found at {checkpoint_path}")
        return

    # Load dataset & splits
    dataset_manager = LEOGraphSnapshotDataset(os.path.join(repo_root, "datasets"))
    dataset_manager.discover_snapshots()
    dataset_manager.validate_snapshots()

    _, _, test_pairs, scenario_test_pairs = dataset_manager.create_aligned_time_splits(
        train_ratio=0.70, val_ratio=0.15
    )

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler_data = pickle.load(f)
        feature_scaler = FeatureScaler()
        feature_scaler.node_scaler = scaler_data["node_scaler"]
        feature_scaler.edge_scaler = scaler_data["edge_scaler"]
        feature_scaler.fitted = True

    test_scaled = [feature_scaler.transform(d) for d in test_pairs]
    scen_test_scaled = {
        scen: [feature_scaler.transform(d) for d in pairs]
        for scen, pairs in scenario_test_pairs.items()
    }

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LEOGATModel(
        node_in_dim=8,
        edge_in_dim=4,
        hidden_dim=128,
        embedding_dim=128,
        heads=4,
        dropout=0.2,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Predict on test set
    test_loader = PyGDataLoader(test_scaled, batch_size=32, shuffle=False)
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            reconstructed_x, _, _ = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
            )
            y_true_list.append(batch.x.cpu().numpy())
            y_pred_list.append(reconstructed_x.cpu().numpy())

    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    # Calculate overall metrics
    overall_r2_weighted = r2_score(y_true, y_pred, multioutput="variance_weighted")
    overall_mse = mean_squared_error(y_true, y_pred)
    overall_mae = mean_absolute_error(y_true, y_pred)

    # Per-feature metrics
    per_feat_r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    per_feat_mse = np.mean((y_true - y_pred) ** 2, axis=0)
    per_feat_mae = np.mean(np.abs(y_true - y_pred), axis=0)

    # Per-scenario metrics
    scen_results = {}
    for scen, pairs in scen_test_scaled.items():
        scen_loader = PyGDataLoader(pairs, batch_size=32, shuffle=False)
        s_true_list, s_pred_list = [], []
        with torch.no_grad():
            for batch in scen_loader:
                batch = batch.to(device)
                reconstructed_x, _, _ = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                )
                s_true_list.append(batch.x.cpu().numpy())
                s_pred_list.append(reconstructed_x.cpu().numpy())

        s_true = np.vstack(s_true_list)
        s_pred = np.vstack(s_pred_list)

        s_r2 = r2_score(s_true, s_pred, multioutput="variance_weighted")
        s_mse = mean_squared_error(s_true, s_pred)
        s_mae = mean_absolute_error(s_true, s_pred)
        scen_results[scen] = (s_r2, s_mse, s_mae)

    # Save exact reconstruction results JSON
    r2_results_dict = {
        "overall_r2_weighted": float(overall_r2_weighted),
        "overall_mse": float(overall_mse),
        "overall_mae": float(overall_mae),
        "per_feature": {
            feat: {
                "r2": float(per_feat_r2[i]),
                "mse": float(per_feat_mse[i]),
                "mae": float(per_feat_mae[i]),
            }
            for i, feat in enumerate(FEATURE_NAMES_8)
        },
        "per_scenario": {
            scen: {
                "r2": float(scen_results[scen][0]),
                "mse": float(scen_results[scen][1]),
                "mae": float(scen_results[scen][2]),
            }
            for scen in scen_results
        },
    }
    with open(artifacts_dir / "exact_reconstruction_r2_results.json", "w", encoding="utf-8") as f:
        json.dump(r2_results_dict, f, indent=2)

    # PRINT TERMINAL REPORT
    print("=" * 80)
    print("        GAT RECONSTRUCTION MODEL EVALUATION REPORT (TEST SET GROUND-TRUTH)       ")
    print("=" * 80)
    print(f"Total Test Snapshots Evaluated : {len(test_pairs)} snapshots")
    print(f"Total Test Node State Vectors : {y_true.shape[0]} satellite states")
    print(f"Ground-Truth Matrix Shape     : {y_true.shape}")
    print("-" * 80)
    print("OVERALL RESULTS")
    print(f"  R² (Variance-Weighted) = {overall_r2_weighted:.6f} ({overall_r2_weighted * 100:.2f}%)")
    print(f"  MSE                    = {overall_mse:.6f}")
    print(f"  MAE                    = {overall_mae:.6f}")
    print("-" * 80)

    print("PER-FEATURE RESULTS (8 NON-TARGET PHYSICAL FEATURES)")
    print(f"{'Feature':<25} | {'R²':<12} | {'MSE':<12} | {'MAE':<12}")
    print("-" * 70)
    for i, feat in enumerate(FEATURE_NAMES_8):
        r2_str = f"{per_feat_r2[i]:<12.6f}" if per_feat_r2[i] >= 0 else f"{per_feat_r2[i]:<12.2f}"
        print(f"{feat:<25} | {r2_str} | {per_feat_mse[i]:<12.6f} | {per_feat_mae[i]:<12.6f}")

    print("-" * 80)
    print("PER-SCENARIO RESULTS")
    print(f"{'Scenario':<20} | {'R²':<12} | {'MSE':<12} | {'MAE':<12}")
    print("-" * 65)
    for scen, (s_r2, s_mse, s_mae) in scen_results.items():
        print(f"{scen:<20} | {s_r2:<12.6f} | {s_mse:<12.6f} | {s_mae:<12.6f}")

    print("=" * 80)

if __name__ == "__main__":
    generate_and_print_report()
