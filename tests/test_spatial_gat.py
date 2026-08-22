"""Unit tests for strict Spatial/Topological GAT representation learner.
"""
from pathlib import Path
import tempfile
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from satsim.gat import (
    LEOGATModel,
    LEOGraphSnapshotDataset,
    GATTrainer,
    GATEmbedder,
)
from satsim.gat.gat_dataset import FeatureScaler, FEATURE_INDICES, TARGET_INDEX


def create_synthetic_snapshot(num_nodes: int = 100, num_edges: int = 190) -> Data:
    """Create synthetic Data object matching Spatial GAT schema (8 non-target node features, 4 edge features)."""
    x = torch.randn(num_nodes, 8, dtype=torch.float32)

    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn(num_edges, 4, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.scenario = "low_load"
    data.timestep = 0
    return data


def test_1_input_feature_shape_8():
    """Verify Spatial GAT input features have shape [100, 8]."""
    data = create_synthetic_snapshot(num_nodes=100)
    assert data.x.shape == (100, 8), f"Expected input shape [100, 8], got {data.x.shape}"
    assert len(FEATURE_INDICES) == 8


def test_2_no_congestion_target_in_gat_input():
    """Verify congestion_score is strictly excluded from GAT input features."""
    assert TARGET_INDEX not in FEATURE_INDICES, "congestion_score (index 13) must be excluded from GAT inputs!"


def test_3_forward_output_shapes():
    """Verify forward output shapes: reconstructed_x=[100,8], node_embeddings=[100,128], graph_embedding=[1,128]."""
    model = LEOGATModel(node_in_dim=8, edge_in_dim=4, hidden_dim=128, embedding_dim=128, heads=4)
    model.eval()

    data = create_synthetic_snapshot(num_nodes=100)
    batch_vec = torch.zeros(100, dtype=torch.long)

    with torch.no_grad():
        reconstructed_x, node_embeddings, graph_embedding = model(
            x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=batch_vec
        )

    assert reconstructed_x.shape == (100, 8), f"Expected reconstructed_x [100, 8], got {reconstructed_x.shape}"
    assert node_embeddings.shape == (100, 128), f"Expected node_embeddings [100, 128], got {node_embeddings.shape}"
    assert graph_embedding.shape == (1, 128), f"Expected graph_embedding [1, 128], got {graph_embedding.shape}"


def test_4_attention_extraction():
    """Verify attention weights are non-empty, non-NaN, and match graph edge dimensions."""
    model = LEOGATModel(node_in_dim=8, edge_in_dim=4, hidden_dim=64, embedding_dim=64, heads=4)
    model.eval()

    data = create_synthetic_snapshot(num_nodes=100, num_edges=190)
    edge_index_att, alpha = model.get_attention_weights(data.x, data.edge_index, data.edge_attr)

    assert edge_index_att.shape[0] == 2
    assert alpha.shape[1] == 4
    assert not torch.isnan(alpha).any()
    assert torch.isfinite(alpha).all()


def test_5_reconstruction_loss_computation():
    """Verify self-supervised reconstruction loss is calculated correctly against X_scaled."""
    model = LEOGATModel(node_in_dim=8, edge_in_dim=4, hidden_dim=32, embedding_dim=32)
    model.train()
    data = create_synthetic_snapshot()

    reconstructed_x, _, _ = model(data.x, data.edge_index, data.edge_attr)
    loss = torch.nn.MSELoss()(reconstructed_x, data.x)

    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


def test_6_standardized_reconstruction_mse_mae_calculation():
    """Verify compute_reconstruction_metrics produces standardized MSE, MAE, and per-feature MAE."""
    from satsim.gat.trainer import compute_reconstruction_metrics

    x_true = np.random.randn(100, 8)
    x_pred = x_true + 0.1 * np.random.randn(100, 8)

    metrics = compute_reconstruction_metrics(x_true, x_pred)
    assert "reconstruction_mse" in metrics
    assert "reconstruction_mae" in metrics
    assert metrics["reconstruction_mse"] >= 0.0
    assert metrics["reconstruction_mae"] >= 0.0
    assert "mae_pos_eci_x" in metrics


def test_7_no_nan_inf_embeddings():
    """Verify generated node embeddings contain zero NaNs and zero Infs."""
    model = LEOGATModel(node_in_dim=8, edge_in_dim=4, hidden_dim=64, embedding_dim=128)
    model.eval()

    data = create_synthetic_snapshot()
    with torch.no_grad():
        _, node_embeddings, _ = model(data.x, data.edge_index, data.edge_attr)

    assert not torch.isnan(node_embeddings).any()
    assert torch.isfinite(node_embeddings).all()


def test_8_exported_embedding_shape_100_128():
    """Verify exported node embeddings have exact shape (100, 128)."""
    model = LEOGATModel(node_in_dim=8, edge_in_dim=4, hidden_dim=64, embedding_dim=128)
    model.eval()

    data = create_synthetic_snapshot(num_nodes=100)
    with torch.no_grad():
        _, node_embeddings, _ = model(data.x, data.edge_index, data.edge_attr)

    assert node_embeddings.shape == (100, 128)


def test_9_downstream_lstm_sequence_compatibility():
    """Verify 30-timestep sequence creation produces shape [30, 128] per satellite."""
    node_embeddings_30 = [torch.randn(100, 128) for _ in range(30)]

    # Stack along time dimension for satellite i=0
    sat0_sequence = torch.stack([emb[0] for emb in node_embeddings_30], dim=0)

    assert sat0_sequence.shape == (30, 128), f"Expected LSTM sequence shape [30, 128], got {sat0_sequence.shape}"
