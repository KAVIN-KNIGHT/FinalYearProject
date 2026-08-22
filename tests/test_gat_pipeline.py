"""Unit tests for GAT model architecture, dataset handling, self-supervised spatial representation training, and embedding generation.
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
from satsim.gat.gat_dataset import FeatureScaler


def create_synthetic_snapshot(num_nodes: int = 100, num_edges: int = 190) -> Data:
    """Create synthetic Data object matching LEO GAT schema (8 non-target node features, 4 edge features)."""
    x = torch.randn(num_nodes, 8, dtype=torch.float32)

    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn(num_edges, 4, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.scenario = "low_load"
    data.timestep = 0
    return data


def test_gat_model_forward_pass_shapes():
    """Verify LEOGATModel produces correct output tensor dimensions for 8 non-target node features."""
    model = LEOGATModel(
        node_in_dim=8,
        edge_in_dim=4,
        hidden_dim=128,
        embedding_dim=128,
        heads=4,
    )
    model.eval()

    data = create_synthetic_snapshot(num_nodes=100, num_edges=190)
    batch_vec = torch.zeros(100, dtype=torch.long)

    with torch.no_grad():
        reconstructed_x, node_embeddings, graph_embedding = model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=batch_vec,
        )

    assert reconstructed_x.shape == (100, 8), f"Expected reconstructed_x [100, 8], got {reconstructed_x.shape}"
    assert node_embeddings.shape == (100, 128), f"Expected node_embeddings [100, 128], got {node_embeddings.shape}"
    assert graph_embedding.shape == (1, 128), f"Expected graph_embedding [1, 128], got {graph_embedding.shape}"


def test_gat_attention_weights_retrieval():
    """Verify LEOGATModel extracts valid edge attention weights for spatial visualization."""
    model = LEOGATModel(node_in_dim=8, edge_in_dim=4, hidden_dim=64, embedding_dim=64, heads=4)
    model.eval()

    data = create_synthetic_snapshot(num_nodes=100, num_edges=190)
    edge_index_att, alpha = model.get_attention_weights(data.x, data.edge_index, data.edge_attr)

    assert edge_index_att.shape[0] == 2
    assert alpha.shape[1] == 4  # 4 attention heads
    assert not torch.isnan(alpha).any()


def test_feature_scaler_fit_transform():
    """Verify FeatureScaler fits on train data and normalizes 8 non-target node features and 4 edge features."""
    scaler = FeatureScaler()
    train_graphs = [create_synthetic_snapshot() for _ in range(5)]

    scaler.fit(train_graphs)
    assert scaler.fitted is True

    test_graph = create_synthetic_snapshot()
    scaled_graph = scaler.transform(test_graph)

    assert scaled_graph.x.shape == test_graph.x.shape
    assert scaled_graph.edge_attr.shape == test_graph.edge_attr.shape
    assert not torch.isnan(scaled_graph.x).any()
    assert not torch.isnan(scaled_graph.edge_attr).any()


def test_gat_trainer_smoke_training_loop():
    """Verify GATTrainer executes self-supervised spatial training loop and saves best checkpoint."""
    model = LEOGATModel(node_in_dim=8, edge_in_dim=4, hidden_dim=32, embedding_dim=32)
    device = torch.device("cpu")

    train_graphs = [create_synthetic_snapshot() for _ in range(4)]
    val_graphs = [create_synthetic_snapshot() for _ in range(2)]

    train_loader = PyGDataLoader(train_graphs, batch_size=2)
    val_loader = PyGDataLoader(val_graphs, batch_size=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = GATTrainer(
            model=model,
            device=device,
            artifacts_dir=tmpdir,
            lr=0.01,
            early_stopping_patience=3,
        )

        trainer.fit(train_loader, val_loader, epochs=2)

        assert (Path(tmpdir) / "gat_best.pt").exists()
        assert (Path(tmpdir) / "gat_last.pt").exists()
        assert (Path(tmpdir) / "training_history.csv").exists()
