"""GAT architecture for 100-satellite LEO network spatial representation learning.

Provides 128-dimensional node-level spatial embeddings and self-supervised
reconstruction of non-target node features.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class LEOGATModel(nn.Module):
    """Graph Attention Network (GAT) for LEO Satellite Constellations.

    Computes 128-dimensional spatial node embeddings and global graph-level
    embeddings using multi-head graph attention layers. Implements self-supervised
    reconstruction of the 8 non-target physical input node features to learn spatial/topological
    representations without using future congestion targets.

    Args:
        node_in_dim: Input non-target node feature dimension (typically 8).
        edge_in_dim: Input edge feature dimension (typically 4).
        hidden_dim: Dimension of intermediate hidden representations (default: 128).
        embedding_dim: Output spatial node embedding dimension (default: 128).
        heads: Number of attention heads in first GAT layer (default: 4).
        dropout: Dropout probability (default: 0.2).
    """

    def __init__(
        self,
        node_in_dim: int = 8,
        edge_in_dim: int = 4,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.dropout = dropout

        # First multi-head GAT layer (concatenates head outputs: heads * (hidden_dim // heads) = hidden_dim)
        head_dim = max(1, hidden_dim // heads)
        self.gat1 = GATConv(
            in_channels=node_in_dim,
            out_channels=head_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_in_dim,
        )

        # Second single-head GAT layer to produce final 128-dim node embedding
        self.gat2 = GATConv(
            in_channels=head_dim * heads,
            out_channels=embedding_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=edge_in_dim,
        )

        # Self-supervised spatial reconstruction head decoding node_embeddings back to non-target input features
        self.reconstructor = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, node_in_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for spatial representation learning and non-target feature reconstruction.

        Args:
            x: Node feature tensor of shape [N, node_in_dim] (16 non-target features).
            edge_index: Graph edge indices of shape [2, E].
            edge_attr: Edge feature tensor of shape [E, edge_in_dim].
            batch: Batch assignment vector of shape [N] (optional).
            return_attention: If True, also returns GAT attention weights (edge_index, alpha).

        Returns:
            Tuple of (reconstructed_x, node_embeddings, graph_embedding) or
            (reconstructed_x, node_embeddings, graph_embedding, (edge_index_att, alpha)) if return_attention=True:
                - reconstructed_x: Reconstructed non-target node features [N, node_in_dim].
                - node_embeddings: Spatial node embeddings [N, embedding_dim].
                - graph_embedding: Global pooled graph embeddings [B, embedding_dim].
        """
        # Layer 1: GATConv + ELU
        if return_attention:
            h, (edge_index_att, alpha) = self.gat1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        else:
            h = self.gat1(x, edge_index, edge_attr=edge_attr)
            edge_index_att, alpha = None, None

        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2: GATConv -> 128-dim spatial node embedding
        node_embeddings = self.gat2(h, edge_index, edge_attr=edge_attr)
        node_embeddings = F.elu(node_embeddings)

        # Self-supervised spatial reconstruction of non-target node features
        reconstructed_x = self.reconstructor(node_embeddings)

        # Global graph pooling
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        graph_embedding = global_mean_pool(node_embeddings, batch)

        if return_attention:
            return reconstructed_x, node_embeddings, graph_embedding, (edge_index_att, alpha)

        return reconstructed_x, node_embeddings, graph_embedding

    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract edge attention weights from Layer 1 for visualization.

        Args:
            x: Node feature tensor [N, node_in_dim].
            edge_index: Graph edge indices [2, E].
            edge_attr: Edge feature tensor [E, edge_in_dim].

        Returns:
            Tuple of (edge_index_att, alpha) where alpha is shape [E, heads].
        """
        self.eval()
        with torch.no_grad():
            _, (edge_index_att, alpha) = self.gat1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        return edge_index_att, alpha
