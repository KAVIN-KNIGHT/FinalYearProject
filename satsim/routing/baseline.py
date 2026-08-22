from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np

from satsim.traffic.flows import PacketFlow


class DijkstraRouter:
    """Shortest-path Dijkstra router for satellite constellation topology graphs."""

    def __init__(self, weight_key: str = "delay_ms"):
        self.weight_key = weight_key

    def compute_all_pairs_routes(self, graph: nx.Graph) -> Dict[Tuple[int, int], List[int]]:
        """Computes all-pairs shortest paths on graph using weight_key."""
        routes = {}
        paths = dict(nx.all_pairs_dijkstra_path(graph, weight=self.weight_key))
        for src, dst_dict in paths.items():
            for dst, path in dst_dict.items():
                routes[(src, dst)] = path
        return routes

    def get_next_hop(self, graph: nx.Graph, current_node: int, dst_node: int) -> Optional[int]:
        """Returns the next-hop node along the shortest path from current_node to dst_node."""
        if current_node == dst_node:
            return current_node
        if not graph.has_node(current_node) or not graph.has_node(dst_node):
            return None
        try:
            path = nx.shortest_path(graph, source=current_node, target=dst_node, weight=self.weight_key)
            return path[1] if len(path) > 1 else current_node
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def route_flow(
        self, graph: nx.Graph, flow: PacketFlow, buffer_delays: Optional[Dict[int, float]] = None
    ) -> Tuple[List[int], float, float, bool]:
        """
        Routes a PacketFlow through graph.
        Returns (path, propagation_delay_ms, total_delay_ms, is_delivered).
        """
        src = flow.src_id
        dst = flow.dst_id

        if not graph.has_node(src) or not graph.has_node(dst):
            return [], 0.0, 0.0, False

        try:
            path = nx.shortest_path(graph, source=src, target=dst, weight=self.weight_key)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], 0.0, 0.0, False

        prop_delay_ms = 0.0
        queuing_delay_ms = 0.0

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            if graph.has_edge(u, v):
                edge_data = graph[u][v]
                prop_delay_ms += edge_data.get("delay_ms", 10.0)

            if buffer_delays and u in buffer_delays:
                queuing_delay_ms += buffer_delays[u]

        total_delay_ms = prop_delay_ms + queuing_delay_ms
        return path, prop_delay_ms, total_delay_ms, True
