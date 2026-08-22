"""GAT dataset exporter for the LEO mega-constellation simulator.

Converts a canonical per-timestep trace (``trace.json``) into PyTorch Geometric
:class:`~torch_geometric.data.Data` snapshots, one ``.pt`` file per timestep.

**Independence guarantee**: This exporter reads only from ``trace.json`` and
``config_used.yaml``.  It does not depend on the LSTM exporter having run first.

**Ground station scope (C4)**: Ground stations are *out of scope* for this graph.
GS nodes are not included in ``data.x`` and GS–satellite links are not included
in ``data.edge_index``.  If GS nodes are added in a future phase, a
``data.node_type`` field must distinguish them from satellite nodes.

**Graph schema** (undirected ISLs stored as bidirectional pairs):

Node features ``data.x`` — shape ``[200, 17]`` (C2, expanded from 11):

+-------+---------------------+-------------------------------------------+
| Index | Feature             | Notes                                     |
+=======+=====================+===========================================+
| 0–2   | ``pos_eci``         | ECI position (km)                         |
+-------+---------------------+-------------------------------------------+
| 3–5   | ``vel_eci``         | ECI velocity (km/s)                       |
+-------+---------------------+-------------------------------------------+
| 6–8   | ``pos_ecef``        | ECEF position (km)                        |
+-------+---------------------+-------------------------------------------+
| 9     | ``is_active``       | 1.0 = nominal, 0.0 = failed               |
+-------+---------------------+-------------------------------------------+
| 10    | ``queue_occupancy`` | Buffer fill ∈ [0, 1]                      |
+-------+---------------------+-------------------------------------------+
| 11    | ``queue_length``    | Packet count (normalised /buffer_cap)     |
+-------+---------------------+-------------------------------------------+
| 12    | ``node_degree``     | Active ISL neighbour count                |
+-------+---------------------+-------------------------------------------+
| 13    | ``congestion_score``| ∈ [0, 2]                                  |
+-------+---------------------+-------------------------------------------+
| 14    | ``cpu_utilization`` | ∈ [0, 1]                                  |
+-------+---------------------+-------------------------------------------+
| 15    | ``memory_utilization`` | ∈ [0, 1]                               |
+-------+---------------------+-------------------------------------------+
| 16    | ``failure_indicator``| 1.0 if SAT_FAILURE active, else 0.0     |
+-------+---------------------+-------------------------------------------+

Edge features ``data.edge_attr`` — shape ``[2E, 6]`` (C3, replaces [2E, 5]):

+-------+---------------------------+------------------------------------------+
| Index | Feature                   | Notes                                    |
+=======+===========================+==========================================+
| 0     | ``distance_km``           | ISL link distance                        |
+-------+---------------------------+------------------------------------------+
| 1     | ``delay_ms``              | Propagation delay (possibly degraded)    |
+-------+---------------------------+------------------------------------------+
| 2     | ``link_utilization``      | ∈ [0, 1]                                 |
+-------+---------------------------+------------------------------------------+
| 3     | ``degradation_factor``    | 1.0 = healthy; >1 = degraded             |
+-------+---------------------------+------------------------------------------+
| 4     | ``link_failure_probability`` | ∈ [0, 1)                              |
+-------+---------------------------+------------------------------------------+
| 5     | ``link_type_code``        | 0=intra-plane, 1=inter-plane, 2=seam     |
+-------+---------------------------+------------------------------------------+
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch_geometric.data import Data

from satsim.export.trace_store import TraceStore
from satsim.logging import get_logger

logger = get_logger("satsim.export.gat_export")

_LINK_TYPE_CODE = {
    "intra_plane": 0,
    "inter_plane": 1,
}

_NODE_FEATURE_DIM = 17
_EDGE_FEATURE_DIM = 6


class GATExporter:
    """Exports canonical simulation trace records into PyTorch Geometric (.pt) graph snapshots."""

    def __init__(self, scenario_dir: Path) -> None:
        """Initialise and validate the scenario directory.

        Args:
            scenario_dir: Directory containing ``trace.json`` and
                ``config_used.yaml`` written by :class:`~satsim.export.trace_store.TraceStore`.

        Raises:
            FileNotFoundError: If ``trace.json`` or ``config_used.yaml`` are absent.
        """
        self.scenario_dir = Path(scenario_dir)
        self.trace_file = self.scenario_dir / "trace.json"
        self.config_file = self.scenario_dir / "config_used.yaml"

        if not self.trace_file.exists() or not self.config_file.exists():
            raise FileNotFoundError(
                f"No canonical trace found in '{self.scenario_dir}'! "
                f"Cannot export GAT dataset. Run "
                f"'python -m satsim.cli.run_scenario --scenario {self.scenario_dir.name}' first."
            )

    def export_scenario(self, output_dir: Optional[Path] = None) -> List[Path]:
        """Export one PyG ``.pt`` snapshot per trace timestep to ``gat/``.

        Returns:
            Sorted list of :class:`pathlib.Path` objects for each written
            ``snapshot_{step:06d}.pt`` file.
        """
        trace_records, config = TraceStore.load_trace(self.scenario_dir)
        num_sats = config.constellation.num_satellites

        gat_dir = Path(output_dir) if output_dir is not None else (self.scenario_dir / "gat")
        gat_dir.mkdir(parents=True, exist_ok=True)

        # Build a sat_id → state lookup for O(1) access per step.
        saved_files: List[Path] = []

        for record in trace_records:
            step = record["timestep"]
            t_s = record["simulation_time_s"]

            # ── Build sat_id → raw state dict (C1: explicit canonical order) ──
            sat_state_map: Dict[int, Any] = {
                s["sat_id"]: s for s in record["satellite_states"]
            }
            # per-satellite metrics from B1
            sat_metrics_map: Dict[str, Any] = record.get("sat_metrics", {})
            # per-edge metrics from B1
            edge_metrics_map: Dict[str, Any] = record.get("edge_metrics", {})

            # ── Node feature matrix [num_sats, 17] ───────────────────────────
            x_list: List[List[float]] = []
            for sat_id in range(num_sats):
                sat = sat_state_map.get(sat_id, {})
                pos_eci = sat.get("pos_eci", [0.0, 0.0, 0.0])
                vel_eci = sat.get("vel_eci", [0.0, 0.0, 0.0])
                pos_ecef = sat.get("pos_ecef", [0.0, 0.0, 0.0])

                sm = sat_metrics_map.get(str(sat_id), {})
                is_active = 1.0 - sm.get("failure_indicator", 0.0)
                queue_occ = sm.get("queue_occupancy", 0.0)
                queue_len = sm.get("queue_length", 0.0)
                node_degree = sm.get("node_degree", 0.0)
                congestion_score = sm.get("congestion_score", 0.0)
                cpu_util = sm.get("cpu_utilization", 0.0)
                mem_util = sm.get("memory_utilization", 0.0)
                fail_ind = sm.get("failure_indicator", 0.0)

                feat = (
                    pos_eci
                    + vel_eci
                    + pos_ecef
                    + [
                        is_active,
                        queue_occ,
                        queue_len,
                        node_degree,
                        congestion_score,
                        cpu_util,
                        mem_util,
                        fail_ind,
                    ]
                )
                x_list.append(feat)

            x = torch.tensor(x_list, dtype=torch.float32)
            assert x.shape == (num_sats, _NODE_FEATURE_DIM), (
                f"Node feature shape mismatch: {x.shape} vs ({num_sats}, {_NODE_FEATURE_DIM})"
            )

            # ── Edge index and feature matrix ─────────────────────────────────
            src_list: List[int] = []
            dst_list: List[int] = []
            edge_attr_list: List[List[float]] = []

            for edge in record["isl_edges"]:
                u = edge["src"]
                v = edge["dst"]
                dist = edge.get("distance_km", 0.0)
                delay = edge.get("delay_ms", 0.0)
                link_type = edge.get("link_type", "intra_plane")
                link_type_code = float(_LINK_TYPE_CODE.get(link_type, 2))

                # Retrieve per-edge metrics (try both (u,v) and (v,u) key forms)
                em = edge_metrics_map.get(f"{u}_{v}") or edge_metrics_map.get(f"{v}_{u}") or {}
                link_util = em.get("link_utilization", 0.0)
                deg_factor = em.get("degradation_factor", 1.0)
                fail_prob = em.get("link_failure_probability", 0.0)

                attr = [dist, delay, link_util, deg_factor, fail_prob, link_type_code]

                # Store bidirectionally (undirected ISL → both directions)
                src_list.extend([u, v])
                dst_list.extend([v, u])
                edge_attr_list.extend([attr, attr])

            if src_list:
                edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, _EDGE_FEATURE_DIM), dtype=torch.float32)

            pyg_data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                timestep=torch.tensor(step, dtype=torch.long),
                simulation_time_s=torch.tensor(t_s, dtype=torch.float32),
            )

            out_path = gat_dir / f"snapshot_{step:06d}.pt"
            torch.save(pyg_data, out_path)
            saved_files.append(out_path)

        logger.info(
            "GAT dataset export complete",
            scenario=self.scenario_dir.name,
            total_snapshots=len(saved_files),
            output_dir=str(gat_dir),
            node_feature_dim=_NODE_FEATURE_DIM,
            edge_feature_dim=_EDGE_FEATURE_DIM,
        )
        return saved_files
