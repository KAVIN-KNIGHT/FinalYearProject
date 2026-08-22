"""Per-timestep telemetry collector for the LEO mega-constellation simulator.

``MetricsCollector`` is called once per simulation timestep and produces three
levels of output, all stored inside a single return dict from :meth:`collect_step`:

- ``"global"``: The original 19 aggregate scalars (unchanged for backward compat).
- ``"per_satellite"``: Dict keyed by satellite ID → per-node metrics dict.
- ``"per_edge"``: Dict keyed by ``"{u}_{v}"`` string → per-link metrics dict.

Per-satellite metrics
---------------------
``queue_length``, ``queue_occupancy``, ``congestion_score``, ``cpu_utilization``,
``memory_utilization``, ``neighbor_count``, ``node_degree``, ``routing_table_age``,
``routing_changes_in_window``, ``failure_indicator``, ``event_flags``,
``end_to_end_delay``, ``throughput``

Notes on routing_table_age and routing_changes_in_window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dijkstra paths are recomputed fresh every timestep against the live topology.
There is no persistent routing table, so ``routing_table_age`` is always 0.0 and
``routing_changes_in_window`` is always 0.  These are placeholder columns for
future adaptive-routing extensions.

Per-edge metrics
----------------
``link_utilization``, ``available_bandwidth_kbps``, ``used_bandwidth_kbps``,
``ber``, ``snr``, ``link_stability``, ``link_lifetime``, ``link_failure_probability``,
``degradation_factor``

Per-timestep routing guarantee (B2)
------------------------------------
``collect_step`` receives the *live* ``active_graph`` produced by
``SimulationEngine`` for that timestep.  This graph already incorporates all
active event state via ``EventInjector.apply_to_graph``.  Therefore every flow
is routed against the current topology, not a cached snapshot — routes are
re-evaluated implicitly on every call.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from satsim.traffic.flows import PacketFlow
from satsim.routing.baseline import DijkstraRouter
from satsim.events.injector import EventInjector

# ── Column lists ─────────────────────────────────────────────────────────────

METRIC_COLUMNS = [
    "timestep",
    "simulation_time_s",
    "active_satellites",
    "active_isls",
    "active_flows",
    "total_packets_sent",
    "total_packets_delivered",
    "total_packets_dropped",
    "packet_delivery_ratio",
    "throughput_kbps",
    "average_delay_ms",
    "max_delay_ms",
    "average_jitter_ms",
    "average_hop_count",
    "average_buffer_utilization",
    "max_buffer_utilization",
    "average_link_utilization",
    "max_link_utilization",
    "active_events_count",
]

#: Base BER for a healthy, non-degraded ISL.
_BASE_BER: float = 1e-9
#: Base SNR (dB) for a healthy, non-degraded ISL.
_BASE_SNR_DB: float = 30.0


class MetricsCollector:
    """Computes all telemetry metrics per timestep at global, satellite, and edge granularity."""

    def __init__(
        self,
        buffer_capacity_packets: int = 1000,
        link_capacity_kbps: float = 10_000.0,
    ) -> None:
        """Initialise the collector.

        Args:
            buffer_capacity_packets: Per-satellite packet buffer size (for queue occupancy).
            link_capacity_kbps: Per-ISL link capacity used to compute utilization.
        """
        self.buffer_capacity_packets = buffer_capacity_packets
        self.link_capacity_kbps = link_capacity_kbps

        self.history: List[Dict[str, Any]] = []
        self.cumulative_packets_sent: int = 0
        self.cumulative_packets_delivered: int = 0
        self.cumulative_packets_dropped: int = 0
        self._last_delays: List[float] = []

    def collect_step(
        self,
        t_s: float,
        step: int,
        graph: nx.Graph,
        active_flows: List[PacketFlow],
        router: DijkstraRouter,
        injector: Optional[EventInjector] = None,
        dt_s: float = 5.0,
    ) -> Dict[str, Any]:
        """Collect all metrics for one simulation timestep.

        Routing note (B2): *graph* is the live topology for this timestep,
        already incorporating all active event state.  Each flow is routed
        against this graph, so routes are re-evaluated on every call.

        Args:
            t_s: Current simulation time in seconds.
            step: Integer timestep index.
            graph: Live ISL graph with event state applied.
            active_flows: Packet flows active during this timestep.
            router: Dijkstra router used to evaluate paths.
            injector: Active event injector (provides per-node/edge failure state).
            dt_s: Timestep duration in seconds.

        Returns:
            Dict with keys ``"global"`` (19 aggregate scalars),
            ``"per_satellite"`` (Dict[int, Dict[str, float]]),
            and ``"per_edge"`` (Dict[str, Dict[str, float]]).
        """
        # ── Pull failure state from injector (safe defaults if absent) ───────
        disabled_nodes = injector.disabled_nodes if injector else set()
        congested_nodes = injector.congested_nodes if injector else set()
        buffer_overflow_nodes = injector.buffer_overflow_nodes if injector else set()

        num_active_sats = graph.number_of_nodes()
        num_active_isls = graph.number_of_edges()
        active_events_count = (
            len([ev for ev in injector.events_history if ev.active]) if injector else 0
        )

        # ── Per-node and per-edge packet counters ─────────────────────────────
        node_packet_counts: Dict[int, int] = {node: 0 for node in graph.nodes()}
        node_bytes_delivered: Dict[int, float] = {node: 0.0 for node in graph.nodes()}
        node_delay_sum: Dict[int, float] = {node: 0.0 for node in graph.nodes()}
        node_delay_count: Dict[int, int] = {node: 0 for node in graph.nodes()}
        edge_packet_counts: Dict[Tuple[int, int], int] = {
            edge: 0 for edge in graph.edges()
        }
        edge_bytes_counts: Dict[Tuple[int, int], float] = {
            edge: 0.0 for edge in graph.edges()
        }

        # ── Route all flows ──────────────────────────────────────────────────
        step_sent = step_delivered = step_dropped = 0
        step_bytes_delivered = 0.0
        delays_ms: List[float] = []
        hop_counts: List[int] = []

        for flow in active_flows:
            step_sent += flow.total_packets
            path, prop_delay_ms, total_delay_ms, is_delivered = router.route_flow(
                graph, flow
            )

            if is_delivered and len(path) > 1:
                step_delivered += flow.total_packets
                step_bytes_delivered += flow.total_bytes

                queuing_ms = sum(
                    0.01 * node_packet_counts.get(n, 0) for n in path[:-1]
                )
                e2e_delay_ms = prop_delay_ms + queuing_ms
                delays_ms.append(e2e_delay_ms)
                hop_counts.append(len(path) - 1)

                for n in path[:-1]:
                    node_packet_counts[n] = node_packet_counts.get(n, 0) + flow.total_packets
                    node_bytes_delivered[n] = node_bytes_delivered.get(n, 0.0) + flow.total_bytes
                    node_delay_sum[n] = node_delay_sum.get(n, 0.0) + e2e_delay_ms
                    node_delay_count[n] = node_delay_count.get(n, 0) + 1

                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge = (u, v) if (u, v) in edge_packet_counts else (v, u)
                    edge_packet_counts[edge] = edge_packet_counts.get(edge, 0) + flow.total_packets
                    edge_bytes_counts[edge] = edge_bytes_counts.get(edge, 0.0) + flow.total_bytes
            else:
                step_dropped += flow.total_packets

        self.cumulative_packets_sent += step_sent
        self.cumulative_packets_delivered += step_delivered
        self.cumulative_packets_dropped += step_dropped

        # ── Global aggregates ─────────────────────────────────────────────────
        pdr = float(np.clip(
            self.cumulative_packets_delivered / max(1, self.cumulative_packets_sent),
            0.0, 1.0,
        ))
        throughput_kbps = (step_bytes_delivered * 8.0 / 1000.0) / dt_s if dt_s > 0 else 0.0

        avg_delay_ms = float(np.mean(delays_ms)) if delays_ms else 0.0
        max_delay_ms = float(np.max(delays_ms)) if delays_ms else 0.0
        avg_jitter_ms = (
            float(abs(avg_delay_ms - float(np.mean(self._last_delays))))
            if delays_ms and self._last_delays
            else 0.0
        )
        self._last_delays = delays_ms

        avg_hop_count = float(np.mean(hop_counts)) if hop_counts else 0.0

        pkt_cap = max(
            1.0,
            (self.link_capacity_kbps * 1000.0 / 8.0 / 512.0) * dt_s,
        )

        buf_utils = [
            min(1.0, node_packet_counts.get(n, 0) / self.buffer_capacity_packets)
            for n in graph.nodes()
        ]
        avg_buf_util = float(np.mean(buf_utils)) if buf_utils else 0.0
        max_buf_util = float(np.max(buf_utils)) if buf_utils else 0.0

        link_utils = [
            min(1.0, edge_packet_counts.get(e, 0) / pkt_cap)
            for e in graph.edges()
        ]
        avg_link_util = float(np.mean(link_utils)) if link_utils else 0.0
        max_link_util = float(np.max(link_utils)) if link_utils else 0.0

        global_record: Dict[str, Any] = {
            "timestep": int(step),
            "simulation_time_s": float(t_s),
            "active_satellites": int(num_active_sats),
            "active_isls": int(num_active_isls),
            "active_flows": len(active_flows),
            "total_packets_sent": int(self.cumulative_packets_sent),
            "total_packets_delivered": int(self.cumulative_packets_delivered),
            "total_packets_dropped": int(self.cumulative_packets_dropped),
            "packet_delivery_ratio": pdr,
            "throughput_kbps": throughput_kbps,
            "average_delay_ms": avg_delay_ms,
            "max_delay_ms": max_delay_ms,
            "average_jitter_ms": avg_jitter_ms,
            "average_hop_count": avg_hop_count,
            "average_buffer_utilization": avg_buf_util,
            "max_buffer_utilization": max_buf_util,
            "average_link_utilization": avg_link_util,
            "max_link_utilization": max_link_util,
            "active_events_count": int(active_events_count),
        }
        self.history.append(global_record)

        # ── Per-satellite metrics (B1 / Section C) ────────────────────────────
        buffer_byte_capacity = float(self.buffer_capacity_packets * 1500)
        max_cpu_packet_rate = float(pkt_cap * 4.0)
        sat_throughput_capacity_kbps = self.link_capacity_kbps * 4.0

        per_satellite: Dict[int, Dict[str, float]] = {}
        for node in graph.nodes():
            q_len = node_packet_counts.get(node, 0)
            q_occ = min(1.0, q_len / self.buffer_capacity_packets)

            # buffer_utilization: actual memory byte occupancy with 64B allocation header overhead
            buf_util = min(1.0, (q_len * 512.0 + 64.0 * q_len) / buffer_byte_capacity)

            degree = graph.degree(node)
            # neighbor_count: operational/active incident neighbors (excluding disabled/failed)
            active_neighbors = sum(
                1 for nbr in graph.neighbors(node)
                if nbr not in disabled_nodes and graph.nodes[nbr].get("is_active", True)
            )

            nd_delay = (
                node_delay_sum.get(node, 0.0) / node_delay_count[node]
                if node_delay_count.get(node, 0) > 0
                else 0.0
            )
            nd_tput = (
                node_bytes_delivered.get(node, 0.0) * 8.0 / 1000.0 / dt_s
                if dt_s > 0
                else 0.0
            )

            # traffic_load: offered/transit load ratio (composite of delivered throughput and queue pressure)
            traffic_load = float(min(1.0, 0.70 * (nd_tput / max(1.0, sat_throughput_capacity_kbps)) + 0.30 * q_occ))

            # cpu_utilization: packet forwarding & queue processing workload
            cpu_util = float(min(1.0, 0.02 + 0.68 * (q_len / max(1.0, max_cpu_packet_rate)) + 0.30 * (nd_tput / max(1.0, sat_throughput_capacity_kbps))))

            # memory_utilization: RAM usage (routing state + packet buffers)
            mem_util = float(min(1.0, 0.08 + 0.72 * buf_util + 0.20 * (active_neighbors / max(1, degree))))

            is_congested = 1.0 if node in congested_nodes else 0.0
            is_overflow = 1.0 if node in buffer_overflow_nodes else 0.0
            fail_ind = 1.0 if node in disabled_nodes else 0.0

            incident_edges = [
                (node, nbr) if (node, nbr) in edge_packet_counts else (nbr, node)
                for nbr in graph.neighbors(node)
            ]
            avg_inc_util = float(np.mean([
                min(1.0, edge_packet_counts.get(e, 0) / pkt_cap) for e in incident_edges
            ])) if incident_edges else 0.0

            is_degraded = 1.0 if any(
                (injector.get_edge_degradation(e) > 1.0) if injector else False
                for e in incident_edges
            ) else 0.0

            # congestion_score: composite of queue occupancy, incident link load, traffic load, and fault flags
            congestion_score = float(
                0.35 * q_occ + 0.35 * avg_inc_util + 0.30 * min(1.0, traffic_load)
                + 0.5 * is_congested + 0.5 * is_overflow
            )

            # event_flags bitmask: bit0=sat_failure, bit1=congestion, bit2=buffer_overflow, bit3=link_degradation
            event_flags = int(fail_ind) | (int(is_congested) << 1) | (int(is_overflow) << 2) | (int(is_degraded) << 3)

            per_satellite[node] = {
                "queue_length": float(q_len),
                "queue_occupancy": float(q_occ),
                "buffer_utilization": float(buf_util),
                "traffic_load": float(traffic_load),
                "cpu_utilization": float(cpu_util),
                "memory_utilization": float(mem_util),
                "congestion_score": float(congestion_score),
                "neighbor_count": float(active_neighbors),
                "node_degree": float(degree),
                # Dijkstra is recomputed fresh each timestep — no persistent table.
                "routing_table_age": 0.0,
                "routing_changes_in_window": 0.0,
                "failure_indicator": fail_ind,
                "event_flags": float(event_flags),
                "end_to_end_delay": float(nd_delay),
                "throughput": float(nd_tput),
            }

        # ── Per-edge metrics (B1) ──────────────────────────────────────────────
        per_edge: Dict[str, Dict[str, float]] = {}
        for u, v, edata in graph.edges(data=True):
            edge_key = f"{u}_{v}"
            e_tup = (u, v) if (u, v) in edge_packet_counts else (v, u)
            pkt_count = edge_packet_counts.get(e_tup, 0)
            byte_count = edge_bytes_counts.get(e_tup, 0.0)
            e_util = min(1.0, pkt_count / pkt_cap)
            avail_bw = self.link_capacity_kbps * (1.0 - e_util)
            used_bw = byte_count * 8.0 / 1000.0 / dt_s if dt_s > 0 else 0.0
            deg_factor = edata.get("degradation_factor", 1.0)
            # BER increases with degradation; SNR degrades inversely.
            ber = _BASE_BER * deg_factor
            snr_db = _BASE_SNR_DB - 10.0 * np.log10(max(deg_factor, 1e-9))
            # link_stability: 1.0 if no degradation/failure active on this edge.
            is_degraded = deg_factor > 1.0
            is_disabled = (
                injector._normalize_edge((u, v)) in injector.disabled_edges
                if injector
                else False
            )
            link_stability = 0.0 if (is_degraded or is_disabled) else 1.0
            link_failure_prob = 1.0 - 1.0 / max(deg_factor, 1.0)

            per_edge[edge_key] = {
                "link_utilization": float(e_util),
                "available_bandwidth_kbps": float(avail_bw),
                "used_bandwidth_kbps": float(used_bw),
                "ber": float(ber),
                "snr": float(snr_db),
                "link_stability": float(link_stability),
                "link_lifetime": 0.0,   # placeholder; no per-edge creation timestamp
                "link_failure_probability": float(link_failure_prob),
                "degradation_factor": float(deg_factor),
            }

        return {
            "global": global_record,
            "per_satellite": per_satellite,
            "per_edge": per_edge,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the global metrics history to a pandas DataFrame.

        Returns:
            DataFrame with one row per timestep and :data:`METRIC_COLUMNS` columns.
        """
        if not self.history:
            return pd.DataFrame(columns=METRIC_COLUMNS)
        df = pd.DataFrame(self.history)
        return df[METRIC_COLUMNS]
