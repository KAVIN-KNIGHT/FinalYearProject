"""Discrete-event simulation engine for the LEO mega-constellation simulator.

The :class:`SimulationEngine` orchestrates orbital propagation, ISL topology,
ground station contacts, stochastic event injection, baseline routing, and
per-timestep telemetry collection into a single reproducible simulation loop.
The output is a list of canonical trace records consumed by the GAT and LSTM
exporters.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional
import networkx as nx
import numpy as np

from satsim.config import SimConfig
from satsim.orbital import WalkerDeltaConstellation
from satsim.topology import ISLManager, GroundStationManager
from satsim.traffic import create_traffic_profile
from satsim.events import EventInjector
from satsim.routing import DijkstraRouter
from satsim.metrics import MetricsCollector
from satsim.logging import get_logger

logger = get_logger("satsim.sim.engine")


class SimulationEngine:
    """Discrete-event simulation loop orchestrating orbital, topology, traffic, events, and metrics."""

    def __init__(self, config: SimConfig) -> None:
        """Initialise all simulation subsystems from *config*.

        Args:
            config: Fully-resolved :class:`~satsim.config.SimConfig` including
                seed, constellation geometry, traffic profile, and event settings.
        """
        self.config = config
        self.seed = config.seed

        self.constellation = WalkerDeltaConstellation(config=config.constellation)
        self.isl_mgr = ISLManager(config=config.isl)
        self.gs_mgr = GroundStationManager(config=config.ground_stations)
        self.traffic_profile = create_traffic_profile(config.traffic.profile, config.traffic, seed=self.seed)
        self.injector = EventInjector(config=config.events, seed=self.seed)
        self.router = DijkstraRouter()
        self.metrics_collector = MetricsCollector()

    def run(self, progress_bar: bool = False) -> List[Dict[str, Any]]:
        """Execute the full discrete-event loop and return per-timestep trace records.

        Args:
            progress_bar: If ``True``, display a ``tqdm`` progress bar during
                simulation.  Safe to set ``False`` in batch / parallel contexts.

        Returns:
            A list of trace record dicts, one per simulated timestep.  Each record
            contains: ``timestep``, ``simulation_time_s``, ``satellite_states``,
            ``isl_edges``, ``gs_contacts``, ``active_events``,
            ``metrics`` (19 global aggregates),
            ``sat_metrics`` (per-satellite metrics keyed by sat_id int), and
            ``edge_metrics`` (per-edge metrics keyed by ``"{u}_{v}"`` string).
        """
        dt_s = self.config.timestep_seconds
        duration_s = self.config.duration_seconds
        num_steps = int(np.ceil(duration_s / dt_s))
        num_sats = self.config.constellation.num_satellites

        all_flows = self.traffic_profile.generate_flows(0.0, duration_s, num_nodes=num_sats)
        trace_records: List[Dict[str, Any]] = []
        gs_names = [gs.name for gs in self.gs_mgr.stations.values()]

        iterator = range(num_steps)
        if progress_bar:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"Simulating ({self.config.traffic.profile})")

        for step in iterator:
            t_s = step * dt_s

            # 1. Propagate satellites to t_s
            sat_states = self.constellation.get_states(t_s)

            # 2. Update ISL graph topology
            base_graph = self.isl_mgr.update_grid_topology(
                sat_states,
                num_planes=self.config.constellation.num_planes,
                sats_per_plane=self.config.constellation.sats_per_plane,
            )

            # 3. Compute Ground Station visibility contacts
            active_contacts = self.gs_mgr.get_active_contacts(
                sat_states, min_elevation_deg=self.config.isl.min_elevation_deg
            )

            # 4. Step event injector and apply active disruptions
            self.injector.step(
                t_s=t_s,
                current_graph=base_graph,
                num_nodes=num_sats,
                gs_names=gs_names,
                dt_s=dt_s,
            )
            active_graph = self.injector.apply_to_graph(base_graph)

            # 5. Extract active flows for current timestep
            step_flows = [
                f for f in all_flows
                if t_s <= f.start_time_s < (t_s + dt_s)
            ]

            # 6. Collect metrics
            metrics = self.metrics_collector.collect_step(
                t_s=t_s,
                step=step,
                graph=active_graph,
                active_flows=step_flows,
                router=self.router,
                injector=self.injector,
                dt_s=dt_s,
            )

            # 7. Construct canonical per-timestep trace record
            sat_states_records = [
                {
                    "sat_id": st.sat_id,
                    "pos_eci": st.position_eci.tolist(),
                    "vel_eci": st.velocity_eci.tolist(),
                    "pos_ecef": st.position_ecef.tolist(),
                    "vel_ecef": st.velocity_ecef.tolist(),
                }
                for st in sat_states.values()
            ]

            isl_edge_records = [
                {
                    "src": u,
                    "dst": v,
                    "distance_km": data.get("distance_km", 0.0),
                    "delay_ms": data.get("delay_ms", 0.0),
                    "link_type": data.get("link_type", "intra_plane"),
                }
                for u, v, data in active_graph.edges(data=True)
            ]

            gs_contact_records = [
                {
                    "gs_id": c.gs_id,
                    "gs_name": c.gs_name,
                    "sat_id": c.sat_id,
                    "elevation_deg": c.elevation_deg,
                    "distance_km": c.distance_km,
                    "delay_ms": c.delay_ms,
                }
                for c in active_contacts
            ]

            active_events_records = [
                {
                    "event_id": ev.event_id,
                    "event_type": str(ev.event_type.value),
                    "start_time_s": ev.start_time_s,
                    "duration_s": ev.duration_s,
                    "target_id": ev.target_id if isinstance(ev.target_id, (int, str)) else list(ev.target_id),
                    "active": ev.active,
                }
                for ev in self.injector.events_history
                if ev.active and ev.start_time_s <= t_s <= ev.end_time_s
            ]

            # metrics is now {"global": {...}, "per_satellite": {...}, "per_edge": {...}}.
            # Keep "metrics" pointing at the 19 global aggregates for backward compat.
            # Per-entity dicts are stored at the top level so exporters can consume them
            # without coupling to each other.
            trace_records.append(
                {
                    "timestep": int(step),
                    "simulation_time_s": float(t_s),
                    "satellite_states": sat_states_records,
                    "isl_edges": isl_edge_records,
                    "gs_contacts": gs_contact_records,
                    "active_events": active_events_records,
                    "metrics": metrics["global"],
                    # sat_metrics: keys are satellite IDs as integers (JSON-serialised as strings).
                    "sat_metrics": {str(k): v for k, v in metrics["per_satellite"].items()},
                    # edge_metrics: keys are "{u}_{v}" strings.
                    "edge_metrics": metrics["per_edge"],
                }
            )

        return trace_records
