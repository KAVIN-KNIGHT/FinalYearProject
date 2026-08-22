"""Event injection engine for the LEO mega-constellation simulator.

Auto-expiry contract
--------------------
Every failure/degradation event auto-recovers at ``start_time_s + duration_s``
via the ``step()`` expiry loop.  An explicit ``RECOVERY`` event is only for
scripted *early* recovery.  The restore operation is idempotent: discarding a
target from a set or removing a key from a dict is safe to call twice.

Overlapping event stacking (A2)
--------------------------------
When two events affect the same edge simultaneously (e.g. ``LINK_DEGRADATION``
followed by ``SOLAR_INTERFERENCE`` on the same ISL), degradation factors are
**multiplied** together.  Each active event contributes its own factor keyed by
its ``event_id``.  Recovering event B removes only B's factor from the dict,
leaving event A's contribution intact.  This prevents the "B's prior state
captured A's degraded state" corruption that a snapshot-based approach produces.

Node failure isolation (A1)
----------------------------
A ``SAT_FAILURE`` event does **not** remove the node from the graph.  Instead
``apply_to_graph`` removes all incident edges from the failed satellite and
annotates ``H.nodes[node]['is_active'] = False``.  Because the ISL manager
rebuilds every edge from live orbital geometry each timestep, recovery is
automatic: once the node is removed from ``disabled_nodes`` its edges reappear
naturally in the next ``ISLManager.update_grid_topology`` call, with no stale
attribute values.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

from satsim.config import EventsConfig
from .types import EventType, SimEvent


class EventInjector:
    """Stochastic event scheduler, state modification, and timestamped query logging engine."""

    def __init__(self, config: Optional[EventsConfig] = None, seed: int = 42) -> None:
        """Initialise the injector.

        Args:
            config: Event configuration controlling which event types are enabled
                and the mean Poisson arrival rate.
            seed: RNG seed for reproducible stochastic scheduling.
        """
        self.config = config or EventsConfig()
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self._next_event_id = 1
        self.events_history: List[SimEvent] = []

        # ── Active network state modifiers ───────────────────────────────────
        # Node / GS sets are still simple sets — membership is all that matters.
        self.disabled_nodes: Set[int] = set()
        self.congested_nodes: Set[int] = set()
        self.buffer_overflow_nodes: Set[int] = set()
        self.gs_congestion: Set[str] = set()

        # For ISL failure: track disabled edges as a set.
        self.disabled_edges: Set[Tuple[int, int]] = set()

        # Multiplicative degradation stacking (A2):
        # Dict[normalized_edge, Dict[event_id, factor]]
        # Effective factor = product of all values.
        self._edge_degradation_factors: Dict[Tuple[int, int], Dict[int, float]] = {}

        # Dict[gs_name, Dict[event_id, attenuation_factor]]
        self._gs_attenuation_factors: Dict[str, Dict[int, float]] = {}

    # ── Public property for backward-compat consumers that read degraded_gs ──
    @property
    def degraded_gs(self) -> Dict[str, float]:
        """Effective (combined) attenuation factor per ground station."""
        result: Dict[str, float] = {}
        for gs, factor_map in self._gs_attenuation_factors.items():
            if factor_map:
                eff = 1.0
                for f in factor_map.values():
                    eff *= f
                result[gs] = eff
        return result

    # ── Public property for backward-compat consumers that read degraded_edges ─
    @property
    def degraded_edges(self) -> Dict[Tuple[int, int], float]:
        """Effective (combined) degradation factor per ISL edge."""
        result: Dict[Tuple[int, int], float] = {}
        for edge, factor_map in self._edge_degradation_factors.items():
            if factor_map:
                eff = 1.0
                for f in factor_map.values():
                    eff *= f
                result[edge] = eff
        return result

    # ─────────────────────────────────────────────────────────────────────────

    def _normalize_edge(self, edge: Tuple[int, int]) -> Tuple[int, int]:
        """Return the canonical (min, max) representation of an undirected edge."""
        u, v = edge
        return (u, v) if u <= v else (v, u)

    def get_edge_degradation(self, edge: Tuple[int, int]) -> float:
        """Return the effective multiplicative degradation factor for *edge*.

        Args:
            edge: An undirected ISL edge ``(u, v)`` (order does not matter).

        Returns:
            Product of all active degradation factors.  Returns ``1.0`` if no
            events are currently degrading this edge.
        """
        e = self._normalize_edge(edge)
        factor_map = self._edge_degradation_factors.get(e, {})
        if not factor_map:
            return 1.0
        result = 1.0
        for f in factor_map.values():
            result *= f
        return result

    def trigger_event(
        self,
        event_type: Union[EventType, str],
        target_id: Union[int, Tuple[int, int], str],
        duration_s: float = 300.0,
        start_time_s: float = 0.0,
        params: Optional[Dict[str, Any]] = None,
        current_time_s: Optional[float] = None,
    ) -> SimEvent:
        """Create and optionally apply a new event immediately.

        Args:
            event_type: The :class:`~satsim.events.types.EventType` to trigger.
            target_id: Satellite ID (int), ISL edge (tuple), or GS name (str).
            duration_s: Active duration of the event.
            start_time_s: Simulation time at which the event starts.
            params: Optional event-specific parameters (e.g. ``multiplier``).
            current_time_s: Current simulation time.  Defaults to ``start_time_s``.

        Returns:
            The created :class:`~satsim.events.types.SimEvent`.
        """
        if isinstance(event_type, str):
            event_type = EventType(event_type.lower())
        if params is None:
            params = {}
        if isinstance(target_id, tuple):
            target_id = self._normalize_edge(target_id)
        if current_time_s is None:
            current_time_s = start_time_s

        event = SimEvent(
            event_id=self._next_event_id,
            event_type=event_type,
            start_time_s=start_time_s,
            duration_s=duration_s,
            target_id=target_id,
            params=params,
            active=True,
        )
        self._next_event_id += 1
        self.events_history.append(event)

        if start_time_s <= current_time_s:
            self._apply_event_state(event)
        return event

    def _apply_event_state(self, event: SimEvent) -> None:
        """Apply the state modification of *event* to the injector's active sets.

        Args:
            event: The event whose effect should be materialised.
        """
        etype = event.event_type
        target = event.target_id
        eid = event.event_id

        if etype == EventType.ISL_FAILURE:
            if isinstance(target, tuple):
                self.disabled_edges.add(self._normalize_edge(target))

        elif etype == EventType.SAT_FAILURE:
            if isinstance(target, int):
                self.disabled_nodes.add(target)

        elif etype == EventType.CONGESTION:
            if isinstance(target, int):
                self.congested_nodes.add(target)
            elif isinstance(target, tuple):
                mult = event.params.get("multiplier", 2.0)
                e = self._normalize_edge(target)
                self._edge_degradation_factors.setdefault(e, {})[eid] = mult

        elif etype == EventType.BUFFER_OVERFLOW:
            if isinstance(target, int):
                self.buffer_overflow_nodes.add(target)

        elif etype == EventType.WEATHER_ATTENUATION:
            if isinstance(target, str):
                factor = event.params.get("attenuation_factor", 0.5)
                self._gs_attenuation_factors.setdefault(target, {})[eid] = factor

        elif etype == EventType.SOLAR_INTERFERENCE:
            if isinstance(target, tuple):
                mult = event.params.get("multiplier", 3.0)
                e = self._normalize_edge(target)
                self._edge_degradation_factors.setdefault(e, {})[eid] = mult

        elif etype == EventType.GS_CONGESTION:
            if isinstance(target, str):
                self.gs_congestion.add(target)

        elif etype == EventType.LINK_DEGRADATION:
            if isinstance(target, tuple):
                mult = event.params.get("multiplier", 2.5)
                e = self._normalize_edge(target)
                self._edge_degradation_factors.setdefault(e, {})[eid] = mult

        elif etype == EventType.RECOVERY:
            self._process_recovery(target, event)

    def _process_recovery(
        self,
        target: Union[int, Tuple[int, int], str],
        recovery_event: SimEvent,
    ) -> None:
        """Restore a target to its un-failed state (idempotent).

        Args:
            target: The target to recover (sat ID, edge, or GS name).
            recovery_event: The RECOVERY event that triggered this call.
        """
        recovery_event.recovered_at_s = recovery_event.start_time_s

        if isinstance(target, int):
            self.disabled_nodes.discard(target)
            self.congested_nodes.discard(target)
            self.buffer_overflow_nodes.discard(target)
        elif isinstance(target, tuple):
            edge = self._normalize_edge(target)
            self.disabled_edges.discard(edge)
            # Remove all event contributions for this edge (full recovery).
            self._edge_degradation_factors.pop(edge, None)
        elif isinstance(target, str):
            self._gs_attenuation_factors.pop(target, None)
            self.gs_congestion.discard(target)

        # Mark all prior active events on the same target as recovered.
        for ev in self.events_history:
            if ev.active and ev.target_id == target and ev.event_id != recovery_event.event_id:
                ev.active = False
                ev.recovered_at_s = recovery_event.start_time_s

    def sync_active_state(self, t_s: float) -> None:
        """Rebuild all active-state sets from scratch for time *t_s*.

        This is called at the start of each ``step()`` to ensure expiry is
        applied correctly even if events were triggered out-of-order.

        Args:
            t_s: Current simulation time in seconds.
        """
        self.disabled_nodes.clear()
        self.disabled_edges.clear()
        self._edge_degradation_factors.clear()
        self.congested_nodes.clear()
        self.buffer_overflow_nodes.clear()
        self._gs_attenuation_factors.clear()
        self.gs_congestion.clear()

        for ev in self.events_history:
            if ev.active and ev.start_time_s <= t_s <= ev.end_time_s:
                self._apply_event_state(ev)

    def step(
        self,
        t_s: float,
        current_graph: Optional[nx.Graph] = None,
        num_nodes: int = 100,
        gs_names: Optional[List[str]] = None,
        dt_s: float = 5.0,
    ) -> List[SimEvent]:
        """Advance injector state to *t_s*, expiring old events and scheduling new ones.

        Args:
            t_s: Current simulation time in seconds.
            current_graph: Live ISL graph (used to pick realistic edge targets).
            num_nodes: Total satellite count (fallback when graph is empty).
            gs_names: Ground station name list (used for GS-related event targets).
            dt_s: Timestep duration in seconds (used to compute Poisson probability).

        Returns:
            List of events triggered this step (including auto-recovery events).
        """
        triggered_now: List[SimEvent] = []

        # 1. Rebuild active state from history for t_s.
        self.sync_active_state(t_s)

        # 2. Auto-expire events that have passed their end time (A3 contract).
        for ev in list(self.events_history):
            if ev.active and ev.event_type != EventType.RECOVERY:
                if t_s >= ev.end_time_s:
                    rec_ev = self.trigger_event(
                        event_type=EventType.RECOVERY,
                        target_id=ev.target_id,
                        duration_s=0.0,
                        start_time_s=t_s,
                        params={"restored_event_id": ev.event_id},
                        current_time_s=t_s,
                    )
                    triggered_now.append(rec_ev)

        # 3. Stochastic event arrival.
        prob_per_step = (self.config.failure_rate_per_hour / 3600.0) * dt_s
        if self.rng.random() < prob_per_step and self.config.enabled_types:
            etype_str = str(self.rng.choice(self.config.enabled_types))
            etype = EventType(etype_str.lower())

            target: Union[int, Tuple[int, int], str] = 0
            if etype in {EventType.ISL_FAILURE, EventType.LINK_DEGRADATION, EventType.SOLAR_INTERFERENCE}:
                if current_graph is not None and current_graph.number_of_edges() > 0:
                    edges = list(current_graph.edges())
                    target = edges[int(self.rng.integers(0, len(edges)))]
                else:
                    u = int(self.rng.integers(0, num_nodes))
                    v = (u + 1) % num_nodes
                    target = (u, v)
            elif etype in {EventType.WEATHER_ATTENUATION, EventType.GS_CONGESTION}:
                target = str(self.rng.choice(gs_names)) if gs_names else "GS_London"
            else:
                target = int(self.rng.integers(0, num_nodes))

            duration = float(self.rng.uniform(60.0, 600.0))
            ev = self.trigger_event(
                event_type=etype,
                target_id=target,
                duration_s=duration,
                start_time_s=t_s,
                current_time_s=t_s,
            )
            triggered_now.append(ev)

        return triggered_now

    def query_log(
        self,
        start_time_s: float = 0.0,
        end_time_s: Optional[float] = None,
        event_type: Optional[Union[EventType, str]] = None,
        target_id: Optional[Any] = None,
    ) -> List[SimEvent]:
        """Return timestamped events matching the given filters.

        Args:
            start_time_s: Exclude events that started before this time.
            end_time_s: Exclude events that started after this time.
            event_type: Filter to a specific :class:`~satsim.events.types.EventType`.
            target_id: Filter to a specific target ID.

        Returns:
            Filtered list of :class:`~satsim.events.types.SimEvent` objects.
        """
        if isinstance(event_type, str):
            event_type = EventType(event_type.lower())
        if target_id is not None and isinstance(target_id, tuple):
            target_id = self._normalize_edge(target_id)

        return [
            ev for ev in self.events_history
            if ev.start_time_s >= start_time_s
            and (end_time_s is None or ev.start_time_s <= end_time_s)
            and (event_type is None or ev.event_type == event_type)
            and (target_id is None or ev.target_id == target_id)
        ]

    def apply_to_graph(self, graph: nx.Graph) -> nx.Graph:
        """Return a modified copy of *graph* with active event state applied.

        Node failure (A1): Failed satellite nodes remain in the graph but
        have all incident edges removed and ``is_active`` set to ``False``.
        This guarantees stable ``data.x`` shapes across all timesteps and
        ensures recovery does not restore stale edge attributes.

        Edge degradation (A2): Multiple concurrent events on the same edge
        multiply their factors together via :meth:`get_edge_degradation`.

        Args:
            graph: The live ISL graph produced by ``ISLManager`` for this
                timestep.  This graph is not mutated.

        Returns:
            A modified copy with failures and degradation applied.
        """
        H = graph.copy()

        # Annotate all nodes with is_active=True by default.
        for node in H.nodes():
            H.nodes[node]["is_active"] = True

        # Isolate failed satellite nodes: remove edges, mark inactive.
        for node in self.disabled_nodes:
            if H.has_node(node):
                H.remove_edges_from(list(H.edges(node)))
                H.nodes[node]["is_active"] = False

        # Remove explicitly disabled ISL edges.
        for u, v in self.disabled_edges:
            if H.has_edge(u, v):
                H.remove_edge(u, v)

        # Apply multiplicative degradation (product of all active factors).
        for edge, factor_map in self._edge_degradation_factors.items():
            u, v = edge
            if H.has_edge(u, v) and factor_map:
                eff_factor = 1.0
                for f in factor_map.values():
                    eff_factor *= f
                H[u][v]["delay_ms"] = H[u][v].get("delay_ms", 0.0) * eff_factor
                H[u][v]["degradation_factor"] = eff_factor

        # Annotate healthy edges with baseline degradation factor.
        for u, v in H.edges():
            e = self._normalize_edge((u, v))
            if "degradation_factor" not in H[u][v]:
                H[u][v]["degradation_factor"] = 1.0

        return H
