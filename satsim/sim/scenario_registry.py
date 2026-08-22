"""Scenario registry for the LEO mega-constellation simulator.

Defines the canonical 13-scenario matrix reconciled from the project specification (§6).
Each scenario is a (traffic_profile, event_condition) pair covering the full spectrum of
load patterns and fault conditions required to produce robust ML training datasets.

Scenario Matrix
---------------
Folder name          Traffic profile              Event condition
low_load             low                          none
medium_load          medium                       none
high_load            high                         none
peak_load            peak                         none
burst                burst                        none
flash_crowd          flash crowd                  none
hotspot              geographic hotspot           none
random_traffic       random                       none
self_similar         self-similar/Poisson         none
mixed                mixed (all profiles)         none
failures             medium                       satellite + ISL failures
weather              medium                       weather attenuation + solar interference
congestion_stress    high                         congestion + buffer overflow + GS congestion
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from satsim.config import SimConfig

# ---------------------------------------------------------------------------
# Canonical 13-scenario matrix (§6, reconciled)
# ---------------------------------------------------------------------------
SCENARIO_MATRIX: Dict[str, Dict[str, Any]] = {
    # ── Pure-load scenarios (no injected events) ──────────────────────────
    "low_load": {
        "traffic_profile": "low",
        "failure_rate_per_hour": 0.0,
        "enabled_event_types": [],
    },
    "medium_load": {
        "traffic_profile": "medium",
        "failure_rate_per_hour": 0.0,
        "enabled_event_types": [],
    },
    "high_load": {
        "traffic_profile": "high",
        "failure_rate_per_hour": 0.0,
        "enabled_event_types": [],
    },
    "peak_load": {
        "traffic_profile": "peak",
        "failure_rate_per_hour": 0.0,
        "enabled_event_types": [],
    },
    "burst": {
        "traffic_profile": "burst",
        "failure_rate_per_hour": 0.0,
        "enabled_event_types": [],
    },
    "flash_crowd": {
        "traffic_profile": "flash_crowd",
        "failure_rate_per_hour": 0.0,
        "enabled_event_types": [],
    },
    "hotspot": {
        "traffic_profile": "hotspot",
        "failure_rate_per_hour": 0.0,
        "enabled_event_types": [],
    },
    "random_traffic": {
        "traffic_profile": "random",
        "failure_rate_per_hour": 0.0,
        "enabled_event_types": [],
    },
    "self_similar": {
        "traffic_profile": "self_similar_poisson",
        "failure_rate_per_hour": 0.0,
        "enabled_event_types": [],
    },
    "mixed": {
        "traffic_profile": "mixed",
        "failure_rate_per_hour": 0.0,
        "enabled_event_types": [],
    },
    # ── Fault-injection scenarios ─────────────────────────────────────────
    "failures": {
        "traffic_profile": "medium",
        "failure_rate_per_hour": 2.0,
        "enabled_event_types": ["sat_failure", "isl_failure", "recovery"],
        # Deterministic events scaled for 100-satellite constellation (IDs 0-99).
        "pre_seeded_events": [
            {"type": "sat_failure", "target": 5,        "start_s": 100.0,  "duration_s": 600.0},
            {"type": "sat_failure", "target": 25,       "start_s": 900.0,  "duration_s": 300.0},
            {"type": "sat_failure", "target": 50,       "start_s": 1800.0, "duration_s": 400.0},
            {"type": "isl_failure", "target": [5, 6],   "start_s": 300.0,  "duration_s": 500.0},
            {"type": "isl_failure", "target": [20, 21], "start_s": 1500.0, "duration_s": 600.0},
        ],
    },
    "weather": {
        "traffic_profile": "medium",
        "failure_rate_per_hour": 1.0,
        "enabled_event_types": ["weather_attenuation", "solar_interference", "sat_failure", "recovery"],
        "pre_seeded_events": [
            {"type": "weather_attenuation", "target": "GS_London",   "start_s": 200.0,  "duration_s": 1200.0, "params": {"attenuation_factor": 0.4}},
            {"type": "weather_attenuation", "target": "GS_Tokyo",    "start_s": 900.0,  "duration_s": 800.0,  "params": {"attenuation_factor": 0.6}},
            {"type": "solar_interference",  "target": [10, 11],     "start_s": 500.0,  "duration_s": 600.0,  "params": {"multiplier": 3.0}},
            {"type": "solar_interference",  "target": [30, 31],     "start_s": 2000.0, "duration_s": 800.0,  "params": {"multiplier": 2.5}},
            {"type": "sat_failure",         "target": 10,           "start_s": 500.0,  "duration_s": 600.0},
        ],
    },
    "congestion_stress": {
        "traffic_profile": "high",
        "failure_rate_per_hour": 1.5,
        "enabled_event_types": [
            "congestion",
            "buffer_overflow",
            "ground_station_congestion",
            "link_degradation",
            "recovery",
        ],
        "pre_seeded_events": [
            {"type": "congestion",      "target": 5,          "start_s": 100.0,  "duration_s": 600.0},
            {"type": "congestion",      "target": 30,         "start_s": 600.0,  "duration_s": 800.0},
            {"type": "buffer_overflow", "target": 15,         "start_s": 300.0,  "duration_s": 400.0},
            {"type": "buffer_overflow", "target": 50,         "start_s": 1200.0, "duration_s": 500.0},
            {"type": "gs_congestion",   "target": "GS_London","start_s": 400.0,  "duration_s": 600.0},
            {"type": "link_degradation","target": [0, 1],     "start_s": 200.0,  "duration_s": 1000.0, "params": {"multiplier": 3.0}},
        ],
    },
}

#: Ordered list of all canonical scenario names.
ALL_SCENARIOS: List[str] = list(SCENARIO_MATRIX.keys())


def validate_event_target(
    target: Any,
    num_satellites: int,
    gs_names: Optional[List[str]] = None,
) -> None:
    """Validate that an event target is strictly valid for *num_satellites*.

    Fails loudly with ValueError if target is out of bounds or invalid.

    Args:
        target: Satellite ID (int), ISL edge tuple/list, or GS name (str).
        num_satellites: Total satellite count in constellation.
        gs_names: Optional list of valid ground station names.
    """
    if isinstance(target, int):
        if not (0 <= target < num_satellites):
            raise ValueError(
                f"[STARTUP ERROR] Invalid satellite event target {target}: "
                f"must be in range [0, {num_satellites - 1}]."
            )
    elif isinstance(target, (tuple, list)):
        if len(target) != 2:
            raise ValueError(f"[STARTUP ERROR] Invalid edge target {target}: must be 2 elements.")
        u, v = int(target[0]), int(target[1])
        if not (0 <= u < num_satellites):
            raise ValueError(
                f"[STARTUP ERROR] Invalid edge target src {u}: "
                f"must be in range [0, {num_satellites - 1}]."
            )
        if not (0 <= v < num_satellites):
            raise ValueError(
                f"[STARTUP ERROR] Invalid edge target dst {v}: "
                f"must be in range [0, {num_satellites - 1}]."
            )
        if u == v:
            raise ValueError(f"[STARTUP ERROR] Invalid self-loop edge target ({u}, {v}).")
    elif isinstance(target, str):
        if gs_names is not None and target not in gs_names:
            raise ValueError(
                f"[STARTUP ERROR] Unknown ground station target '{target}'. "
                f"Valid GS choices: {gs_names}"
            )
    else:
        raise ValueError(f"[STARTUP ERROR] Unrecognized target type: {type(target)}")


def get_scenario_config(
    scenario_name: str,
    seed: int = 42,
    base_config: Optional[SimConfig] = None,
) -> SimConfig:
    """Return a fully resolved :class:`~satsim.config.SimConfig` for *scenario_name*.

    Args:
        scenario_name: One of the 13 canonical scenario names defined in
            :data:`SCENARIO_MATRIX`.
        seed: RNG seed to embed in the config.  Identical seed → identical trace.
        base_config: Optional starting config to clone and patch.  If ``None``,
            defaults are loaded from ``satsim/config/defaults.yaml``.

    Returns:
        A :class:`SimConfig` with the scenario's traffic profile, event condition,
        and seed applied.

    Raises:
        ValueError: If *scenario_name* is not in :data:`SCENARIO_MATRIX`.
    """
    name_clean = scenario_name.lower().strip()
    if name_clean not in SCENARIO_MATRIX:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Choose from: {ALL_SCENARIOS}"
        )

    config = SimConfig.load_yaml() if base_config is None else base_config.model_copy(deep=True)
    spec = SCENARIO_MATRIX[name_clean]
    config.seed = seed
    config.traffic.profile = spec["traffic_profile"]
    config.events.failure_rate_per_hour = spec["failure_rate_per_hour"]
    if spec["enabled_event_types"]:
        config.events.enabled_types = spec["enabled_event_types"]

    # Validate all pre-seeded event targets against the constellation size
    gs_names = [gs.name for gs in config.ground_stations.locations]
    for ev in spec.get("pre_seeded_events", []):
        validate_event_target(ev["target"], config.constellation.num_satellites, gs_names)

    return config
