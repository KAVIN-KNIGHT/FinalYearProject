# SatSim — LEO Mega-Constellation Network Simulator

A modular, reproducible, **research-grade Python simulator** of a 100-satellite LEO mega-constellation.
It generates three independent ML-ready dataset products from the same underlying orbital, traffic, and event dynamics:

| Dataset | Format | Description |
|---------|--------|-------------|
| **GAT** | PyTorch Geometric `.pt` per timestep | Graph snapshots of constellation topology (8 node + 4 edge features) |
| **LSTM** | `CSV` / `Parquet` per satellite×window | Synchronized multivariate sliding-window sequences |
| **PPO** | Gymnasium env `SatelliteRouting-v0` | RL next-hop routing environment |

Both GAT and LSTM datasets are exported **independently** — training the GAT does not require the LSTM export to have run, and vice versa.

---

## Quick-Start (Fresh Clone)

```bash
git clone <repo>
cd satsim
pip install -e .
python -m satsim.cli.batch_generate --scenarios low_load --duration 15.0 --seed 42
```

That's all that's needed. The command produces `datasets/low_load/` with every required sub-folder populated.

To run the full test suite:

```bash
pip install -e .[dev]
pytest
```

---

## Project Layout

```
satsim/
├── cli/                    # CLI entry points
│   ├── batch_generate.py   # Parallel batch scenario generation
│   ├── export_gat.py       # Standalone GAT export
│   ├── export_lstm.py      # Standalone LSTM export
│   └── run_scenario.py     # Single-scenario simulation
├── config/
│   ├── defaults.yaml       # Default configuration values
│   └── schema.py           # Pydantic config models
├── envs/
│   └── routing_env.py      # Gymnasium PPO routing environment
├── events/
│   ├── injector.py         # Stochastic event scheduler
│   └── types.py            # Event type enum + SimEvent dataclass
├── export/
│   ├── gat_export.py       # GAT dataset exporter
│   ├── lstm_export.py      # LSTM dataset exporter
│   └── trace_store.py      # Canonical per-timestep trace store
├── metrics/
│   └── collector.py        # 19-field per-timestep telemetry collector
├── orbital/
│   ├── constellation.py    # Walker-Delta constellation geometry
│   └── propagation.py      # Keplerian orbital propagator
├── routing/
│   └── baseline.py         # Dijkstra baseline router
├── sim/
│   ├── engine.py           # Discrete-event simulation loop
│   └── scenario_registry.py# 13-scenario canonical matrix
├── topology/
│   ├── ground_stations.py  # Ground station visibility
│   └── isl_manager.py      # ISL topology manager
└── traffic/
    ├── flows.py             # Packet flow model
    └── profiles.py          # 10 vectorized traffic profiles
```

---

## CLI Reference

### `batch_generate` — Parallel Batch Scenario Generation

```
python -m satsim.cli.batch_generate [OPTIONS]

Options:
  --scenarios TEXT       Comma-separated scenario names or 'all' (default: all)
  --seed INT             RNG seed for reproducibility (default: 42)
  --duration FLOAT       Simulation duration override in seconds (default: from config)
  --output-dir TEXT      Root output directory (default: datasets)
  --num-workers INT      Parallel worker count (-1 = all CPUs, default: -1)
```

**Example:**
```bash
python -m satsim.cli.batch_generate --scenarios low_load,failures --seed 42 --duration 3600
```

### `run_scenario` — Single Scenario

```
python -m satsim.cli.run_scenario --scenario low_load --seed 42
```

### `export_gat` — Standalone GAT Export

```
python -m satsim.cli.export_gat --scenario low_load
```

Requires `datasets/low_load/trace.json` to exist. Fails cleanly with `FileNotFoundError` otherwise.

### `export_lstm` — Standalone LSTM Export

```
python -m satsim.cli.export_lstm --scenario low_load
```

Requires `datasets/low_load/trace.json` to exist. Fails cleanly with `FileNotFoundError` otherwise.

---

## Scenario Matrix (13 Canonical Scenarios, §6)

| Folder | Traffic Profile | Event Condition |
|--------|----------------|-----------------|
| `low_load` | low | none |
| `medium_load` | medium | none |
| `high_load` | high | none |
| `peak_load` | peak | none |
| `burst` | burst | none |
| `flash_crowd` | flash crowd | none |
| `hotspot` | geographic hotspot | none |
| `random_traffic` | random | none |
| `self_similar` | self-similar/Poisson | none |
| `mixed` | mixed (all profiles) | none |
| `failures` | medium | satellite + ISL failures active |
| `weather` | medium | weather attenuation + solar interference |
| `congestion_stress` | high | congestion + buffer overflow + GS congestion |

> **Note**: `congestion_stress` replaces the undefined `ddos` folder from the original spec. It models high load plus every congestion-related event type, which captures flash-crowd / DoS-like training conditions without implying an attack-traffic model that was never specified.

---

## Dataset Layout

Every scenario produces:

```
datasets/<scenario_name>/
├── config_used.yaml          # Exact YAML config that produced this dataset
├── trace.json                # Canonical per-timestep trace (source of truth)
├── global_metrics/
│   ├── metrics.csv           # 19-field telemetry per timestep
│   └── metrics.parquet
├── gat/
│   ├── snapshot_000000.pt    # PyG Data snapshot (one per timestep)
│   └── ...
├── lstm/
│   ├── lstm_sequences.csv    # Per-satellite sliding-window sequences
│   ├── lstm_sequences.parquet
│   └── window_metadata.json  # Window/stride config and row count
└── routing_history/
    └── routes_summary.json   # Routing metrics summary
```

Batch summary logs:
```
datasets/
├── batch_run_log.json
└── batch_run_log.csv
```

---

## Config Schema

All configuration is Pydantic-validated and YAML-serializable. The top-level model is `SimConfig`.

### `SimConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seed` | `int` | `42` | Global RNG seed |
| `timestep_seconds` | `float` | `5.0` | Simulation step size |
| `duration_seconds` | `float` | `3600.0` | Total simulation duration |
| `constellation` | `ConstellationConfig` | — | Orbital plane geometry |
| `isl` | `ISLConfig` | — | Inter-satellite link parameters |
| `ground_stations` | `GroundStationsConfig` | — | Ground station placement |
| `traffic` | `TrafficConfig` | — | Traffic generation profile |
| `events` | `EventsConfig` | — | Event injection parameters |
| `export` | `ExportConfig` | — | Dataset export settings |
| `logging` | `LoggingConfig` | — | Logging verbosity |

### `ConstellationConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_satellites` | `int` | `200` | Total satellites (must be divisible by `num_planes`) |
| `num_planes` | `int` | `10` | Orbital planes |
| `altitude_km` | `float` | `550.0` | Orbital altitude |
| `inclination_deg` | `float` | `53.0` | Orbital inclination. Peak latitude equals this value |
| `eccentricity` | `float` | `0.0` | Orbital eccentricity (0 = circular) |
| `propagation` | `"keplerian"\|"sgp4"` | `"keplerian"` | Propagation model |

### `ISLConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_range_km` | `float` | `5000.0` | Maximum ISL range |
| `min_elevation_deg` | `float` | `10.0` | Minimum elevation to avoid Earth occlusion |

### `EventsConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled_types` | `List[str]` | `["isl_failure","sat_failure","congestion","weather_attenuation"]` | Active event types |
| `failure_rate_per_hour` | `float` | `0.5` | Mean Poisson event arrival rate |

**Event types:** `isl_failure`, `sat_failure`, `congestion`, `buffer_overflow`, `weather_attenuation`, `solar_interference`, `ground_station_congestion`, `link_degradation`, `recovery`

### `LSTMExportConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `window_size` | `int` | `30` | Sliding window length in timesteps |
| `stride` | `int` | `5` | Stride between consecutive windows |
| `format` | `"parquet"\|"csv"` | `"parquet"` | Output file format |

---

## LSTM Dataset Column Reference (§7)

One row per `(satellite_id, window_start_timestep)`.

| Column | Type | Description |
|--------|------|-------------|
| `satellite_id` | `int` | Satellite index (0–199) |
| `timestep` | `int` | Step within window (0 .. window_size-1) |
| `simulation_time_s` | `float` | Absolute simulation time |
| `pos_eci_x/y/z` | `float` | ECI position (km) |
| `vel_eci_x/y/z` | `float` | ECI velocity (km/s) |
| `pos_ecef_x/y/z` | `float` | ECEF position (km) |
| `is_active` | `float` | 1.0 = nominal, 0.0 = failed |
| `buffer_utilization` | `float` | Queue occupancy ∈ [0, 1] |
| `degree` | `int` | Current ISL node degree |
| `avg_isl_delay_ms` | `float` | Mean delay of active ISLs (ms) |
| `window_id` | `int` | Window index for this satellite |
| `step_in_window` | `int` | Position within the window |

> **Critical**: `is_active = 0.0` rows are **never dropped** for satellites failing mid-window. The failure indicator is the training signal — excluding it would defeat the purpose of the failures/weather/congestion_stress scenarios.

---

## GAT Dataset Schema (§8)

**Nodes** = 100 satellites. Ground stations are not currently included as graph nodes (this decision is documented here explicitly — no silent mixing of node schemas).

**Node features** `data.x` — shape `[100, 8]`:

| Index | Feature | Description |
|-------|---------|-------------|
| 0–2 | `pos_eci_x/y/z` | ECI position (km) |
| 3–5 | `vel_eci_x/y/z` | ECI velocity (km/s) |
| 6 | `buffer_utilization` | Queue occupancy ∈ [0, 1] |
| 7 | `degree` | Active ISL node degree [0..4] |

**Edges** = 380 active ISLs.
`data.edge_index` — shape `[2, 380]`.

**Edge features** `data.edge_attr` — shape `[380, 4]`:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `distance_km` | Physical ISL link distance (km) |
| 1 | `delay_ms` | Propagation + link delay (ms) |
| 2 | `link_utilization` | Bandwidth utilization fraction ∈ [0, 1] |
| 3 | `link_failure_probability` | Real-time failure probability ∈ [0, 1) |

One `.pt` file per timestep snapshot: `snapshot_{step:06d}.pt` (9,360 total snapshots).

---

## PPO Environment Spec (§9)

**ID:** `SatelliteRouting-v0`

**Observation space:** `Box(low=-1e8, high=1e8, shape=(22,), dtype=float32)`

| Indices | Content |
|---------|---------|
| 0–2 | Current node ECI position |
| 3–5 | Current node ECI velocity |
| 6–8 | Current node ECEF position |
| 9–11 | Destination node ECI position |
| 12 | Destination node ID (normalised) |
| 13 | Local buffer utilization |
| 14–17 | Neighbour ISL delays (up to 4 neighbours; 0 if fewer) |
| 18–21 | Neighbour active flags (1.0 active, 0.0 absent) |

**Action space:** `Discrete(200)` — select next-hop target satellite ID.

**Reward** (each term logged independently in `info["reward_components"]`):

| Term | Signal |
|------|--------|
| `delivery` | +1.0 on reaching destination |
| `throughput` | +0.5 on successful delivery |
| `latency` | −0.01 × edge_delay_ms per hop |
| `congestion` | 0.0 (extendable via config) |
| `loss` | −1.0 on invalid hop (no edge / failed node) |
| `hop_count` | −0.05 per hop |
| `energy` | −0.01 per hop |
| `success` | +5.0 on successful delivery |

---

## Metrics Reference (§10)

All 19 metrics computed every timestep by `MetricsCollector`:

`queue_length`, `queue_occupancy`, `buffer_utilization`, `packet_delay`, `end_to_end_delay`, `throughput`, `packet_delivery_ratio`, `packet_loss_ratio`, `jitter`, `link_utilization`, `available_bandwidth`, `used_bandwidth`, `propagation_delay`, `transmission_delay`, `ber`, `snr`, `link_lifetime`, `link_stability`, `congestion_score`

---

## Reproducibility

Running `python -m satsim.cli.batch_generate --seed 42` twice produces **byte-identical** `trace.json` and `config_used.yaml` files for every scenario. The `batch_run_log.csv` will also have identical row statistics.

---

## Coding Standards (§11)

- **Python 3.11+**, full type hints on every public function/method.
- **Pydantic** for all config and data-transfer objects — no bare `dict`s passed between modules.
- **Structured logging** (`structlog` JSON-formatted), not `print()`.
- Every module has a **module-level docstring**; every public function has a docstring with `Args:/Returns:`.
- **No silent fallbacks**: if a config value is missing or a computation can't produce a real number, raise — don't return 0/None and continue.
- Every exported dataset file is accompanied by the **exact config** that produced it.
- New functionality ships with **at least one test** that would fail if the implementation were replaced with a stub.
#   F i n a l Y e a r P r o j e c t  
 #   F i n a l Y e a r P r o j e c t  
 