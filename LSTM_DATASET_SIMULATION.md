# LSTM Dataset — Complete Simulation & Data Reference

> **Scope**: Everything about how the LSTM training dataset (`datasets/lstm_all_scenarios.csv`)
> is generated — the constellation geometry, orbital algorithms, ISL topology, traffic models,
> routing, event injection, metric computation, and the congestion score formula.
> Every number and formula here is sourced directly from the production simulation code.

---

## Table of Contents

1. [Dataset at a Glance](#1-dataset-at-a-glance)
2. [End-to-End Pipeline Overview](#2-end-to-end-pipeline-overview)
3. [Stage 1 — Constellation Geometry (Walker-Delta)](#3-stage-1--constellation-geometry)
4. [Stage 2 — Orbital Propagation Algorithm (Keplerian + Newton-Raphson)](#4-stage-2--orbital-propagation-algorithm)
5. [Stage 3 — ISL Topology Construction](#5-stage-3--isl-topology-construction)
6. [Stage 4 — Traffic Injection (13 Scenarios, 10 Algorithms)](#6-stage-4--traffic-injection)
7. [Stage 5 — Event Injection (Failures, Degradations, Overloads)](#7-stage-5--event-injection)
8. [Stage 6 — Routing Algorithm (Dijkstra)](#8-stage-6--routing-algorithm)
9. [Stage 7 — Metrics Collection & congestion_score Formula](#9-stage-7--metrics-collection)
10. [CSV Schema — Every Column Explained](#10-csv-schema)
11. [Feature Selection — From CSV to 24 LSTM Inputs](#11-feature-selection)
12. [Sliding Window Construction — How LSTM Samples Are Built](#12-sliding-window-construction)
13. [Time-Aware Train/Val/Test Split](#13-time-aware-split)
14. [Reproducibility & Seeding](#14-reproducibility)

---

## 1. Dataset at a Glance

| Property | Value |
|---|---|
| **File** | `datasets/lstm_all_scenarios.csv` (330 MB) |
| **Also available as** | `datasets/lstm_all_scenarios.parquet` (59 MB) |
| **Total rows** | **936,000** |
| **Formula** | 100 satellites x 13 scenarios x 720 timesteps |
| **Row meaning** | One satellite at one timestep in one scenario |
| **Timestep duration** | 5 seconds (configurable via `SimConfig.timestep_seconds`) |
| **Total simulated time** | 720 x 5 s = **3,600 seconds (1 hour) per scenario** |
| **LSTM features (after audit)** | **24** |
| **LSTM target** | `congestion_score(t+1)` — predicted one step ahead |

---

## 2. End-to-End Pipeline Overview

Each of the 13 scenarios runs this pipeline independently:

```
Walker-Delta geometry (static, computed once)
        |
        v
For each timestep t = 0 to 719:
    +----------------------------------------------------------+
    | 1. Keplerian propagator -> pos_eci, vel_eci, pos_ecef   |
    | 2. ISLManager.update_grid_topology -> NetworkX graph    |
    | 3. GroundStationManager -> gs_contacts (visibility)     |
    | 4. EventInjector.step -> apply active events to graph   |
    | 5. TrafficProfile.generate_flows -> packet flows        |
    | 6. DijkstraRouter.route_flow -> path, delay, delivery   |
    | 7. MetricsCollector.collect_step -> per-sat metrics     |
    | 8. Append 100 rows (one per satellite) to trace         |
    +----------------------------------------------------------+
        |
        v
Export trace -> CSV (100 x 720 = 72,000 rows per scenario)
        |
        v
Concatenate all 13 scenarios -> 936,000 rows total
```

Source: `satsim/sim/engine.py`

---

## 3. Stage 1 — Constellation Geometry

**Algorithm**: Walker-Delta constellation pattern `53deg: 100/10/1`

Source: `satsim/orbital/constellation.py`

### Parameters

| Parameter | Value |
|---|---|
| Satellites | 100 |
| Orbital planes | 10 |
| Satellites per plane | 10 |
| Altitude | 550 km |
| Inclination | 53 degrees (Starlink-like) |
| Orbit type | Circular (eccentricity = 0) |
| Phasing factor (f) | 1 |

### Initial Orbital Elements per Satellite

For plane `p` (0-9), satellite `s` (0-9):

```
a    = 6371.0 + 550.0 = 6921.0 km          (semi-major axis)
e    = 0.0                                  (circular orbit)
inc  = radians(53 degrees)
RAAN = p x (2*pi / 10)                      (evenly spaced orbital planes)
M0   = (s x 2*pi/10 + p x 2*pi/100) mod 2*pi   (phased mean anomaly)
```

These elements are **computed once at startup** and stored per satellite.
The propagator advances each satellite's position from these elements at every timestep.

---

## 4. Stage 2 — Orbital Propagation Algorithm

**Algorithm**: Keplerian Analytical Propagator with Newton-Raphson solver

Source: `satsim/orbital/propagation.py`

> An SGP4 fallback interface also exists but uses Keplerian as its backend.

### Step-by-Step at Each Timestep `t_s`

**Step 1 — Advance Mean Anomaly**
```
dt = t_s - epoch_0
n  = sqrt(mu / a^3)        mean motion [rad/s], mu = 398600.4418 km^3/s^2
M  = (M0 + n*dt) mod 2*pi  mean anomaly [rad]
```

**Step 2 — Solve Kepler's Equation (Newton-Raphson)**
```
Equation:  M = E - e*sin(E)        (Eccentric Anomaly E is unknown)
Iteration: E_new = E - (E - e*sin(E) - M) / (1 - e*cos(E))
Tolerance: |delta_E| < 1e-10,  max 100 iterations
For e=0 (circular): E = M  (no iteration needed)
```

**Step 3 — Compute True Anomaly and Radius**
```
sin_nu = (sqrt(1-e^2)*sin(E)) / (1 - e*cos(E))
cos_nu = (cos(E) - e)         / (1 - e*cos(E))
nu     = arctan2(sin_nu, cos_nu)    [true anomaly]
r      = a*(1 - e*cos(E))           [orbital radius in km]
```

**Step 4 — Orbital Plane Coordinates**
```
u      = arg_perigee + nu
x_orb  = r*cos(u)
y_orb  = r*sin(u)
z_orb  = 0
vx_orb = (x_orb*h*e)/(r*p) - (h/r)*sin(u)
vy_orb = (y_orb*h*e)/(r*p) + (h/r)*cos(u)
where h = sqrt(mu*p),  p = a*(1-e^2)
```

**Step 5 — Rotate to Earth-Centered Inertial (ECI)**
```
Active Euler rotation (right-hand, Z-X ordering):
R = R_z(RAAN) @ R_x(inc)

r_ECI = R @ r_orb    -> (pos_eci_x, pos_eci_y, pos_eci_z)  [km]
v_ECI = R @ v_orb    -> (vel_eci_x, vel_eci_y, vel_eci_z)  [km/s]
```

**Step 6 — Rotate ECI to ECEF (Earth rotation)**
```
theta  = omega_E * t_s         omega_E = 7.2921159e-5 rad/s
R_ecef = [[cos(theta),  sin(theta), 0],
          [-sin(theta), cos(theta), 0],
          [0,           0,          1]]

r_ECEF = R_ecef @ r_ECI
v_ECEF = (R_ecef @ v_ECI) - (omega_E_vec x r_ECEF)   (Coriolis correction)
```

### Output per Satellite per Timestep

```
pos_eci_x, pos_eci_y, pos_eci_z    [km]    -> 3 LSTM features (KEPT)
vel_eci_x, vel_eci_y, vel_eci_z    [km/s]  -> 3 LSTM features (KEPT)
pos_ecef_x, pos_ecef_y, pos_ecef_z [km]    -> EXCLUDED (exact duplicate of ECI after audit)
```

---

## 5. Stage 3 — ISL Topology Construction

**Algorithm**: Walker-Delta fixed grid — intra-plane + inter-plane nearest-neighbour

Source: `satsim/topology/`

At every timestep the `ISLManager` rebuilds the ISL graph:
- Each satellite connects to its **same-plane neighbour** (plus/minus 1 in orbital phase)
- Each satellite connects to its **adjacent-plane neighbour** (same index in adjacent plane)
- Maximum degree = **4** ISL links per satellite
- Edge attributes: `distance_km`, `delay_ms` (speed-of-light), `link_type`

```
delay_ms = distance_km / 299792.458 * 1000    (speed of light = 299,792.458 km/s)
```

The result is a **NetworkX Graph** with 100 nodes and up to 380 edges.
Events (Stage 5) can remove edges from this graph before routing.

---

## 6. Stage 4 — Traffic Injection

Source: `satsim/traffic/profiles.py`

Each scenario uses a **specific traffic generation algorithm**.
All use `numpy.random.Generator` seeded from `SimConfig.seed` for reproducibility.

### Scenario to Algorithm Mapping

| Scenario | Algorithm | Key Parameters |
|---|---|---|
| `low_load` | **Poisson constant-rate** | mean_rate = 2 pkts/s |
| `medium_load` | **Poisson constant-rate** | mean_rate = 10 pkts/s |
| `high_load` | **Poisson constant-rate** | mean_rate = 40 pkts/s |
| `peak_load` | **Poisson constant-rate** | mean_rate = 100 pkts/s |
| `burst` | **ON/OFF Poisson** | 15% ON prob; ON=30x, OFF=0.1x rate |
| `flash_crowd` | **Poisson + surge window** | Middle 33-66% duration: rate x20, 80% traffic to satellite 0 |
| `hotspot` | **Poisson + geographic focus** | 80% of flows to satellites 0-4 |
| `random_traffic` | **Uniform random** | Uniform arrival times, sizes, rates |
| `self_similar` | **Poisson arrivals + Pareto durations** | lambda=15/s; Pareto alpha=1.5, heavy tails, H>0.5 |
| `mixed` | **Composite (Low + Burst + Hotspot)** | Three profiles superimposed |
| `failures` | Constant-rate + EventInjector | SAT_FAILURE + ISL_FAILURE events |
| `weather` | Constant-rate + EventInjector | SOLAR_INTERFERENCE + WEATHER_ATTENUATION |
| `congestion_stress` | Peak-rate + EventInjector | CONGESTION + BUFFER_OVERFLOW injected |

### Constant-Rate Poisson (low/medium/high/peak)

```python
expected_flows = duration * num_nodes * (mean_rate / 50.0)
n_flows = Poisson(expected_flows)

sources      = Uniform(0, 99)
destinations = (sources + Uniform(1, 99)) % 100    # src != dst guaranteed

priority = choice(['high','medium','low'], p=[0.1, 0.3, 0.6])
duration = Uniform(1.0, 5.0) seconds
rate     = Uniform(5.0, 20.0) packets/sec
```

### ON/OFF Burst Traffic

```python
# Each 1-second window: ON with 15% probability, OFF otherwise
is_burst   = Uniform() < 0.15
multiplier = 30.0 if is_burst else 0.1    # 300x contrast ratio
n_flows    = Poisson(multiplier)
# Burst rates: Uniform(10, 50) pkts/sec during ON windows
```

### Flash Crowd

```python
surge_start = t_start + duration * 0.33
surge_end   = t_start + duration * 0.66
# During surge: n_flows = Poisson(20.0), outside: Poisson(1.0)
# During surge: 80% of destinations forced to satellite 0
is_target    = in_surge AND (Uniform() < 0.8)
destination  = 0 if is_target else random
```

### Hotspot

```python
hotspot_nodes = [0, 1, 2, 3, 4]    # 5 fixed geographic hotspot satellites
is_hotspot    = Uniform() < 0.80
destination   = hotspot_nodes[random] if is_hotspot else random
# 3 flows/sec on average (Poisson(3.0))
```

### Self-Similar (Pareto Heavy Tails — Internet-like)

```python
# Poisson arrival process:
inter_arrivals = Exponential(1/15)    # 15 arrivals/sec aggregate
start_times    = cumsum(inter_arrivals)

# Heavy-tailed flow durations:
# alpha=1.5 is in (1,2) => infinite variance => long-range dependence (Hurst H > 0.5)
duration = (Pareto(alpha=1.5) + 1.0) * 1.0    clipped to [0.5, 60] sec
```

### Mixed (Composite)

```python
f_low     = LowTrafficProfile.generate_flows(...)     # seed
f_burst   = BurstTrafficProfile.generate_flows(...)   # seed+1
f_hotspot = HotspotProfile.generate_flows(...)        # seed+2
all_flows = f_low + f_burst + f_hotspot
all_flows.sort(key=lambda x: x.start_time_s)
```

---

## 7. Stage 5 — Event Injection

**Algorithm**: Poisson-scheduled stochastic event injector with multiplicative degradation stacking

Source: `satsim/events/injector.py`

### Event Types

| Event Type | Effect on Graph | Effect on Metrics |
|---|---|---|
| `SAT_FAILURE` | Removes all incident edges; is_active=False | failure_indicator=1; sat isolated |
| `ISL_FAILURE` | Removes specific edge from graph | Routing detours; higher hop count |
| `CONGESTION` | Adds node to congested_nodes set | is_congested=1; +0.5 to congestion_score |
| `BUFFER_OVERFLOW` | Adds node to buffer_overflow_nodes | is_overflow=1; +0.5 to congestion_score |
| `LINK_DEGRADATION` | Multiplies degradation_factor on edge | Higher BER, lower SNR |
| `SOLAR_INTERFERENCE` | Link degradation (stacks multiplicatively) | Same as LINK_DEGRADATION |
| `WEATHER_ATTENUATION` | Ground station attenuation factor | Reduced GS throughput |
| `RECOVERY` | Removes from disabled sets | Auto-expires at start_time + duration |

### Multiplicative Degradation Stacking

When two events affect the same ISL edge simultaneously:
```
effective_degradation = factor_A * factor_B    (each event contributes independently)

Recovering event B removes only B's factor:
effective_degradation = factor_A               (A's degradation intact)
```

This prevents the "snapshot corruption" problem where recovering B would accidentally
restore A's degraded state.

### event_flags Bitmask (stored in CSV)

```
bit 0 = 1  ->  satellite failure (SAT_FAILURE active)
bit 1 = 1  ->  congestion event (CONGESTION active)
bit 2 = 1  ->  buffer overflow (BUFFER_OVERFLOW active)
bit 3 = 1  ->  link degradation (any incident ISL has degradation_factor > 1.0)

Example: event_flags = 6 = binary 0110 -> congestion + buffer overflow simultaneously
```

### Node Failure Isolation

A failed satellite does NOT disappear from the graph — its node stays, but all incident
edges are removed. When the event expires, ISL edges reappear automatically at the next
`ISLManager.update_grid_topology` call (fresh orbital geometry). No stale attributes.

---

## 8. Stage 6 — Routing Algorithm

**Algorithm**: Dijkstra Shortest Path (weighted by `delay_ms`)

Source: `satsim/routing/baseline.py`

```python
path = networkx.shortest_path(graph, source=src, target=dst, weight="delay_ms")
```

### Key Properties

- Re-computed **every timestep** on the live post-event graph
- No cached routing tables (hence routing_table_age = 0, routing_changes = 0 always)
- Weight = `delay_ms` (propagation delay, not hop count)
- Unroutable flows (isolated satellite) are dropped: `is_delivered = False`

### Per-Flow Output Used by MetricsCollector

```
path             [sat_a, sat_b, sat_c, ...]   satellite hop sequence
prop_delay_ms  = sum(delay_ms for each edge in path)
queuing_ms     = sum(0.01 * queue_length for each intermediate node)
e2e_delay_ms   = prop_delay_ms + queuing_ms
is_delivered   = True / False
```

---

## 9. Stage 7 — Metrics Collection

**Source**: `satsim/metrics/collector.py`

Constants used: `buffer_capacity_packets = 1000`, `link_capacity_kbps = 10,000 kbps`

After routing all packet flows, MetricsCollector computes per-satellite values:

### All Per-Satellite Metric Formulas

**queue_length** (`q_len`):
```
q_len = total packets forwarded through this satellite as intermediate node this timestep
```

**queue_occupancy**:
```
q_occ = min(1.0, q_len / 1000)
```

**buffer_utilization**:
```
# 512-byte payload + 64-byte allocation header per packet
buf_util = min(1.0, (q_len * 576) / (1000 * 1500))
         = min(1.0, q_len * 576 / 1,500,000)
```

**traffic_load**:
```
sat_capacity_kbps = 10,000 * 4 = 40,000    (4 ISL links * link_capacity)
node_kbps         = bytes_delivered * 8 / 1000 / dt_s
traffic_load      = min(1.0, 0.70 * (node_kbps / 40,000) + 0.30 * q_occ)
```

**cpu_utilization**:
```
pkt_capacity    = (10,000 * 1000/8/512) * 5   (pkts per 5s window at full link rate)
max_cpu_rate    = pkt_capacity * 4
cpu_util        = min(1.0, 0.02 + 0.68*(q_len/max_cpu_rate) + 0.30*(node_kbps/40,000))
```

**memory_utilization**:
```
active_neighbors = count of ISL neighbors that are NOT in disabled_nodes AND is_active=True
mem_util = min(1.0, 0.08 + 0.72*buf_util + 0.20*(active_neighbors/degree))
```

**avg_isl_delay_ms**:
```
avg_isl_delay_ms = mean(delay_ms for each incident ISL edge)
```

**link_utilization** (per-node average of incident ISL utilizations):
```
e_util_per_edge  = min(1.0, edge_packet_count / pkt_capacity)
avg_inc_util     = mean(e_util_per_edge for each of 4 incident edges)
```

**end_to_end_delay**:
```
nd_delay = total_e2e_delay_ms_for_flows_through_node / count_of_flows_through_node
         = 0.0 if no flows passed through this satellite
```

**throughput**:
```
nd_tput = bytes_delivered_through_node * 8 / 1000 / dt_s   [kbps]
```

---

### THE LSTM TARGET: congestion_score Formula

```python
congestion_score = (
    0.35 * q_occ          # queue occupancy pressure        [0.0, 1.0]
  + 0.35 * avg_inc_util   # average incident link load      [0.0, 1.0]
  + 0.30 * traffic_load   # composite traffic pressure      [0.0, 1.0]
  + 0.50 * is_congested   # CONGESTION event active?   0 or 1
  + 0.50 * is_overflow    # BUFFER_OVERFLOW event?     0 or 1
)
```

| Component | Weight | Meaning |
|---|---|---|
| `q_occ` | 35% | How full is the packet buffer? |
| `avg_inc_util` | 35% | How saturated are the ISL links? |
| `traffic_load` | 30% | Throughput + queue composite pressure |
| `is_congested` | +0.50 | Injected CONGESTION event flag |
| `is_overflow` | +0.50 | Injected BUFFER_OVERFLOW event flag |

**Normal range (no events)**: 0.0 to 1.0  
**With events**: up to 2.0 (both flags = 1 on top of fully saturated network)  
**TargetScaler** (StandardScaler) normalises this range before LSTM training.

---

## 10. CSV Schema — Every Column Explained

### Indexing Columns (excluded from LSTM)

| Column | Type | Description |
|---|---|---|
| `scenario` | str | Scenario name e.g. "low_load" |
| `seed` | int | Random seed for reproducibility |
| `satellite_id` | int 0-99 | Satellite identifier |
| `timestep` | int 0-719 | Simulation timestep index |
| `simulation_time_s` | float | timestep * 5.0 seconds |

### Orbital Features (from Keplerian propagator)

| Column | Unit | LSTM? | Reason |
|---|---|---|---|
| `pos_eci_x` | km | YES | ECI X position |
| `pos_eci_y` | km | YES | ECI Y position |
| `pos_eci_z` | km | YES | ECI Z position |
| `vel_eci_x` | km/s | YES | ECI X velocity |
| `vel_eci_y` | km/s | YES | ECI Y velocity |
| `vel_eci_z` | km/s | YES | ECI Z velocity |
| `pos_ecef_x` | km | NO | Exact duplicate of ECI (r=1.0) |
| `pos_ecef_y` | km | NO | Exact duplicate of ECI (r=1.0) |
| `pos_ecef_z` | km | NO | Exact duplicate of ECI (r=1.0) |

### Network State Features (from MetricsCollector)

| Column | Range | LSTM? | Reason if excluded |
|---|---|---|---|
| `is_active` | 0 or 1 | YES | — |
| `buffer_utilization` | [0,1] | YES | — |
| `degree` | 0-4 | YES | — |
| `avg_isl_delay_ms` | float | YES | — |
| `queue_length` | int | NO | Exact duplicate of queue_occupancy (r=1.0) |
| `queue_occupancy` | [0,1] | YES | — |
| `end_to_end_delay` | float | YES | — |
| `throughput` | float | YES | — |
| `link_utilization` | [0,1] | YES | — |
| `traffic_load` | [0,1] | YES | — |
| `cpu_utilization` | [0,1] | YES | — |
| `memory_utilization` | [0,1] | YES | — |
| `routing_table_age` | float | YES | Always 0.0 (placeholder) |
| `routing_changes_in_window` | int | YES | Always 0 (placeholder) |
| `event_flags` | bitmask | YES | — |
| `neighbor_count` | int | NO | Exact duplicate of degree (r=1.0) |
| `node_degree` | int | NO | Exact duplicate of degree (r=1.0) |
| `failure_indicator` | 0 or 1 | NO | Exact duplicate of is_active (r=1.0) |

### Target Column

| Column | Range | Description |
|---|---|---|
| `congestion_score` | [0.0, ~2.0] | Composite congestion index. **LSTM predicts this at t+1. Strictly excluded from inputs.** |

---

## 11. Feature Selection — From CSV to 24 LSTM Inputs

`LEOLSTMDataset.audit_features()` in `satsim/lstm/lstm_dataset.py` runs a Pearson correlation audit:

1. Compute pairwise Pearson r for all numeric, non-index, non-target columns
2. Flag any pair with |r| > 0.99999 as an exact duplicate
3. Remove the second column of each exact-duplicate pair (keep first)
4. Save audit results to `artifacts/lstm/feature_audit.csv`

### Removed Exact Duplicates (5 columns removed)

| Removed Column | Kept Column | |r| |
|---|---|---|
| `pos_ecef_x` | `pos_eci_x` | ~1.0 |
| `pos_ecef_y` | `pos_eci_y` | ~1.0 |
| `pos_ecef_z` | `pos_eci_z` | ~1.0 |
| `neighbor_count` | `degree` | 1.0 |
| `node_degree` | `degree` | 1.0 |
| `failure_indicator` | `is_active` | 1.0 |
| `queue_length` | `queue_occupancy` | 1.0 |

### Final 24 LSTM Input Features

| # | Feature | Feature Group |
|---|---|---|
| 1 | `pos_eci_x` | Orbital state |
| 2 | `pos_eci_y` | Orbital state |
| 3 | `pos_eci_z` | Orbital state |
| 4 | `vel_eci_x` | Orbital state |
| 5 | `vel_eci_y` | Orbital state |
| 6 | `vel_eci_z` | Orbital state |
| 7 | `is_active` | System health |
| 8 | `buffer_utilization` | Buffer/Queue |
| 9 | `degree` | Link topology |
| 10 | `avg_isl_delay_ms` | Link topology |
| 11 | `queue_occupancy` | Buffer/Queue |
| 12 | `end_to_end_delay` | Traffic |
| 13 | `throughput` | Traffic |
| 14 | `link_utilization` | Link topology |
| 15 | `traffic_load` | Traffic |
| 16 | `cpu_utilization` | System health |
| 17 | `memory_utilization` | System health |
| 18 | `routing_table_age` | Routing dynamics |
| 19 | `routing_changes_in_window` | Routing dynamics |
| 20 | `event_flags` | Routing dynamics |
| 21-24 | remaining non-collinear columns | Other telemetry |

> **CRITICAL**: `congestion_score` is strictly excluded from LSTM inputs.
> A programmatic assertion (`assert t_end < t_target`) fires at every window creation
> to guarantee no target leakage is possible.

---

## 12. Sliding Window Construction

Source: `satsim/lstm/lstm_dataset.py` — `build_time_aware_sequences()`

Windows are created **independently per (scenario, seed, satellite_id) group**
so sequences never cross satellite or scenario boundaries.

```python
for idx in range(0, n_timesteps - window_size, stride=1):

    X      = feature_matrix[idx : idx + 30]      # shape [30, 24]  -- the input
    y      = congestion_score[idx + 30]           # real score at t+1 from CSV
    y_curr = congestion_score[idx + 29]           # last score in window (for persistence baseline)

    # Enforced anti-leakage checks:
    assert input_end_t < target_t                 # input must end BEFORE target timestep
    assert target_t == input_end_t + 1            # must be exactly 1 step ahead (no gaps)
```

### Example for satellite_id=42, scenario=high_load

```
Window 0:   X=timesteps[0-29],   y=congestion_score[30]
Window 1:   X=timesteps[1-30],   y=congestion_score[31]
...
Window 689: X=timesteps[689-718], y=congestion_score[719]
```

Per satellite per scenario: 720 - 30 = **690 windows**

---

## 13. Time-Aware Train/Val/Test Split

Timesteps are split **chronologically** (no shuffling) before window creation:

```
Timesteps   0 - 503   (70%)  ->  training
Timesteps 504 - 611   (15%)  ->  validation
Timesteps 612 - 719   (15%)  ->  test
```

| Split | Timesteps | Windows/sat/scenario | Total Windows |
|---|---|---|---|
| Train | 0-503 (504 steps) | 504-30 = 474 | 474 x 100 x 13 = 615,600 |
| Val | 504-611 (108 steps) | 78 | 78 x 100 x 13 = 101,400 |
| Test | 612-719 (109 steps) | 79 | 79 x 100 x 13 = 102,700 |

**Scalers are fit on training data only:**

```python
FeatureScaler.fit(train_samples)   # StandardScaler across all [30, 24] rows in train
TargetScaler.fit(train_samples)    # StandardScaler on all y values in train

# Validation/test: transform only, never fit
x_scaled = feature_scaler.transform(x)
y_raw    = target_scaler.inverse_transform(y_pred_scaled)  # back to [0, ~2] for metrics
```

### How MSE/RMSE/MAE/R2 Are Computed

```
For each test sample:
    y_pred_scaled  = model.forward(x_scaled)
    y_pred_raw     = target_scaler.inverse_transform(y_pred_scaled)   # [0, ~2] range
    y_true_raw     = sample.y                                          # directly from CSV

MSE  = mean((y_pred_raw - y_true_raw)^2)
RMSE = sqrt(MSE)
MAE  = mean(|y_pred_raw - y_true_raw|)
R2   = 1 - SS_residual / SS_total
```

**Achieved test set performance**:  
MSE = 0.002902, RMSE = 0.053866, MAE = 0.036090, R2 = 0.88839

---

## 14. Reproducibility

Every simulation component is seeded independently:

| Component | Algorithm | Seeding |
|---|---|---|
| `WalkerDeltaConstellation` | Deterministic geometry | No randomness |
| `KeplerianPropagator` | Newton-Raphson | No randomness |
| `ISLManager` | Proximity-based grid | No randomness |
| `TrafficProfile` | Poisson/Pareto/Uniform | `numpy.random.default_rng(seed)` |
| `EventInjector` | Poisson event scheduler | `numpy.random.default_rng(seed)` |
| `DijkstraRouter` | NetworkX shortest path | No randomness |
| `MetricsCollector` | Pure arithmetic | No randomness |

Running `python -m satsim.cli.generate_dataset --seed 42` twice produces
**byte-identical** `lstm_all_scenarios.csv` files.

---

*All formulas and parameters in this document are sourced directly from:*
- `satsim/sim/engine.py` — simulation loop orchestration
- `satsim/orbital/constellation.py` + `propagation.py` — orbital mechanics
- `satsim/topology/` — ISL graph construction
- `satsim/traffic/profiles.py` — all 10 traffic generation algorithms
- `satsim/events/injector.py` — stochastic event injection
- `satsim/routing/baseline.py` — Dijkstra routing
- `satsim/metrics/collector.py` — all metric formulas including congestion_score
- `satsim/lstm/lstm_dataset.py` — feature audit, sliding windows, splits
