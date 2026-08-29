# LSTM in the SatSim Project â€” A Complete A-to-Z Learning Guide

> **Scope**: Self-contained technical reference for the LSTM (Long Short-Term Memory)
> component of the **100-Satellite LEO Constellation Dynamic Routing System**.
> After reading this you will understand *why* LSTM is used, *what* data it consumes,
> *how* the model is built and trained, and *what* it produces.

---

## Table of Contents

1. [Why LSTM?](#1-why-lstm)
2. [LSTM Fundamentals â€” The Mathematics](#2-lstm-fundamentals)
3. [The Dataset â€” Source of Truth](#3-the-dataset)
4. [The 13 Simulation Scenarios](#4-the-13-scenarios)
5. [Dataset Schema â€” Every Column Explained](#5-dataset-schema)
6. [Feature Selection â€” The 24 LSTM Input Features](#6-feature-selection)
7. [Sliding Window Sequence Construction](#7-sliding-window)
8. [Time-Aware Train / Val / Test Split](#8-time-aware-split)
9. [Scaling â€” Feature and Target Normalization](#9-scaling)
10. [Model Architecture â€” LEOLSTMModel](#10-model-architecture)
11. [Hyperparameters â€” Full Reference Table](#11-hyperparameters)
12. [Training Loop â€” Step by Step](#12-training-loop)
13. [Optimizer, Scheduler, and Early Stopping](#13-optimizer)
14. [Evaluation Metrics and Results](#14-metrics)
15. [Baseline Comparisons](#15-baselines)
16. [Temporal Embedding Extraction](#16-embeddings)
17. [Downstream Usage â€” GAT + PPO Fusion](#17-downstream)
18. [Output Artifacts](#18-artifacts)
19. [Diagnostic Plots](#19-plots)
20. [Data Leakage Audit](#20-leakage-audit)
21. [Reproducibility and Seeding](#21-reproducibility)
22. [How to Run â€” CLI Reference](#22-cli)
23. [Source Files Quick Reference](#23-source-files)
24. [End-to-End Pipeline Flow Diagram](#24-flow-diagram)

---

## 1. Why LSTM?

### The Context

A **Low Earth Orbit (LEO) constellation** of 100 satellites is constantly moving.
Each satellite maintains Inter-Satellite Links (ISLs) with its neighbours, forming a
dynamic mesh network. Packets are routed through this mesh. The critical operational question:

> **"How congested will a satellite be at the *next* timestep?"**

Predicting this one step ahead enables:
- A PPO-based RL routing agent to **pre-emptively avoid** congested nodes.
- A network controller to **re-balance load** before queues overflow.

### Why Not a Feedforward Network?

Each satellite's congestion is **not independent of its history**. A satellite at 80%
buffer utilization 5 seconds ago is far more likely to be congested now than one that
was at 10%. Congestion builds, peaks, and subsides over time. A feedforward network
sees only the *current snapshot* and misses all temporal patterns.

### Why LSTM Specifically?

LSTM (Long Short-Term Memory) is a Recurrent Neural Network that:
- **Remembers** information across many past timesteps via its gated hidden state.
- **Forgets** irrelevant old information via the *forget gate*.
- **Selectively updates** memory via the *input gate*.
- **Outputs** relevant context via the *output gate*.

This gating mechanism makes it ideal for satellite telemetry sequences where congestion
can develop and decay across 30+ seconds.

---

## 2. LSTM Fundamentals â€” The Mathematics

A single LSTM cell maintains two state vectors at each timestep `t`:
- `h_t` â€” **hidden state** (short-term working memory), shape `[hidden_dim]`
- `c_t` â€” **cell state** (long-term memory), shape `[hidden_dim]`

Given input `x_t` (one row of features at timestep `t`):

```
Forget gate:  f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
Input gate:   i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
Candidate:    g_t = tanh(W_g   * [h_{t-1}, x_t] + b_g)
Output gate:  o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)

Cell update:  c_t = f_t (*) c_{t-1}  +  i_t (*) g_t
Hidden state: h_t = o_t (*) tanh(c_t)
```

Where `(*)` is element-wise multiplication.

**In plain English:**
- `f_t` â€” decides how much of the *old* memory to keep (0=forget all, 1=keep all)
- `i_t` â€” decides how much *new* candidate information to write into memory
- `g_t` â€” the new candidate information (what could be learned at this step)
- `o_t` â€” decides how much of the memory to expose as output
- `c_t` â€” the updated long-term memory (blend of old and new)
- `h_t` â€” final output for this timestep (becomes input to the next layer)

### Two-Layer Stacking

This project stacks **2 LSTM layers**. The `h_t` from Layer 1 becomes the `x_t` of Layer 2:
- **Layer 1** captures low-level patterns (immediate queue spikes, link failures).
- **Layer 2** captures higher-level patterns (gradual congestion build-up over 30 steps).

---

## 3. The Dataset â€” Source of Truth

### File Location

```
datasets/lstm_all_scenarios.csv        (330 MB, primary)
datasets/lstm_all_scenarios.parquet    ( 59 MB, faster I/O alternative)
```

### Scale

| Dimension | Value |
|---|---|
| **Total rows** | **936,000** |
| Satellites | 100 (IDs 0 to 99) |
| Scenarios | 13 |
| Timesteps per (scenario, satellite) | 720 (t = 0 to 719) |
| Formula | 100 x 13 x 720 = **936,000** |

Each row represents **one satellite at one timestep in one scenario**.

### How the Data Was Generated

1. 100 satellites placed in a Walker-Delta constellation at ~550 km altitude.
2. SGP4/SDP4 orbital propagator (`satsim/orbital/`) computes ECI/ECEF positions.
3. ISL topology dynamically updated based on proximity (`satsim/topology/`).
4. Synthetic traffic generators (`satsim/traffic/`) inject packets per scenario profile.
5. At every timestep, telemetry for all 100 satellites is written as 100 rows.
6. 720 steps x 100 satellites = 72,000 rows/scenario x 13 scenarios = **936,000 total**.

---

## 4. The 13 Simulation Scenarios

Each scenario represents a different real-world LEO network operating condition:

| # | Scenario | Traffic Regime | Key Characteristic |
|---|---|---|---|
| 1 | `low_load` | Sparse | Network mostly idle; baseline |
| 2 | `medium_load` | Nominal | Typical steady-state |
| 3 | `high_load` | Heavy | Sustained high utilization |
| 4 | `peak_load` | Stress | Near-saturation bandwidth |
| 5 | `burst` | Transient surges | Short sharp traffic spikes |
| 6 | `flash_crowd` | Localized floods | Many packets to same region |
| 7 | `hotspot` | Geographic focus | Persistent overload on one belt |
| 8 | `random_traffic` | Stochastic | Poisson-distributed arrivals |
| 9 | `self_similar` | Heavy-tailed | Bursty Internet-like traffic |
| 10 | `mixed` | Multi-modal | Multiple patterns simultaneously |
| 11 | `failures` | Outages | Random link/node failures |
| 12 | `weather` | Atmospheric | Rain fade, Doppler effects |
| 13 | `congestion_stress` | Extreme | Max stress; score near 1.0 |

Each scenario runs for **720 timesteps on all 100 satellites** (72,000 rows).
Per-scenario split: **503 train / 108 val / 109 test timesteps**.

Training across all 13 ensures generalisation from calm (`low_load`) to crisis
(`congestion_stress`) conditions.

---

## 5. Dataset Schema â€” Every Column Explained

### Indexing Columns (NOT used as model features)

| Column | Type | Description |
|---|---|---|
| `scenario` | str | Scenario name (e.g. "low_load") |
| `seed` | int | Random seed for reproducibility |
| `satellite_id` | int | Satellite index 0-99 |
| `timestep` | int | Simulation timestep 0-719 |
| `simulation_time_s` | float | Wall-clock simulation time (seconds) |

### Orbital / Positional Features

| Column | Type | Description |
|---|---|---|
| `pos_eci_x` | float32 | Earth-Centered Inertial position X (km) |
| `pos_eci_y` | float32 | ECI position Y (km) |
| `pos_eci_z` | float32 | ECI position Z (km) |
| `vel_eci_x` | float32 | ECI velocity X (km/s) |
| `vel_eci_y` | float32 | ECI velocity Y (km/s) |
| `vel_eci_z` | float32 | ECI velocity Z (km/s) |
| `pos_ecef_x/y/z` | float32 | ECEF positions â€” **EXCLUDED** (exact duplicate of ECI after audit) |

> ECI = fixed to stars (inertial). ECEF = rotates with Earth. They are linearly
> related for any satellite trajectory, so ECEF adds no new information.

### Network State Features

| Column | Type | Description |
|---|---|---|
| `is_active` | int 0/1 | 1 = operational, 0 = failed/occluded |
| `buffer_utilization` | float [0,1] | Fraction of packet buffer filled |
| `degree` | int 0-4 | Active ISL connection count |
| `avg_isl_delay_ms` | float | Mean propagation+queuing delay across ISLs (ms) |
| `queue_length` | int | Raw packet count â€” **EXCLUDED** (duplicate of queue_occupancy) |
| `queue_occupancy` | float [0,1] | Queue fill fraction |
| `end_to_end_delay` | float | Average end-to-end delivery delay (ms) |
| `throughput` | float | Packets/sec forwarded |
| `link_utilization` | float [0,1] | Average ISL bandwidth fraction in use |
| `traffic_load` | float | Offered traffic injection rate |
| `cpu_utilization` | float [0,1] | On-board processor load |
| `memory_utilization` | float [0,1] | On-board memory load |
| `routing_table_age` | float | Seconds since last routing table update |
| `routing_changes_in_window` | int | Routing updates in last observation window |
| `event_flags` | int bitmask | Bit 0=link failure, Bit 1=handover, Bit 2=reroute |

### Target Column (what LSTM predicts)

| Column | Type | Description |
|---|---|---|
| `congestion_score` | float [0,1] | **Composite congestion index**. 0=fully free, 1=fully congested. Weighted combination of buffer_utilization, link_utilization, cpu_utilization, queue_occupancy, and normalized delay. **Predicted one step ahead.** |

---

## 6. Feature Selection â€” The 24 LSTM Input Features

`LEOLSTMDataset.audit_features()` runs an automated **correlation audit**:

1. List all numeric columns except indexing columns and the target.
2. Compute pairwise Pearson correlation for all candidate pairs.
3. Flag any column with |correlation| > 0.99999 as an **exact duplicate**.
4. Remove exact duplicates (keep first of each pair).

**Excluded exact duplicates:**
- `pos_ecef_x/y/z` â€” |r| ~1.0 with ECI coordinates
- `neighbor_count` â€” |r|=1.0 with `degree`
- `failure_indicator` â€” |r|=1.0 with `is_active`
- `node_degree` â€” |r|=1.0 with `degree`
- `queue_length` â€” |r|=1.0 with `queue_occupancy`

After de-duplication, **24 features** remain. Saved to: `artifacts/lstm/feature_audit.csv`

The 24 features span:
- **Orbital state (6)**: pos_eci_x/y/z, vel_eci_x/y/z
- **Link topology (3)**: degree, avg_isl_delay_ms, link_utilization
- **Buffer/Queue (2)**: buffer_utilization, queue_occupancy
- **Traffic (3)**: traffic_load, throughput, end_to_end_delay
- **System health (3)**: cpu_utilization, memory_utilization, is_active
- **Routing dynamics (3)**: routing_table_age, routing_changes_in_window, event_flags
- **Other telemetry (4)**: remaining non-collinear numeric features

> **CRITICAL**: `congestion_score` is **strictly excluded** from all LSTM inputs.
> Enforced by programmatic assertion at every sliding window creation.

---

## 7. Sliding Window Sequence Construction

### The Core Idea

The LSTM receives a **sliding window** of 30 consecutive timesteps per satellite
and predicts the congestion score at timestep **t+1**:

```
Input window:   X(t-29, t-28, ..., t-1, t)   shape [30, 24]
Target label:   congestion_score(t+1)          scalar float
```

This is **one-step-ahead temporal prediction**.

### Why Window Size = 30?

30 timesteps at 1-second simulation resolution = **30 seconds of history**. Enough to:
- Capture a full burst event (typically 10-20 seconds in the burst scenario)
- Observe routing table update cycles
- See early warning signs of a congestion build-up before it peaks

### Stride = 1

With stride 1, consecutive windows overlap by 29 timesteps.
For a 504-timestep training portion (per satellite): 474 windows.

### The SequenceSample Dataclass

```python
class SequenceSample:
    x: np.ndarray    # shape [30, 24] â€” 30-step input window
    y: float         # congestion_score(t+1) â€” the prediction target
    y_curr: float    # congestion_score(t) â€” for Persistence Baseline only
    scenario: str
    seed: int
    satellite_id: int
    input_start_t: int   # timestep of first row in window
    input_end_t: int     # timestep of last row in window
    target_t: int        # always = input_end_t + 1
```

### Anti-Leakage Assertions (at every window)

```python
assert input_end_t < target_t           # target strictly in the future
assert target_t == input_end_t + 1      # no timestep gaps
# Failure count must be ZERO
```

### Approximate Sequence Counts

| Split | Sequences |
|---|---|
| Training (70%) | ~6,539,000 (across all 13 scenarios x 100 satellites) |
| Validation (15%) | ~1,404,000 |
| Test (15%) | ~1,417,000 |

---

## 8. Time-Aware Train / Val / Test Split

### Why Time-Aware?

A naive random split lets future timesteps appear in training â€” that is temporal
data leakage. The split must be **chronologically strict**.

### How It Works

For each unique group `(scenario, seed, satellite_id)`:

1. Sort rows by `timestep` (ascending).
2. First 70% of timesteps â†’ **training** portion.
3. Next 15% â†’ **validation** portion.
4. Remaining 15% â†’ **test** portion.
5. Sliding windows are created **within each portion only** â€” no window spans a boundary.

```
Timestep:  0 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ 503 | 504 â”€â”€â”€â”€ 611 | 612 â”€â”€ 719
                TRAIN (70%)         VAL (15%)    TEST (15%)
```

### Guarantees

- Model **never sees future data** during training.
- Validation and test represent genuinely unseen future network states.
- Split is applied per satellite trajectory, not globally.
- Scalers are fitted only on training data.

---

## 9. Scaling â€” Feature and Target Normalization

### Why Scale?

Raw features span different magnitudes:
- `pos_eci_x`: -7,000 to +7,000 km
- `buffer_utilization`: 0.0 to 1.0
- `throughput`: 0 to thousands of packets/sec

Without scaling, large-magnitude features dominate gradient updates.

### FeatureScaler â€” Normalizes Input Features

```python
class FeatureScaler:
    scaler = StandardScaler()   # zero mean, unit variance per feature

    def fit(train_samples):
        x_mat = vstack([s.x for s in train_samples])  # [N*30, 24]
        scaler.fit(x_mat)    # per-feature mean and std

    def transform(x):       # x: [30, 24] or [B, 30, 24]
        return (x - mean_feature) / std_feature
```

Saved to: `artifacts/lstm/feature_scaler.pkl`

### TargetScaler â€” Normalizes congestion_score

```python
class TargetScaler:
    scaler = StandardScaler()

    def fit(train_samples):
        all_y = [s.y for s in train_samples]
        scaler.fit(all_y)    # training congestion score mean and std

    def transform(y):        # standardize before computing MSE loss
        return (y - mean_y) / std_y

    def inverse_transform(y_scaled):   # restore to [0,1] for metrics
        return y_scaled * std_y + mean_y
```

Saved to: `artifacts/lstm/target_scaler.pkl`

> **RULE**: Both scalers are fitted **ONLY on training data**. Validation and test
> data only ever call `transform()`, never `fit()`. Refitting on val/test would be
> data leakage and would artificially inflate reported performance.

---

## 10. Model Architecture â€” LEOLSTMModel

**Source file**: `satsim/lstm/lstm_model.py`

### Layer-by-Layer Architecture

```
INPUT TENSOR: [batch_size, 30, 24]  = [B, W, F]
              (Batch x Window x Features)
                      |
    +-----------------v-----------------------------------------+
    |  nn.LSTM(                                                  |
    |    input_size  = 24,   # F features per timestep          |
    |    hidden_size = 128,  # H hidden dimension               |
    |    num_layers  = 2,    # stacked LSTM layers              |
    |    batch_first = True, # input is [B,W,F] not [W,B,F]    |
    |    dropout     = 0.2,  # applied between layer 1 and 2   |
    |  )                                                         |
    |                                                            |
    |  Layer 1: [B,30,24]  -> hidden [B,30,128]                 |
    |  Dropout(0.2) between layers                               |
    |  Layer 2: [B,30,128] -> output lstm_out [B,30,128]        |
    +----------------------------+-------------------------------+
                                 |
              Extract LAST timestep:
              h_last = lstm_out[:, -1, :]   shape: [B, 128]
                       |                              |
                       |                              |
              TEMPORAL EMBEDDING         PREDICTION HEAD
              (returned as-is)           Linear(128 -> 64)
              shape: [B, 128]            ReLU()
                                         Dropout(0.2)
                                         Linear(64 -> 1)
                                                |
                                        pred_congestion [B, 1]
                                        (standardized scale)
```

### Why the Last Timestep?

`lstm_out[:, -1, :]` is the hidden state **after processing all 30 input timesteps**.
It is the LSTM's compressed summary of the entire sequence â€” the richest representation
of everything learned from 30 historical observations.

### Prediction Head Design

`Linear(128->64) -> ReLU -> Dropout(0.2) -> Linear(64->1)`:
- The 64-unit intermediate layer allows non-linear feature combination.
- Dropout(0.2) in the head provides additional regularisation.
- Final Linear(64->1) outputs the standardized predicted congestion score.

### Forward Pass Returns Both Outputs

```python
def forward(x: Tensor[B, 30, 24]) -> Tuple[Tensor[B, 1], Tensor[B, 128]]:
    # pred_congestion    â€” predicted congestion_score(t+1), standardized
    # temporal_embedding â€” 128-dim hidden state for downstream GAT+PPO fusion
```

During training only `pred_congestion` is used for the loss.
During embedding extraction, `temporal_embedding` is saved to disk.

---

## 11. Hyperparameters â€” Full Reference Table

| Hyperparameter | Default | CLI Flag | Description |
|---|---|---|---|
| `input_dim` | 24 | auto-detected | Features per timestep (from feature audit) |
| `hidden_dim` | 128 | `--hidden-dim` | LSTM hidden size AND embedding dimension |
| `num_layers` | 2 | `--num-layers` | Stacked LSTM layers |
| `dropout` | 0.2 | `--dropout` | Dropout between LSTM layers and in head |
| `window_size` | 30 | `--window-size` | Historical timesteps per sequence |
| `stride` | 1 | `--stride` | Step between consecutive windows |
| `batch_size` | 128 | `--batch-size` | Samples per gradient update |
| `learning_rate` | 0.001 | `--lr` | Initial Adam learning rate |
| `weight_decay` | 0.0001 | `--weight-decay` | L2 regularisation on weights |
| `epochs` | 50 | `--epochs` | Maximum training epochs |
| `patience` | 7 | `--patience` | Early stopping patience (epochs) |
| `scheduler_patience` | 3 | hardcoded | Epochs before halving LR |
| `scheduler_factor` | 0.5 | hardcoded | LR multiplier when reducing |
| `seed` | 42 | `--seed` | Global random seed |
| `train_ratio` | 0.70 | hardcoded | Fraction of timesteps for training |
| `val_ratio` | 0.15 | hardcoded | Fraction for validation |

---

## 12. Training Loop â€” Step by Step

### Phase 1: Data Loading and Validation

```
1. Load datasets/lstm_all_scenarios.csv (936,000 rows)
2. Assert: 13 scenarios, 100 satellites, 0 NaN/Inf, 0 duplicate rows
3. Correlation audit -> 24 non-collinear features selected
4. Build sliding windows: W=30, stride=1, per (scenario, satellite)
5. Fit FeatureScaler on training windows ONLY
6. Fit TargetScaler on training targets ONLY
7. Save both scalers to artifacts/lstm/
```

### Phase 2: Smoke Test (before training starts)

```python
smoke_x = next(iter(train_loader))[0]   # shape [128, 30, 24]
smoke_pred, smoke_emb = model(smoke_x)
assert smoke_pred.shape == (128, 1)     # correct output shape
assert smoke_emb.shape  == (128, 128)  # correct embedding shape
assert no NaN, no Inf                   # numerically stable
```

### Phase 3: Per-Epoch Training (for each epoch 1 to max_epochs)

```
A. model.train()
B. For each batch (x_b [128,30,24], y_scaled_b [128,1]):
   i.  pred_scaled, _ = model(x_b)
   ii. loss = MSELoss(pred_scaled, y_scaled_b)  # on standardized targets
   iii.loss.backward()
   iv. optimizer.step()
   v.  optimizer.zero_grad()

C. model.eval() -> evaluate validation set:
   i.  Forward pass for all val batches
   ii. inverse_transform predictions -> raw [0,1] scale
   iii.Compute raw-scale MSE, RMSE, MAE, R2

D. scheduler.step(val_loss)        # halve LR if plateau for 3 epochs

E. Save lstm_last.pt               # always

F. If val_loss < best_val_loss:
       Save lstm_best.pt
       patience_counter = 0
   Else:
       patience_counter += 1
       If patience_counter >= 7: EARLY STOP

G. Append to training_history.csv
```

### Phase 4: Final Evaluation

```
Load lstm_best.pt
Run on full test_loader -> inverse_transform -> raw [0,1] scale
Compute: MSE, RMSE, MAE, R2, MedianAE, MaxAE
Compute Mean Baseline and Persistence Baseline
Save: test_metrics.json, baseline_metrics.json, scenario_metrics.csv
```

---

## 13. Optimizer, Scheduler, and Early Stopping

### Optimizer: Adam

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,           # initial learning rate
    weight_decay=1e-4   # L2 regularisation (prevents large weights)
)
```

Adam maintains per-parameter adaptive learning rates using first and second moment
estimates of gradients. Combines Momentum and RMSProp for fast stable convergence.

### Loss Function: MSELoss on Standardized Targets

```python
criterion = nn.MSELoss()
loss = criterion(pred_scaled, y_scaled)
```

Computed in **standardized space** (zero mean, unit variance). Keeps gradient
magnitudes consistent. All reported metrics use the **raw [0,1] scale**.

### Learning Rate Scheduler: ReduceLROnPlateau

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",     # monitor for decrease
    factor=0.5,     # multiply LR by 0.5 when triggered
    patience=3      # wait 3 epochs before reducing
)
```

LR follows a staircase descent:
- Start: LR = 0.001
- After 3 epochs no improvement: LR = 0.0005
- After another 3: LR = 0.00025

### Early Stopping (patience = 7)

If the best validation loss is not beaten for 7 consecutive epochs, training stops.
`lstm_best.pt` (the best checkpoint) is always preserved.
In practice the model converged at approximately **epoch 20** of the 50-epoch budget.

---

## 14. Evaluation Metrics and Results

All final metrics are on the **raw, original [0,1] scale** of `congestion_score`:

| Metric | Formula | Interpretation |
|---|---|---|
| **MSE** | mean((y_true - y_pred)^2) | Squared error; penalises large errors |
| **RMSE** | sqrt(MSE) | Same units as congestion_score |
| **MAE** | mean(|y_true - y_pred|) | Average absolute error |
| **R2** | 1 - MSE/Var(y_true) | Fraction of variance explained (1.0=perfect) |
| **Median AE** | median(|y_true - y_pred|) | Error at 50th percentile |
| **Max AE** | max(|y_true - y_pred|) | Worst single prediction error |

### Achieved Test Performance (saved to `artifacts/lstm/test_metrics.json`)

| Metric | Value |
|---|---|
| **Test MSE** | **0.002902** |
| **Test RMSE** | **0.053866** |
| **Test MAE** | **0.036090** |
| **Test R2** | **0.88839 (88.84%)** |

R2 = 0.888 means the LSTM explains **88.8% of the variance** in future congestion.
MAE = 0.036 means predictions are on average only **3.6 percentage points** from truth.

---

## 15. Baseline Comparisons

Two naive baselines contextualize LSTM performance (saved to `baseline_metrics.json`):

### Baseline 1: Mean Baseline

```python
mean_train = mean([s.y for s in train_samples])
preds = [mean_train] * len(test_samples)
```

R2 = 0.0 by definition. Absolute minimum performance floor.

### Baseline 2: Persistence Baseline

```python
preds = [s.y_curr for s in test_samples]   # y(t+1) = y(t)
```

Stronger because congestion has temporal autocorrelation.

### Comparison Summary

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Mean Baseline | ~0.150 | ~0.120 | ~0.00 |
| Persistence Baseline | ~0.080 | ~0.060 | ~0.60 |
| **LSTM Model** | **0.054** | **0.036** | **0.888** |

LSTM significantly outperforms both baselines, proving genuine temporal modelling.

---

## 16. Temporal Embedding Extraction

**Source file**: `satsim/lstm/embedder.py`

### What Is a Temporal Embedding?

For any input window `X(t-29...t)` for satellite `s`, the LSTM's final hidden state is:

```
temporal_embedding = h_last  in R^128
```

This 128-dimensional vector compresses the satellite's 30-second temporal history.
It encodes patterns such as:
- "Buffer utilization has been climbing steadily for 20 steps."
- "This satellite just recovered from a routing table update."
- "Traffic has had 3 burst spikes in the last 10 steps."

### Extraction Process

```python
embedder = LSTMEmbedder(
    model_path  = "artifacts/lstm/lstm_best.pt",
    scaler_path = "artifacts/lstm/feature_scaler.pkl",
)

# Process ALL samples (train + val + test), batch_size=512:
for batch in all_samples:
    x_scaled = feature_scaler.transform(batch.x)     # [512, 30, 24]
    _, temporal_embedding = model(x_scaled)           # [512, 128]
    # Save indexed by (scenario, satellite_id, target_timestep)
```

**Alignment assertion before saving:**
```python
assert sample.input_end_t == sample.target_t - 1
```

### Output Files

13 pickle files (one per scenario) in `artifacts/lstm/embeddings/`:

```python
# Each file payload:
{
    "scenario":               "low_load",
    "seed":                   42,
    "satellite_ids":          np.int64[N],
    "input_start_timesteps":  np.int64[N],
    "input_end_timesteps":    np.int64[N],     # = target_timestep - 1
    "target_timesteps":       np.int64[N],
    "temporal_embeddings":    np.float32[N, 128],  # THE EMBEDDINGS
}
```

Master index: `artifacts/lstm/embedding_index.csv`
Alignment preview: `artifacts/lstm/gat_lstm_alignment_preview.csv`

---

## 17. Downstream Usage â€” GAT + PPO Fusion

The LSTM is **Stage 2 of a 3-stage pipeline**:

```
Stage 1: GAT Spatial Representation Learner
  Input:  [100 satellites] x [8 features: pos_eci_x/y/z, vel_eci_x/y/z,
                                           buffer_utilization, degree]
          + [380 ISL edges] x [4 edge features]
  Output: spatial_embedding in R^(100 x 128)
  Meaning: "Where are satellites and how is the topology structured?"

Stage 2: LSTM Temporal Representation Learner  (THIS DOCUMENT)
  Input:  [24 features x 30 timesteps] per satellite
  Output: temporal_embedding in R^(100 x 128)
  Meaning: "How has each satellite behaved over the last 30 seconds?"

Stage 3: PPO Reinforcement Learning Router
  Input:  concat(spatial, temporal) in R^(100 x 256)
  Output: routing policy (next-hop per packet)
  Meaning: "Given topology AND temporal dynamics, what is the optimal route?"
```

### The 256-Dimensional Fused Representation

```python
fused = torch.cat([
    gat_spatial_embedding,    # [100, 128]
    lstm_temporal_embedding,  # [100, 128]
], dim=-1)
# fused: [100, 256]
```

This gives the PPO agent both **spatial** (current topology) and
**temporal** (congestion dynamics) context simultaneously.

---

## 18. Output Artifacts

All files written to `artifacts/lstm/` after a complete pipeline run:

| File | Contents |
|---|---|
| `lstm_best.pt` | Best model weights (model_state_dict, optimizer_state_dict, epoch, val_loss, model_config) |
| `lstm_last.pt` | Final epoch weights |
| `feature_scaler.pkl` | StandardScaler on 24 training input features |
| `target_scaler.pkl` | StandardScaler on training congestion_score |
| `test_metrics.json` | MSE, RMSE, MAE, R2, MedianAE, MaxAE on test set (raw scale) |
| `baseline_metrics.json` | Mean and Persistence baseline metrics |
| `scenario_metrics.csv` | Per-scenario test metrics (13 rows) |
| `feature_audit.csv` | Feature selection: mean, std, min, max, selected flag |
| `training_history.csv` | Epoch-by-epoch: train_loss, val_loss, RMSE, MAE, R2, LR, time |
| `config.yaml` | Complete hyperparameter config for reproducibility |
| `results.json` | Consolidated summary (sequences, metrics, embedding info) |
| `LSTM_EVALUATION_REPORT.md` | Human-readable markdown evaluation report |
| `LSTM_DATA_LEAKAGE_AUDIT.md` | Audit confirming zero data leakage |
| `embedding_index.csv` | Master index of all extracted temporal embeddings |
| `gat_lstm_alignment_preview.csv` | GAT/LSTM timestep alignment preview |
| `embeddings/embedding_<scenario>.pt` | 13 pickle files, each [N, 128] embeddings |
| `plots/` | 8 diagnostic PNG plots at 300 DPI |

---

## 19. Diagnostic Plots

8 plots in `artifacts/lstm/plots/` at 300 DPI:

### Plot 1: `training_validation_loss.png`
Train (blue) vs Val (orange dashed) MSE loss per epoch.
Look for: smooth convergence; large gap = overfitting.

### Plot 2: `actual_vs_predicted.png`
Scatter: Actual vs Predicted congestion_score(t+1), raw scale.
Red dashed = perfect y=x diagonal. Look for: tight cluster on diagonal.

### Plot 3: `baseline_comparison.png`
Grouped bars: RMSE and MAE for Mean Baseline, Persistence Baseline, LSTM.
Look for: LSTM bars significantly shorter than baselines.

### Plot 4: `prediction_error_distribution.png`
Density histogram of residuals (y_true - y_pred).
Red line at zero. Look for: tight distribution centered at 0.

### Plot 5: `scenario_performance.png`
Test RMSE per scenario. Blue = normal, red = hard
(failures, weather, congestion_stress).

### Plot 6: `target_distribution.png`
Box plots of raw congestion_score per scenario.
Confirms data quality: congestion_stress near 1.0, low_load near 0.0.

### Plot 7: `temporal_prediction_example.png`
Time-series: Actual (blue) vs Predicted (red dashed) for one satellite.
Look for: predicted line tracking real peaks and troughs.

### Plot 8: `lstm_embedding_pca.png`
2D PCA of 128-dim temporal embeddings colored by scenario.
Look for: distinct per-scenario clusters.

---

## 20. Data Leakage Audit

4-checkpoint audit performed after training. Result in `LSTM_DATA_LEAKAGE_AUDIT.md`.

### Check 1: Target Isolation
```python
assert input_end_t < target_t           # target in the future
assert target_t == input_end_t + 1      # no gaps
# Failures: ZERO (required)
```

### Check 2: Split Boundary Integrity
```python
max(train target_t)  <  min(val input_start_t)
max(val target_t)    <  min(test input_start_t)
```

### Check 3: Scaler Isolation
FeatureScaler and TargetScaler: `fit()` called once on training data only.
Val/test: only `transform()`, never `fit()`.

### Check 4: No Cross-Group Contamination
Windows created independently per `(scenario, seed, satellite_id)`.
No window spans data from a different satellite.

**Status: PASS (NO DATA LEAKAGE DETECTED)**

---

## 21. Reproducibility and Seeding

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

Full config saved to `artifacts/lstm/config.yaml`:

```yaml
seed: 42
dataset: datasets/lstm_all_scenarios.csv
lstm:
  input_dim: 24
  hidden_dim: 128
  num_layers: 2
  window_size: 30
  stride: 1
  dropout: 0.2
  learning_rate: 0.001
  weight_decay: 0.0001
  batch_size: 128
  epochs: 50
  early_stopping_patience: 7
```

Running the CLI twice with `--seed 42` on identical hardware produces
**byte-identical outputs**.

---

## 22. How to Run â€” CLI Reference

**Source file**: `satsim/cli/train_lstm.py`

### Full Training

```bash
python -m satsim.cli.train_lstm \
    --dataset datasets/lstm_all_scenarios.csv \
    --artifacts-dir artifacts/lstm \
    --epochs 50 --batch-size 128 --window-size 30 \
    --hidden-dim 128 --num-layers 2 \
    --lr 0.001 --weight-decay 0.0001 --dropout 0.2 \
    --patience 7 --seed 42
```

### Skip Training (lstm_best.pt already exists)

```bash
python -m satsim.cli.train_lstm --skip-training
```

Goes directly to: evaluation -> plots -> embedding extraction -> report generation.

### Parquet Format (faster I/O)

```bash
python -m satsim.cli.train_lstm --dataset datasets/lstm_all_scenarios.parquet
```

### Export Embeddings Only

```bash
python -m satsim.cli.export_lstm
```

### Run Tests

```bash
pytest tests/test_lstm_pipeline.py -v
```

Tests cover: forward pass shapes, scaler fit/transform/inverse, training smoke loop.

---

## 23. Source Files Quick Reference

| File | Class / Role |
|---|---|
| `satsim/lstm/__init__.py` | Module public exports |
| `satsim/lstm/lstm_model.py` | `LEOLSTMModel` â€” 2-layer LSTM + prediction head |
| `satsim/lstm/lstm_dataset.py` | `LEOLSTMDataset`, `FeatureScaler`, `TargetScaler`, `SequenceSample`, `PyGSequenceDataset` |
| `satsim/lstm/trainer.py` | `LSTMTrainer` â€” training loop, early stopping, evaluation, baselines |
| `satsim/lstm/embedder.py` | `LSTMEmbedder` â€” 128-dim temporal embedding extraction |
| `satsim/lstm/plotter.py` | `LSTMPlotter` â€” 8 diagnostic plots |
| `satsim/cli/train_lstm.py` | End-to-end CLI orchestrator (14 pipeline stages) |
| `satsim/cli/export_lstm.py` | Standalone embedding export |
| `tests/test_lstm_pipeline.py` | Unit tests |
| `datasets/lstm_all_scenarios.csv` | Primary dataset (936,000 rows, 330 MB) |
| `datasets/lstm_all_scenarios.parquet` | Parquet format (59 MB) |
| `artifacts/lstm/` | All model checkpoints, scalers, metrics, embeddings, plots |

---

## 24. End-to-End Pipeline Flow Diagram

```
datasets/lstm_all_scenarios.csv  (936,000 rows)
          |
          v
+----------------------------------------------------------+
|  LEOLSTMDataset                                          |
|  load_and_validate() -> 13 scenarios, 100 sats, 0 NaN   |
|  audit_features()    -> 24 features selected             |
|  build_time_aware_sequences(W=30, stride=1)              |
|  -> ~9.36M SequenceSample objects                        |
|  -> Train 70% / Val 15% / Test 15% (time-ordered)       |
+-----------------------+----------------------------------+
                        |
                        v
+----------------------------------------------------------+
|  Scalers (fitted ONLY on training split)                 |
|  FeatureScaler: StandardScaler on [30, 24] windows       |
|  TargetScaler:  StandardScaler on congestion_score(t+1)  |
+-----------------------+----------------------------------+
                        |
                        v
+----------------------------------------------------------+
|  PyGSequenceDataset + DataLoader (batch_size=128)         |
|  Each batch: x_scaled[128,30,24], y_scaled[128,1],       |
|              y_raw[128,1], y_curr[128]                   |
+-----------------------+----------------------------------+
                        |
                        v
+----------------------------------------------------------+
|  LEOLSTMModel                                            |
|  LSTM(24->128, 2 layers, dropout=0.2)                    |
|  Head: Linear(128->64)->ReLU->Dropout->Linear(64->1)     |
|  forward(x): -> pred_congestion[B,1], embedding[B,128]   |
+-------+---------------------------+-----------------------+
        |                           |
        v                           v
+---------------+      +---------------------------+
|  LSTMTrainer  |      |  LSTMEmbedder             |
|  MSELoss      |      |  h_last[B,128] for all    |
|  Adam lr=1e-3 |      |  train+val+test samples   |
|  ReduceLROnP  |      |  -> embeddings/*.pt        |
|  EarlyStop p=7|      |  -> embedding_index.csv   |
|  -> ~epoch 20 |      +-------------+-------------+
|  -> R2=0.888  |                    |
+---------------+                    v
        |             +---------------------------+
        v             |  PPO Routing Agent        |
+---------------+     |  fused = cat(             |
|  Metrics &    |     |    gat_embed  [100, 128], |
|  Baselines    |     |    lstm_embed [100, 128]  |
|  Plots (x8)   |     |  ) -> [100, 256]          |
+---------------+     |  -> Routing decisions      |
                      +---------------------------+
```

---

*Document generated: 2026-08-29. Reflects the current SatSim LSTM pipeline.*
*Source: satsim/lstm/ -- LEOLSTMModel, LEOLSTMDataset, LSTMTrainer, LSTMEmbedder, LSTMPlotter*
*Dataset: datasets/lstm_all_scenarios.csv -- 936,000 rows, 13 scenarios, 100 satellites, 720 timesteps*
