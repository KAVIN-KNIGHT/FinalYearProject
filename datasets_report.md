# Datasets and Folder Structure Report — LEO Satellite Network Dynamic Routing Project

## Executive Summary
This report provides a comprehensive reference of the repository folder structure, dataset schemas, canonical simulation scenario specifications, and model artifact storage for the **100-Satellite LEO Constellation Dynamic Routing System** migrated to the **8-Node / 4-Edge Physical GAT Architecture**.

All datasets and generated embeddings adhere to strict data-integrity, reproducible split, and zero-target-leakage protocols.

---

## 1. Comprehensive Repository Directory Structure

```
c:/projects/Final year project 2/
├── artifacts/                           # Saved model checkpoints, embeddings, metrics, and plots
│   ├── gat/                             # Graph Attention Network (GAT) artifacts
│   │   ├── spatial/                     # ★ PRIMARY: 8+4 Spatial/Topological GAT Representation Learner
│   │   │   ├── gat_best.pt              # Best model weights checkpoint (validation MSE loss: 0.007803)
│   │   │   ├── gat_last.pt              # Final epoch model weights checkpoint
│   │   │   ├── feature_scaler.pkl       # StandardScaler fitted ONLY on training split (8 node + 4 edge features)
│   │   │   ├── exact_reconstruction_r2_results.json # Exact R² results (Overall R² = 99.21%)
│   │   │   ├── validation_metrics.json  # Validation standardized reconstruction MSE (0.007803) & MAE (0.041248)
│   │   │   ├── test_metrics.json        # Test standardized reconstruction MSE (0.007820) & MAE (0.041289)
│   │   │   ├── per_feature_metrics.csv  # Standardized MAE breakdown per feature across 8 node features
│   │   │   ├── scenario_metrics.csv     # Per-scenario reconstruction MSE and MAE breakdown
│   │   │   ├── training_history.csv     # Epoch-by-epoch train/val loss & LR history over 50 epochs
│   │   │   ├── embedding_index.csv      # Master metadata index for all 9,360 exported spatial embeddings
│   │   │   ├── embeddings/              # 9,360 exported 128-D spatial node embedding payloads (.pt)
│   │   │   │   ├── embedding_000000.pt  # Payload: {scenario, seed, timestep, satellite_ids, node_embeddings[100,128]}
│   │   │   │   └── ...                  # (9,360 total payload files)
│   │   │   └── plots/                   # Spatial diagnostic & visual evidence plots
│   │   │       ├── training_validation_loss.png       # Reconstruction MSE loss convergence curve
│   │   │       ├── gat_topology_attention.png         # Top 25% highest-attention ISL edges visualization
│   │   │       ├── gat_embedding_visualization.png    # 2D PCA projection scatter of 128-D embeddings
│   │   │       └── gat_embedding_similarity_heatmap.png # Pairwise satellite cosine similarity matrix
│   └── lstm/                            # Long Short-Term Memory (LSTM) artifacts
│       ├── lstm_best.pt                 # Best LSTM model weights checkpoint (20 epochs)
│       ├── lstm_last.pt                 # Final epoch LSTM checkpoint
│       ├── feature_scaler.pkl           # Feature scaler for 24 sequence input features
│       ├── target_scaler.pkl            # Target scaler for congestion_score(t+1)
│       ├── test_metrics.json            # LSTM Test MSE (0.002902), RMSE (0.053866), MAE (0.036090), R² (0.88839)
│       ├── baseline_metrics.json        # Mean baseline & Persistence baseline performance comparison
│       ├── scenario_metrics.csv         # Per-scenario LSTM test performance
│       ├── feature_audit.csv            # Correlation matrix audit excluding exact duplicate columns
│       └── plots/                       # LSTM diagnostic plots (actual vs predicted, loss curves, residual dist)
├── comparison/                          # Comparative Benchmark Suite vs. Base Paper (Zhang et al. TVT 2025)
│   ├── README.md                        # Comparison suite navigation guide
│   ├── base_paper_vs_our_gat_report.md  # Exhaustive mathematical & architectural benchmark report
│   ├── compare_metrics.py               # Automated evaluation & 300 DPI plot generator
│   ├── comparison_metrics.json          # Structured comparative benchmark data
│   ├── comparison_summary.csv           # Side-by-side metric comparison table
│   └── plots/                           # Publication-grade comparative figures
│       ├── feature_resolution_comparison.png
│       ├── radar_architecture_comparison.png
│       ├── reconstruction_mse_scenarios.png
│       ├── routing_delay_hops_benchmark.png
│       └── training_convergence_comparison.png
├── datasets/                            # Simulation raw datasets and PyG graph snapshots
│   ├── lstm_all_scenarios.csv           # ★ SINGLE SOURCE OF TRUTH: 936,000 raw LSTM rows (100 sats × 13 scens × 720 steps)
│   ├── lstm_all_scenarios.parquet       # Optional high-performance Parquet format
│   ├── low_load/                        # Scenario 1: Raw simulation logs & PyG snapshots
│   │   └── gat/                         # 720 snapshot files: snapshot_0.pt ... snapshot_719.pt
│   ├── medium_load/                     # Scenario 2: 720 snapshots
│   ├── high_load/                       # Scenario 3: 720 snapshots
│   ├── peak_load/                       # Scenario 4: 720 snapshots
│   ├── burst/                           # Scenario 5: 720 snapshots
│   ├── flash_crowd/                     # Scenario 6: 720 snapshots
│   ├── hotspot/                         # Scenario 7: 720 snapshots
│   ├── random_traffic/                  # Scenario 8: 720 snapshots
│   ├── self_similar/                    # Scenario 9: 720 snapshots
│   ├── mixed/                           # Scenario 10: 720 snapshots
│   ├── failures/                        # Scenario 11: 720 snapshots
│   ├── weather/                         # Scenario 12: 720 snapshots
│   └── congestion_stress/               # Scenario 13: 720 snapshots
├── satsim/                              # Main Satsim Python Package (Python 3.11+)
│   ├── gat/                             # 8+4 GAT module (LEOGATModel, LEOGraphSnapshotDataset, GATTrainer, GATEmbedder, GATPlotter)
│   ├── lstm/                            # LSTM module (LEOLSTMModel, LEOLSTMDataset, LSTMTrainer, LSTMPlotter)
│   ├── envs/                            # Gymnasium satellite routing environment (routing_env.py)
│   ├── sim/                             # Core constellation discrete-event simulation engine
│   ├── orbital/                         # SGP4/SDP4 orbital dynamics & satellite coordinate transforms
│   ├── topology/                        # Dynamic ISL topology graph management
│   ├── traffic/                         # Synthetic packet traffic generators
│   └── logging.py                       # Structlog JSON logging system
├── tests/                               # Comprehensive unit & pipeline test suite
│   ├── test_spatial_gat.py              # Dedicated unit tests for Spatial GAT 8+4 pipeline (9/9 passed)
│   ├── test_gat_pipeline.py             # GAT 8+4 architecture and dataset tests (4/4 passed)
│   ├── test_lstm_pipeline.py            # LSTM sequence and training tests
│   └── ...                              # Orbital, topology, routing, and simulation engine tests
├── pyproject.toml                       # Pinned production dependencies (gymnasium, torch, torch_geometric, pydantic)
├── README.md                            # High-level project documentation
├── GAT_ARCHITECTURE.md                  # Comprehensive GAT 8+4 architecture specification
├── CONSTELLATION_TOPOLOGY.md            # Orbital mechanics & ISL mesh topology specification
└── WORKFLOW_SIMULATION_TO_GAT.md        # End-to-end technical workflow reference
```

---

## 2. Canonical Datasets Breakdown

### A. Raw Consolidated LSTM Dataset (`datasets/lstm_all_scenarios.csv`)
- **Total Row Count**: Exactly **936,000 rows**.
- **Data Structure**: Single source-of-truth table where each row represents one satellite at one timestep in one scenario:
  $$\text{Total Rows} = 100 \text{ satellites} \times 13 \text{ scenarios} \times 720 \text{ timesteps} = 936,000$$
- **Primary Schema Columns**:
  - Indexing: `scenario`, `seed`, `satellite_id`, `timestep`, `simulation_time_s`
  - Satellite Motion: `pos_eci_x/y/z`, `vel_eci_x/y/z`, `pos_ecef_x/y/z`
  - Network State: `is_active`, `buffer_utilization`, `degree`, `avg_isl_delay_ms`, `queue_length`, `queue_occupancy`, `end_to_end_delay`, `throughput`, `link_utilization`, `traffic_load`, `cpu_utilization`, `memory_utilization`, `routing_table_age`, `routing_changes_in_window`, `event_flags`
  - Target: `congestion_score`

### B. PyG Graph Snapshots (`datasets/<scenario>/gat/snapshot_<t>.pt`)
- **Total File Count**: Exactly **9,360 files** (720 snapshot `.pt` files per scenario across 13 scenarios).
- **Snapshot Payload Structure** (`torch_geometric.data.Data`):
  - `x`: Node feature matrix of shape **`[100, 8]`** (8 non-target physical node features).
  - `edge_index`: Graph edge indices tensor of shape **`[2, 380]`** (380 active ISLs).
  - `edge_attr`: Dynamic edge attributes matrix of shape **`[380, 4]`** (4 physical link attributes).
  - `scenario`: Scenario string identifier.
  - `timestep`: Timestep integer ($0 \dots 719$).

---

## 3. Canonical LEO Simulation Scenarios (13 Scenarios)

All 13 simulation scenarios run for **720 timesteps** ($t = 0 \dots 719$) on a **100-satellite constellation** (IDs 0–99):

| Scenario Name | Traffic Regime | Active Satellites | Timesteps | Snapshot Files | Raw LSTM Rows | Split (Train / Val / Test) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`low_load`** | Low / Sparse Traffic | 100 | 720 | 720 | 72,000 | 503 / 108 / 109 |
| **`medium_load`** | Nominal Load | 100 | 720 | 720 | 72,000 | 503 / 108 / 109 |
| **`high_load`** | Heavy Load | 100 | 720 | 720 | 72,000 | 503 / 108 / 109 |
| **`peak_load`** | Capacity Stress | 100 | 720 | 720 | 72,000 | 503 / 108 / 109 |
| **`burst`** | Transient Surges | 100 | 720 | 720 | 72,000 | 503 / 108 / 109 |
| **`flash_crowd`** | Sudden Localized Floods | 100 | 720 | 720 | 72,000 | 503 / 108 / 109 |
| **`hotspot`** | Geographic Concentration | 100 | 720 | 720 | 72,000 | 503 / 108 / 109 |
| **`random_traffic`** | Stochastic Traffic | 100 | 720 | 720 | 72,000 | 503 / 108 / 109 |
| **`self_similar`** | Heavy-Tailed Pareto | 100 | 720 | 720 | 72,000 | 503 / 108 / 109 |
| **`mixed`** | Composite Multi-Modal | 100 | 720 | 720 | 72,000 | 503 / 108 / 109 |
| **`failures`** | Sat & ISL Link Outages | 100 | 720 | 720 | 72,000 | 503 / 108 / 109 |
| **`weather`** | Atmospheric Attenuation | 100 | 720 | 720 | 72,000 | 503 / 108 / 109 |
| **`congestion_stress`**| Extreme Network Overload| 100 | 720 | 720 | 72,000 | 503 / 108 / 109 |
| **Total** | — | **100** | **9,360** | **9,360** | **936,000** | **6,539 / 1,404 / 1,417** |

---

## 4. Feature & Target Specifications

### Streamlined Spatial GAT Input Features (8 Non-Target Physical Features)

$$\text{FEATURE\_INDICES} = [0, 1, 2, 3, 4, 5, 10, 12]$$

The GAT input matrix $X \in \mathbb{R}^{100 \times 8}$ consists of **8 non-target physical node features**:

| Index in Snapshot | Feature Name | Data Type | Physical Description | $R^2$ Score |
| :---: | :--- | :---: | :--- | :---: |
| `0` (idx 0) | `pos_eci_x` | `float32` | Earth-Centered Inertial position X (km) | **0.9983** (99.83%) |
| `1` (idx 1) | `pos_eci_y` | `float32` | Earth-Centered Inertial position Y (km) | **0.9989** (99.89%) |
| `2` (idx 2) | `pos_eci_z` | `float32` | Earth-Centered Inertial position Z (km) | **0.9997** (99.97%) |
| `3` (idx 3) | `vel_eci_x` | `float32` | Earth-Centered Inertial velocity X (km/s) | **0.9985** (99.85%) |
| `4` (idx 4) | `vel_eci_y` | `float32` | Earth-Centered Inertial velocity Y (km/s) | **0.9988** (99.88%) |
| `5` (idx 5) | `vel_eci_z` | `float32` | Earth-Centered Inertial velocity Z (km/s) | **0.9994** (99.94%) |
| `6` (idx 10) | `buffer_utilization` | `float32` | Packet buffer utilization fraction $[0.0, 1.0]$ | **0.9443** (94.43%) |
| `7` (idx 12) | `degree` | `float32` | Active graph node connectivity degree ($0 \dots 4$) | **0.9996** (99.96%) |

### Excluded Columns & Target Isolation
- **Eliminated Collinear Replicas**: `pos_ecef_x/y/z`, `neighbor_count`, `failure_indicator`, `node_degree`, `queue_length`, `simulation_time_s`.
- **Target Column**: `congestion_score` (column index 13) is **strictly excluded** from GAT inputs. Target $y = \text{NONE}$ for Spatial GAT representation learning.

### ISL Edge Features (4 Physical Attributes)
$$\text{EDGE\_INDICES} = [0, 1, 2, 4]$$
Each edge feature vector $E_{ij} \in \mathbb{R}^4$ represents physical link attributes:
1. `distance_km`: Inter-satellite distance (km).
2. `delay_ms`: Speed-of-light propagation and link delay (ms).
3. `link_utilization`: Bandwidth utilization fraction $[0.0, 1.0]$.
4. `link_failure_probability`: Real-time stochastic link failure risk $[0.0, 1.0)$.

---

## 5. Model Artifact Storage Breakdown

### A. Spatial/Topological GAT Artifacts ([`artifacts/gat/spatial/`](file:///c:/projects/Final%20year%20project%202/artifacts/gat/spatial))
- **`gat_best.pt`**: Best model weights checkpoint (validation MSE loss: `0.007803`).
- **`gat_last.pt`**: Final epoch 50 model checkpoint.
- **`feature_scaler.pkl`**: `StandardScaler` fitted **ONLY on 6,539 training snapshots** (8 node + 4 edge features).
- **`exact_reconstruction_r2_results.json`**: Ground-truth test evaluation (**Overall Variance-Weighted $R^2 = 99.21\%$**).
- **`validation_metrics.json`**: Standardized validation reconstruction MSE (`0.007803`) & MAE (`0.041248`).
- **`test_metrics.json`**: Standardized test reconstruction MSE (`0.007820`) & MAE (`0.041289`).
- **`per_feature_metrics.csv`**: Standardized MAE breakdown across all 8 node features.
- **`scenario_metrics.csv`**: Per-scenario reconstruction MSE and MAE breakdown across 13 scenarios.
- **`embedding_index.csv`**: Master metadata index tracking all 9,360 exported spatial embeddings.
- **`embeddings/`**: Directory containing **9,360 spatial embedding payload files** (`embedding_000000.pt` ... `embedding_009359.pt`), each containing a node embedding tensor of shape **`[100, 128]`** (0 NaNs, 0 Infs).
- **`plots/`**: Visual evidence PNG images:
  - `training_validation_loss.png`
  - `gat_topology_attention.png` (Top 25% highest-attention ISL edges)
  - `gat_embedding_visualization.png` (2D PCA scatter plot)
  - `gat_embedding_similarity_heatmap.png` (Cosine similarity matrix)

### B. Baseline Congestion-Prediction GAT Artifacts ([`artifacts/gat/corrected/`](file:///c:/projects/Final%20year%20project%202/artifacts/gat/corrected))
- Preserved baseline model checkpoints (`gat_best.pt`), feature scalers, target scalers, and evaluation reports comparing GAT against Naïve Mean baselines.

### C. LSTM Temporal Model Artifacts ([`artifacts/lstm/`](file:///c:/projects/Final%20year%20project%202/artifacts/lstm))
- Model checkpoints (`lstm_best.pt`, 20 epochs), sequence feature scalers, target scalers, test metrics (MSE: `0.002902`, RMSE: `0.053866`, MAE: `0.036090`, $R^2$: `0.88839`), baseline comparisons, feature audit CSVs, and prediction scatter plots.
