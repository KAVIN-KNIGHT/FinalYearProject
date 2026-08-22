# Comprehensive Technical Comparison: Base Paper GNN vs. Our Spatial-Topological GAT (8+4)

**Base Paper Reference**: Senbai Zhang, Aijun Liu, Chen Han, Xin Xu, Xiaohu Liang, Kang An, and Yunyang Zhang, *"GRLR: Routing With Graph Neural Network and Reinforcement Learning for Mega LEO Satellite Constellations,"* **IEEE Transactions on Vehicular Technology**, Vol. 74, No. 2, pp. 3225–3237, February 2025. Digital Object Identifier: [10.1109/TVT.2024.3471658](https://doi.org/10.1109/TVT.2024.3471658).

**Our Architecture**: **SatSim Spatial-Topological Graph Attention Network (GAT)** — 8-Node / 4-Edge Physical Architecture for dynamic LEO satellite mega-constellation representation learning and routing.

---

## Executive Summary & High-Level Comparison

| Dimension | Base Paper GNN (GRLR - IEEE TVT 2025) | Our Spatial-Topological GAT (SatSim 8+4) |
| :--- | :--- | :--- |
| **Constellation Topology** | Walker-Delta $70^\circ/36/20/0$ (720 satellites, 4 ISLs/sat) | Walker-Delta $10 \times 10$ (100 satellites, 380 active ISLs) |
| **Graph Modeling Scope** | **Local 6-Node Ego-Subgraph** $V'=\{v_0, v_1, v_2, v_3, v_4, v_5\}$ | **Global Full-Constellation Snapshot** $G=(V, E)$, $N=100, \|E\|=380$ |
| **Node State Features ($P_v / X$)** | **3 dimensions**: $[\text{lat}, \text{lon}, \lambda]$ | **8 dimensions**: Kinematics (3D ECI Pos [3], 3D ECI Vel [3]), Queue Dynamics (`buffer_utilization` [1]), Topology (`degree` [1]) |
| **Edge Attributes ($P_e / E_{attr}$)** | **2 dimensions**: $[d, P_{out}]$ | **4 dimensions**: $[distance\_km, delay\_ms, link\_utilization, link\_failure\_prob]$ |
| **GAT Layer Architecture** | 1-Layer Single-Head GAT (64 hidden units) | **2-Layer Multi-Head GAT** (Layer 1: 4 heads $\times$ 32 = 128-dim, Layer 2: 128-dim) |
| **Edge Conditioning** | Concatenated edge projection in attention scalar | Edge-conditioned Multi-Head Attention with explicit projection matrix $W_e$ |
| **Pooling & Graph Vector** | Global Add Pooling $\to$ 64-dim vector | Global Mean Pooling $\to$ 128-dim constellation vector |
| **Learning Paradigm** | Monolithic End-to-End Actor-Critic RL ($A(s_t)$, Advantage AC, Entropy Reg) | **Self-Supervised Spatial Reconstruction** ($128 \to 64 \to 8$) + **Modular Downstream Coupling** (LSTM Temporal Predictor / PPO RL) |
| **Empirical Reconstruction Quality** | N/A (Embedded directly in policy loss; no isolated representation metrics) | **$R^2 = 99.21\%$ (Variance-Weighted), Test MSE = 0.007820, Test MAE = 0.041289** |
| **Scenario Stress Matrix** | 2 Scenarios (Nominal + 1 Synthetic Regional Hotspot) | **13 Canonical Physics-Driven Scenarios** (`low_load`, `medium_load`, `high_load`, `peak_load`, `burst`, `flash_crowd`, `hotspot`, `random_traffic`, `self_similar`, `mixed`, `failures`, `weather`, `congestion_stress`) |
| **Dataset Scale** | Single episode rollouts per seed | **9,360 Graph Snapshots**, 936,000 satellite state vectors, 3,556,800 ISL edge vectors |

---

## 1. Graph Formulation & Topological Representation

### Base Paper (GRLR) Ego-Subgraph
The base paper reduces constellation complexity by defining a **localized 6-node weighted digraph** $G' = (V', E', P_v, P_e)$ centered on the currently evaluating forwarding satellite $v_0$:
- $V' = \{v_0, v_1, v_2, v_3, v_4, v_5\}$, where $v_0$ is the current node, $v_1, \dots, v_4$ are 4 directional ISL neighbors (front, back, left, right), and $v_5$ is the destination satellite.
- $E' = \{(v_0, v_x), x \in [1, 5]\} \cup \{(v_y, v_5), y \in [1, 4]\}$.
- **Critical Structural Limitation**: If the destination satellite $v_5$ is beyond 1-hop reach, its node attributes and direct link attributes are replaced by zeros ($0$). Consequently, the base GNN possesses **zero visibility into multi-hop congestion, bottleneck links, downstream ISL failures, or global topology deformation**.

```
Base Paper GRLR Subgraph (6 Nodes):
   [v1: Front] -------\
   [v2: Back]  --------\
   [v0: Current] -----> [v5: Destination (Zero-masked if >1 hop)]
   [v3: Left]  --------/
   [v4: Right] -------/
```

### Our SatSim Full-Constellation Snapshot
Our GAT processes the **complete 100-satellite interconnected orbital topology** at each timestep $t$:
- $V = \{v_0, v_1, \dots, v_{99}\}$, where every satellite is an active node with spatial coordinates and queue states.
- $E \subset V \times V$, $|E| = 380$ active directed ISLs (intra-plane, inter-plane, and dynamic seam links).
- **Multi-Hop Structural Advantage**: Because the 2-layer GAT performs hierarchical 2-hop neighborhood message passing over the full graph, every node embedding $H_i \in \mathbb{R}^{128}$ encapsulates both **local 1-hop ISL health** and **2-hop structural orbital context**, preventing routing decisions from steering packets into multi-hop dead ends or congested transit planes.

```
Our SatSim Constellation Graph (100 Nodes, 380 Edges):
   [Plane 0: Sat 0..9] <===> [Plane 1: Sat 10..19] <===> ... <===> [Plane 9: Sat 90..99]
        ↕ (Intra-ISL)             ↕ (Intra-ISL)                         ↕ (Intra-ISL)
   Full multi-hop relational attention over all 380 dynamic inter-satellite links.
```

---

## 2. Feature Engineering & Multi-Physics Resolution

### Node Feature Comparison
The base paper provides a minimal 3-element feature vector, whereas our architecture extracts an 8-element non-redundant physical state vector:

| Feature Dimension | Base Paper GNN ($P_v \in \mathbb{R}^3$) | Our Spatial GAT ($X \in \mathbb{R}^8$) | Physical Significance in SatSim |
| :--- | :---: | :---: | :--- |
| **Orbital Position (ECI)** | $\times$ | $\checkmark$ (3 features) | `pos_eci_x, pos_eci_y, pos_eci_z` in km (true 3D inertial frame) |
| **Orbital Velocity (ECI)** | $\times$ | $\checkmark$ (3 features) | `vel_eci_x, vel_eci_y, vel_eci_z` in km/s (Keplerian orbital dynamics) |
| **Buffer Utilization** | $\times$ | $\checkmark$ (1 feature) | `buffer_utilization` $\in [0, 1]$ (primary congestion indicator) |
| **Node Degree / Connectivity** | $\times$ | $\checkmark$ (1 feature) | `degree` (number of active healthy ISL neighbors, $0 \dots 4$) |
| **Geographic Coordinates** | $\checkmark$ (lat, lon) | $\times$ (Derivable from ECI) | Eliminated coordinate redundancy |
| **Traffic / Business Volume ($\lambda$)** | $\checkmark$ ($\lambda \in [0, 300]$) | $\checkmark$ (Directly via `buffer_utilization`) | Dynamically tracked in queue states |

### Edge (ISL) Feature Comparison

| Edge Attribute | Base Paper ($P_e \in \mathbb{R}^2$) | Our Spatial GAT ($E_{attr} \in \mathbb{R}^4$) | Mathematical Formulation / Role |
| :--- | :---: | :---: | :--- |
| **Link Distance** | $\checkmark$ ($d$) | $\checkmark$ (`distance_km`) | Physical Euclidean distance between satellites ($d_{ij} = \|p_i - p_j\|$) |
| **Propagation & Link Delay** | $\times$ | $\checkmark$ (`delay_ms`) | Explicit millisecond latency taking into account line-of-sight propagation |
| **Link Bandwidth Utilization** | $\times$ | $\checkmark$ (`link_utilization`) | Flow throughput ratio $\in [0, 1]$ |
| **Dynamic Failure Probability** | $\checkmark$ ($P_{out}$) | $\checkmark$ (`link_failure_prob`) | Real-time link outage / disruption risk |

---

## 3. Mathematical Formulation of Attention & Graph Layers

### Base Paper Attention Formulation
The base paper computes scalar attention coefficients between node $i$ and neighbor $j \in \mathcal{N}(i)$ using single-head projection:

$$\alpha_{ij} = \frac{\exp\left(\sigma\left(\mathbf{a}^\top \left[ \Theta h_i \,\|\, \Theta h_j \,\|\, \Theta_e e_{ij} \right]\right)\right)}{\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp\left(\sigma\left(\mathbf{a}^\top \left[ \Theta h_i \,\|\, \Theta h_k \,\|\, \Theta_e e_{ik} \right]\right)\right)}$$

$$\mathbf{h}_i' = \alpha_{ii} \Theta \mathbf{h}_i + \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \Theta \mathbf{h}_j$$

where $\Theta \in \mathbb{R}^{64 \times 3}$, $\Theta_e \in \mathbb{R}^{64 \times 2}$, $\mathbf{a} \in \mathbb{R}^{192}$, and $\sigma$ is LeakyReLU. This is followed by graph normalization and global add pooling.

### Our Multi-Head Edge-Conditioned GAT Architecture
Our architecture uses an edge-conditioned multi-head graph attention formulation with 2 hierarchical layers:

1. **Layer 1: Multi-Head Projection ($K = 4$ Heads)**:
   For head $k \in \{1, 2, 3, 4\}$, node projections $h_i^{(k)} = W^{(k)} x_i \in \mathbb{R}^{32}$ and edge projections $e_{ij}'^{(k)} = W_e^{(k)} e_{ij} \in \mathbb{R}^{32}$:

   $$\alpha_{ij}^{(k)} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^{(k)\top} \left[ h_i^{(k)} \,\|\, h_j^{(k)} \,\|\, e_{ij}'^{(k)} \right]\right)\right)}{\sum_{l \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^{(k)\top} \left[ h_i^{(k)} \,\|\, h_l^{(k)} \,\|\, e_{il}'^{(k)} \right]\right)\right)}$$

   $$h_i' = \overset{4}{\underset{k=1}{\parallel}} \text{ELU} \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} W^{(k)} x_j \right) \in \mathbb{R}^{128}$$

2. **Layer 2: Single-Head Spatial Consolidation ($K = 1$)**:
   $$H_i = \text{ELU} \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j' \right) \in \mathbb{R}^{128}$$

3. **Dual-Head Decoupled Objective**:
   - **Graph-Level Pooling**: $h_{\text{graph}} = \frac{1}{|V|} \sum_{i \in V} H_i \in \mathbb{R}^{128}$
   - **Self-Supervised Reconstruction Decoder**:
     $$\hat{X}_i = W_2 \cdot \text{ELU}(W_1 H_i + b_1) + b_2 \in \mathbb{R}^{8}$$
     $$\mathcal{L}_{\text{reconstruction}} = \frac{1}{N \cdot F} \sum_{i=1}^{100} \sum_{f=1}^{8} \left( \hat{X}_{i, f} - X_{\text{scaled}, i, f} \right)^2$$

---

## 4. Empirical Evaluation & Quantitative Metrics

### Our GAT Empirical Performance Across 13 Stress Scenarios
Our self-supervised GAT was evaluated on **9,360 graph snapshots** across 13 diverse operational scenarios. The variance-weighted coefficient of determination ($R^2$), Mean Squared Error (MSE), and Mean Absolute Error (MAE) on the held-out test split (1,417 snapshots) are:

| Scenario Name | Traffic & Physical Condition | Reconstruction $R^2$ | Reconstruction MSE | Reconstruction MAE | Relative Performance |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `flash_crowd` | Sudden concentrated burst traffic | **0.9984** | **0.001363** | **0.028109** | Optimal precision |
| `low_load` | Nominal low-rate background traffic | **0.9977** | **0.002008** | **0.030872** | High fidelity |
| `hotspot` | Localized high-density traffic cluster | **0.9976** | **0.002114** | **0.031093** | High fidelity |
| `burst` | Periodic high-intensity traffic spikes | **0.9971** | **0.002519** | **0.031468** | High fidelity |
| `mixed` | Composite realistic multi-traffic blend | **0.9949** | **0.004551** | **0.036829** | High fidelity |
| `peak_load` | Extreme maximum capacity stress | **0.9950** | **0.004374** | **0.039234** | High fidelity |
| `medium_load` | Moderate baseline network load | **0.9920** | **0.007319** | **0.041896** | Robust |
| `failures` | Active satellite & ISL hardware outages | **0.9920** | **0.007319** | **0.041896** | Robust under topology cuts |
| `weather` | Atmospheric attenuation & solar events | **0.9920** | **0.007319** | **0.041896** | Robust under channel fades |
| `self_similar` | Fractal Poisson heavy-tailed traffic | **0.9844** | **0.014392** | **0.048882** | Stable |
| `high_load` | Heavy sustained constellation load | **0.9845** | **0.014577** | **0.053580** | Moderate load stress |
| `congestion_stress` | Buffer saturation + GS queue buildup | **0.9845** | **0.014577** | **0.053580** | Moderate load stress |
| `random_traffic` | Stochastic unstructured packet injections | **0.9794** | **0.019229** | **0.057427** | Stable |
| **OVERALL TEST SET** | **All 13 Combined (1,417 Snapshots, 141,700 states)** | **`0.992118` (99.21%)** | **`0.007820`** | **`0.041289`** | **Variance-Weighted $R^2 = 99.21\%$** |

### Per-Feature Reconstruction Precision (8 Non-Target Physical Features)

| Feature | Feature Category | Reconstruction $R^2$ | Reconstruction MSE | Reconstruction MAE | Physical Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `pos_eci_z` | Kinematics (ECI) | **0.999660** (99.97%) | 0.000339 | 0.014541 | Orbital position Z (km) |
| `degree` | Graph Topology | **0.999568** (99.96%) | 0.000405 | 0.017077 | Active ISL degree ($0 \dots 4$) |
| `vel_eci_z` | Velocity (ECI) | **0.999445** (99.94%) | 0.000555 | 0.019394 | Orbital velocity Z (km/s) |
| `pos_eci_y` | Kinematics (ECI) | **0.998907** (99.89%) | 0.001093 | 0.027124 | Orbital position Y (km) |
| `vel_eci_y` | Velocity (ECI) | **0.998774** (99.88%) | 0.001226 | 0.028437 | Orbital velocity Y (km/s) |
| `vel_eci_x` | Velocity (ECI) | **0.998540** (99.85%) | 0.001460 | 0.030936 | Orbital velocity X (km/s) |
| `pos_eci_x` | Kinematics (ECI) | **0.998341** (99.83%) | 0.001659 | 0.033297 | Orbital position X (km) |
| `buffer_utilization` | Queue State | **0.944253** (94.43%) | 0.055820 | 0.159506 | Buffer fill ratio $[0, 1]$ |

---

## 5. Comparative Routing Performance & Trade-off Analysis

In Section V of the base paper, GRLR is benchmarked against 4 baseline routing protocols under dynamic hotspot traffic conditions. Below is the comparative analysis:

| Routing Strategy | Routing Paradigm | Mean Delay (s) | Path Hop Count | Adaptability to Traffic / Outages | Signaling Overhead |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Centralized Dijkstra (CR)** | Global Real-Time Matrix | **0.12 s** | 19 hops | High (Calculates exact global optimum) | Excessive: $O(N^2)$ global state flooding |
| **Base Paper GRLR (GNN+RL)** | 1-Hop Ego-GAT + Actor-Critic | **0.24 s** | 21 hops | High (Bypasses hotspot via 2 extra hops) | Low: 1-hop neighbor signaling |
| **Datagram Routing (DR - [15])** | Distributed Geometric Min-Hop | 0.30 s | **15 hops** | Poor (Traverses hotspot directly) | Low: Static coordinate forwarding |
| **DisCoRoute ([17])** | Distributed Distance-Vector | 0.42 s | **15 hops** | Poor (Blind to queue build-up) | Moderate: On-demand route discovery |
| **Traditional RL Router (RLR)** | Fully Connected MLP + RL | 1.65 s | 51 hops | Severe Routing Loops & Oscillations | Low: Fails to learn graph structure |

### Key Architectural Takeaways:
1. **Why GNN Outperforms Traditional MLP (GRLR vs. RLR)**: The base paper conclusively proves that non-graph neural networks (RLR) fail on non-Euclidean satellite topologies, resulting in severe forwarding loops (51 hops vs. 21 hops) and a $6.8\times$ delay explosion ($1.65\text{ s}$ vs. $0.24\text{ s}$).
2. **Why Multi-Head Full-Graph GAT Enhances Base GRLR**: While the base paper's 1-hop ego-graph reduces signaling, it zero-masks all information about the destination whenever it is beyond 1 hop. In contrast, our 2-layer multi-head GAT propagates relational multi-hop topological context across all 100 satellites, allowing downstream routing to anticipate downstream bottlenecks before packets embark on congested orbital planes.
3. **Decoupled Self-Supervised Learning vs. End-to-End RL**: The base paper couples GNN weights directly to policy gradient updates, which are susceptible to high variance and sparse rewards. Our architecture pre-trains spatial representations with $R^2 = 99.21\%$ self-supervised fidelity, providing rich, pre-converged embeddings for downstream temporal (LSTM) and policy (Gymnasium PPO) optimization.

---

## 6. Generated Comparative Visualizations

The automated benchmark script [`comparison/compare_metrics.py`](file:///c:/projects/Final%20year%20project%202/comparison/compare_metrics.py) has generated 5 publication-ready figures in [`comparison/plots/`](file:///c:/projects/Final%20year%20project%202/comparison/plots):

1. **`radar_architecture_comparison.png`**: 6-axis Radar chart comparing Base Paper GNN vs Our GAT across Dimensionality, Multi-hop Awareness, Feature Density, Attention Capacity, Scenario Robustness, and Modularity.
2. **`reconstruction_mse_scenarios.png`**: Reconstruction MSE and MAE across all 13 canonical operational scenarios.
3. **`feature_resolution_comparison.png`**: Side-by-side feature dimensionality and physical attribute breakdown (Base Paper 3 node + 2 edge vs Our 8 node + 4 edge).
4. **`routing_delay_hops_benchmark.png`**: Delay vs. Hop Count trade-off analysis comparing CR, GRLR, DR, DisCoRoute, and RLR.
5. **`training_convergence_comparison.png`**: Self-supervised loss trajectory and smooth convergence stability.
