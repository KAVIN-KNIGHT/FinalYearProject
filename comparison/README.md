# SatSim Comparative Analysis & Architectural Evolution Suite

This directory contains the comparative benchmarking frameworks, empirical metrics, automated evaluation scripts, and publication-ready 300 DPI visualizations for two core benchmarks:

1. **Base Paper GNN (GRLR - IEEE TVT 2025) vs. Our Spatial-Topological GAT**:
   - Reference: Senbai Zhang et al., *"GRLR: Routing With Graph Neural Network and Reinforcement Learning for Mega LEO Satellite Constellations"*, *IEEE Transactions on Vehicular Technology*, Vol. 74, No. 2, pp. 3225–3237, Feb 2025.
   - Comprehensive Technical Report: [`base_paper_vs_our_gat_report.md`](file:///c:/projects/Final%20year%20project%202/comparison/base_paper_vs_our_gat_report.md)
   - Benchmark Script: [`compare_metrics.py`](file:///c:/projects/Final%20year%20project%202/comparison/compare_metrics.py)

2. **Previous Architecture (16+6) vs. Current Architecture (8+4)**:
   - Deep ablation and architectural evolution benchmark demonstrating why streamlining to 8 physical node features and 4 physical link features reduced test error by **$24.9\%$ MSE** / **$31.9\%$ MAE** and tightened generalization by **$10\times$**.
   - Comprehensive Technical Report: [`previous_vs_current_architecture_report.md`](file:///c:/projects/Final%20year%20project%202/comparison/previous_vs_current_architecture_report.md)
   - Benchmark Script: [`compare_architectures.py`](file:///c:/projects/Final%20year%20project%202/comparison/compare_architectures.py)

---

## Directory Structure

```
comparison/
├── README.md                                    # This navigation guide
├── base_paper_vs_our_gat_report.md              # Base Paper vs. Our GAT comparison report
├── previous_vs_current_architecture_report.md   # Previous (16+6) vs. Current (8+4) architecture report
├── compare_metrics.py                           # Base Paper vs. Our GAT comparison generator
├── compare_architectures.py                     # 16+6 vs. 8+4 architecture comparison generator
├── comparison_metrics.json                      # Base Paper comparison metrics (JSON)
├── comparison_summary.csv                       # Base Paper comparison table (CSV)
├── architecture_comparison_metrics.json         # 16+6 vs. 8+4 comparison metrics (JSON)
├── architecture_comparison_summary.csv          # 16+6 vs. 8+4 comparison table (CSV)
└── plots/                                       # 300 DPI Publication-Grade Comparative Charts
    ├── architecture_evolution_metrics_bar.png   # 16+6 vs 8+4 Error & Generalization Bar Chart
    ├── feature_r2_improvement_matrix.png        # 16+6 vs 8+4 Per-Feature R² Fidelity Matrix
    ├── scenario_mse_delta_comparison.png        # 16+6 vs 8+4 Scenario-by-Scenario MSE Comparison
    ├── radar_architecture_tradeoffs.png         # 16+6 vs 8+4 Efficiency Radar Chart
    ├── generalization_gap_breakdown.png         # 16+6 vs 8+4 Train/Val Generalization Gap Breakdown
    ├── feature_resolution_comparison.png        # Base Paper vs Our GAT Feature Granularity
    ├── radar_architecture_comparison.png        # Base Paper vs Our GAT Capacity Radar Chart
    ├── reconstruction_mse_scenarios.png         # Our GAT 13-Scenario Reconstruction MSE/MAE
    ├── routing_delay_hops_benchmark.png         # Base Paper Routing Delay vs. Hops Trade-offs
    └── training_convergence_comparison.png      # GAT Self-Supervised Learning Convergence Curve
```

---

## Quick Execution

To regenerate all comparative benchmarks and high-resolution charts:

```bash
# 1. Generate Base Paper vs. Our GAT comparison
python comparison/compare_metrics.py

# 2. Generate Previous (16+6) vs. Current (8+4) Architecture comparison
python comparison/compare_architectures.py
```

---

## Key Architecture Evolution Summary (16+6 vs. 8+4)

| Metric | Previous Architecture (16+6) | Current Architecture (8+4) | Impact |
| :--- | :---: | :---: | :--- |
| **Node Input Dimension** | 16 features | **8 features** | $-50.0\%$ (Non-redundant physical features) |
| **Edge Attribute Dimension** | 6 features | **4 features** | $-33.3\%$ (Physical ISL only) |
| **Variance-Weighted $R^2$** | $98.80\%$ | **`99.21%`** | **$+0.41\%$ Accuracy Gain** |
| **Test Reconstruction MSE** | $0.010417$ | **`0.007820`** | **$-24.9\%$ Error Reduction** |
| **Test Reconstruction MAE** | $0.060633$ | **`0.041289`** | **$-31.9\%$ Error Reduction** |
| **Train/Val Generalization Gap** | $0.021463$ | **`0.002144`** | **$10.0\times$ Tighter Generalization** |
| **Peak-Load Scenario MSE** | $0.026041$ | **`0.004374`** | **$-83.2\%$ MSE Improvement** |
