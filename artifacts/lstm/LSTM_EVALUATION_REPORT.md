# LSTM Evaluation Report — 100-Satellite LEO Temporal Congestion Prediction

## Executive Summary
A **leak-free 2-layer LSTM model** was trained across all 13 canonical LEO simulation scenarios using the single consolidated dataset `datasets/lstm_all_scenarios.csv` (936,000 raw rows). The model predicts future congestion score at timestep $t+1$ ($X(t-29 \dots t) \to \text{congestion\_score}(t+1)$) and extracts **128-dimensional node temporal embeddings** for downstream GAT + PPO fusion.

## 1. Dataset & Split Specifications
- **Raw Dataset Rows**: 936,000
- **Satellites**: 100 (IDs 0–99)
- **Scenarios (13)**: low_load, medium_load, high_load, peak_load, burst, flash_crowd, hotspot, random_traffic, self_similar, mixed, failures, weather, congestion_stress
- **Window Size**: 30 historical timesteps
- **Time-Aware Split**: Train: 308,100 (70%), Val: 50,700 (15%), Test: 52,000 (15%)

## 2. Input Features & Target Definition
- **Target**: `congestion_score(t+1)`
- **Input Features**:
```text
 1. simulation_time_s
 2. pos_eci_x
 3. pos_eci_y
 4. pos_eci_z
 5. vel_eci_x
 6. vel_eci_y
 7. vel_eci_z
 8. pos_ecef_x
 9. pos_ecef_y
10. is_active
11. buffer_utilization
12. degree
13. avg_isl_delay_ms
14. queue_length
15. queue_occupancy
16. end_to_end_delay
17. throughput
18. link_utilization
19. traffic_load
20. cpu_utilization
21. memory_utilization
22. routing_table_age
23. routing_changes_in_window
24. event_flags
```

## 3. Model Architecture & Training Hyperparameters
```yaml
batch_size: 256
dropout: 0.2
early_stopping_patience: 7
epochs: 20
hidden_dim: 128
input_dim: 24
learning_rate: 0.001
num_layers: 2
stride: 2
weight_decay: 0.0001
window_size: 30

```

## 4. Test Performance Comparison: Baselines vs LSTM (Raw Scale)
| Model | Test MSE (Raw) | Test MAE (Raw) | Test RMSE (Raw) | Test R² Score |
|---|---|---|---|---|
| **Mean Baseline** | 0.026026 | 0.132372 | 0.161327 | -0.001105 |
| **Persistence Baseline** ($y_{t+1} = y_t$) | 0.005517 | 0.046553 | 0.074279 | 0.787771 |
| **LSTM Model** | 0.002902 | 0.036090 | 0.053866 | 0.888390 |

- **LSTM Improvement vs Mean Baseline**: **+66.61% RMSE**, **+72.74% MAE**
- **LSTM Improvement vs Persistence Baseline**: **+27.48% RMSE**, **+22.48% MAE**

## 5. Per-Scenario Evaluation Breakdown (Raw Scale)
| Scenario | Test Samples | MAE | RMSE | MSE | R² Score |
|---|---|---|---|---|---|
| `low_load` | 0 | 0.016286 | 0.021555 | 0.000465 | 0.291493 |
| `medium_load` | 0 | 0.038507 | 0.051177 | 0.002619 | 0.707341 |
| `high_load` | 0 | 0.052257 | 0.068885 | 0.004745 | 0.694437 |
| `peak_load` | 0 | 0.023334 | 0.034613 | 0.001198 | 0.650215 |
| `burst` | 0 | 0.027952 | 0.044065 | 0.001942 | 0.111058 |
| `flash_crowd` | 0 | 0.010238 | 0.012346 | 0.000152 | -0.127654 |
| `hotspot` | 0 | 0.016275 | 0.022124 | 0.000489 | 0.388978 |
| `random_traffic` | 0 | 0.070191 | 0.089238 | 0.007963 | 0.325258 |
| `self_similar` | 0 | 0.050157 | 0.075004 | 0.005626 | 0.456460 |
| `mixed` | 0 | 0.034697 | 0.050379 | 0.002538 | 0.434111 |
| `failures` | 0 | 0.038507 | 0.051177 | 0.002619 | 0.707341 |
| `weather` | 0 | 0.038507 | 0.051177 | 0.002619 | 0.707341 |
| `congestion_stress` | 0 | 0.052257 | 0.068885 | 0.004745 | 0.694437 |

## 6. Artifact Locations & Diagnostic Plots
- **Model Weights**: [artifacts\lstm\lstm_best.pt](file:///artifacts/lstm/lstm_best.pt)
- **Feature Scaler**: [artifacts\lstm\feature_scaler.pkl](file:///artifacts/lstm/feature_scaler.pkl)
- **Target Scaler**: [artifacts\lstm\target_scaler.pkl](file:///artifacts/lstm/target_scaler.pkl)
- **Feature Audit CSV**: [artifacts\lstm\feature_audit.csv](file:///artifacts/lstm/feature_audit.csv)
- **Scenario Metrics CSV**: [artifacts\lstm\scenario_metrics.csv](file:///artifacts/lstm/scenario_metrics.csv)
- **Embeddings Directory**: `artifacts/lstm/embeddings/` (410,800 files)
- **Embedding Index**: [artifacts\lstm\embedding_index.csv](file:///artifacts/lstm/embedding_index.csv)
- **GAT/LSTM Alignment Preview**: [artifacts\lstm\gat_lstm_alignment_preview.csv](file:///artifacts/lstm/gat_lstm_alignment_preview.csv)
- **Training/Val Loss Plot**: ![](artifacts/lstm/plots/training_validation_loss.png)
- **Actual vs Predicted Plot**: ![](artifacts/lstm/plots/actual_vs_predicted.png)
- **Baseline Comparison Plot**: ![](artifacts/lstm/plots/baseline_comparison.png)
- **Error Distribution Plot**: ![](artifacts/lstm/plots/prediction_error_distribution.png)
- **Scenario Performance Plot**: ![](artifacts/lstm/plots/scenario_performance.png)
- **Target Distribution Plot**: ![](artifacts/lstm/plots/target_distribution.png)
- **Temporal Prediction Plot**: ![](artifacts/lstm/plots/temporal_prediction_example.png)
- **Embedding PCA Plot**: ![](artifacts/lstm/plots/lstm_embedding_pca.png)
