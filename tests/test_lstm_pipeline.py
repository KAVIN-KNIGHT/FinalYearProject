"""Unit tests for LSTM model architecture, dataset handling, time-aware sequence windowing, training, baseline calculation, and embedding generation.
"""
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from satsim.lstm import (
    LEOLSTMModel,
    LEOLSTMDataset,
    FeatureScaler,
    TargetScaler,
    SequenceSample,
    LSTMTrainer,
    LSTMEmbedder,
)


def create_synthetic_sequence_sample(
    scenario: str = "low_load",
    seed: int = 42,
    sat_id: int = 0,
    start_t: int = 0,
    window_size: int = 30,
    num_features: int = 15,
) -> SequenceSample:
    """Create a synthetic SequenceSample."""
    x = np.random.randn(window_size, num_features).astype(np.float32)
    y = float(np.random.rand())
    y_curr = float(np.random.rand())
    return SequenceSample(
        x=x,
        y=y,
        y_curr=y_curr,
        scenario=scenario,
        seed=seed,
        satellite_id=sat_id,
        input_start_t=start_t,
        input_end_t=start_t + window_size - 1,
        target_t=start_t + window_size,
    )


def test_lstm_model_forward_pass_shapes():
    """Verify LEOLSTMModel produces correct output tensor dimensions for 30 historical timesteps."""
    model = LEOLSTMModel(input_dim=15, hidden_dim=128, num_layers=2)
    model.eval()

    x = torch.randn(8, 30, 15)  # batch size 8, 30 timesteps, 15 features

    with torch.no_grad():
        pred_congestion, temporal_emb = model(x)

    assert pred_congestion.shape == (8, 1)
    assert temporal_emb.shape == (8, 128)
    assert not torch.isnan(pred_congestion).any()
    assert not torch.isnan(temporal_emb).any()


def test_lstm_scalers_fit_transform():
    """Verify FeatureScaler and TargetScaler fit on training samples and transform features/targets."""
    samples = [create_synthetic_sequence_sample() for _ in range(5)]

    f_scaler = FeatureScaler()
    f_scaler.fit(samples)
    assert f_scaler.fitted is True

    t_scaler = TargetScaler()
    t_scaler.fit(samples)
    assert t_scaler.fitted is True

    test_sample = create_synthetic_sequence_sample()
    scaled_x = f_scaler.transform(test_sample.x)
    assert scaled_x.shape == (30, 15)

    scaled_y = t_scaler.transform(test_sample.y)
    assert scaled_y.shape == (1, 1)

    raw_y = t_scaler.inverse_transform(scaled_y)
    assert np.isclose(raw_y[0, 0], test_sample.y, atol=1e-4)


def test_lstm_trainer_smoke_training_loop():
    """Verify LSTMTrainer executes training loop and saves best model checkpoint."""
    model = LEOLSTMModel(input_dim=15, hidden_dim=32, num_layers=2)
    device = torch.device("cpu")

    train_samples = [create_synthetic_sequence_sample() for _ in range(8)]
    val_samples = [create_synthetic_sequence_sample() for _ in range(4)]

    t_scaler = TargetScaler()
    t_scaler.fit(train_samples)

    f_scaler = FeatureScaler()
    f_scaler.fit(train_samples)

    from satsim.lstm.lstm_dataset import PyGSequenceDataset
    train_loader = DataLoader(PyGSequenceDataset(train_samples, f_scaler, t_scaler), batch_size=4)
    val_loader = DataLoader(PyGSequenceDataset(val_samples, f_scaler, t_scaler), batch_size=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = LSTMTrainer(
            model=model,
            device=device,
            target_scaler=t_scaler,
            artifacts_dir=tmpdir,
            lr=0.01,
            early_stopping_patience=3,
        )

        trainer.fit(train_loader, val_loader, epochs=2)

        assert (Path(tmpdir) / "lstm_best.pt").exists()
        assert (Path(tmpdir) / "lstm_last.pt").exists()
        assert (Path(tmpdir) / "training_history.csv").exists()
