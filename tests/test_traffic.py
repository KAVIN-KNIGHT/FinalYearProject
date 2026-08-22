from collections import Counter
import numpy as np
import pytest

from satsim.config import TrafficConfig
from satsim.traffic import (
    create_traffic_profile,
    LowTrafficProfile,
    MediumTrafficProfile,
    HighTrafficProfile,
    PeakTrafficProfile,
    BurstTrafficProfile,
    FlashCrowdProfile,
    HotspotProfile,
    RandomTrafficProfile,
    SelfSimilarPoissonProfile,
    MixedTrafficProfile,
    PROFILE_REGISTRY,
)


def test_registry_contains_all_10_profiles():
    expected_profiles = [
        "low",
        "medium",
        "high",
        "peak",
        "burst",
        "flash_crowd",
        "hotspot",
        "random",
        "self_similar_poisson",
        "mixed",
    ]
    assert len(PROFILE_REGISTRY) == 10
    for p in expected_profiles:
        assert p in PROFILE_REGISTRY


def test_traffic_load_ordering():
    config = TrafficConfig()
    duration_s = 600.0

    p_low = LowTrafficProfile(config, seed=42)
    p_med = MediumTrafficProfile(config, seed=42)
    p_high = HighTrafficProfile(config, seed=42)
    p_peak = PeakTrafficProfile(config, seed=42)

    flows_low = p_low.generate_flows(0.0, duration_s, num_nodes=50)
    flows_med = p_med.generate_flows(0.0, duration_s, num_nodes=50)
    flows_high = p_high.generate_flows(0.0, duration_s, num_nodes=50)
    flows_peak = p_peak.generate_flows(0.0, duration_s, num_nodes=50)

    assert len(flows_low) < len(flows_med) < len(flows_high) < len(flows_peak)


def test_burst_profile_coefficient_of_variation():
    config = TrafficConfig()
    duration_s = 600.0
    profile = BurstTrafficProfile(config, seed=42)
    flows = profile.generate_flows(0.0, duration_s, num_nodes=50)

    bins = np.zeros(int(duration_s))
    for f in flows:
        idx = int(f.start_time_s)
        if 0 <= idx < len(bins):
            bins[idx] += f.total_packets

    mean_load = np.mean(bins)
    std_load = np.std(bins)
    cv = std_load / mean_load if mean_load > 0 else 0.0

    assert cv > 1.2


def test_flash_crowd_surge_ratio():
    config = TrafficConfig()
    duration_s = 600.0
    profile = FlashCrowdProfile(config, seed=42)
    flows = profile.generate_flows(0.0, duration_s, num_nodes=50)

    baseline_flows = [f for f in flows if f.start_time_s < 180.0]
    surge_flows = [f for f in flows if 220.0 <= f.start_time_s <= 380.0]

    rate_baseline = len(baseline_flows) / 180.0
    rate_surge = len(surge_flows) / 160.0

    # Tightened assertion threshold: rate_surge >= 8.0 * rate_baseline
    assert rate_surge >= 8.0 * rate_baseline


def test_hotspot_destination_concentration():
    config = TrafficConfig()
    duration_s = 600.0
    profile = HotspotProfile(config, seed=42)
    flows = profile.generate_flows(0.0, duration_s, num_nodes=50)

    hotspot_nodes = {0, 1, 2, 3, 4}
    dst_counts = Counter(f.dst_id for f in flows)
    hotspot_traffic_count = sum(dst_counts[h] for h in hotspot_nodes)
    total_traffic_count = len(flows)

    ratio = hotspot_traffic_count / total_traffic_count if total_traffic_count > 0 else 0.0
    assert ratio >= 0.70


def test_self_similar_poisson_statistical_sanity():
    config = TrafficConfig()
    duration_s = 600.0
    profile = SelfSimilarPoissonProfile(config, seed=42)
    flows = profile.generate_flows(0.0, duration_s, num_nodes=50)

    arr_times = [f.start_time_s for f in flows]
    inter_arrivals = np.diff(arr_times)

    mean_inter_arrival = np.mean(inter_arrivals)
    expected_mean = 1.0 / 15.0

    assert pytest.approx(mean_inter_arrival, rel=0.25) == expected_mean


def test_self_similarity_hurst_exponent():
    """
    Variance-Time plot test verifying long-range dependence / self-similarity.
    Aggregates arrival counts into block sizes m in [1, 2, 4, 8, 16, 32].
    Fits log(Var(X^(m))) vs log(m) slope -beta and asserts Hurst parameter H = 1 - beta/2 > 0.5.
    """
    config = TrafficConfig()
    duration_s = 3600.0  # 1 hour simulation for clean Hurst estimation
    profile = SelfSimilarPoissonProfile(config, seed=42)
    flows = profile.generate_flows(0.0, duration_s, num_nodes=50)

    time_series = np.zeros(int(duration_s))
    for f in flows:
        t_start_idx = int(f.start_time_s)
        t_end_idx = min(int(duration_s), int(f.start_time_s + f.duration_s))
        if t_end_idx > t_start_idx:
            time_series[t_start_idx:t_end_idx] += f.packets_per_sec

    block_sizes = [1, 2, 4, 8, 16, 32]
    variances = []

    for m in block_sizes:
        num_blocks = len(time_series) // m
        blocked = time_series[: num_blocks * m].reshape(-1, m).mean(axis=1)
        variances.append(np.var(blocked))

    log_m = np.log(block_sizes)
    log_var = np.log(variances)

    slope, _ = np.polyfit(log_m, log_var, 1)
    beta = -slope
    hurst = 1.0 - (beta / 2.0)

    # Assert long-range dependence (Hurst parameter H > 0.5, slope beta < 1.0)
    assert hurst > 0.5
    assert beta < 1.0


def test_mixed_profile_compositing():
    """
    Verifies that MixedTrafficProfile successfully composites baseline, burst, and hotspot traffic:
    1. CV > 1.0 confirming burst component presence.
    2. Hotspot destination concentration >= 30% confirming hotspot component presence.
    """
    config = TrafficConfig()
    duration_s = 600.0
    profile = MixedTrafficProfile(config, seed=42)
    flows = profile.generate_flows(0.0, duration_s, num_nodes=50)

    bins = np.zeros(int(duration_s))
    for f in flows:
        idx = int(f.start_time_s)
        if 0 <= idx < len(bins):
            bins[idx] += f.total_packets

    mean_load = np.mean(bins)
    std_load = np.std(bins)
    cv = std_load / mean_load if mean_load > 0 else 0.0

    hotspot_nodes = {0, 1, 2, 3, 4}
    dst_counts = Counter(f.dst_id for f in flows)
    hotspot_count = sum(dst_counts[h] for h in hotspot_nodes)
    ratio = hotspot_count / len(flows)

    assert cv > 1.0
    assert ratio >= 0.30


def test_all_10_profiles_generation_smoke():
    config = TrafficConfig()
    duration_s = 600.0

    for name in PROFILE_REGISTRY:
        prof = create_traffic_profile(name, config, seed=42)
        flows = prof.generate_flows(0.0, duration_s, num_nodes=20)
        assert len(flows) > 0, f"Profile '{name}' generated zero flows!"
        for f in flows[:5]:
            assert f.start_time_s >= 0.0
            assert f.duration_s > 0.0
            assert f.packet_size_bytes in range(64, 1501)
            assert f.priority in ["high", "medium", "low"]
