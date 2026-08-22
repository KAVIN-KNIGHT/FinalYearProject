from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Type, Optional
import numpy as np

from satsim.config import TrafficConfig
from .flows import PacketFlow, sample_packet_size, PRIORITY_LEVELS


class TrafficProfile(ABC):
    """Abstract Base Class for all network traffic profiles."""

    def __init__(self, config: TrafficConfig, seed: int = 42):
        self.config = config
        self.seed = seed

    @abstractmethod
    def generate_flows(
        self,
        t_start: float,
        duration: float,
        num_nodes: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> List[PacketFlow]:
        """Generates list of PacketFlow objects over [t_start, t_start + duration]."""
        pass


class ConstantRateProfile(TrafficProfile):
    """Vectorized steady-state traffic profile generator."""

    def __init__(self, config: TrafficConfig, mean_rate_pkts_s: float, seed: int = 42):
        super().__init__(config, seed)
        self.mean_rate_pkts_s = mean_rate_pkts_s

    def generate_flows(
        self,
        t_start: float,
        duration: float,
        num_nodes: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> List[PacketFlow]:
        if rng is None:
            rng = np.random.default_rng(self.seed)

        expected_flows = int(duration * num_nodes * (self.mean_rate_pkts_s / 50.0))
        n_flows = max(1, int(rng.poisson(max(1, expected_flows))))

        start_times = rng.uniform(t_start, t_start + duration, size=n_flows)
        start_times.sort()

        sources = rng.integers(0, num_nodes, size=n_flows)
        offsets = rng.integers(1, num_nodes, size=n_flows)
        destinations = (sources + offsets) % num_nodes

        priorities = rng.choice(PRIORITY_LEVELS, p=[0.1, 0.3, 0.6], size=n_flows)
        durations = rng.uniform(1.0, 5.0, size=n_flows)
        rates = rng.uniform(5.0, 20.0, size=n_flows)

        flows = []
        for i in range(n_flows):
            prio = str(priorities[i])
            pkt_size = sample_packet_size(prio, rng)
            flows.append(
                PacketFlow(
                    flow_id=i,
                    src_id=int(sources[i]),
                    dst_id=int(destinations[i]),
                    priority=prio,
                    packet_size_bytes=pkt_size,
                    start_time_s=float(start_times[i]),
                    duration_s=float(durations[i]),
                    packets_per_sec=float(rates[i]),
                )
            )
        return flows


class LowTrafficProfile(ConstantRateProfile):
    def __init__(self, config: TrafficConfig, seed: int = 42):
        super().__init__(config, mean_rate_pkts_s=2.0, seed=seed)


class MediumTrafficProfile(ConstantRateProfile):
    def __init__(self, config: TrafficConfig, seed: int = 42):
        super().__init__(config, mean_rate_pkts_s=10.0, seed=seed)


class HighTrafficProfile(ConstantRateProfile):
    def __init__(self, config: TrafficConfig, seed: int = 42):
        super().__init__(config, mean_rate_pkts_s=40.0, seed=seed)


class PeakTrafficProfile(ConstantRateProfile):
    def __init__(self, config: TrafficConfig, seed: int = 42):
        super().__init__(config, mean_rate_pkts_s=100.0, seed=seed)


class BurstTrafficProfile(TrafficProfile):
    """ON/OFF burst generator producing high variance & coefficient of variation CV > 1.2."""

    def generate_flows(
        self,
        t_start: float,
        duration: float,
        num_nodes: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> List[PacketFlow]:
        if rng is None:
            rng = np.random.default_rng(self.seed)

        flows = []
        flow_id = 0
        step_dt = 1.0
        num_steps = int(np.ceil(duration / step_dt))

        is_burst = rng.random(size=num_steps) < 0.15
        mults = np.where(is_burst, 30.0, 0.1)

        for step in range(num_steps):
            t_curr = t_start + step * step_dt
            n_flows = rng.poisson(mults[step])
            if n_flows == 0:
                continue

            sources = rng.integers(0, num_nodes, size=n_flows)
            offsets = rng.integers(1, num_nodes, size=n_flows)
            destinations = (sources + offsets) % num_nodes
            priorities = rng.choice(PRIORITY_LEVELS, p=[0.1, 0.3, 0.6], size=n_flows)
            durations = rng.uniform(0.5, 2.0, size=n_flows)
            rates = rng.uniform(10.0, 50.0, size=n_flows)

            for i in range(n_flows):
                prio = str(priorities[i])
                pkt_size = sample_packet_size(prio, rng)
                flows.append(
                    PacketFlow(
                        flow_id=flow_id,
                        src_id=int(sources[i]),
                        dst_id=int(destinations[i]),
                        priority=prio,
                        packet_size_bytes=pkt_size,
                        start_time_s=t_curr,
                        duration_s=float(durations[i]),
                        packets_per_sec=float(rates[i]),
                    )
                )
                flow_id += 1
        return flows


class FlashCrowdProfile(TrafficProfile):
    """Normal baseline with a massive surge window during the middle third of duration."""

    def generate_flows(
        self,
        t_start: float,
        duration: float,
        num_nodes: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> List[PacketFlow]:
        if rng is None:
            rng = np.random.default_rng(self.seed)

        flows = []
        flow_id = 0
        step_dt = 1.0
        num_steps = int(np.ceil(duration / step_dt))

        surge_start = t_start + duration * 0.33
        surge_end = t_start + duration * 0.66
        target_dst = 0

        for step in range(num_steps):
            t_curr = t_start + step * step_dt
            in_flash = surge_start <= t_curr <= surge_end
            n_flows = rng.poisson(20.0 if in_flash else 1.0)
            if n_flows == 0:
                continue

            sources = rng.integers(0, num_nodes, size=n_flows)
            offsets = rng.integers(1, num_nodes, size=n_flows)
            dest_random = (sources + offsets) % num_nodes

            is_target = in_flash & (rng.random(size=n_flows) < 0.8)
            destinations = np.where(is_target, target_dst, dest_random)

            priorities = rng.choice(PRIORITY_LEVELS, p=[0.1, 0.3, 0.6], size=n_flows)
            durations = rng.uniform(1.0, 5.0, size=n_flows)
            rates = rng.uniform(5.0, 20.0, size=n_flows)

            for i in range(n_flows):
                prio = str(priorities[i])
                pkt_size = sample_packet_size(prio, rng)
                flows.append(
                    PacketFlow(
                        flow_id=flow_id,
                        src_id=int(sources[i]),
                        dst_id=int(destinations[i]),
                        priority=prio,
                        packet_size_bytes=pkt_size,
                        start_time_s=t_curr,
                        duration_s=float(durations[i]),
                        packets_per_sec=float(rates[i]),
                    )
                )
                flow_id += 1
        return flows


class HotspotProfile(TrafficProfile):
    """Concentrates >= 70% of traffic to a small set of hotspot destination nodes."""

    def generate_flows(
        self,
        t_start: float,
        duration: float,
        num_nodes: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> List[PacketFlow]:
        if rng is None:
            rng = np.random.default_rng(self.seed)

        flows = []
        flow_id = 0
        step_dt = 1.0
        num_steps = int(np.ceil(duration / step_dt))
        hotspot_nodes = np.arange(min(5, num_nodes))

        for step in range(num_steps):
            t_curr = t_start + step * step_dt
            n_flows = rng.poisson(3.0)
            if n_flows == 0:
                continue

            sources = rng.integers(0, num_nodes, size=n_flows)
            offsets = rng.integers(1, num_nodes, size=n_flows)
            dest_random = (sources + offsets) % num_nodes
            dest_hotspot = rng.choice(hotspot_nodes, size=n_flows)

            is_hotspot = rng.random(size=n_flows) < 0.8
            destinations = np.where(is_hotspot, dest_hotspot, dest_random)

            priorities = rng.choice(PRIORITY_LEVELS, p=[0.1, 0.3, 0.6], size=n_flows)
            durations = rng.uniform(1.0, 5.0, size=n_flows)
            rates = rng.uniform(5.0, 20.0, size=n_flows)

            for i in range(n_flows):
                prio = str(priorities[i])
                pkt_size = sample_packet_size(prio, rng)
                flows.append(
                    PacketFlow(
                        flow_id=flow_id,
                        src_id=int(sources[i]),
                        dst_id=int(destinations[i]),
                        priority=prio,
                        packet_size_bytes=pkt_size,
                        start_time_s=t_curr,
                        duration_s=float(durations[i]),
                        packets_per_sec=float(rates[i]),
                    )
                )
                flow_id += 1
        return flows


class RandomTrafficProfile(TrafficProfile):
    """Uniformly random arrival times, sources, destinations, and sizes."""

    def generate_flows(
        self,
        t_start: float,
        duration: float,
        num_nodes: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> List[PacketFlow]:
        if rng is None:
            rng = np.random.default_rng(self.seed)

        num_total_flows = int(duration * 5.0)

        start_times = rng.uniform(t_start, t_start + duration, size=num_total_flows)
        start_times.sort()

        sources = rng.integers(0, num_nodes, size=num_total_flows)
        offsets = rng.integers(1, num_nodes, size=num_total_flows)
        destinations = (sources + offsets) % num_nodes

        priorities = rng.choice(PRIORITY_LEVELS, p=[0.1, 0.3, 0.6], size=num_total_flows)
        durations = rng.uniform(0.5, 10.0, size=num_total_flows)
        rates = rng.uniform(1.0, 50.0, size=num_total_flows)

        flows = []
        for i in range(num_total_flows):
            prio = str(priorities[i])
            pkt_size = sample_packet_size(prio, rng)
            flows.append(
                PacketFlow(
                    flow_id=i,
                    src_id=int(sources[i]),
                    dst_id=int(destinations[i]),
                    priority=prio,
                    packet_size_bytes=pkt_size,
                    start_time_s=float(start_times[i]),
                    duration_s=float(durations[i]),
                    packets_per_sec=float(rates[i]),
                )
            )
        return flows


class SelfSimilarPoissonProfile(TrafficProfile):
    """
    Poisson inter-arrival session process combined with heavy-tailed Pareto flow durations.
    Uses shape parameter alpha = 1.5 in (1, 2), establishing heavy tails and long-range dependence
    (Hurst parameter H > 0.5) according to the Hurst-Mandelbrot theorem for aggregated ON/OFF processes.
    """

    PARETO_ALPHA: float = 1.5  # Pin alpha in (1, 2) for infinite variance / long-range dependence

    def generate_flows(
        self,
        t_start: float,
        duration: float,
        num_nodes: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> List[PacketFlow]:
        if rng is None:
            rng = np.random.default_rng(self.seed)

        target_lambda = 15.0  # 15 arrivals/sec aggregate
        num_expected = int(duration * target_lambda)

        inter_arrivals = rng.exponential(1.0 / target_lambda, size=num_expected)
        start_times = t_start + np.cumsum(inter_arrivals)

        mask = start_times < (t_start + duration)
        start_times = start_times[mask]
        n_flows = len(start_times)

        sources = rng.integers(0, num_nodes, size=n_flows)
        offsets = rng.integers(1, num_nodes, size=n_flows)
        destinations = (sources + offsets) % num_nodes

        priorities = rng.choice(PRIORITY_LEVELS, p=[0.1, 0.3, 0.6], size=n_flows)
        # Heavy-tailed Pareto distribution with alpha = 1.5
        pareto_vals = (rng.pareto(self.PARETO_ALPHA, size=n_flows) + 1.0) * 1.0
        durations = np.clip(pareto_vals, 0.5, 60.0)
        rates = rng.uniform(5.0, 30.0, size=n_flows)

        flows = []
        for i in range(n_flows):
            prio = str(priorities[i])
            pkt_size = sample_packet_size(prio, rng)
            flows.append(
                PacketFlow(
                    flow_id=i,
                    src_id=int(sources[i]),
                    dst_id=int(destinations[i]),
                    priority=prio,
                    packet_size_bytes=pkt_size,
                    start_time_s=float(start_times[i]),
                    duration_s=float(durations[i]),
                    packets_per_sec=float(rates[i]),
                )
            )
        return flows


class MixedTrafficProfile(TrafficProfile):
    """Composite mix of background traffic + bursts + hotspot demand."""

    def generate_flows(
        self,
        t_start: float,
        duration: float,
        num_nodes: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> List[PacketFlow]:
        if rng is None:
            rng = np.random.default_rng(self.seed)

        low_prof = LowTrafficProfile(self.config, seed=self.seed)
        burst_prof = BurstTrafficProfile(self.config, seed=self.seed + 1)
        hotspot_prof = HotspotProfile(self.config, seed=self.seed + 2)

        f_low = low_prof.generate_flows(t_start, duration, num_nodes, rng)
        f_burst = burst_prof.generate_flows(t_start, duration, num_nodes, rng)
        f_hotspot = hotspot_prof.generate_flows(t_start, duration, num_nodes, rng)

        all_flows = f_low + f_burst + f_hotspot
        for idx, f in enumerate(all_flows):
            f.flow_id = idx

        all_flows.sort(key=lambda x: x.start_time_s)
        return all_flows


PROFILE_REGISTRY: Dict[str, Type[TrafficProfile]] = {
    "low": LowTrafficProfile,
    "medium": MediumTrafficProfile,
    "high": HighTrafficProfile,
    "peak": PeakTrafficProfile,
    "burst": BurstTrafficProfile,
    "flash_crowd": FlashCrowdProfile,
    "hotspot": HotspotProfile,
    "random": RandomTrafficProfile,
    "self_similar_poisson": SelfSimilarPoissonProfile,
    "mixed": MixedTrafficProfile,
}


def create_traffic_profile(name: str, config: TrafficConfig, seed: int = 42) -> TrafficProfile:
    name_clean = name.lower().strip()
    if name_clean not in PROFILE_REGISTRY:
        raise ValueError(
            f"Unknown traffic profile '{name}'. Choose from: {list(PROFILE_REGISTRY.keys())}"
        )
    return PROFILE_REGISTRY[name_clean](config, seed=seed)
