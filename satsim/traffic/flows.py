from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

PRIORITY_LEVELS = ["high", "medium", "low"]


@dataclass
class PacketFlow:
    flow_id: int
    src_id: int
    dst_id: int
    priority: str  # "high", "medium", "low"
    packet_size_bytes: int  # 64 to 1500
    start_time_s: float
    duration_s: float
    packets_per_sec: float

    @property
    def total_packets(self) -> int:
        return max(1, int(np.round(self.duration_s * self.packets_per_sec)))

    @property
    def total_bytes(self) -> int:
        return self.total_packets * self.packet_size_bytes

    @property
    def rate_bps(self) -> float:
        return self.packets_per_sec * self.packet_size_bytes * 8.0


def sample_priority(rng: np.random.Generator) -> str:
    """Samples priority based on realistic network mix: 10% high, 30% medium, 60% low."""
    return str(rng.choice(PRIORITY_LEVELS, p=[0.1, 0.3, 0.6]))


def sample_packet_size(priority: str, rng: np.random.Generator) -> int:
    """Samples packet size based on flow priority."""
    if priority == "high":
        return int(rng.integers(64, 256))
    elif priority == "medium":
        return int(rng.integers(256, 1024))
    else:
        return int(rng.integers(1024, 1501))
