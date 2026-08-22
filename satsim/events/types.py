from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union


class EventType(str, Enum):
    ISL_FAILURE = "isl_failure"
    SAT_FAILURE = "sat_failure"
    CONGESTION = "congestion"
    BUFFER_OVERFLOW = "buffer_overflow"
    WEATHER_ATTENUATION = "weather_attenuation"
    SOLAR_INTERFERENCE = "solar_interference"
    GS_CONGESTION = "gs_congestion"
    LINK_DEGRADATION = "link_degradation"
    RECOVERY = "recovery"


@dataclass
class SimEvent:
    event_id: int
    event_type: EventType
    start_time_s: float
    duration_s: float
    target_id: Union[int, Tuple[int, int], str]  # Sat ID (int), Edge (u, v), or GS name (str)
    params: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    recovered_at_s: Optional[float] = None

    @property
    def end_time_s(self) -> float:
        return self.start_time_s + self.duration_s
