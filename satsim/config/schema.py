from __future__ import annotations
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field, model_validator


class ConstellationConfig(BaseModel):
    num_satellites: int = Field(default=100, ge=1)
    num_planes: int = Field(default=10, ge=1)
    satellites_per_plane: Optional[int] = Field(default=None, ge=1)
    altitude_km: float = Field(default=550.0, gt=0.0)
    inclination_deg: float = Field(default=53.0, ge=0.0, le=180.0)
    eccentricity: float = Field(default=0.0, ge=0.0, lt=1.0)
    propagation: Literal["keplerian", "sgp4"] = "keplerian"

    @property
    def sats_per_plane(self) -> int:
        if self.satellites_per_plane is not None:
            return self.satellites_per_plane
        return self.num_satellites // self.num_planes

    @model_validator(mode="after")
    def validate_sat_plane_divisibility(self) -> ConstellationConfig:
        if self.satellites_per_plane is not None:
            if self.num_satellites != self.num_planes * self.satellites_per_plane:
                raise ValueError(
                    f"num_satellites ({self.num_satellites}) must equal num_planes ({self.num_planes}) * satellites_per_plane ({self.satellites_per_plane})."
                )
        if self.num_satellites % self.num_planes != 0:
            raise ValueError(
                f"num_satellites ({self.num_satellites}) must be evenly divisible by num_planes ({self.num_planes})."
            )
        return self


class ISLConfig(BaseModel):
    max_range_km: float = Field(default=5000.0, gt=0.0)
    min_elevation_deg: float = Field(default=10.0, ge=0.0, le=90.0)


class GroundStationLocation(BaseModel):
    name: str
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    alt_km: float = Field(default=0.0, ge=0.0)


class GroundStationsConfig(BaseModel):
    count: int = Field(default=12, ge=1)
    placement: Literal["config_list", "random", "uniform"] = "config_list"
    locations: List[GroundStationLocation] = Field(default_factory=list)


class TrafficConfig(BaseModel):
    profile: str = "mixed"
    priorities: List[str] = Field(default_factory=lambda: ["high", "medium", "low"])
    packet_size_bytes: List[int] = Field(default_factory=lambda: [64, 1500])


class EventsConfig(BaseModel):
    enabled_types: List[str] = Field(
        default_factory=lambda: [
            "isl_failure",
            "sat_failure",
            "congestion",
            "weather_attenuation",
        ]
    )
    failure_rate_per_hour: float = Field(default=0.5, ge=0.0)


class GATExportConfig(BaseModel):
    enabled: bool = True
    snapshot_interval_steps: int = Field(default=1, ge=1)


class LSTMExportConfig(BaseModel):
    enabled: bool = True
    window_size: int = Field(default=30, ge=1)
    stride: int = Field(default=5, ge=1)
    format: Literal["parquet", "csv"] = "parquet"


class ExportConfig(BaseModel):
    gat: GATExportConfig = Field(default_factory=GATExportConfig)
    lstm: LSTMExportConfig = Field(default_factory=LSTMExportConfig)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    structured: bool = True


class SimConfig(BaseModel):
    seed: int = 42
    constellation: ConstellationConfig = Field(default_factory=ConstellationConfig)
    timestep_seconds: float = Field(default=5.0, gt=0.0)
    duration_seconds: float = Field(default=3600.0, gt=0.0)
    isl: ISLConfig = Field(default_factory=ISLConfig)
    ground_stations: GroundStationsConfig = Field(default_factory=GroundStationsConfig)
    traffic: TrafficConfig = Field(default_factory=TrafficConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load_yaml(cls, path: Optional[str | Path] = None) -> SimConfig:
        if path is None:
            path = Path(__file__).parent / "defaults.yaml"
        else:
            path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Optional[str | Path] = None) -> str:
        data = self.model_dump()
        yaml_str = yaml.dump(data, sort_keys=False)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_str)
        return yaml_str
