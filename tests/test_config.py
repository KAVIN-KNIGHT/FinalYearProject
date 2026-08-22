import tempfile
from pathlib import Path
import pytest
from satsim.config import SimConfig, ConstellationConfig


def test_load_defaults():
    config = SimConfig.load_yaml()
    assert config.seed == 42
    assert config.constellation.num_satellites == 100
    assert config.constellation.num_planes == 10
    assert config.constellation.sats_per_plane == 10
    assert len(config.ground_stations.locations) == 12
    assert config.traffic.profile == "mixed"
    assert config.export.gat.enabled is True
    assert config.export.lstm.format == "parquet"


def test_yaml_roundtrip():
    config = SimConfig.load_yaml()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "config_used.yaml"
        config.to_yaml(out_file)

        reloaded = SimConfig.load_yaml(out_file)
        assert reloaded.model_dump() == config.model_dump()


def test_validation_bounds():
    with pytest.raises(Exception):
        ConstellationConfig(num_satellites=0)

    with pytest.raises(Exception):
        ConstellationConfig(inclination_deg=200.0)
