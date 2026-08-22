from gymnasium.envs.registration import register
from .routing_env import SatelliteRoutingEnv

register(
    id="SatelliteRouting-v0",
    entry_point="satsim.envs.routing_env:SatelliteRoutingEnv",
)

__all__ = ["SatelliteRoutingEnv"]
