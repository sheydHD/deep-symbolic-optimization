import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

from dso.task.control.envs.continuous_cartpole import CustomCartPoleContinuous
from dso.task.control.envs.lander import LunarLanderMultiDiscrete

try:
    import pybullet_envs
except ImportError as e:
    warnings.warn(f"pybullet_envs could not be imported or registered: {e}")

# Register custom environments
register(
    id="CustomCartPoleContinuous-v0",
    entry_point="dso.task.control.envs.continuous_cartpole:CustomCartPoleContinuous",
    max_episode_steps=500
)

# Register LunarLanderMultiDiscrete environment
register(
    id="LunarLanderMultiDiscrete-v0",
    entry_point="dso.task.control.envs.lander:LunarLanderMultiDiscrete",
    max_episode_steps=1000,
    reward_threshold=200
)