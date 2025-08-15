import warnings
import gymnasium
from gymnasium.envs.registration import register

from dso.task.control.envs.continuous_cartpole import CustomCartPoleContinuous
from dso.task.control.envs.lander import LunarLanderMultiDiscrete

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