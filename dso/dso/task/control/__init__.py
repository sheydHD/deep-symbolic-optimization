import warnings
import gymnasium
from gymnasium.envs.registration import register
# the above import is here to register the pybullet environments to Gym. Don't delete!
# without the import you won't be able to use pybullet environments.

from dso.task.control.envs.continuous_cartpole import CustomCartPoleContinuous
from dso.task.control.envs.lander import LunarLanderMultiDiscrete

# Legacy/optional environment registrations (commented out for reference):
# from dso.task.control.envs.pendulum import CustomPendulumEnv
# from dso.task.control.envs.cartpole_bullet import CustomCartPoleContinuousBulletEnv
# from dso.task.control.envs.lander import CustomLunarLander

# register(
#     id="CustomPendulum-v0",
#     entry_point="dso.task.control.envs.pendulum:CustomPendulumEnv",
#     max_episode_steps=200
# )
# register(
#     id="CustomCartPoleContinuousBulletEnv-v0",
#     entry_point="dso.task.control.envs.cartpole_bullet:CustomCartPoleContinuousBulletEnv",
#     max_episode_steps=500
# )
# register(
#     id="LunarLanderNoRewardShaping-v0",
#     entry_point="dso.task.control.envs.lander:CustomLunarLander",
#     max_episode_steps=1000,
#     reward_threshold=200,
#     kwargs={"reward_shaping_coef": 0, "continuous": False}
# )
# register(
#     id="LunarLanderContinuousNoRewardShaping-v0",
#     entry_point="dso.task.control.envs.lander:CustomLunarLander",
#     max_episode_steps=1000,
#     reward_threshold=200,
#     kwargs={"reward_shaping_coef": 0, "continuous": True}
# )
# register(
#     id="LunarLanderCustom-v0",
#     entry_point="dso.task.control.envs.lander:CustomLunarLander",
#     max_episode_steps=1000,
#     reward_threshold=200
# )

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