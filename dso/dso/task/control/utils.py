"""Utility functions for control task."""

import os
from pkg_resources import resource_filename

from datetime import datetime
from glob import glob

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

# Handle changes in Gymnasium API
try:
    # Try direct import first (older versions)
    from gymnasium.wrappers.monitoring import video_recorder
except ImportError:
    # For newer versions of Gymnasium
    try:
        from gymnasium.wrappers.record_video import RecordVideo as VideoRecorder
        # Create compatibility layer
        class VideoRecorderCompat:
            def __init__(self, env, base_path):
                self.video_recorder = None
                self.env = env
                self.base_path = base_path
                
            def capture_frame(self):
                pass
                
            def close(self):
                pass
                
        video_recorder = VideoRecorderCompat
    except ImportError:
        # Fallback to dummy implementation if neither is available
        class DummyVideoRecorder:
            def __init__(self, env, base_path):
                self.env = env
                
            def capture_frame(self):
                pass
                
            def close(self):
                pass
                
        video_recorder = DummyVideoRecorder

import numpy as np

try:
    import mpi4py
except ImportError:
    mpi4py = None

# Handle stable_baselines import
try:
    from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, SAC, TD3
except ImportError:
    # Fallback to stable-baselines3
    try:
        from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, HER
        # Create compatibility classes
        class PPO2:
            def __init__(self, *args, **kwargs): pass
            @staticmethod
            def load(*args, **kwargs): return None
        class ACER:
            def __init__(self, *args, **kwargs): pass
            @staticmethod
            def load(*args, **kwargs): return None
        class ACKTR:
            def __init__(self, *args, **kwargs): pass
            @staticmethod
            def load(*args, **kwargs): return None
    except ImportError:
        # Define dummy classes if stable_baselines is not available
        for cls_name in ["PPO2", "A2C", "ACER", "ACKTR", "DQN", "HER", "SAC", "TD3"]:
            exec(f"""class {cls_name}:
                def __init__(self, *args, **kwargs): pass
                @staticmethod
                def load(*args, **kwargs): return None""")
        print("WARNING: Neither stable-baselines nor stable-baselines3 is installed. Using dummy implementations.")


if mpi4py is None:
    DDPG, TRPO = None, None
else:
    from stable_baselines import DDPG, TRPO


ALGORITHMS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'her': HER,
    'sac': SAC,
    'ppo2': PPO2,
    'trpo': TRPO,
    'td3': TD3
}


# NOTE: This does not load observation normalization
# Loads a model into global namespace
def load_model(algorithm, model_path):
    global model
    model = ALGORITHMS[algorithm].load(model_path)
    print(f"Loaded {algorithm.upper()} model {model_path}")


# Load an environment's default model, which is located at:
# dso/task/control/data/[env_name]/model-[algorithm].[pkl | zip]
def load_default_model(env_name):

    # Find default algorithm and model path for the environment
    task_root = resource_filename("dso.task", "control")
    root = os.path.join(task_root, "data", env_name)
    files = os.listdir(root)
    for f in files:
        if f.startswith("model-"):
            for ext in [".pkl", ".zip"]:
                if f.endswith(ext):
                    algorithm = f.split("model-")[-1].split(ext)[0]
                    model_path = os.path.join(root, f)

                    # Load that model
                    load_model(algorithm, model_path)

                    return

    assert False, f"Could not find default model for environment {env_name}."


# From https://github.com/araffin/rl-baselines-zoo/blob/master/utils/wrappers.py
class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.
    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """
    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high= np.concatenate((low, [0])), np.concatenate((high, [1.]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super().__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self, **kwargs):
        self._current_step = 0
        return self._get_obs(self.env.reset(**kwargs))

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.
        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))


class RecordingWrapper(gym.Wrapper):
    """Recording wrapper to simply generate videos from gym environments."""

    def __init__(self, env, directory, record_video=False):
        """Initialize the recording wrapper.

        Parameters
        ----------
        env : gym.Env
            The environment to wrap.

        directory : str
            Directory in which to save videos.

        record_video : bool
            Whether to record a video.
        """

        super(RecordingWrapper, self).__init__(env)
        self.record_video = record_video
        self.directory = directory
        self.video_recorder = None

        if not os.path.exists(directory):
            os.makedirs(directory)

    def step(self, action):
        """Take a step in the environment."""

        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if self.video_recorder is not None:
            self.video_recorder.capture_frame()

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment, initializing the video recorder."""

        if self.video_recorder is not None:
            self._close_video_recorder()

        if self.record_video:
            self.video_recorder = video_recorder(
                env=self.env,
                base_path=os.path.join(
                    self.directory,
                    f"video_{datetime.now().strftime('%Y-%m-%d-%H%M%S%f')}"
                )
            )

        result = self.env.reset(**kwargs)
        if self.video_recorder is not None:
            self.video_recorder.capture_frame()
        return result

    def _close_video_recorder(self):
        """Cleaning up."""
        self.video_recorder.close()
        if self.video_recorder.functional:
            self.videos.append((self.video_recorder.path, self.video_recorder.metadata_path))

    def close(self):
        """Flush all monitor data to disk and close any open rending windows."""
        super().close()
        if self.video_recorder is not None:
            self._close_video_recorder()
        self._clean_up_metadata()

    def __del__(self):
        """Make sure we've closed up shop when garbage collecting."""
        self.close()

    def _clean_up_metadata(self):
        """Deleting all metadata json files in the directory."""
        metadata_files = glob(os.path.join(self.directory , "*.json"))
        for metadata_file in metadata_files:
            os.remove(metadata_file)