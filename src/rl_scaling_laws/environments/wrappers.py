"""Environment wrappers for preprocessing and normalization."""

from collections import deque
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class FrameStack(gym.Wrapper):
    """Stack consecutive frames for temporal information."""

    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)

        low = np.repeat(env.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], num_stack, axis=0)

        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        return np.array(self.frames)


class NormalizeObservation(gym.Wrapper):
    """Normalize observations using running mean and std."""

    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ):
        super().__init__(env)
        self.epsilon = epsilon
        self.clip = clip
        self.obs_rms = RunningMeanStd(shape=env.observation_space.shape)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        self.obs_rms.update(obs)
        normalized = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        return np.clip(normalized, -self.clip, self.clip)


class NormalizeReward(gym.Wrapper):
    """Normalize rewards using running std."""

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.return_rms = RunningMeanStd(shape=())
        self.returns = 0.0

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        self.returns = 0.0
        return self.env.reset(**kwargs)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.returns = self.returns * self.gamma + reward
        self.return_rms.update(np.array([self.returns]))

        normalized_reward = reward / np.sqrt(self.return_rms.var + self.epsilon)
        normalized_reward = np.clip(normalized_reward, -self.clip, self.clip)

        if terminated or truncated:
            self.returns = 0.0

        return obs, normalized_reward, terminated, truncated, info


class RecordEpisodeStatistics(gym.Wrapper):
    """Record episode statistics like reward and length."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_reward = 0.0
        self.episode_length = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        self.episode_reward = 0.0
        self.episode_length = 0
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.episode_reward += reward
        self.episode_length += 1

        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_length,
            }

        return obs, reward, terminated, truncated, info


class ClipReward(gym.Wrapper):
    """Clip rewards to [-1, 1]."""

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, np.sign(reward), terminated, truncated, info


class ResizeObservation(gym.ObservationWrapper):
    """Resize image observations."""

    def __init__(self, env: gym.Env, size: int = 84):
        super().__init__(env)
        self.size = size

        obs_shape = (size, size) + env.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.resize(
            obs, (self.size, self.size), interpolation=cv2.INTER_AREA
        )


class GrayscaleObservation(gym.ObservationWrapper):
    """Convert RGB observations to grayscale."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        obs_shape = env.observation_space.shape[:2] + (1,)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        import cv2

        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(gray, -1)


class RunningMeanStd:
    """Running mean and standard deviation."""

    def __init__(self, shape: Tuple = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count
