"""Environment factory for creating RL environments."""

from typing import Any, Callable, Dict, List, Optional

import gymnasium as gym
import numpy as np

from rl_scaling_laws.environments.wrappers import (
    ClipReward,
    FrameStack,
    GrayscaleObservation,
    NormalizeObservation,
    NormalizeReward,
    RecordEpisodeStatistics,
    ResizeObservation,
)
from rl_scaling_laws.utils.logging import get_logger

logger = get_logger(__name__)


def make_env(
    domain: str,
    env_name: str,
    seed: int = 0,
    **kwargs,
) -> gym.Env:
    """Create a single environment instance.

    Args:
        domain: Environment domain (atari, dmc, metaworld, procgen).
        env_name: Environment name.
        seed: Random seed.
        **kwargs: Additional environment arguments.

    Returns:
        Gymnasium environment.
    """
    if domain == "atari":
        return _make_atari_env(env_name, seed, **kwargs)
    elif domain == "dmc":
        return _make_dmc_env(env_name, seed, **kwargs)
    elif domain == "metaworld":
        return _make_metaworld_env(env_name, seed, **kwargs)
    elif domain == "procgen":
        return _make_procgen_env(env_name, seed, **kwargs)
    else:
        raise ValueError(f"Unknown domain: {domain}")


def make_vec_env(
    domain: str,
    env_name: str,
    num_envs: int,
    seed: int = 0,
    **kwargs,
) -> gym.vector.VectorEnv:
    """Create vectorized environments.

    Args:
        domain: Environment domain.
        env_name: Environment name.
        num_envs: Number of parallel environments.
        seed: Base random seed.
        **kwargs: Additional environment arguments.

    Returns:
        Vectorized gymnasium environment.
    """
    def make_env_fn(idx: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            return make_env(domain, env_name, seed=seed + idx, **kwargs)
        return _init

    env_fns = [make_env_fn(i) for i in range(num_envs)]
    return gym.vector.AsyncVectorEnv(env_fns)


def _make_atari_env(
    game: str,
    seed: int,
    frame_stack: int = 4,
    screen_size: int = 84,
    terminal_on_life_loss: bool = True,
    clip_rewards: bool = True,
    noop_max: int = 30,
    **kwargs,
) -> gym.Env:
    """Create Atari environment with standard preprocessing."""
    try:
        # Create base environment
        env_id = f"ALE/{game}-v5"
        env = gym.make(
            env_id,
            frameskip=1,  # We handle frame skip ourselves
            repeat_action_probability=0.0,
            full_action_space=False,
        )

        # Apply wrappers
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=noop_max,
            frame_skip=4,
            screen_size=screen_size,
            terminal_on_life_loss=terminal_on_life_loss,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,
        )

        if frame_stack > 1:
            env = gym.wrappers.FrameStack(env, frame_stack)

        if clip_rewards:
            env = ClipReward(env)

        env = RecordEpisodeStatistics(env)
        env.reset(seed=seed)

        logger.info(f"Created Atari env: {game}")
        return env

    except Exception as e:
        logger.error(f"Failed to create Atari env {game}: {e}")
        raise


def _make_dmc_env(
    task_name: str,
    seed: int,
    from_pixels: bool = False,
    height: int = 84,
    width: int = 84,
    camera_id: int = 0,
    frame_stack: int = 3,
    action_repeat: int = 2,
    normalize_obs: bool = True,
    **kwargs,
) -> gym.Env:
    """Create DeepMind Control Suite environment."""
    try:
        from dm_control import suite

        # Parse task name
        if "_" in task_name:
            domain_name, task = task_name.rsplit("_", 1)
        else:
            domain_name = kwargs.get("domain_name", "walker")
            task = kwargs.get("task_name", "walk")

        # Create DMC environment
        dm_env = suite.load(domain_name, task, task_kwargs={"random": seed})

        # Wrap with gymnasium interface
        env = DMCWrapper(
            dm_env,
            from_pixels=from_pixels,
            height=height,
            width=width,
            camera_id=camera_id,
            action_repeat=action_repeat,
        )

        if from_pixels and frame_stack > 1:
            env = FrameStack(env, frame_stack)

        if normalize_obs and not from_pixels:
            env = NormalizeObservation(env)

        env = RecordEpisodeStatistics(env)

        logger.info(f"Created DMC env: {domain_name}_{task}")
        return env

    except ImportError:
        logger.error("dm_control not installed. Install with: pip install dm-control")
        raise
    except Exception as e:
        logger.error(f"Failed to create DMC env {task_name}: {e}")
        raise


def _make_metaworld_env(
    task: str,
    seed: int,
    normalize_obs: bool = True,
    **kwargs,
) -> gym.Env:
    """Create MetaWorld environment."""
    try:
        import metaworld

        # Get environment class
        ml1 = metaworld.ML1(task)
        env_cls = ml1.train_classes[task]
        env = env_cls()

        # Set task
        tasks = ml1.train_tasks
        env.set_task(tasks[0])

        # Wrap with gymnasium interface
        env = MetaWorldWrapper(env)

        if normalize_obs:
            env = NormalizeObservation(env)

        env = RecordEpisodeStatistics(env)
        env.reset(seed=seed)

        logger.info(f"Created MetaWorld env: {task}")
        return env

    except ImportError:
        logger.error("metaworld not installed")
        raise
    except Exception as e:
        logger.error(f"Failed to create MetaWorld env {task}: {e}")
        raise


def _make_procgen_env(
    game: str,
    seed: int,
    distribution_mode: str = "easy",
    num_levels: int = 200,
    start_level: int = 0,
    normalize_obs: bool = True,
    **kwargs,
) -> gym.Env:
    """Create Procgen environment."""
    try:
        from procgen import ProcgenEnv

        # Create Procgen environment
        env = ProcgenEnv(
            num_envs=1,
            env_name=game,
            distribution_mode=distribution_mode,
            num_levels=num_levels,
            start_level=start_level,
            rand_seed=seed,
        )

        # Wrap to remove batch dimension
        env = ProcgenWrapper(env)

        if normalize_obs:
            env = NormalizeObservation(env)

        env = RecordEpisodeStatistics(env)

        logger.info(f"Created Procgen env: {game}")
        return env

    except ImportError:
        logger.error("procgen not installed. Install with: pip install procgen")
        raise
    except Exception as e:
        logger.error(f"Failed to create Procgen env {game}: {e}")
        raise


class DMCWrapper(gym.Env):
    """Gymnasium wrapper for DeepMind Control Suite."""

    def __init__(
        self,
        env,
        from_pixels: bool = False,
        height: int = 84,
        width: int = 84,
        camera_id: int = 0,
        action_repeat: int = 2,
    ):
        self._env = env
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._action_repeat = action_repeat

        # Set up observation space
        if from_pixels:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(3, height, width), dtype=np.uint8
            )
        else:
            obs_spec = env.observation_spec()
            obs_dim = sum(np.prod(v.shape) for v in obs_spec.values())
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

        # Set up action space
        action_spec = env.action_spec()
        self.action_space = gym.spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        time_step = self._env.reset()
        obs = self._get_obs(time_step)
        return obs, {}

    def step(self, action):
        reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.0
            if time_step.last():
                break

        obs = self._get_obs(time_step)
        terminated = time_step.last()
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self._env.physics.render(
                height=self._height, width=self._width, camera_id=self._camera_id
            )
            obs = np.transpose(obs, (2, 0, 1))
            return obs
        else:
            obs = []
            for v in time_step.observation.values():
                obs.append(v.flatten())
            return np.concatenate(obs).astype(np.float32)


class MetaWorldWrapper(gym.Env):
    """Gymnasium wrapper for MetaWorld environments."""

    def __init__(self, env):
        self._env = env

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=env.action_space.low,
            high=env.action_space.high,
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs = self._env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return obs.astype(np.float32), {}

    def step(self, action):
        result = self._env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result

        return obs.astype(np.float32), reward, terminated, truncated, info


class ProcgenWrapper(gym.Env):
    """Gymnasium wrapper for Procgen (removes batch dimension)."""

    def __init__(self, env):
        self._env = env

        # Remove batch dimension from spaces
        obs_shape = env.observation_space.shape[1:]  # Remove first dim
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(env.action_space.n)

    def reset(self, **kwargs):
        obs = self._env.reset()
        return obs[0], {}

    def step(self, action):
        obs, reward, done, info = self._env.step(np.array([action]))
        return obs[0], reward[0], done[0], False, info[0] if info else {}
