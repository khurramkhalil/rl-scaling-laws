"""Environment registry and configuration."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EnvConfig:
    """Configuration for an environment."""

    domain: str
    name: str
    obs_type: str  # "vector", "image"
    action_type: str  # "discrete", "continuous"
    max_episode_steps: int
    kwargs: Dict[str, Any]


class EnvironmentRegistry:
    """Registry for environment configurations."""

    _envs: Dict[str, EnvConfig] = {}

    @classmethod
    def register(cls, config: EnvConfig) -> None:
        """Register an environment configuration."""
        key = f"{config.domain}/{config.name}"
        cls._envs[key] = config

    @classmethod
    def get(cls, domain: str, name: str) -> EnvConfig:
        """Get environment configuration."""
        key = f"{domain}/{name}"
        if key not in cls._envs:
            raise KeyError(f"Environment {key} not found")
        return cls._envs[key]

    @classmethod
    def list_envs(cls, domain: Optional[str] = None) -> List[str]:
        """List registered environments."""
        if domain:
            return [k for k in cls._envs.keys() if k.startswith(f"{domain}/")]
        return list(cls._envs.keys())


def get_env_config(domain: str, name: str) -> EnvConfig:
    """Get environment configuration."""
    return EnvironmentRegistry.get(domain, name)


# Register default environments
def _register_defaults():
    """Register default environment configurations."""

    # Atari games
    for game in ["Breakout", "Pong", "SpaceInvaders", "Qbert", "Seaquest"]:
        EnvironmentRegistry.register(EnvConfig(
            domain="atari",
            name=game,
            obs_type="image",
            action_type="discrete",
            max_episode_steps=27000,
            kwargs={
                "frame_stack": 4,
                "screen_size": 84,
                "terminal_on_life_loss": True,
            }
        ))

    # DMC tasks
    for domain_name, tasks in [
        ("walker", ["walk", "run"]),
        ("humanoid", ["walk", "run", "stand"]),
        ("cheetah", ["run"]),
    ]:
        for task in tasks:
            EnvironmentRegistry.register(EnvConfig(
                domain="dmc",
                name=f"{domain_name}_{task}",
                obs_type="vector",
                action_type="continuous",
                max_episode_steps=1000,
                kwargs={
                    "domain_name": domain_name,
                    "task_name": task,
                }
            ))

    # MetaWorld tasks
    for task in ["reach-v2", "push-v2", "pick-place-v2", "door-open-v2"]:
        EnvironmentRegistry.register(EnvConfig(
            domain="metaworld",
            name=task,
            obs_type="vector",
            action_type="continuous",
            max_episode_steps=500,
            kwargs={"task": task}
        ))

    # Procgen games
    for game in ["starpilot", "coinrun", "bigfish", "bossfight"]:
        EnvironmentRegistry.register(EnvConfig(
            domain="procgen",
            name=game,
            obs_type="image",
            action_type="discrete",
            max_episode_steps=1000,
            kwargs={
                "distribution_mode": "easy",
                "num_levels": 200,
            }
        ))


_register_defaults()
