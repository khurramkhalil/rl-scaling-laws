"""Main training entry point using Hydra."""

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from rl_scaling_laws.algorithms import get_algorithm
from rl_scaling_laws.environments.factory import make_env
from rl_scaling_laws.training.trainer import Trainer
from rl_scaling_laws.utils.device import get_device
from rl_scaling_laws.utils.logging import setup_logging, get_logger
from rl_scaling_laws.utils.seed import set_seed

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Main training function.

    Args:
        cfg: Hydra configuration.

    Returns:
        Final evaluation reward.
    """
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set up directories
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log_dir = output_dir / "logs"
    save_dir = output_dir / "checkpoints"

    # Set up logging
    setup_logging(log_dir)

    # Set random seed
    set_seed(cfg.seed, deterministic=cfg.get("deterministic", False))

    # Get device
    device = get_device(cfg.device if cfg.device != "auto" else None)
    logger.info(f"Using device: {device}")

    # Create environments
    env_cfg = cfg.environment
    train_env = make_env(
        domain=env_cfg.domain,
        env_name=env_cfg.get("game", env_cfg.get("task", "default")),
        seed=cfg.seed,
        **OmegaConf.to_container(env_cfg, resolve=True),
    )

    eval_env = make_env(
        domain=env_cfg.domain,
        env_name=env_cfg.get("game", env_cfg.get("task", "default")),
        seed=cfg.seed + 1000,
        **OmegaConf.to_container(env_cfg, resolve=True),
    )

    logger.info(f"Environment: {env_cfg.domain}/{env_cfg.get('game', env_cfg.get('task'))}")
    logger.info(f"Observation space: {train_env.observation_space}")
    logger.info(f"Action space: {train_env.action_space}")

    # Create algorithm
    algo_cfg = cfg.algorithm
    algo_name = algo_cfg.name

    # Build algorithm kwargs
    algo_kwargs = {
        "observation_space": train_env.observation_space,
        "action_space": train_env.action_space,
        "architecture": cfg.scaling.architecture_type,
        "target_params": cfg.scaling.target_params,
        "device": device,
        "seed": cfg.seed,
    }

    # Add algorithm-specific kwargs
    algo_specific = OmegaConf.to_container(algo_cfg, resolve=True)
    algo_specific.pop("name", None)
    algo_kwargs.update(algo_specific)

    # Add training kwargs
    algo_kwargs.update({
        "batch_size": cfg.training.batch_size,
        "learning_starts": cfg.training.learning_starts,
    })

    if "utd_ratio" in algo_specific or "gradient_steps" in cfg.training:
        algo_kwargs["utd_ratio"] = algo_specific.get(
            "utd_ratio", cfg.training.gradient_steps
        )

    # Create algorithm
    algorithm = get_algorithm(algo_name, **algo_kwargs)

    logger.info(f"Algorithm: {algo_name}")
    logger.info(f"Target parameters: {cfg.scaling.target_params:,}")
    logger.info(f"Actual parameters: {algorithm.get_param_count()}")

    # Set up WandB config
    wandb_config = None
    if cfg.logging.use_wandb:
        wandb_config = {
            "project": cfg.experiment.project,
            "name": f"{cfg.experiment.name}_{cfg.scaling.target_params}",
            "config": OmegaConf.to_container(cfg, resolve=True),
            "group": cfg.experiment.group,
            "tags": cfg.experiment.tags,
        }

    # Create trainer
    trainer = Trainer(
        algorithm=algorithm,
        train_env=train_env,
        eval_env=eval_env,
        total_timesteps=cfg.training.total_timesteps,
        eval_freq=cfg.training.eval_freq,
        n_eval_episodes=cfg.training.n_eval_episodes,
        save_freq=cfg.training.save_freq,
        log_freq=cfg.training.log_freq,
        log_dir=str(log_dir),
        save_dir=str(save_dir),
        use_wandb=cfg.logging.use_wandb,
        wandb_config=wandb_config,
    )

    # Run training
    results = trainer.train()

    # Clean up
    train_env.close()
    eval_env.close()

    return results["final_reward"]


if __name__ == "__main__":
    main()
