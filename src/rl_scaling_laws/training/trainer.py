"""Main trainer class for RL experiments."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from rl_scaling_laws.algorithms.base import BaseAlgorithm
from rl_scaling_laws.evaluation.evaluator import Evaluator
from rl_scaling_laws.training.callbacks import (
    BaseCallback,
    CheckpointCallback,
    WandbCallback,
)
from rl_scaling_laws.utils.logging import get_logger, setup_logging
from rl_scaling_laws.utils.param_count import format_param_count

logger = get_logger(__name__)


class Trainer:
    """Main trainer for RL experiments."""

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        train_env,
        eval_env=None,
        # Training settings
        total_timesteps: int = 1_000_000,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        save_freq: int = 50_000,
        log_freq: int = 1_000,
        # Directories
        log_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        # Callbacks
        callbacks: Optional[List[BaseCallback]] = None,
        # Logging
        use_wandb: bool = False,
        wandb_config: Optional[Dict] = None,
    ):
        """Initialize trainer.

        Args:
            algorithm: RL algorithm to train.
            train_env: Training environment.
            eval_env: Evaluation environment (optional).
            total_timesteps: Total training timesteps.
            eval_freq: Steps between evaluations.
            n_eval_episodes: Number of evaluation episodes.
            save_freq: Steps between checkpoints.
            log_freq: Steps between logging.
            log_dir: Directory for logs.
            save_dir: Directory for checkpoints.
            callbacks: List of callbacks.
            use_wandb: Use Weights & Biases logging.
            wandb_config: WandB configuration.
        """
        self.algorithm = algorithm
        self.train_env = train_env
        self.eval_env = eval_env or train_env

        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_freq = save_freq
        self.log_freq = log_freq

        self.log_dir = Path(log_dir) if log_dir else None
        self.save_dir = Path(save_dir) if save_dir else None

        # Set up directories
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            setup_logging(self.log_dir)

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set up callbacks
        self.callbacks = callbacks or []

        if use_wandb:
            wandb_config = wandb_config or {}
            self.callbacks.append(WandbCallback(**wandb_config))

        if self.save_dir:
            self.callbacks.append(
                CheckpointCallback(str(self.save_dir), save_freq=save_freq)
            )

        # Set up evaluator
        self.evaluator = Evaluator(
            self.eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )

        # Training state
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def train(self) -> Dict[str, Any]:
        """Run training loop.

        Returns:
            Dictionary of training results.
        """
        logger.info(f"Starting training for {self.total_timesteps} timesteps")
        logger.info(f"Algorithm: {self.algorithm.__class__.__name__}")
        logger.info(f"Parameters: {self.algorithm.get_param_count()}")

        # Initialize callbacks
        locals_ = {"model": self.algorithm}
        for callback in self.callbacks:
            callback.on_training_start(locals_)

        # Training loop
        start_time = time.time()
        last_log_time = start_time
        last_eval_time = 0

        episode_reward = 0.0
        episode_length = 0

        pbar = tqdm(total=self.total_timesteps, desc="Training")

        while self.algorithm.num_timesteps < self.total_timesteps:
            # Collect rollouts
            n_collected = self.algorithm.collect_rollouts(self.train_env, n_steps=1)

            # Track episode stats (from info)
            episode_length += n_collected
            # Note: Episode tracking handled by environment wrapper

            # Train
            if self.algorithm.num_timesteps >= getattr(
                self.algorithm, "learning_starts", 0
            ):
                train_metrics = self.algorithm.train()

                # Log training metrics
                if (
                    self.algorithm.num_timesteps % self.log_freq == 0
                    and train_metrics
                ):
                    self._log_metrics(train_metrics, "train")

            # Evaluate
            if self.algorithm.num_timesteps - last_eval_time >= self.eval_freq:
                eval_metrics = self.evaluator.evaluate(
                    self.algorithm, self.algorithm.num_timesteps
                )
                self._log_metrics(eval_metrics, "eval")
                last_eval_time = self.algorithm.num_timesteps

                # Notify callbacks
                for callback in self.callbacks:
                    if isinstance(callback, CheckpointCallback):
                        callback.save_best_model(eval_metrics["eval/mean_reward"])

            # Update callbacks
            for callback in self.callbacks:
                callback.update_locals({"num_timesteps": self.algorithm.num_timesteps})
                callback.on_step()

            # Update progress bar
            pbar.update(n_collected)
            pbar.set_postfix({
                "reward": f"{eval_metrics.get('eval/mean_reward', 0):.1f}"
                if "eval_metrics" in dir() else "N/A"
            })

        pbar.close()

        # Final evaluation
        final_metrics = self.evaluator.evaluate(
            self.algorithm, self.algorithm.num_timesteps
        )

        # End callbacks
        for callback in self.callbacks:
            callback.on_training_end()

        # Training summary
        total_time = time.time() - start_time
        fps = self.algorithm.num_timesteps / total_time

        results = {
            "total_timesteps": self.algorithm.num_timesteps,
            "total_time": total_time,
            "fps": fps,
            "final_reward": final_metrics["eval/mean_reward"],
            "final_std": final_metrics["eval/std_reward"],
        }

        logger.info(
            f"Training complete: {self.algorithm.num_timesteps} steps in {total_time:.1f}s "
            f"({fps:.0f} fps)"
        )
        logger.info(
            f"Final performance: {results['final_reward']:.2f} +/- {results['final_std']:.2f}"
        )

        return results

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """Log metrics to callbacks."""
        # Add timestep to metrics
        metrics["timestep"] = self.algorithm.num_timesteps

        # Log to WandB callback
        for callback in self.callbacks:
            if isinstance(callback, WandbCallback):
                callback.log(metrics, step=self.algorithm.num_timesteps)

    def save(self, path: str) -> None:
        """Save trainer state."""
        self.algorithm.save(path)

    def load(self, path: str) -> None:
        """Load trainer state."""
        self.algorithm.load(path)
