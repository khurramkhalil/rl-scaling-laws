"""Training callbacks for logging, checkpointing, and evaluation."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from rl_scaling_laws.utils.logging import get_logger

logger = get_logger(__name__)


class BaseCallback(ABC):
    """Base class for training callbacks."""

    def __init__(self):
        self.num_timesteps = 0
        self.n_calls = 0

    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        """Called at the start of training."""
        pass

    def on_rollout_start(self) -> None:
        """Called at the start of a rollout."""
        pass

    def on_step(self) -> bool:
        """Called at each step. Return False to stop training."""
        self.n_calls += 1
        return True

    def on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        pass

    def on_training_end(self) -> None:
        """Called at the end of training."""
        pass

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """Update local variables."""
        self.num_timesteps = locals_.get("num_timesteps", self.num_timesteps)


class WandbCallback(BaseCallback):
    """Callback for Weights & Biases logging."""

    def __init__(
        self,
        project: str = "rl-scaling-laws",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        group: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        log_freq: int = 1000,
        save_model: bool = False,
        gradient_save_freq: int = 0,
    ):
        super().__init__()
        self.project = project
        self.name = name
        self.config = config or {}
        self.group = group
        self.tags = tags
        self.notes = notes
        self.log_freq = log_freq
        self.save_model = save_model
        self.gradient_save_freq = gradient_save_freq
        self._run = None

    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        """Initialize WandB run."""
        try:
            import wandb

            self._run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                group=self.group,
                tags=self.tags,
                notes=self.notes,
                reinit=True,
            )

            # Log model architecture if available
            if "model" in locals_:
                wandb.watch(
                    locals_["model"],
                    log="gradients" if self.gradient_save_freq > 0 else None,
                    log_freq=self.gradient_save_freq if self.gradient_save_freq > 0 else 1000,
                )

            logger.info(f"WandB run initialized: {self._run.url}")

        except ImportError:
            logger.warning("wandb not installed, skipping WandB logging")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to WandB."""
        if self._run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def on_training_end(self) -> None:
        """Finish WandB run."""
        if self._run is not None:
            import wandb

            if self.save_model:
                # Save final model as artifact
                pass

            wandb.finish()


class CheckpointCallback(BaseCallback):
    """Callback for saving model checkpoints."""

    def __init__(
        self,
        save_dir: str,
        save_freq: int = 10000,
        save_best: bool = True,
        save_last: bool = True,
        keep_last_n: int = 3,
        verbose: bool = True,
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_freq = save_freq
        self.save_best = save_best
        self.save_last = save_last
        self.keep_last_n = keep_last_n
        self.verbose = verbose
        self.best_reward = float("-inf")
        self._model = None
        self._saved_checkpoints: List[Path] = []

    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        """Set up checkpoint directory."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._model = locals_.get("model")

    def on_step(self) -> bool:
        """Save checkpoint at specified frequency."""
        if self.num_timesteps % self.save_freq == 0 and self._model is not None:
            self._save_checkpoint(f"checkpoint_{self.num_timesteps}")
        return True

    def _save_checkpoint(self, name: str) -> None:
        """Save a checkpoint."""
        path = self.save_dir / f"{name}.pt"

        if isinstance(self._model, nn.Module):
            torch.save(self._model.state_dict(), path)
        else:
            # Assume it's an algorithm with a save method
            self._model.save(path)

        self._saved_checkpoints.append(path)

        # Clean up old checkpoints
        if self.keep_last_n > 0 and len(self._saved_checkpoints) > self.keep_last_n:
            old_checkpoint = self._saved_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

        if self.verbose:
            logger.info(f"Saved checkpoint: {path}")

    def save_best_model(self, reward: float) -> None:
        """Save model if it's the best so far."""
        if self.save_best and reward > self.best_reward:
            self.best_reward = reward
            path = self.save_dir / "best_model.pt"

            if isinstance(self._model, nn.Module):
                torch.save(self._model.state_dict(), path)
            else:
                self._model.save(path)

            if self.verbose:
                logger.info(f"New best model saved: {path} (reward: {reward:.2f})")

    def on_training_end(self) -> None:
        """Save final checkpoint."""
        if self.save_last and self._model is not None:
            path = self.save_dir / "last_model.pt"
            if isinstance(self._model, nn.Module):
                torch.save(self._model.state_dict(), path)
            else:
                self._model.save(path)
            if self.verbose:
                logger.info(f"Saved final model: {path}")


class EvalCallback(BaseCallback):
    """Callback for periodic evaluation."""

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        verbose: bool = True,
        callback_on_new_best: Optional[BaseCallback] = None,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render
        self.verbose = verbose
        self.callback_on_new_best = callback_on_new_best
        self.best_mean_reward = float("-inf")
        self._model = None
        self.evaluations_results: List[float] = []
        self.evaluations_timesteps: List[int] = []

    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        """Get model reference."""
        self._model = locals_.get("model")

    def on_step(self) -> bool:
        """Run evaluation at specified frequency."""
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        """Run evaluation episodes."""
        if self._model is None:
            return

        episode_rewards = []
        episode_lengths = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action = self._model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

                if self.render:
                    self.eval_env.render()

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        mean_reward = sum(episode_rewards) / len(episode_rewards)
        std_reward = (
            sum((r - mean_reward) ** 2 for r in episode_rewards) / len(episode_rewards)
        ) ** 0.5
        mean_length = sum(episode_lengths) / len(episode_lengths)

        self.evaluations_results.append(mean_reward)
        self.evaluations_timesteps.append(self.num_timesteps)

        if self.verbose:
            logger.info(
                f"Eval @ {self.num_timesteps}: "
                f"reward={mean_reward:.2f} +/- {std_reward:.2f}, "
                f"length={mean_length:.0f}"
            )

        # Check for new best
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.callback_on_new_best is not None:
                self.callback_on_new_best.save_best_model(mean_reward)


class TensorboardCallback(BaseCallback):
    """Callback for TensorBoard logging."""

    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = log_dir
        self._writer = None

    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(self.log_dir)
            logger.info(f"TensorBoard logging to: {self.log_dir}")
        except ImportError:
            logger.warning("tensorboard not installed, skipping TensorBoard logging")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log a histogram."""
        if self._writer is not None:
            self._writer.add_histogram(tag, values, step)

    def on_training_end(self) -> None:
        """Close TensorBoard writer."""
        if self._writer is not None:
            self._writer.close()
