"""Evaluation utilities for RL experiments."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from rl_scaling_laws.utils.logging import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evaluator for RL algorithms."""

    def __init__(
        self,
        env,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        record_video: bool = False,
        video_folder: Optional[str] = None,
    ):
        """Initialize evaluator.

        Args:
            env: Environment for evaluation.
            n_eval_episodes: Number of evaluation episodes.
            deterministic: Use deterministic policy.
            render: Render environment.
            record_video: Record evaluation videos.
            video_folder: Folder to save videos.
        """
        self.env = env
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render
        self.record_video = record_video
        self.video_folder = Path(video_folder) if video_folder else None

        if record_video and video_folder:
            self.video_folder.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        model,
        num_timesteps: int = 0,
    ) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            model: RL model to evaluate.
            num_timesteps: Current training timesteps (for logging).

        Returns:
            Dictionary of evaluation metrics.
        """
        episode_rewards = []
        episode_lengths = []
        episode_successes = []

        for episode in range(self.n_eval_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            success = False

            while not done:
                action = model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                # Check for success (MetaWorld/robotics)
                if "success" in info:
                    success = success or info["success"]

                if self.render:
                    self.env.render()

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_successes.append(float(success))

        # Compute statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)
        mean_length = np.mean(episode_lengths)

        metrics = {
            "eval/mean_reward": mean_reward,
            "eval/std_reward": std_reward,
            "eval/min_reward": min_reward,
            "eval/max_reward": max_reward,
            "eval/mean_ep_length": mean_length,
        }

        if any(episode_successes):
            metrics["eval/success_rate"] = np.mean(episode_successes)

        logger.info(
            f"Evaluation @ {num_timesteps}: "
            f"reward={mean_reward:.2f} +/- {std_reward:.2f}, "
            f"length={mean_length:.0f}"
        )

        return metrics

    def record_evaluation(
        self,
        model,
        video_name: str = "eval",
        max_steps: int = 1000,
    ) -> None:
        """Record a video of model evaluation.

        Args:
            model: RL model to evaluate.
            video_name: Name for the video file.
            max_steps: Maximum steps to record.
        """
        if not self.record_video or self.video_folder is None:
            return

        try:
            from rl_scaling_laws.evaluation.video import VideoRecorder

            recorder = VideoRecorder(
                self.env,
                str(self.video_folder / f"{video_name}.mp4"),
            )

            obs, _ = self.env.reset()
            recorder.record(self.env)

            for _ in range(max_steps):
                action = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                recorder.record(self.env)

                if terminated or truncated:
                    break

            recorder.save()
            logger.info(f"Saved evaluation video: {video_name}.mp4")

        except Exception as e:
            logger.warning(f"Failed to record video: {e}")


def evaluate_policy(
    model,
    env,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
) -> Tuple[float, float]:
    """Simple function to evaluate a policy.

    Args:
        model: Model with predict method.
        env: Gymnasium environment.
        n_eval_episodes: Number of episodes.
        deterministic: Use deterministic policy.

    Returns:
        Tuple of (mean_reward, std_reward).
    """
    evaluator = Evaluator(env, n_eval_episodes, deterministic)
    metrics = evaluator.evaluate(model)
    return metrics["eval/mean_reward"], metrics["eval/std_reward"]
