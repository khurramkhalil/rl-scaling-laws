"""Video recording utilities."""

from pathlib import Path
from typing import List, Optional

import numpy as np


class VideoRecorder:
    """Record environment frames to video."""

    def __init__(
        self,
        env,
        path: str,
        fps: int = 30,
    ):
        """Initialize video recorder.

        Args:
            env: Environment to record.
            path: Output video path.
            fps: Frames per second.
        """
        self.env = env
        self.path = Path(path)
        self.fps = fps
        self.frames: List[np.ndarray] = []

        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, env=None) -> None:
        """Record a frame.

        Args:
            env: Environment (uses self.env if None).
        """
        env = env or self.env

        try:
            frame = env.render()
            if frame is not None:
                self.frames.append(frame)
        except Exception:
            pass

    def save(self) -> None:
        """Save recorded frames to video."""
        if not self.frames:
            return

        try:
            import imageio

            imageio.mimsave(
                str(self.path),
                self.frames,
                fps=self.fps,
            )
        except ImportError:
            try:
                from moviepy.editor import ImageSequenceClip

                clip = ImageSequenceClip(self.frames, fps=self.fps)
                clip.write_videofile(str(self.path), logger=None)
            except ImportError:
                print("Install imageio or moviepy for video recording")

    def reset(self) -> None:
        """Clear recorded frames."""
        self.frames = []


def record_episode(
    env,
    model,
    path: str,
    max_steps: int = 1000,
    deterministic: bool = True,
) -> float:
    """Record a single episode.

    Args:
        env: Environment.
        model: Model with predict method.
        path: Output video path.
        max_steps: Maximum steps.
        deterministic: Use deterministic policy.

    Returns:
        Episode reward.
    """
    recorder = VideoRecorder(env, path)

    obs, _ = env.reset()
    recorder.record(env)

    total_reward = 0.0

    for _ in range(max_steps):
        action = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        recorder.record(env)

        if terminated or truncated:
            break

    recorder.save()
    return total_reward
