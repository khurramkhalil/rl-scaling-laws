"""Experiment management and monitoring utilities."""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rl_scaling_laws.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    algorithm: str
    environment: str
    target_params: int
    seed: int
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    estimated_hours: float = 1.0


@dataclass
class ExperimentStatus:
    """Status of an experiment."""

    experiment_id: str
    status: str  # pending, running, completed, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    progress: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None


class ExperimentManager:
    """Manage and monitor experiments."""

    def __init__(
        self,
        base_dir: str,
        project_name: str = "rl-scaling-laws",
    ):
        """Initialize experiment manager.

        Args:
            base_dir: Base directory for experiments.
            project_name: Project name.
        """
        self.base_dir = Path(base_dir)
        self.project_name = project_name

        # Create directories
        self.experiments_dir = self.base_dir / "experiments"
        self.queue_dir = self.base_dir / "queue"
        self.logs_dir = self.base_dir / "logs"

        for d in [self.experiments_dir, self.queue_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Track experiments
        self.experiments: Dict[str, ExperimentStatus] = {}
        self._load_existing_experiments()

    def _load_existing_experiments(self) -> None:
        """Load existing experiment statuses."""
        status_file = self.base_dir / "experiment_status.json"
        if status_file.exists():
            with open(status_file) as f:
                data = json.load(f)
                for exp_id, status_dict in data.items():
                    self.experiments[exp_id] = ExperimentStatus(**status_dict)

    def _save_status(self) -> None:
        """Save experiment statuses to disk."""
        status_file = self.base_dir / "experiment_status.json"
        data = {
            exp_id: {
                "experiment_id": status.experiment_id,
                "status": status.status,
                "start_time": status.start_time,
                "end_time": status.end_time,
                "progress": status.progress,
                "metrics": status.metrics,
                "error_message": status.error_message,
            }
            for exp_id, status in self.experiments.items()
        }
        with open(status_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_experiment(
        self,
        config: ExperimentConfig,
    ) -> str:
        """Create a new experiment.

        Args:
            config: Experiment configuration.

        Returns:
            Experiment ID.
        """
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{config.name}_{config.target_params}_{config.seed}_{timestamp}"

        # Create experiment directory
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_dict = {
            "name": config.name,
            "algorithm": config.algorithm,
            "environment": config.environment,
            "target_params": config.target_params,
            "seed": config.seed,
            "config_overrides": config.config_overrides,
            "priority": config.priority,
            "estimated_hours": config.estimated_hours,
        }
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Create status
        self.experiments[exp_id] = ExperimentStatus(
            experiment_id=exp_id,
            status="pending",
        )

        self._save_status()
        logger.info(f"Created experiment: {exp_id}")

        return exp_id

    def queue_experiment(
        self,
        exp_id: str,
        partition: str = "gpu",
        gpus: int = 1,
        time_limit: str = "24:00:00",
    ) -> str:
        """Queue experiment for execution.

        Args:
            exp_id: Experiment ID.
            partition: SLURM partition.
            gpus: Number of GPUs.
            time_limit: Time limit.

        Returns:
            Job ID.
        """
        exp_dir = self.experiments_dir / exp_id
        config_path = exp_dir / "config.json"

        with open(config_path) as f:
            config = json.load(f)

        # Build command
        overrides = " ".join(
            f"{k}={v}" for k, v in config["config_overrides"].items()
        )

        cmd = f"""python -m rl_scaling_laws.train \\
            algorithm={config['algorithm']} \\
            environment={config['environment']} \\
            scaling.target_params={config['target_params']} \\
            seed={config['seed']} \\
            experiment.name={exp_id} \\
            {overrides}
        """

        # Create SLURM script
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={exp_id}
#SBATCH --output={self.logs_dir}/{exp_id}.out
#SBATCH --error={self.logs_dir}/{exp_id}.err
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time_limit}

cd {os.getcwd()}
source venv/bin/activate

{cmd}
"""

        script_path = self.queue_dir / f"{exp_id}.sh"
        with open(script_path, "w") as f:
            f.write(slurm_script)

        logger.info(f"Queued experiment: {exp_id}")
        return str(script_path)

    def submit_batch(
        self,
        exp_ids: List[str],
        max_concurrent: int = 10,
    ) -> List[str]:
        """Submit a batch of experiments.

        Args:
            exp_ids: List of experiment IDs.
            max_concurrent: Maximum concurrent jobs.

        Returns:
            List of job IDs.
        """
        job_ids = []

        for exp_id in exp_ids:
            script_path = self.queue_dir / f"{exp_id}.sh"
            if not script_path.exists():
                self.queue_experiment(exp_id)

            # Submit to SLURM
            try:
                result = subprocess.run(
                    ["sbatch", str(script_path)],
                    capture_output=True,
                    text=True,
                )
                job_id = result.stdout.strip().split()[-1]
                job_ids.append(job_id)

                self.experiments[exp_id].status = "running"
                self.experiments[exp_id].start_time = datetime.now().isoformat()

                logger.info(f"Submitted {exp_id} as job {job_id}")

            except Exception as e:
                logger.error(f"Failed to submit {exp_id}: {e}")

        self._save_status()
        return job_ids

    def get_status(self, exp_id: str) -> ExperimentStatus:
        """Get experiment status.

        Args:
            exp_id: Experiment ID.

        Returns:
            Experiment status.
        """
        return self.experiments.get(exp_id)

    def get_summary(self) -> Dict[str, int]:
        """Get summary of experiment statuses.

        Returns:
            Dictionary with counts per status.
        """
        summary = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        for status in self.experiments.values():
            if status.status in summary:
                summary[status.status] += 1
        return summary

    def get_failed_experiments(self) -> List[str]:
        """Get list of failed experiments.

        Returns:
            List of failed experiment IDs.
        """
        return [
            exp_id for exp_id, status in self.experiments.items()
            if status.status == "failed"
        ]

    def retry_failed(self) -> List[str]:
        """Retry all failed experiments.

        Returns:
            List of resubmitted experiment IDs.
        """
        failed = self.get_failed_experiments()
        for exp_id in failed:
            self.experiments[exp_id].status = "pending"
            self.experiments[exp_id].error_message = None

        self._save_status()
        return failed

    def update_progress(
        self,
        exp_id: str,
        progress: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update experiment progress.

        Args:
            exp_id: Experiment ID.
            progress: Progress (0-1).
            metrics: Current metrics.
        """
        if exp_id in self.experiments:
            self.experiments[exp_id].progress = progress
            if metrics:
                self.experiments[exp_id].metrics.update(metrics)
            self._save_status()

    def mark_completed(
        self,
        exp_id: str,
        metrics: Dict[str, float],
    ) -> None:
        """Mark experiment as completed.

        Args:
            exp_id: Experiment ID.
            metrics: Final metrics.
        """
        if exp_id in self.experiments:
            self.experiments[exp_id].status = "completed"
            self.experiments[exp_id].end_time = datetime.now().isoformat()
            self.experiments[exp_id].progress = 1.0
            self.experiments[exp_id].metrics = metrics
            self._save_status()
            logger.info(f"Completed experiment: {exp_id}")

    def mark_failed(
        self,
        exp_id: str,
        error_message: str,
    ) -> None:
        """Mark experiment as failed.

        Args:
            exp_id: Experiment ID.
            error_message: Error message.
        """
        if exp_id in self.experiments:
            self.experiments[exp_id].status = "failed"
            self.experiments[exp_id].end_time = datetime.now().isoformat()
            self.experiments[exp_id].error_message = error_message
            self._save_status()
            logger.error(f"Failed experiment {exp_id}: {error_message}")


def create_phase2_experiments(
    manager: ExperimentManager,
    algorithms: List[str] = ["sac", "dqn", "ppo"],
    environments: List[str] = ["atari", "dmc"],
    scales: List[int] = [10_000_000, 30_000_000, 100_000_000, 300_000_000, 1_000_000_000],
    seeds: List[int] = [1, 2, 3, 4, 5],
) -> List[str]:
    """Create all Phase 2 experiments.

    Args:
        manager: Experiment manager.
        algorithms: List of algorithms.
        environments: List of environments.
        scales: Parameter scales.
        seeds: Random seeds.

    Returns:
        List of experiment IDs.
    """
    exp_ids = []

    for algo in algorithms:
        for env in environments:
            for scale in scales:
                for seed in seeds:
                    # Estimate hours based on scale
                    hours = scale / 100_000_000 * 10  # ~10h per 100M

                    config = ExperimentConfig(
                        name=f"phase2_{algo}_{env}",
                        algorithm=algo,
                        environment=env,
                        target_params=scale,
                        seed=seed,
                        estimated_hours=hours,
                        priority=1 if scale <= 100_000_000 else 0,  # Prioritize smaller
                    )

                    exp_id = manager.create_experiment(config)
                    exp_ids.append(exp_id)

    logger.info(f"Created {len(exp_ids)} Phase 2 experiments")
    return exp_ids
