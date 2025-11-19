#!/usr/bin/env python
"""Job submission script for cluster computing."""

import argparse
import os
import subprocess
from pathlib import Path


def create_slurm_script(
    job_name: str,
    config_overrides: str,
    output_dir: str,
    partition: str = "gpu",
    gpus: int = 1,
    cpus: int = 8,
    memory: str = "32G",
    time: str = "24:00:00",
) -> str:
    """Create a SLURM submission script.

    Args:
        job_name: Job name.
        config_overrides: Hydra config overrides.
        output_dir: Output directory.
        partition: SLURM partition.
        gpus: Number of GPUs.
        cpus: Number of CPUs.
        memory: Memory allocation.
        time: Time limit.

    Returns:
        SLURM script content.
    """
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/slurm_%j.out
#SBATCH --error={output_dir}/slurm_%j.err
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --time={time}

# Load modules (adjust for your cluster)
module load python/3.10
module load cuda/11.8

# Activate environment
source ~/venvs/rl-scaling/bin/activate

# Run training
python -m rl_scaling_laws.train {config_overrides}

echo "Job completed"
"""
    return script


def submit_sweep(
    sweep_config: str,
    output_dir: str,
    dry_run: bool = False,
) -> None:
    """Submit a sweep of jobs.

    Args:
        sweep_config: Path to sweep configuration.
        output_dir: Base output directory.
        dry_run: If True, print commands without submitting.
    """
    # Example sweep configurations
    param_scales = [10_000_000, 30_000_000, 100_000_000, 300_000_000, 1_000_000_000]
    seeds = [1, 2, 3, 4, 5]

    for scale in param_scales:
        for seed in seeds:
            job_name = f"rl_scale_{scale // 1_000_000}M_s{seed}"
            config = f"scaling.target_params={scale} seed={seed}"

            job_dir = Path(output_dir) / job_name
            job_dir.mkdir(parents=True, exist_ok=True)

            script = create_slurm_script(
                job_name=job_name,
                config_overrides=config,
                output_dir=str(job_dir),
            )

            script_path = job_dir / "submit.sh"
            with open(script_path, "w") as f:
                f.write(script)

            if dry_run:
                print(f"Would submit: {job_name}")
                print(f"  Config: {config}")
            else:
                result = subprocess.run(
                    ["sbatch", str(script_path)],
                    capture_output=True,
                    text=True,
                )
                print(f"Submitted {job_name}: {result.stdout.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Submit RL scaling law jobs")
    parser.add_argument("--sweep", type=str, help="Sweep configuration")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print without submitting")

    # Single job options
    parser.add_argument("--config", type=str, help="Config overrides for single job")
    parser.add_argument("--job-name", type=str, default="rl_train", help="Job name")

    args = parser.parse_args()

    if args.sweep:
        submit_sweep(args.sweep, args.output_dir, args.dry_run)
    elif args.config:
        job_dir = Path(args.output_dir) / args.job_name
        job_dir.mkdir(parents=True, exist_ok=True)

        script = create_slurm_script(
            job_name=args.job_name,
            config_overrides=args.config,
            output_dir=str(job_dir),
        )

        script_path = job_dir / "submit.sh"
        with open(script_path, "w") as f:
            f.write(script)

        if args.dry_run:
            print(f"Would submit: {args.job_name}")
            print(script)
        else:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
            )
            print(f"Submitted: {result.stdout.strip()}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
