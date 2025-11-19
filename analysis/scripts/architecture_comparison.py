"""Architecture comparison utilities for scaling experiments."""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_scaling_laws.models import NetworkBuilder
from rl_scaling_laws.utils.param_count import count_parameters, format_param_count
from rl_scaling_laws.utils.logging import get_logger

logger = get_logger(__name__)


class ArchitectureComparison:
    """Compare different architectures at various scales."""

    def __init__(self):
        self.architectures = ["mlp", "cnn", "transformer", "moe"]
        self.results: Dict[str, Dict] = {}

    def build_comparison_table(
        self,
        input_dim: int,
        output_dim: int,
        target_params: List[int],
    ) -> pd.DataFrame:
        """Build comparison table for architectures.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            target_params: List of target parameter counts.

        Returns:
            DataFrame with architecture comparisons.
        """
        records = []

        for arch in self.architectures:
            for target in target_params:
                try:
                    builder = NetworkBuilder(arch, target_params=target)
                    network = builder.build_actor(input_dim, output_dim)

                    actual_params = count_parameters(network)

                    record = {
                        "architecture": arch,
                        "target_params": target,
                        "actual_params": actual_params,
                        "param_error": abs(actual_params - target) / target,
                    }

                    # Get architecture-specific info
                    if hasattr(network, "architecture_info"):
                        record.update(network.architecture_info)

                    records.append(record)

                except Exception as e:
                    logger.warning(f"Failed to build {arch} with {target} params: {e}")

        return pd.DataFrame(records)

    def estimate_flops(
        self,
        input_dim: int,
        output_dim: int,
        target_params: int,
        batch_size: int = 256,
    ) -> Dict[str, int]:
        """Estimate FLOPs for each architecture.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            target_params: Target parameters.
            batch_size: Batch size.

        Returns:
            Dictionary mapping architecture to estimated FLOPs.
        """
        from rl_scaling_laws.utils.param_count import compute_flops_per_forward

        flops = {}

        for arch in self.architectures:
            try:
                builder = NetworkBuilder(arch, target_params=target_params)
                network = builder.build_actor(input_dim, output_dim)

                flops[arch] = compute_flops_per_forward(
                    network, (input_dim,)
                ) * batch_size

            except Exception as e:
                logger.warning(f"Failed to estimate FLOPs for {arch}: {e}")
                flops[arch] = 0

        return flops

    def compare_memory_usage(
        self,
        input_dim: int,
        output_dim: int,
        target_params: int,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """Compare memory usage across architectures.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            target_params: Target parameters.
            batch_size: Batch size.

        Returns:
            DataFrame with memory comparisons.
        """
        from rl_scaling_laws.utils.memory import estimate_memory_usage

        records = []

        for arch in self.architectures:
            try:
                builder = NetworkBuilder(arch, target_params=target_params)
                network = builder.build_actor(input_dim, output_dim)

                memory = estimate_memory_usage(
                    network, batch_size, (input_dim,)
                )

                records.append({
                    "architecture": arch,
                    **memory,
                })

            except Exception as e:
                logger.warning(f"Failed to estimate memory for {arch}: {e}")

        return pd.DataFrame(records)

    def find_saturation_point(
        self,
        results: pd.DataFrame,
        performance_col: str = "final_reward",
        param_col: str = "target_params",
        threshold: float = 0.95,
    ) -> Dict[str, int]:
        """Find parameter count where each architecture saturates.

        Args:
            results: DataFrame with experimental results.
            performance_col: Performance column name.
            param_col: Parameter count column name.
            threshold: Fraction of max performance to consider saturated.

        Returns:
            Dictionary mapping architecture to saturation point.
        """
        saturation_points = {}

        for arch, arch_data in results.groupby("architecture"):
            sorted_data = arch_data.sort_values(param_col)
            max_perf = sorted_data[performance_col].max()
            threshold_perf = max_perf * threshold

            # Find first point exceeding threshold
            for _, row in sorted_data.iterrows():
                if row[performance_col] >= threshold_perf:
                    saturation_points[arch] = row[param_col]
                    break

        return saturation_points

    def plot_architecture_comparison(
        self,
        results: pd.DataFrame,
        metric: str = "final_reward",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot performance vs parameters for all architectures.

        Args:
            results: DataFrame with results.
            metric: Metric to plot.
            save_path: Path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.tab10.colors
        markers = ["o", "s", "^", "D"]

        for i, (arch, arch_data) in enumerate(results.groupby("architecture")):
            sorted_data = arch_data.groupby("target_params")[metric].agg(
                ["mean", "std"]
            ).reset_index()

            ax.errorbar(
                sorted_data["target_params"],
                sorted_data["mean"],
                yerr=sorted_data["std"],
                label=arch.upper(),
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                markersize=8,
                capsize=5,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Parameters", fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title("Architecture Comparison", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def compute_efficiency_metrics(
        self,
        results: pd.DataFrame,
        perf_col: str = "final_reward",
        param_col: str = "target_params",
        compute_col: str = "total_flops",
    ) -> pd.DataFrame:
        """Compute efficiency metrics for architectures.

        Args:
            results: DataFrame with results.
            perf_col: Performance column.
            param_col: Parameter column.
            compute_col: Compute column.

        Returns:
            DataFrame with efficiency metrics.
        """
        efficiency_records = []

        for arch, arch_data in results.groupby("architecture"):
            for _, row in arch_data.iterrows():
                efficiency_records.append({
                    "architecture": arch,
                    "target_params": row[param_col],
                    "performance": row[perf_col],
                    "perf_per_param": row[perf_col] / row[param_col] * 1e6,
                    "perf_per_flop": (
                        row[perf_col] / row[compute_col] * 1e9
                        if compute_col in row and row[compute_col] > 0
                        else None
                    ),
                })

        return pd.DataFrame(efficiency_records)


def generate_architecture_report(
    results_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Generate comprehensive architecture comparison report.

    Args:
        results_path: Path to results CSV/JSON.
        output_dir: Output directory for report.

    Returns:
        Report dictionary.
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    if results_path.endswith(".csv"):
        results = pd.read_csv(results_path)
    else:
        results = pd.read_json(results_path)

    comparison = ArchitectureComparison()

    # Generate plots
    fig = comparison.plot_architecture_comparison(
        results,
        save_path=str(output_dir / "architecture_comparison.png"),
    )
    plt.close(fig)

    # Find saturation points
    saturation = comparison.find_saturation_point(results)

    # Compute efficiency
    efficiency = comparison.compute_efficiency_metrics(results)

    # Summary
    report = {
        "saturation_points": saturation,
        "best_architecture_per_scale": {},
        "efficiency_summary": efficiency.groupby("architecture").agg({
            "perf_per_param": "mean",
        }).to_dict(),
    }

    # Find best architecture per scale
    for params in results["target_params"].unique():
        scale_data = results[results["target_params"] == params]
        best_arch = scale_data.loc[scale_data["final_reward"].idxmax(), "architecture"]
        report["best_architecture_per_scale"][int(params)] = best_arch

    # Save report
    import json
    with open(output_dir / "architecture_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Generated architecture report in {output_dir}")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare architectures")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="architecture_report")
    args = parser.parse_args()

    generate_architecture_report(args.results, args.output_dir)
