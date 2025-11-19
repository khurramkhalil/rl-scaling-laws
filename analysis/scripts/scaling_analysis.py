"""Analysis tools for scaling law experiments."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from rl_scaling_laws.utils.logging import get_logger

logger = get_logger(__name__)


class ScalingLawAnalyzer:
    """Analyze scaling law experimental results."""

    def __init__(self, results_dir: str):
        """Initialize analyzer.

        Args:
            results_dir: Directory containing experiment results.
        """
        self.results_dir = Path(results_dir)
        self.data = None

    def load_results(self, pattern: str = "**/metrics.json") -> pd.DataFrame:
        """Load all experimental results.

        Args:
            pattern: Glob pattern for result files.

        Returns:
            DataFrame with all results.
        """
        records = []

        for path in self.results_dir.glob(pattern):
            try:
                with open(path) as f:
                    metrics = json.load(f)

                # Extract config from path or metrics
                record = {
                    "path": str(path),
                    **metrics,
                }
                records.append(record)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        self.data = pd.DataFrame(records)
        logger.info(f"Loaded {len(self.data)} experiment results")
        return self.data

    def fit_power_law(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weighted: bool = False,
    ) -> Dict[str, float]:
        """Fit power law: y = A * x^alpha.

        Args:
            x: Independent variable (e.g., parameters).
            y: Dependent variable (e.g., performance).
            weighted: Use weighted least squares.

        Returns:
            Dictionary with alpha, A, r_squared, and confidence intervals.
        """
        log_x = np.log(x)
        log_y = np.log(y)

        # Linear regression in log space
        if weighted:
            # Weight by 1/variance (if we had multiple seeds)
            weights = np.ones_like(log_x)
        else:
            weights = None

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

        alpha = slope
        A = np.exp(intercept)
        r_squared = r_value ** 2

        # Confidence intervals (95%)
        n = len(x)
        t_val = stats.t.ppf(0.975, n - 2)
        alpha_ci = (alpha - t_val * std_err, alpha + t_val * std_err)

        return {
            "alpha": alpha,
            "A": A,
            "r_squared": r_squared,
            "p_value": p_value,
            "std_err": std_err,
            "alpha_ci": alpha_ci,
        }

    def fit_chinchilla(
        self,
        params: np.ndarray,
        compute: np.ndarray,
        performance: np.ndarray,
    ) -> Dict[str, float]:
        """Fit Chinchilla-style scaling law.

        L = E + A/N^alpha + B/D^beta

        Where N is parameters, D is data/compute, L is loss.

        Args:
            params: Parameter counts.
            compute: Compute/data amounts.
            performance: Performance values (lower is better for loss).

        Returns:
            Fitted parameters.
        """
        def chinchilla_loss(X, E, A, alpha, B, beta):
            N, D = X
            return E + A / (N ** alpha) + B / (D ** beta)

        X = np.vstack([params, compute])

        try:
            popt, pcov = curve_fit(
                chinchilla_loss,
                X,
                performance,
                p0=[0.1, 1.0, 0.5, 1.0, 0.5],
                bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, 2, np.inf, 2]),
                maxfev=10000,
            )

            E, A, alpha, B, beta = popt
            perr = np.sqrt(np.diag(pcov))

            return {
                "E": E,
                "A": A,
                "alpha": alpha,
                "B": B,
                "beta": beta,
                "param_errors": perr,
            }
        except Exception as e:
            logger.error(f"Failed to fit Chinchilla law: {e}")
            return {}

    def compute_pareto_frontier(
        self,
        x: np.ndarray,
        y: np.ndarray,
        maximize_y: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Pareto frontier.

        Args:
            x: First objective (e.g., compute).
            y: Second objective (e.g., performance).
            maximize_y: Whether to maximize y (vs minimize).

        Returns:
            Tuple of (pareto_x, pareto_y, pareto_indices).
        """
        # Sort by x
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        pareto_indices = [0]
        best_y = y_sorted[0]

        for i in range(1, len(x_sorted)):
            if maximize_y:
                if y_sorted[i] > best_y:
                    pareto_indices.append(i)
                    best_y = y_sorted[i]
            else:
                if y_sorted[i] < best_y:
                    pareto_indices.append(i)
                    best_y = y_sorted[i]

        pareto_indices = np.array(pareto_indices)

        return (
            x_sorted[pareto_indices],
            y_sorted[pareto_indices],
            sorted_indices[pareto_indices],
        )

    def find_optimal_allocation(
        self,
        performance_data: pd.DataFrame,
        compute_budget: float,
        param_col: str = "target_params",
        compute_col: str = "total_compute",
        perf_col: str = "final_reward",
    ) -> Dict[str, Any]:
        """Find optimal parameter/compute allocation for given budget.

        Args:
            performance_data: DataFrame with results.
            compute_budget: Total compute budget.
            param_col: Column name for parameters.
            compute_col: Column name for compute.
            perf_col: Column name for performance.

        Returns:
            Optimal configuration.
        """
        # Filter to configurations within budget
        within_budget = performance_data[
            performance_data[compute_col] <= compute_budget
        ]

        if len(within_budget) == 0:
            return {"error": "No configurations within budget"}

        # Find best performance
        best_idx = within_budget[perf_col].idxmax()
        best_config = within_budget.loc[best_idx]

        return {
            "optimal_params": best_config[param_col],
            "optimal_compute": best_config[compute_col],
            "optimal_performance": best_config[perf_col],
            "efficiency": best_config[perf_col] / best_config[compute_col],
        }

    def plot_scaling_curve(
        self,
        params: np.ndarray,
        performance: np.ndarray,
        fit_result: Dict[str, float],
        title: str = "Scaling Curve",
        xlabel: str = "Parameters",
        ylabel: str = "Performance",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot scaling curve with power law fit.

        Args:
            params: Parameter counts.
            performance: Performance values.
            fit_result: Result from fit_power_law.
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            save_path: Path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data points
        ax.scatter(params, performance, s=100, alpha=0.7, label="Data")

        # Plot fit
        x_fit = np.logspace(
            np.log10(params.min()),
            np.log10(params.max()),
            100
        )
        y_fit = fit_result["A"] * x_fit ** fit_result["alpha"]
        ax.plot(
            x_fit, y_fit, "--",
            label=f'Fit: y = {fit_result["A"]:.2e} × x^{fit_result["alpha"]:.3f}'
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title}\n(R² = {fit_result["r_squared"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {save_path}")

        return fig

    def plot_pareto_frontier(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pareto_x: np.ndarray,
        pareto_y: np.ndarray,
        xlabel: str = "Compute",
        ylabel: str = "Performance",
        title: str = "Pareto Frontier",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot Pareto frontier.

        Args:
            x: All x values.
            y: All y values.
            pareto_x: Pareto optimal x values.
            pareto_y: Pareto optimal y values.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            title: Plot title.
            save_path: Path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot all points
        ax.scatter(x, y, alpha=0.5, label="All configurations")

        # Plot Pareto frontier
        ax.plot(pareto_x, pareto_y, "r-o", linewidth=2, markersize=8,
                label="Pareto frontier")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def generate_report(
        self,
        output_dir: str,
        group_by: List[str] = ["algorithm", "environment"],
    ) -> Dict[str, Any]:
        """Generate comprehensive scaling analysis report.

        Args:
            output_dir: Directory to save report.
            group_by: Columns to group analysis by.

        Returns:
            Report summary.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.data is None:
            self.load_results()

        report = {"groups": {}}

        # Analyze each group
        for group_vals, group_data in self.data.groupby(group_by):
            group_name = "_".join(str(v) for v in group_vals)

            params = group_data["target_params"].values
            perf = group_data["final_reward"].values

            # Fit scaling law
            if len(params) >= 3:
                fit = self.fit_power_law(params, perf)

                # Plot
                fig = self.plot_scaling_curve(
                    params, perf, fit,
                    title=f"Scaling Curve: {group_name}",
                    save_path=str(output_dir / f"scaling_{group_name}.png"),
                )
                plt.close(fig)

                report["groups"][group_name] = {
                    "n_experiments": len(group_data),
                    "scaling_exponent": fit["alpha"],
                    "r_squared": fit["r_squared"],
                    "best_performance": perf.max(),
                    "param_range": [int(params.min()), int(params.max())],
                }

        # Save report
        with open(output_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Generated report in {output_dir}")
        return report


def compute_scaling_exponents_by_domain(
    results: pd.DataFrame,
    param_col: str = "target_params",
    perf_col: str = "final_reward",
    domain_col: str = "environment",
) -> pd.DataFrame:
    """Compute scaling exponents for each domain.

    Args:
        results: DataFrame with results.
        param_col: Parameter column.
        perf_col: Performance column.
        domain_col: Domain column.

    Returns:
        DataFrame with scaling exponents per domain.
    """
    analyzer = ScalingLawAnalyzer("")
    records = []

    for domain, domain_data in results.groupby(domain_col):
        params = domain_data[param_col].values
        perf = domain_data[perf_col].values

        if len(params) >= 3:
            fit = analyzer.fit_power_law(params, perf)
            records.append({
                "domain": domain,
                "alpha": fit["alpha"],
                "A": fit["A"],
                "r_squared": fit["r_squared"],
                "n_points": len(params),
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze scaling law results")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="analysis_output")
    args = parser.parse_args()

    analyzer = ScalingLawAnalyzer(args.results_dir)
    analyzer.load_results()
    analyzer.generate_report(args.output_dir)
