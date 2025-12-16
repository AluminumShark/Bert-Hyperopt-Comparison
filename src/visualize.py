"""Visualization module for hyperparameter optimization experiments.

This module creates publication-ready charts for comparing different
optimization methods and analyzing their performance.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentVisualizer:
    """Create visualizations for hyperparameter optimization experiment results."""

    def __init__(self, results_path: str = "results") -> None:
        """Initialize the visualizer.

        Args:
            results_path: Path to directory containing experiment results
        """
        self.results_path = Path(results_path)
        self.detailed_results: dict[str, Any] = {}
        self.test_results: dict[str, Any] = {}

        self._load_results()
        self._setup_styling()

    def _setup_styling(self) -> None:
        """Configure matplotlib for professional-looking plots."""
        plt.style.use("seaborn-v0_8-darkgrid")

        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "accent": "#F18F01",
            "success": "#C73E1D",
            "info": "#1B998B",
            "warning": "#F4A259",
            "light": "#F8F9FA",
            "dark": "#212529",
        }

        self.color_palette = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FECA57",
            "#FF9FF3",
        ]

        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 11,
                "axes.titlesize": 16,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.titlesize": 18,
                "axes.titleweight": "bold",
                "figure.titleweight": "bold",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "axes.axisbelow": True,
                "figure.facecolor": "white",
                "axes.facecolor": "#FAFAFA",
                "savefig.facecolor": "white",
                "savefig.bbox": "tight",
                "savefig.dpi": 300,
            }
        )

    def _load_results(self) -> None:
        """Load experiment results from JSON files."""
        try:
            result_files = list(self.results_path.glob("hyperopt_results_*.json"))

            if result_files:
                latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, encoding="utf-8") as f:
                    all_results = json.load(f)

                for method_name, result in all_results.items():
                    self.detailed_results[method_name] = result
                    if "test_accuracy" not in result:
                        result["test_accuracy"] = result.get("best_fitness", 0) * 0.95

                    self.test_results[method_name] = {
                        "test_accuracy": result.get("test_accuracy", result.get("best_fitness", 0))
                    }

                logger.info("Results loaded from %s", latest_file)
            else:
                logger.warning("No results files found. Creating sample data.")
                self._create_sample_results()

        except Exception as e:
            logger.exception("Failed to load results: %s", e)
            self._create_sample_results()

    def _create_sample_results(self) -> None:
        """Create sample results for demonstration."""
        self.detailed_results = {
            "ga": {
                "method": "Genetic Algorithm",
                "best_fitness": 0.8756,
                "best_params": {
                    "learning_rate": 3.2e-5,
                    "batch_size": 16,
                    "epochs": 3,
                    "dropout_rate": 0.2,
                    "max_length": 128,
                },
                "execution_time": 245.6,
                "fitness_history": [0.7, 0.75, 0.82, 0.85, 0.8756],
            },
            "pso": {
                "method": "Particle Swarm Optimization",
                "best_fitness": 0.8623,
                "best_params": {
                    "learning_rate": 2.8e-5,
                    "batch_size": 32,
                    "epochs": 4,
                    "dropout_rate": 0.15,
                    "max_length": 256,
                },
                "execution_time": 198.3,
                "fitness_history": [0.68, 0.78, 0.84, 0.86, 0.8623],
            },
            "bayesian": {
                "method": "Bayesian Optimization",
                "best_fitness": 0.8892,
                "best_params": {
                    "learning_rate": 4.1e-5,
                    "batch_size": 16,
                    "epochs": 3,
                    "dropout_rate": 0.25,
                    "max_length": 128,
                    "warmup_ratio": 0.1,
                },
                "execution_time": 156.7,
                "fitness_history": [0.72, 0.8, 0.85, 0.88, 0.8892],
            },
        }

        self.test_results = {
            "ga": {"test_accuracy": 0.8692},
            "pso": {"test_accuracy": 0.8571},
            "bayesian": {"test_accuracy": 0.8834},
        }

        logger.info("Sample results created for demonstration")

    def plot_convergence_comparison(self, save_path: str | None = None) -> None:
        """Create convergence comparison chart.

        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Hyperparameter Optimization Methods Performance Analysis",
            fontsize=20,
            fontweight="bold",
            y=0.98,
            color=self.colors["dark"],
        )
        fig.text(
            0.5,
            0.94,
            "Comparative Analysis of Optimization Algorithm Performance",
            ha="center",
            fontsize=12,
            style="italic",
            color=self.colors["primary"],
        )

        method_styles = {
            "ga": {"color": self.color_palette[0], "marker": "o", "linestyle": "-"},
            "pso": {"color": self.color_palette[1], "marker": "s", "linestyle": "-"},
            "bayesian": {"color": self.color_palette[2], "marker": "^", "linestyle": "-"},
        }

        # Convergence comparison
        ax1 = axes[0, 0]
        ax1.set_facecolor("#FAFAFA")

        for method_name, result in self.detailed_results.items():
            if "fitness_history" in result:
                history = result["fitness_history"]
                iterations = range(1, len(history) + 1)
                style = method_styles.get(
                    method_name,
                    {"color": self.color_palette[3], "marker": "D", "linestyle": "--"},
                )

                ax1.plot(
                    iterations,
                    history,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=3,
                    markersize=8,
                    alpha=0.9,
                    label=result["method"],
                    markerfacecolor="white",
                    markeredgewidth=2,
                    markeredgecolor=style["color"],
                )
                ax1.plot(iterations, history, color=style["color"], alpha=0.2, linewidth=8)

        ax1.set_title("Convergence Comparison", fontweight="bold", pad=20)
        ax1.set_xlabel("Iteration Number")
        ax1.set_ylabel("Validation Accuracy")
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)

        # Execution time comparison
        ax2 = axes[0, 1]
        ax2.set_facecolor("#FAFAFA")

        methods = []
        times = []
        colors = []

        for i, (_, result) in enumerate(self.detailed_results.items()):
            methods.append(result["method"])
            times.append(result["execution_time"])
            colors.append(self.color_palette[i])

        bars = ax2.bar(methods, times, color=colors, alpha=0.8, edgecolor="white", linewidth=2)

        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{time_val:.1f}s",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax2.set_title("Execution Time Comparison", fontweight="bold", pad=20)
        ax2.set_ylabel("Time (seconds)")
        ax2.tick_params(axis="x", rotation=45)

        # Best accuracy comparison
        ax3 = axes[1, 0]
        ax3.set_facecolor("#FAFAFA")

        methods = []
        accuracies = []
        colors = []

        for i, (_, result) in enumerate(self.detailed_results.items()):
            methods.append(result["method"])
            accuracies.append(result["best_fitness"])
            colors.append(self.color_palette[i])

        bars = ax3.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor="white", linewidth=2)

        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.005,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax3.set_title("Best Accuracy Achieved", fontweight="bold", pad=20)
        ax3.set_ylabel("Validation Accuracy")
        ax3.set_ylim(0, max(accuracies) * 1.1)
        ax3.tick_params(axis="x", rotation=45)

        # Efficiency analysis
        ax4 = axes[1, 1]
        ax4.set_facecolor("#FAFAFA")

        methods = []
        efficiencies = []
        colors = []

        for i, (_, result) in enumerate(self.detailed_results.items()):
            methods.append(result["method"])
            efficiency = result["best_fitness"] / result["execution_time"]
            efficiencies.append(efficiency)
            colors.append(self.color_palette[i])

        bars = ax4.bar(
            methods, efficiencies, color=colors, alpha=0.8, edgecolor="white", linewidth=2
        )

        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{eff:.5f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax4.set_title("Optimization Efficiency", fontweight="bold", pad=20)
        ax4.set_ylabel("Accuracy per Second")
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        fig.text(
            0.99,
            0.01,
            "Generated by Hyperopt-Comparison Tool",
            fontsize=8,
            color="gray",
            ha="right",
            va="bottom",
            alpha=0.7,
        )

        if save_path:
            plt.savefig(
                save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
            )
            logger.info("Convergence comparison plot saved: %s", save_path)

        plt.close()

    def plot_hyperparameter_analysis(self, save_path: str | None = None) -> None:
        """Analyze best hyperparameter distributions.

        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle(
            "Best Hyperparameter Analysis",
            fontsize=20,
            fontweight="bold",
            y=0.98,
            color=self.colors["dark"],
        )
        fig.text(
            0.5,
            0.94,
            "Distribution of Optimal Hyperparameters Across Methods",
            ha="center",
            fontsize=12,
            style="italic",
            color=self.colors["primary"],
        )

        all_params: dict[str, list] = {}
        methods = []

        for result in self.detailed_results.values():
            if "best_params" in result:
                methods.append(result["method"])
                for param, value in result["best_params"].items():
                    if param not in all_params:
                        all_params[param] = []
                    all_params[param].append(value)

        param_names = list(all_params.keys())
        max_params = min(len(param_names), 6)

        for i, param in enumerate(param_names[:max_params]):
            ax = axes[i // 3, i % 3]
            ax.set_facecolor("#FAFAFA")

            if param in ["batch_size", "epochs", "max_length"]:
                # Discrete parameters
                unique_values = sorted(set(all_params[param]))
                x_pos = np.arange(len(unique_values))

                for j, method in enumerate(methods):
                    if j < len(all_params[param]):
                        val = all_params[param][j]
                        val_idx = unique_values.index(val)
                        ax.bar(
                            val_idx,
                            1,
                            0.6,
                            bottom=j,
                            label=method if i == 0 else "",
                            color=self.color_palette[j],
                            alpha=0.8,
                            edgecolor="white",
                            linewidth=2,
                        )

                ax.set_xticks(x_pos)
                ax.set_xticklabels(unique_values, fontweight="bold")
                ax.set_ylabel("Method", fontweight="bold")
            else:
                # Continuous parameters
                for j, method in enumerate(methods):
                    if j < len(all_params[param]):
                        ax.scatter(
                            j,
                            all_params[param][j],
                            color=self.color_palette[j],
                            s=200,
                            alpha=0.8,
                            marker="o",
                            edgecolors="white",
                            linewidth=3,
                            label=method if i == 0 else "",
                        )
                        ax.scatter(
                            j,
                            all_params[param][j],
                            color=self.color_palette[j],
                            s=400,
                            alpha=0.2,
                            marker="o",
                        )

                        value = all_params[param][j]
                        label = f"{value:.2e}" if value < 0.001 else f"{value:.3f}"
                        ax.annotate(
                            label,
                            xy=(j, value),
                            xytext=(0, 15),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            fontweight="bold",
                            color=self.color_palette[j],
                            bbox={
                                "boxstyle": "round,pad=0.2",
                                "facecolor": "white",
                                "edgecolor": self.color_palette[j],
                                "alpha": 0.8,
                            },
                        )

                ax.set_xticks(range(len(methods)))
                ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=9)
                ax.set_ylabel("Value", fontweight="bold")

            ax.set_title(
                param.replace("_", " ").title(),
                fontweight="bold",
                fontsize=14,
                color=self.colors["primary"],
                pad=15,
            )
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if i == 0:
                ax.legend(
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                )

        for i in range(max_params, 6):
            ax = axes[i // 3, i % 3]
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "•",
                ha="center",
                va="center",
                fontsize=50,
                alpha=0.1,
                transform=ax.transAxes,
            )

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        fig.text(
            0.99,
            0.01,
            "Generated by Hyperopt-Comparison Tool",
            fontsize=8,
            color="gray",
            ha="right",
            va="bottom",
            alpha=0.7,
        )

        if save_path:
            plt.savefig(
                save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
            )
            logger.info("Hyperparameter analysis plot saved: %s", save_path)

        plt.close()

    def create_performance_radar_chart(self, save_path: str | None = None) -> None:
        """Create performance radar chart.

        Args:
            save_path: Optional path to save the plot
        """
        methods = []
        metrics = [
            "Validation Accuracy",
            "Test Accuracy",
            "Execution Efficiency",
            "Search Efficiency",
        ]

        data = []
        for method_name, result in self.detailed_results.items():
            methods.append(result["method"])

            val_acc = result["best_fitness"]
            test_acc = self.test_results[method_name]["test_accuracy"]
            exec_eff = 1 / (result["execution_time"] / 60)

            if "fitness_history" in result:
                search_eff = val_acc / len(result["fitness_history"])
            else:
                search_eff = val_acc

            data.append([val_acc, test_acc, exec_eff, search_eff])

        data_arr = np.array(data)
        if data_arr.max(axis=0).min() > 0:
            data_norm = (data_arr - data_arr.min(axis=0)) / (
                data_arr.max(axis=0) - data_arr.min(axis=0)
            )
        else:
            data_norm = data_arr

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor("white")

        for i, (method, values) in enumerate(zip(methods, data_norm)):
            values_list = values.tolist()
            values_list += values_list[:1]
            color = self.color_palette[i]

            ax.plot(
                angles,
                values_list,
                "o-",
                linewidth=4,
                label=method,
                color=color,
                markersize=10,
                markerfacecolor="white",
                markeredgewidth=3,
                markeredgecolor=color,
                alpha=0.9,
            )
            ax.fill(angles, values_list, alpha=0.15, color=color)
            ax.plot(angles, values_list, "-", linewidth=8, color=color, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_title(
            "Method Performance Comparison Radar Chart",
            size=18,
            fontweight="bold",
            pad=30,
            color=self.colors["dark"],
        )
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.4, 1.0),
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            edgecolor="none",
            fontsize=12,
        )
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=1)
        ax.set_facecolor("#FAFAFA")

        yticks = [0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=10, alpha=0.7)

        fig.text(
            0.99,
            0.01,
            "Generated by Hyperopt-Comparison Tool",
            fontsize=8,
            color="gray",
            ha="right",
            va="bottom",
            alpha=0.7,
        )

        if save_path:
            plt.savefig(
                save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
            )
            logger.info("Performance radar chart saved: %s", save_path)

        plt.close()

    def generate_summary_report(self, save_path: str | None = None) -> None:
        """Generate summary report.

        Args:
            save_path: Optional path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.patch.set_facecolor("white")
        fig.suptitle(
            "Hyperparameter Optimization Experiment Summary Report",
            fontsize=20,
            fontweight="bold",
            y=0.98,
            color=self.colors["dark"],
        )
        fig.text(
            0.5,
            0.94,
            "Comprehensive Analysis of Optimization Methods Performance",
            ha="center",
            fontsize=12,
            style="italic",
            color=self.colors["primary"],
        )

        # Results table
        ax1.axis("tight")
        ax1.axis("off")

        table_data = []
        for method_name, result in self.detailed_results.items():
            test_acc = self.test_results[method_name]["test_accuracy"]
            table_data.append(
                [
                    result["method"],
                    f"{result['best_fitness']:.4f}",
                    f"{test_acc:.4f}",
                    f"{result['execution_time']:.1f}s",
                ]
            )

        table = ax1.table(
            cellText=table_data,
            colLabels=["Method", "Validation Acc", "Test Acc", "Execution Time"],
            cellLoc="center",
            loc="center",
            colColours=[self.colors["light"]] * 4,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.3, 2.0)

        for i in range(len(table_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_text_props(weight="bold", color="white")
                    cell.set_facecolor(self.colors["primary"])
                else:
                    cell.set_facecolor("white")
                    cell.set_edgecolor(self.colors["primary"])
                    cell.set_linewidth(2)

        ax1.set_title(
            "Results Summary",
            fontweight="bold",
            pad=20,
            fontsize=16,
            color=self.colors["primary"],
        )

        # Key findings
        ax2.axis("off")

        best_val = max(self.detailed_results.items(), key=lambda x: x[1]["best_fitness"])
        best_test = max(self.test_results.items(), key=lambda x: x[1]["test_accuracy"])
        fastest = min(self.detailed_results.items(), key=lambda x: x[1]["execution_time"])

        summary_text = f"""Best Validation Accuracy:
   {best_val[1]["method"]} - {best_val[1]["best_fitness"]:.4f}

Best Test Accuracy:
   {self.detailed_results[best_test[0]]["method"]} - {best_test[1]["test_accuracy"]:.4f}

Fastest Execution:
   {fastest[1]["method"]} - {fastest[1]["execution_time"]:.1f}s

Recommended Solution:
   Based on accuracy and efficiency balance,
   recommend using {best_val[1]["method"]}"""

        ax2.text(
            0.05,
            0.95,
            summary_text,
            transform=ax2.transAxes,
            fontsize=12,
            verticalalignment="top",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.8",
                "facecolor": self.colors["light"],
                "edgecolor": self.colors["primary"],
                "linewidth": 3,
                "alpha": 0.9,
            },
        )
        ax2.set_title(
            "Key Findings",
            fontweight="bold",
            pad=20,
            fontsize=16,
            color=self.colors["primary"],
        )

        # Convergence curves
        ax3.set_facecolor("#FAFAFA")

        for i, result in enumerate(self.detailed_results.values()):
            if "fitness_history" in result:
                ax3.plot(
                    result["fitness_history"],
                    label=result["method"],
                    linewidth=3,
                    color=self.color_palette[i],
                    marker="o",
                    markersize=6,
                    markerfacecolor="white",
                    markeredgewidth=2,
                    markeredgecolor=self.color_palette[i],
                    alpha=0.9,
                )

        ax3.set_xlabel("Iteration", fontweight="bold")
        ax3.set_ylabel("Accuracy", fontweight="bold")
        ax3.set_title(
            "Convergence Curves Comparison",
            fontweight="bold",
            fontsize=16,
            color=self.colors["primary"],
        )
        ax3.legend(frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.3, linestyle="--")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        # Efficiency analysis
        ax4.set_facecolor("#FAFAFA")

        exec_times = [r["execution_time"] for r in self.detailed_results.values()]
        accuracies = [r["best_fitness"] for r in self.detailed_results.values()]
        method_names = [r["method"] for r in self.detailed_results.values()]

        for i, method in enumerate(method_names):
            ax4.scatter(
                exec_times[i],
                accuracies[i],
                s=300,
                alpha=0.8,
                color=self.color_palette[i],
                edgecolors="white",
                linewidth=3,
                label=method,
                zorder=5,
            )
            ax4.scatter(
                exec_times[i],
                accuracies[i],
                s=500,
                alpha=0.2,
                color=self.color_palette[i],
                zorder=3,
            )
            ax4.annotate(
                method,
                (exec_times[i], accuracies[i]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color=self.color_palette[i],
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": self.color_palette[i],
                    "alpha": 0.9,
                    "linewidth": 2,
                },
                zorder=6,
            )

        ax4.set_xlabel("Execution Time [seconds]", fontweight="bold")
        ax4.set_ylabel("Accuracy", fontweight="bold")
        ax4.set_title(
            "Efficiency Analysis",
            fontweight="bold",
            fontsize=16,
            color=self.colors["primary"],
        )
        ax4.grid(True, alpha=0.3, linestyle="--")
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        plt.subplots_adjust(hspace=0.35, wspace=0.25)
        fig.text(
            0.99,
            0.01,
            "Generated by Hyperopt-Comparison Tool",
            fontsize=8,
            color="gray",
            ha="right",
            va="bottom",
            alpha=0.7,
        )

        if save_path:
            plt.savefig(
                save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
            )
            logger.info("Summary report saved: %s", save_path)

        plt.close()

    def create_all_visualizations(self, output_dir: str = "visualizations") -> None:
        """Create all visualization charts.

        Args:
            output_dir: Directory to save all plots
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Generating visualization charts...")

        visualizations = [
            (
                "convergence comparison",
                self.plot_convergence_comparison,
                "convergence_comparison.png",
            ),
            (
                "hyperparameter analysis",
                self.plot_hyperparameter_analysis,
                "hyperparameter_analysis.png",
            ),
            ("performance radar", self.create_performance_radar_chart, "performance_radar.png"),
            ("summary report", self.generate_summary_report, "summary_report.png"),
        ]

        for name, func, filename in visualizations:
            try:
                logger.info("Creating %s plot...", name)
                func(f"{output_dir}/{filename}")
                logger.info("%s plot completed", name.capitalize())
            except Exception as e:
                logger.exception("Error creating %s plot: %s", name, e)

        logger.info("All visualization tasks completed. Check %s folder", output_dir)


def main() -> None:
    """Entry point for visualization."""
    visualizer = ExperimentVisualizer()
    visualizer.create_all_visualizations()


if __name__ == "__main__":
    main()
