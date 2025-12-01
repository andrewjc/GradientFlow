# -*- coding: utf-8 -*-
"""
Comparative Analyzer
====================

Analyzer for comparing gradient flow between different models,
checkpoints, or training stages. Useful for tracking how gradient
health evolves during training or comparing architectural choices.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

from ..core.engine import FlowAnalyzer
from ..core.metrics import FlowMetrics, LayerHealth
from ..visualizers.report import FlowReport


@dataclass
class LayerComparison:
    """Comparison between a layer across two analyses."""
    layer_name: str
    layer_type: str

    # Metrics from each analysis
    pressure_a: float
    pressure_b: float
    health_a: float
    health_b: float

    @property
    def pressure_change(self) -> float:
        """Relative change in pressure (positive = increased)."""
        if self.pressure_a > 0:
            return (self.pressure_b - self.pressure_a) / self.pressure_a
        return 0.0

    @property
    def health_change(self) -> float:
        """Absolute change in health score."""
        return self.health_b - self.health_a

    @property
    def improved(self) -> bool:
        """Check if the layer improved."""
        return self.health_b > self.health_a

    @property
    def degraded(self) -> bool:
        """Check if the layer degraded significantly."""
        return self.health_b < self.health_a - 10


@dataclass
class ComparisonReport:
    """Full comparison between two analyses."""
    name_a: str
    name_b: str
    layer_comparisons: List[LayerComparison] = field(default_factory=list)

    @property
    def improved_layers(self) -> List[LayerComparison]:
        """Layers that improved."""
        return [c for c in self.layer_comparisons if c.improved]

    @property
    def degraded_layers(self) -> List[LayerComparison]:
        """Layers that degraded."""
        return [c for c in self.layer_comparisons if c.degraded]

    @property
    def avg_health_change(self) -> float:
        """Average health change across all layers."""
        if not self.layer_comparisons:
            return 0.0
        return np.mean([c.health_change for c in self.layer_comparisons])

    @property
    def avg_pressure_change(self) -> float:
        """Average pressure change across all layers."""
        if not self.layer_comparisons:
            return 0.0
        return np.mean([c.pressure_change for c in self.layer_comparisons])


class ComparativeAnalyzer:
    """
    Analyzer for comparing gradient flow between models or checkpoints.

    This analyzer is designed for:
    - Comparing checkpoints during training
    - Comparing architectural variations
    - Tracking gradient health evolution
    - A/B testing model modifications

    Key diagnostics:
    - Per-layer health changes
    - Pressure evolution tracking
    - Improvement/degradation detection
    - Statistical significance testing

    Example:
        >>> analyzer = ComparativeAnalyzer()
        >>>
        >>> # Compare two checkpoints
        >>> report = analyzer.compare_checkpoints(
        ...     model_class,
        ...     "checkpoint_epoch_10.pt",
        ...     "checkpoint_epoch_50.pt",
        ...     sample_input,
        ...     sample_target,
        ...     loss_fn,
        ... )
        >>>
        >>> print(f"Improved layers: {len(report.improved_layers)}")
        >>> print(f"Degraded layers: {len(report.degraded_layers)}")
        >>> print(f"Average health change: {report.avg_health_change:.1f}")
    """

    def __init__(self):
        """Initialize comparative analyzer."""
        self._reports: Dict[str, FlowReport] = {}
        self._flow_analyzer: Optional[FlowAnalyzer] = None

    def analyze_model(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        sample_target: torch.Tensor,
        loss_fn: Callable,
        name: str = "model",
        num_samples: int = 1,
    ) -> FlowReport:
        """
        Analyze a model and store the report.

        Args:
            model: The neural network to analyze
            sample_input: Sample input tensor
            sample_target: Sample target tensor
            loss_fn: Loss function
            name: Identifier for this analysis
            num_samples: Number of analysis iterations

        Returns:
            FlowReport for this model
        """
        self._flow_analyzer = FlowAnalyzer(model, name)
        report = self._flow_analyzer.analyze(
            sample_input,
            sample_target,
            loss_fn,
            num_samples=num_samples,
        )
        self._reports[name] = report
        self._flow_analyzer.cleanup()
        return report

    def compare(
        self,
        name_a: str,
        name_b: str,
    ) -> ComparisonReport:
        """
        Compare two previously analyzed models.

        Args:
            name_a: Name of first analysis
            name_b: Name of second analysis

        Returns:
            ComparisonReport with layer-by-layer comparison
        """
        if name_a not in self._reports or name_b not in self._reports:
            raise ValueError(f"Models not found. Available: {list(self._reports.keys())}")

        report_a = self._reports[name_a]
        report_b = self._reports[name_b]

        return self._compare_reports(report_a, report_b, name_a, name_b)

    def compare_models(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        sample_input: torch.Tensor,
        sample_target: torch.Tensor,
        loss_fn: Callable,
        name_a: str = "model_a",
        name_b: str = "model_b",
        num_samples: int = 1,
    ) -> ComparisonReport:
        """
        Analyze and compare two models.

        Args:
            model_a: First model to analyze
            model_b: Second model to analyze
            sample_input: Sample input tensor
            sample_target: Sample target tensor
            loss_fn: Loss function
            name_a: Name for first model
            name_b: Name for second model
            num_samples: Number of analysis iterations

        Returns:
            ComparisonReport with layer-by-layer comparison
        """
        report_a = self.analyze_model(model_a, sample_input, sample_target, loss_fn, name_a, num_samples)
        report_b = self.analyze_model(model_b, sample_input, sample_target, loss_fn, name_b, num_samples)

        return self._compare_reports(report_a, report_b, name_a, name_b)

    def compare_checkpoints(
        self,
        model_factory: Callable[[], nn.Module],
        checkpoint_a: Union[str, Path],
        checkpoint_b: Union[str, Path],
        sample_input: torch.Tensor,
        sample_target: torch.Tensor,
        loss_fn: Callable,
        device: str = "cuda",
        num_samples: int = 1,
    ) -> ComparisonReport:
        """
        Load and compare two checkpoints.

        Args:
            model_factory: Function that creates a fresh model instance
            checkpoint_a: Path to first checkpoint
            checkpoint_b: Path to second checkpoint
            sample_input: Sample input tensor
            sample_target: Sample target tensor
            loss_fn: Loss function
            device: Device to load models on
            num_samples: Number of analysis iterations

        Returns:
            ComparisonReport with layer-by-layer comparison
        """
        # Load first checkpoint
        model_a = model_factory()
        state_a = torch.load(checkpoint_a, map_location=device, weights_only=False)
        if isinstance(state_a, dict) and "model_state_dict" in state_a:
            model_a.load_state_dict(state_a["model_state_dict"])
        else:
            model_a.load_state_dict(state_a)
        model_a.to(device)
        model_a.eval()

        # Load second checkpoint
        model_b = model_factory()
        state_b = torch.load(checkpoint_b, map_location=device, weights_only=False)
        if isinstance(state_b, dict) and "model_state_dict" in state_b:
            model_b.load_state_dict(state_b["model_state_dict"])
        else:
            model_b.load_state_dict(state_b)
        model_b.to(device)
        model_b.eval()

        name_a = Path(checkpoint_a).stem
        name_b = Path(checkpoint_b).stem

        return self.compare_models(
            model_a, model_b,
            sample_input.to(device),
            sample_target.to(device),
            loss_fn,
            name_a, name_b,
            num_samples,
        )

    def _compare_reports(
        self,
        report_a: FlowReport,
        report_b: FlowReport,
        name_a: str,
        name_b: str,
    ) -> ComparisonReport:
        """Compare two flow reports."""
        comparisons = []

        # Find common layers
        common_layers = set(report_a.metrics.keys()) & set(report_b.metrics.keys())

        for layer_name in sorted(common_layers):
            metrics_a = report_a.metrics[layer_name]
            metrics_b = report_b.metrics[layer_name]

            health_a = report_a.health.get(layer_name)
            health_b = report_b.health.get(layer_name)

            comparisons.append(LayerComparison(
                layer_name=layer_name,
                layer_type=health_a.layer_type if health_a else "Unknown",
                pressure_a=metrics_a.mean_pressure,
                pressure_b=metrics_b.mean_pressure,
                health_a=health_a.score if health_a else 0.0,
                health_b=health_b.score if health_b else 0.0,
            ))

        return ComparisonReport(
            name_a=name_a,
            name_b=name_b,
            layer_comparisons=comparisons,
        )

    def track_training(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        sample_target: torch.Tensor,
        loss_fn: Callable,
        step: int,
    ) -> FlowReport:
        """
        Track gradient flow at a training step.

        Call this periodically during training to build a history
        of gradient flow evolution.

        Args:
            model: The model being trained
            sample_input: Sample input tensor
            sample_target: Sample target tensor
            loss_fn: Loss function
            step: Current training step

        Returns:
            FlowReport for this step
        """
        name = f"step_{step}"
        return self.analyze_model(model, sample_input, sample_target, loss_fn, name)

    def get_training_history(self) -> List[Tuple[int, FlowReport]]:
        """
        Get all tracked training steps.

        Returns:
            List of (step, report) tuples sorted by step
        """
        history = []
        for name, report in self._reports.items():
            if name.startswith("step_"):
                step = int(name.split("_")[1])
                history.append((step, report))
        return sorted(history, key=lambda x: x[0])

    def get_health_evolution(self, layer_name: str) -> List[Tuple[int, float]]:
        """
        Get health score evolution for a specific layer.

        Args:
            layer_name: Name of the layer to track

        Returns:
            List of (step, health_score) tuples
        """
        evolution = []
        for step, report in self.get_training_history():
            health = report.health.get(layer_name)
            if health:
                evolution.append((step, health.score))
        return evolution

    def get_pressure_evolution(self, layer_name: str) -> List[Tuple[int, float]]:
        """
        Get pressure evolution for a specific layer.

        Args:
            layer_name: Name of the layer to track

        Returns:
            List of (step, pressure) tuples
        """
        evolution = []
        for step, report in self.get_training_history():
            metrics = report.metrics.get(layer_name)
            if metrics:
                evolution.append((step, metrics.mean_pressure))
        return evolution

    def print_comparison(self, comparison: ComparisonReport) -> None:
        """Print a comparison report to console."""
        print(f"\n{'='*80}")
        print(f"COMPARISON: {comparison.name_a} -> {comparison.name_b}")
        print(f"{'='*80}\n")

        print(f"Total layers compared: {len(comparison.layer_comparisons)}")
        print(f"Improved layers: {len(comparison.improved_layers)}")
        print(f"Degraded layers: {len(comparison.degraded_layers)}")
        print(f"Average health change: {comparison.avg_health_change:+.1f}")
        print(f"Average pressure change: {comparison.avg_pressure_change:+.1%}")

        if comparison.degraded_layers:
            print(f"\n{'Degraded Layers':^80}")
            print("-" * 80)
            print(f"{'Layer':<40} {'Health A':>10} {'Health B':>10} {'Change':>10}")
            print("-" * 80)

            for c in sorted(comparison.degraded_layers, key=lambda x: x.health_change)[:10]:
                print(f"{c.layer_name[:40]:<40} {c.health_a:>10.1f} {c.health_b:>10.1f} {c.health_change:>+10.1f}")

        if comparison.improved_layers:
            print(f"\n{'Improved Layers':^80}")
            print("-" * 80)
            print(f"{'Layer':<40} {'Health A':>10} {'Health B':>10} {'Change':>10}")
            print("-" * 80)

            for c in sorted(comparison.improved_layers, key=lambda x: x.health_change, reverse=True)[:10]:
                print(f"{c.layer_name[:40]:<40} {c.health_a:>10.1f} {c.health_b:>10.1f} {c.health_change:>+10.1f}")

    def clear_history(self) -> None:
        """Clear all stored reports."""
        self._reports.clear()
