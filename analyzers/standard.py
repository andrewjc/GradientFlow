# -*- coding: utf-8 -*-
"""
Standard Network Analyzer
=========================

Analyzer for feedforward networks including MLPs, CNNs, and
standard sequential architectures. Focuses on layer-by-layer
gradient flow patterns and common issues like vanishing/exploding
gradients in deep stacks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass

from ..core.engine import FlowAnalyzer
from ..core.metrics import FlowMetrics, LayerHealth
from ..visualizers.report import FlowReport


@dataclass
class LayerTypeStats:
    """Statistics aggregated by layer type."""
    layer_type: str
    count: int
    avg_pressure: float
    max_pressure: float
    min_pressure: float
    avg_health: float
    problematic_count: int


class StandardAnalyzer:
    """
    Analyzer for standard feedforward neural networks.

    This analyzer is designed for:
    - Multi-Layer Perceptrons (MLPs)
    - Convolutional Neural Networks (CNNs)
    - Any sequential feedforward architecture

    Key diagnostics:
    - Gradient magnitude decay through depth
    - Layer type pressure analysis
    - Bottleneck detection (compression layers)
    - Activation function gradient behavior

    Example:
        >>> analyzer = StandardAnalyzer()
        >>> report = analyzer.analyze(model, sample_input, sample_target, loss_fn)
        >>> report.print_summary()
        >>>
        >>> # Get layer type statistics
        >>> stats = analyzer.get_layer_type_stats()
        >>> for s in stats:
        ...     print(f"{s.layer_type}: avg_pressure={s.avg_pressure:.2e}")
    """

    # Layer types that commonly cause gradient issues
    COMPRESSION_LAYERS = (nn.Linear,)
    NORMALIZATION_LAYERS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)
    ACTIVATION_LAYERS = (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)
    POOLING_LAYERS = (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)

    def __init__(
        self,
        track_activations: bool = True,
        depth_analysis: bool = True,
        bottleneck_threshold: float = 0.25,
    ):
        """
        Initialize standard analyzer.

        Args:
            track_activations: Also track forward activations
            depth_analysis: Analyze gradient decay through depth
            bottleneck_threshold: Fan-in/fan-out ratio threshold for bottleneck detection
        """
        self.track_activations = track_activations
        self.depth_analysis = depth_analysis
        self.bottleneck_threshold = bottleneck_threshold

        self._flow_analyzer: Optional[FlowAnalyzer] = None
        self._layer_depths: Dict[str, int] = {}
        self._layer_types: Dict[str, str] = {}
        self._compression_ratios: Dict[str, float] = {}

    def analyze(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        sample_target: torch.Tensor,
        loss_fn: Callable,
        model_name: str = "StandardNetwork",
        num_samples: int = 1,
    ) -> FlowReport:
        """
        Analyze gradient flow in a standard network.

        Args:
            model: The neural network to analyze
            sample_input: Sample input tensor
            sample_target: Sample target tensor
            loss_fn: Loss function
            model_name: Name for the report
            num_samples: Number of analysis iterations

        Returns:
            FlowReport with analysis results
        """
        # Pre-analyze network structure
        self._analyze_structure(model)

        # Run flow analysis
        self._flow_analyzer = FlowAnalyzer(model, model_name)
        report = self._flow_analyzer.analyze(
            sample_input,
            sample_target,
            loss_fn,
            num_samples=num_samples,
        )

        # Add standard-specific analysis
        self._add_depth_analysis(report)
        self._add_bottleneck_analysis(report)
        self._add_layer_type_analysis(report)

        return report

    def _analyze_structure(self, model: nn.Module) -> None:
        """Pre-analyze network structure for depth and compression."""
        self._layer_depths.clear()
        self._layer_types.clear()
        self._compression_ratios.clear()

        depth = 0
        for name, module in model.named_modules():
            if name == "":
                continue

            self._layer_types[name] = type(module).__name__

            # Track depth for leaf modules
            if len(list(module.children())) == 0:
                self._layer_depths[name] = depth
                depth += 1

            # Calculate compression ratio for Linear layers
            if isinstance(module, nn.Linear):
                fan_in = module.in_features
                fan_out = module.out_features
                if fan_in > 0:
                    ratio = fan_out / fan_in
                    self._compression_ratios[name] = ratio

    def _add_depth_analysis(self, report: FlowReport) -> None:
        """Add depth-based gradient decay analysis."""
        if not self.depth_analysis:
            return

        # Group metrics by depth
        depth_pressures: Dict[int, List[float]] = {}

        for name, metrics in report.metrics.items():
            if name in self._layer_depths:
                depth = self._layer_depths[name]
                if depth not in depth_pressures:
                    depth_pressures[depth] = []
                depth_pressures[depth].append(metrics.mean_pressure)

        if len(depth_pressures) < 2:
            return

        # Calculate decay rate
        depths = sorted(depth_pressures.keys())
        pressures = [np.mean(depth_pressures[d]) for d in depths]

        if len(pressures) >= 2 and pressures[0] > 0 and pressures[-1] > 0:
            # Log-scale decay rate
            decay_rate = np.log(pressures[-1] / pressures[0]) / len(depths)

            if decay_rate < -0.5:
                report.issues.append({
                    "type": "DECAY",
                    "severity": "HIGH" if decay_rate < -1.0 else "MEDIUM",
                    "layer": "network-wide",
                    "layer_type": "depth",
                    "info": f"Gradient decay rate: {decay_rate:.3f} (log-scale per layer)"
                })

    def _add_bottleneck_analysis(self, report: FlowReport) -> None:
        """Add bottleneck detection for compression layers."""
        for name, ratio in self._compression_ratios.items():
            if ratio < self.bottleneck_threshold:
                # This is a compression bottleneck
                metrics = report.metrics.get(name)
                if metrics and metrics.max_pressure > 10:
                    severity = "CRITICAL" if metrics.max_pressure > 100 else (
                        "HIGH" if metrics.max_pressure > 50 else "MEDIUM"
                    )
                    report.issues.append({
                        "type": "BOTTLENECK",
                        "severity": severity,
                        "layer": name,
                        "layer_type": self._layer_types.get(name, "Linear"),
                        "info": f"Compression {ratio:.2f}x with pressure {metrics.max_pressure:.1f}"
                    })

    def _add_layer_type_analysis(self, report: FlowReport) -> None:
        """Add analysis grouped by layer type."""
        # Group by type
        type_metrics: Dict[str, List[FlowMetrics]] = {}

        for name, metrics in report.metrics.items():
            layer_type = self._layer_types.get(name, "Unknown")
            if layer_type not in type_metrics:
                type_metrics[layer_type] = []
            type_metrics[layer_type].append(metrics)

        # Check for problematic layer types
        for layer_type, metrics_list in type_metrics.items():
            avg_pressure = np.mean([m.mean_pressure for m in metrics_list])
            max_pressure = max(m.max_pressure for m in metrics_list)

            if max_pressure > 100 and len(metrics_list) > 1:
                report.issues.append({
                    "type": "TYPE_PRESSURE",
                    "severity": "HIGH",
                    "layer": f"all {layer_type}",
                    "layer_type": layer_type,
                    "info": f"{len(metrics_list)} layers, avg={avg_pressure:.1f}, max={max_pressure:.1f}"
                })

    def get_layer_type_stats(self) -> List[LayerTypeStats]:
        """
        Get statistics aggregated by layer type.

        Returns:
            List of LayerTypeStats for each layer type
        """
        if self._flow_analyzer is None:
            return []

        report = self._flow_analyzer._last_report
        if report is None:
            return []

        # Group by type
        type_data: Dict[str, Dict[str, Any]] = {}

        for name, metrics in report.metrics.items():
            layer_type = self._layer_types.get(name, "Unknown")
            if layer_type not in type_data:
                type_data[layer_type] = {
                    "pressures": [],
                    "health_scores": [],
                }
            type_data[layer_type]["pressures"].append(metrics.mean_pressure)

            health = report.health.get(name)
            if health:
                type_data[layer_type]["health_scores"].append(health.score)

        # Build stats
        stats = []
        for layer_type, data in type_data.items():
            pressures = data["pressures"]
            health_scores = data["health_scores"]

            stats.append(LayerTypeStats(
                layer_type=layer_type,
                count=len(pressures),
                avg_pressure=np.mean(pressures),
                max_pressure=max(pressures),
                min_pressure=min(pressures),
                avg_health=np.mean(health_scores) if health_scores else 0.0,
                problematic_count=sum(1 for h in health_scores if h < 50),
            ))

        return sorted(stats, key=lambda s: s.avg_pressure, reverse=True)

    def get_depth_profile(self) -> List[Tuple[int, float, float]]:
        """
        Get gradient pressure profile by depth.

        Returns:
            List of (depth, mean_pressure, std_pressure) tuples
        """
        if self._flow_analyzer is None:
            return []

        report = self._flow_analyzer._last_report
        if report is None:
            return []

        depth_pressures: Dict[int, List[float]] = {}

        for name, metrics in report.metrics.items():
            if name in self._layer_depths:
                depth = self._layer_depths[name]
                if depth not in depth_pressures:
                    depth_pressures[depth] = []
                depth_pressures[depth].append(metrics.mean_pressure)

        profile = []
        for depth in sorted(depth_pressures.keys()):
            pressures = depth_pressures[depth]
            profile.append((
                depth,
                np.mean(pressures),
                np.std(pressures) if len(pressures) > 1 else 0.0
            ))

        return profile

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._flow_analyzer:
            self._flow_analyzer.cleanup()
