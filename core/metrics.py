# -*- coding: utf-8 -*-
"""
Gradient Metrics and Health Assessment
=======================================

This module defines the core data structures for tracking and assessing
gradient propagation through neural networks.

Key Concepts:
    - FlowMetrics: Raw statistics collected during analysis
    - LayerHealth: Computed health scores and diagnostics
    - Magnitude: Gradient magnitude (L2 norm)
    - Variance: Gradient variance over time
    - Change Rate: Rate of change in gradient patterns
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class HealthStatus(Enum):
    """Health status categories for layers."""
    CRITICAL = "critical"
    UNHEALTHY = "unhealthy"
    MARGINAL = "marginal"
    HEALTHY = "healthy"
    OPTIMAL = "optimal"


@dataclass
class FlowMetrics:
    """
    Raw gradient flow statistics for a single layer.

    This dataclass accumulates statistics during the analysis simulation.
    After simulation completes, computed properties provide derived metrics.

    Attributes:
        name: Layer name (from model.named_modules())
        layer_type: Type of layer (Linear, Conv2d, RMSNorm, etc.)
        param_count: Number of parameters in this layer
        grad_norms: L2 norms of gradients at each step
        grad_means: Mean gradient values at each step
        grad_stds: Gradient standard deviations at each step
        grad_maxs: Maximum absolute gradient values at each step
        weight_norms: Weight L2 norms at each step (if tracked)
        activation_norms: Activation L2 norms at each step (if tracked)
        activation_means: Activation means at each step (if tracked)
        zero_count: Number of steps with zero/negligible gradients
        nan_count: Number of steps with NaN gradients
        inf_count: Number of steps with Inf gradients
    """

    name: str
    layer_type: str
    param_count: int = 0

    # Gradient statistics (collected each step)
    grad_norms: List[float] = field(default_factory=list)
    grad_means: List[float] = field(default_factory=list)
    grad_stds: List[float] = field(default_factory=list)
    grad_maxs: List[float] = field(default_factory=list)

    # Weight tracking
    weight_norms: List[float] = field(default_factory=list)

    # Activation tracking
    activation_norms: List[float] = field(default_factory=list)
    activation_means: List[float] = field(default_factory=list)

    # Issue counters
    zero_count: int = 0
    nan_count: int = 0
    inf_count: int = 0

    # =========================================================================
    # Computed Properties - Gradient Statistics
    # =========================================================================

    @property
    def mean_pressure(self) -> float:
        """
        Mean gradient magnitude (average L2 norm).

        Low values indicate potential vanishing gradients.
        High values indicate strong gradient signals.
        """
        valid = [g for g in self.grad_norms if np.isfinite(g)]
        return float(np.mean(valid)) if valid else 0.0

    @property
    def max_pressure(self) -> float:
        """
        Maximum gradient magnitude (peak L2 norm).

        Very high values indicate exploding gradients.
        """
        valid = [g for g in self.grad_norms if np.isfinite(g)]
        return float(np.max(valid)) if valid else 0.0

    @property
    def min_pressure(self) -> float:
        """Minimum gradient magnitude across all steps."""
        valid = [g for g in self.grad_norms if np.isfinite(g)]
        return float(np.min(valid)) if valid else 0.0

    @property
    def turbulence(self) -> float:
        """
        Gradient temporal variance (standard deviation of magnitudes over time).

        High variance relative to mean magnitude indicates training instability
        and inconsistent gradient signals across iterations.
        """
        valid = [g for g in self.grad_norms if np.isfinite(g)]
        return float(np.std(valid)) if valid else 0.0

    @property
    def flow_velocity(self) -> float:
        """
        Gradient temporal change rate (mean absolute change between steps).

        Measures how quickly gradient patterns evolve during training.
        Very high values can indicate oscillating or unstable gradients.
        """
        valid = [g for g in self.grad_norms if np.isfinite(g)]
        if len(valid) < 2:
            return 0.0
        diffs = np.diff(valid)
        return float(np.mean(np.abs(diffs)))

    @property
    def pressure_range(self) -> float:
        """Range between maximum and minimum gradient magnitudes."""
        return self.max_pressure - self.min_pressure

    @property
    def coefficient_of_variation(self) -> float:
        """Temporal variance normalized by mean magnitude (CV = std/mean)."""
        if self.mean_pressure > 0:
            return self.turbulence / self.mean_pressure
        return 0.0

    @property
    def signal_to_noise(self) -> float:
        """
        Signal-to-noise ratio: mean magnitude divided by variance.

        High values indicate stable, consistent gradients.
        Low values indicate noisy, unstable gradients.
        """
        if self.turbulence > 0:
            return self.mean_pressure / self.turbulence
        return float('inf') if self.mean_pressure > 0 else 0.0

    @property
    def zero_ratio(self) -> float:
        """Fraction of steps with zero gradients."""
        total = len(self.grad_norms)
        return self.zero_count / total if total > 0 else 0.0

    @property
    def numerical_issues_ratio(self) -> float:
        """Fraction of steps with NaN or Inf gradients."""
        total = len(self.grad_norms)
        if total == 0:
            return 0.0
        return (self.nan_count + self.inf_count) / total

    # =========================================================================
    # Weight and Activation Analysis
    # =========================================================================

    @property
    def weight_drift(self) -> float:
        """How much weights have changed over the simulation."""
        if len(self.weight_norms) < 2:
            return 0.0
        return abs(self.weight_norms[-1] - self.weight_norms[0])

    @property
    def weight_stability(self) -> float:
        """Stability of weight norms (1 - normalized std)."""
        if len(self.weight_norms) < 2:
            return 1.0
        mean = np.mean(self.weight_norms)
        if mean == 0:
            return 1.0
        std = np.std(self.weight_norms)
        return max(0.0, 1.0 - std / mean)

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "name": self.name,
            "layer_type": self.layer_type,
            "param_count": self.param_count,
            "mean_pressure": self.mean_pressure,
            "max_pressure": self.max_pressure,
            "min_pressure": self.min_pressure,
            "turbulence": self.turbulence,
            "flow_velocity": self.flow_velocity,
            "coefficient_of_variation": self.coefficient_of_variation,
            "zero_ratio": self.zero_ratio,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "num_samples": len(self.grad_norms),
        }


@dataclass
class LayerHealth:
    """
    Health assessment for a single layer.

    Computed from FlowMetrics using configurable thresholds.
    Provides both a numeric score (0-100) and categorical status.
    """

    name: str
    layer_type: str
    score: float  # 0-100
    status: HealthStatus
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @classmethod
    def from_metrics(cls, metrics: FlowMetrics, config: Any) -> "LayerHealth":
        """
        Compute health from metrics using config thresholds.

        The health score is computed by starting at 100 and applying
        penalties for various pathological conditions.
        """
        score = 100.0
        issues = []
        recommendations = []

        # Penalty for vanishing gradients
        if metrics.mean_pressure < config.vanishing_threshold:
            if metrics.mean_pressure < 1e-8:
                score -= 50
                issues.append("Severely vanishing gradients")
                recommendations.append("Add skip connections or use residual architecture")
            elif metrics.mean_pressure < 1e-6:
                score -= 30
                issues.append("Vanishing gradients detected")
                recommendations.append("Consider layer normalization or different activation")
            else:
                score -= 15
                issues.append("Low gradient magnitude")

        # Penalty for exploding gradients
        if metrics.max_pressure > config.exploding_threshold:
            if metrics.max_pressure > 1000:
                score -= 50
                issues.append("Severely exploding gradients")
                recommendations.append("Reduce learning rate and add gradient clipping")
            elif metrics.max_pressure > 100:
                score -= 35
                issues.append("Exploding gradients detected")
                recommendations.append("Apply gradient clipping (max_grad_norm)")
            else:
                score -= 20
                issues.append("High gradient magnitude")
                recommendations.append("Consider gradient scaling")

        # Penalty for turbulence
        cv = metrics.coefficient_of_variation
        if cv > config.turbulence_threshold:
            if cv > 10:
                score -= 25
                issues.append("Extremely unstable gradients")
                recommendations.append("Increase batch size or add normalization")
            elif cv > 5:
                score -= 15
                issues.append("Unstable gradient flow")
            else:
                score -= 10
                issues.append("Moderate gradient instability")

        # Penalty for numerical issues
        if metrics.nan_count > 0:
            score -= 40
            issues.append(f"NaN gradients detected ({metrics.nan_count} occurrences)")
            recommendations.append("Check for division by zero or log of zero")

        if metrics.inf_count > 0:
            score -= 35
            issues.append(f"Inf gradients detected ({metrics.inf_count} occurrences)")
            recommendations.append("Add value clamping before problematic operations")

        # Penalty for dead layers
        if metrics.zero_ratio > config.dead_layer_threshold:
            score -= 30
            issues.append(f"Layer appears dead ({metrics.zero_ratio:.1%} zero gradients)")
            recommendations.append("Verify layer is connected to loss computation")

        # Clamp score
        score = max(0.0, min(100.0, score))

        # Determine status
        if score >= 90:
            status = HealthStatus.OPTIMAL
        elif score >= 75:
            status = HealthStatus.HEALTHY
        elif score >= 50:
            status = HealthStatus.MARGINAL
        elif score >= 25:
            status = HealthStatus.UNHEALTHY
        else:
            status = HealthStatus.CRITICAL

        return cls(
            name=metrics.name,
            layer_type=metrics.layer_type,
            score=score,
            status=status,
            issues=issues,
            recommendations=recommendations
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export health assessment as dictionary."""
        return {
            "name": self.name,
            "layer_type": self.layer_type,
            "score": self.score,
            "status": self.status.value,
            "issues": self.issues,
            "recommendations": self.recommendations,
        }
