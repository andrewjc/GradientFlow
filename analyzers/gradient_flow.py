"""
    High-level gradient analysis interface.

    Usage:
        analyzer = GradientFlowAnalyzer(model)
        issues = analyzer.analyze(input_fn, loss_fn, steps=20)

        # Print issues
        for issue in issues:
            print(issue)

        # Get specific severity
        critical = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
"""

import torch
import torch.nn as nn
from typing import List, Callable, Optional, Any, Dict
from dataclasses import dataclass
from enum import Enum

from ..core.engine import FlowAnalyzer as LowLevelAnalyzer
from ..core.hooks import HookManager, VectorBackwardHook
from ..core.fluid_dynamics import VectorMetrics


class IssueSeverity(Enum):
    """Severity levels for gradient issues."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class IssueType(Enum):
    """Types of gradient issues."""
    VANISHING = "VANISHING"
    EXPLODING = "EXPLODING"
    DEAD = "DEAD"
    UNSTABLE = "UNSTABLE"
    NUMERICAL = "NUMERICAL"
    BOTTLENECK = "BOTTLENECK"
    SATURATION = "SATURATION"


@dataclass
class GradientIssue:
    """
    Represents a detected gradient issue.

    Attributes:
        layer: Name of the affected layer
        layer_type: Type of layer (Linear, Conv2d, etc.)
        issue_type: Type of issue (VANISHING, EXPLODING, etc.)
        severity: How serious the issue is
        description: Human-readable description
        magnitude: Current gradient magnitude
        recommended_actions: List of suggested fixes
        metrics: Detailed metrics for debugging
    """
    layer: str
    layer_type: str
    issue_type: IssueType
    severity: IssueSeverity
    description: str
    magnitude: float
    recommended_actions: List[str]
    metrics: Dict[str, Any]

    def __str__(self) -> str:
        """String representation for easy printing."""
        actions = "\n    - ".join(self.recommended_actions)
        return (
            f"[{self.severity.value}] {self.issue_type.value} in {self.layer} ({self.layer_type})\n"
            f"  Description: {self.description}\n"
            f"  Magnitude: {self.magnitude:.2e}\n"
            f"  Recommended Actions:\n    - {actions}"
        )


class GradientFlowAnalyzer:
    def __init__(
        self,
        model: nn.Module,
        name: Optional[str] = None,
        enable_rnn_analyzer: bool = False,
        enable_circular_flow_analyser: bool = False,
        vanishing_threshold: float = 1e-6,
        exploding_threshold: float = 10.0,
        dead_threshold: float = 0.9,
        unstable_cv_threshold: float = 3.0
    ):
        """
        Initialize gradient flow analyzer.

        Args:
            model: PyTorch model to analyze
            name: Optional name for the model
            enable_rnn_analyzer: Enable only for RNN spectral analysis.
            enable_circular_flow_analyser: Enable only for detecting circular gradient flows.
            vanishing_threshold: Threshold for vanishing gradient detection
            exploding_threshold: Threshold for exploding gradient detection
            dead_threshold: Ratio of zero gradients to mark layer as dead
            unstable_cv_threshold: Coefficient of variation threshold for instability
        """
        self.model = model
        self.name = name or model.__class__.__name__
        self.enable_rnn_analyser = enable_rnn_analyzer
        self.enable_circular_flow_analyser = enable_circular_flow_analyser

        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.dead_threshold = dead_threshold
        self.unstable_cv_threshold = unstable_cv_threshold

        self._scalar_analyzer = LowLevelAnalyzer(model, name=name)
        self._hook_manager = HookManager()
        self._vector_metrics: Dict[str, VectorMetrics] = {}
        self._layer_types: Dict[str, str] = {}

    def analyze(
        self,
        input_fn: Callable[[], torch.Tensor],
        loss_fn: Callable[[Any], torch.Tensor],
        steps: int = 20,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> List[GradientIssue]:
        """
        Analyze gradient flow and return detected issues.

        Args:
            input_fn: Function that returns a batch of input tensors
            loss_fn: Function that takes model output and returns scalar loss
            steps: Number of simulation steps
            optimizer: Optional optimizer (creates dummy if None)

        Returns:
            List of detected gradient issues, sorted by severity
        """

        scalar_metrics = self._scalar_analyzer.analyze(
            input_fn=input_fn,
            loss_fn=loss_fn,
            steps=steps,
            optimizer=optimizer
        )


        if self.enable_circular_flow_analyser:
            self._run_vector_analysis(input_fn, loss_fn, steps, optimizer)

        issues = self._detect_issues(scalar_metrics)

        severity_order = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.HIGH: 1,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 3,
            IssueSeverity.INFO: 4
        }
        issues.sort(key=lambda x: severity_order[x.severity])

        return issues

    def _run_vector_analysis(
        self,
        input_fn: Callable[[], torch.Tensor],
        loss_fn: Callable[[Any], torch.Tensor],
        steps: int,
        optimizer: Optional[torch.optim.Optimizer]
    ):

        self._vector_metrics.clear()
        self._hook_manager.remove_all()


        for name, module in self.model.named_modules():
            self._layer_types[name] = type(module).__name__

        for name, module in self.model.named_modules():
            is_leaf = len(list(module.children())) == 0
            if is_leaf and name:
                hook = VectorBackwardHook.fluid_metrics_collector(
                    storage=self._vector_metrics,
                    name=name,
                    layer_type=type(module).__name__,
                    compute_divergence=True,
                    compute_curl=True
                )
                handle = module.register_full_backward_hook(hook)
                self._hook_manager.add_hook(name, handle, "backward")

        if optimizer is None:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        self.model.train()
        for step in range(steps):
            optimizer.zero_grad()
            inputs = input_fn()
            output = self.model(inputs)
            loss = loss_fn(output)
            loss.backward()
            optimizer.step()

        self._hook_manager.remove_all()

    def _detect_issues(self, scalar_metrics) -> List[GradientIssue]:

        issues = []

        for name, metrics in scalar_metrics.items():
            layer_type = metrics.layer_type

            # Get vector metrics if available
            vector_metrics = self._vector_metrics.get(name)

            # Check for vanishing gradients
            if 0 < metrics.mean_pressure < self.vanishing_threshold:
                severity = IssueSeverity.CRITICAL if metrics.mean_pressure < 1e-8 else IssueSeverity.HIGH

                actions = [
                    "Use residual/skip connections",
                    "Try different activation functions (ReLU to LeakyReLU)",
                    "Reduce network depth or increase layer width",
                    "Check weight initialization (use Xavier/He initialization)"
                ]

                description = (
                    f"Gradient magnitude ({metrics.mean_pressure:.2e}) is extremely small. "
                    f"Layer may not be learning effectively."
                )

                issues.append(GradientIssue(
                    layer=name,
                    layer_type=layer_type,
                    issue_type=IssueType.VANISHING,
                    severity=severity,
                    description=description,
                    magnitude=metrics.mean_pressure,
                    recommended_actions=actions,
                    metrics={
                        'mean_magnitude': metrics.mean_pressure,
                        'max_magnitude': metrics.max_pressure,
                        'variance': metrics.turbulence
                    }
                ))

            # Check for exploding gradients
            if metrics.max_pressure > self.exploding_threshold:
                severity = IssueSeverity.CRITICAL if metrics.max_pressure > 100 else IssueSeverity.HIGH

                actions = [
                    "Enable gradient clipping (torch.nn.utils.clip_grad_norm_)",
                    "Reduce learning rate",
                    "Use batch normalization or layer normalization",
                    "Check for numerical instability in loss function"
                ]

                description = (
                    f"Gradient magnitude ({metrics.max_pressure:.2f}) is very large. "
                    f"May cause training instability or NaN values."
                )

                issues.append(GradientIssue(
                    layer=name,
                    layer_type=layer_type,
                    issue_type=IssueType.EXPLODING,
                    severity=severity,
                    description=description,
                    magnitude=metrics.max_pressure,
                    recommended_actions=actions,
                    metrics={
                        'mean_magnitude': metrics.mean_pressure,
                        'max_magnitude': metrics.max_pressure,
                        'variance': metrics.turbulence
                    }
                ))

            # Check for dead layers
            if len(metrics.grad_norms) > 0:
                zero_ratio = metrics.zero_count / len(metrics.grad_norms)
                if zero_ratio > self.dead_threshold:
                    severity = IssueSeverity.HIGH if zero_ratio > 0.99 else IssueSeverity.MEDIUM

                    actions = [
                        "Check if layer is connected to loss",
                        "Verify weight initialization (all zeros?)",
                        "Inspect activation function (dead ReLU?)",
                        "Ensure layer receives non-zero inputs"
                    ]

                    description = (
                        f"{zero_ratio:.1%} of gradients are zero. "
                        f"Layer appears to be dead or disconnected."
                    )

                    issues.append(GradientIssue(
                        layer=name,
                        layer_type=layer_type,
                        issue_type=IssueType.DEAD,
                        severity=severity,
                        description=description,
                        magnitude=metrics.mean_pressure,
                        recommended_actions=actions,
                        metrics={
                            'zero_ratio': zero_ratio,
                            'mean_magnitude': metrics.mean_pressure
                        }
                    ))

            # Check for unstable gradients
            if metrics.mean_pressure > 0 and metrics.turbulence > 0:
                cv = metrics.turbulence / metrics.mean_pressure
                if cv > self.unstable_cv_threshold:
                    severity = IssueSeverity.HIGH if cv > 5.0 else IssueSeverity.MEDIUM

                    actions = [
                        "Reduce learning rate for stability",
                        "Add batch normalization or layer normalization",
                        "Increase batch size to reduce variance",
                        "Check for data preprocessing issues"
                    ]

                    description = (
                        f"Gradient variance is high (CV={cv:.2f}). "
                        f"Training may be unstable."
                    )

                    issues.append(GradientIssue(
                        layer=name,
                        layer_type=layer_type,
                        issue_type=IssueType.UNSTABLE,
                        severity=severity,
                        description=description,
                        magnitude=metrics.mean_pressure,
                        recommended_actions=actions,
                        metrics={
                            'coefficient_of_variation': cv,
                            'variance': metrics.turbulence,
                            'mean_magnitude': metrics.mean_pressure
                        }
                    ))

            # Check for numerical issues
            if metrics.nan_count > 0 or metrics.inf_count > 0:
                actions = [
                    "Check for division by zero in loss or model",
                    "Inspect for overflow in exponential/power operations",
                    "Reduce learning rate significantly",
                    "Add numerical stability epsilons where needed"
                ]

                description = (
                    f"Detected {metrics.nan_count} NaN and {metrics.inf_count} Inf values. "
                    f"Critical numerical instability."
                )

                issues.append(GradientIssue(
                    layer=name,
                    layer_type=layer_type,
                    issue_type=IssueType.NUMERICAL,
                    severity=IssueSeverity.CRITICAL,
                    description=description,
                    magnitude=metrics.mean_pressure,
                    recommended_actions=actions,
                    metrics={
                        'nan_count': metrics.nan_count,
                        'inf_count': metrics.inf_count
                    }
                ))

            # Add vector-based diagnostics if available
            if vector_metrics:
                # Check for severe bottlenecks (strong convergence)
                if vector_metrics.mean_divergence < -0.5:
                    actions = [
                        "Consider widening the layer",
                        "Add skip connections around bottleneck",
                        "Use gradient scaling for this layer"
                    ]

                    description = (
                        f"Strong gradient convergence detected (div={vector_metrics.mean_divergence:.3f}). "
                        f"May indicate a severe bottleneck."
                    )

                    issues.append(GradientIssue(
                        layer=name,
                        layer_type=layer_type,
                        issue_type=IssueType.BOTTLENECK,
                        severity=IssueSeverity.MEDIUM,
                        description=description,
                        magnitude=metrics.mean_pressure,
                        recommended_actions=actions,
                        metrics={
                            'divergence': vector_metrics.mean_divergence,
                            'curl': vector_metrics.mean_curl,
                            'coherence': vector_metrics.flow_coherence
                        }
                    ))

        return issues

    def get_healthy_layers(self, all_issues: List[GradientIssue]) -> List[str]:
        """Get list of layers with no issues."""
        problematic = {issue.layer for issue in all_issues}
        all_layers = set(self._scalar_analyzer.metrics.keys())
        return sorted(all_layers - problematic)

    def print_summary(self, issues: List[GradientIssue]):
        """Print a summary of detected issues."""
        if not issues:
            print("\n[OK] No gradient issues detected! Model looks healthy.")
            return

        print(f"\n[WARNING] Found {len(issues)} gradient issue(s):\n")

        by_severity = {}
        for issue in issues:
            if issue.severity not in by_severity:
                by_severity[issue.severity] = []
            by_severity[issue.severity].append(issue)

        for severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH, IssueSeverity.MEDIUM, IssueSeverity.LOW]:
            if severity in by_severity:
                print(f"{severity.value}: {len(by_severity[severity])} issue(s)")

        print("\nDetails:\n" + "="*80)
        for issue in issues:
            print(issue)
            print("="*80)
