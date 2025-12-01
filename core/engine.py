# -*- coding: utf-8 -*-
"""
Core Gradient Flow Analysis Engine
===================================

The heart of the Gradient Hydrodynamics toolkit. This module provides:

1. GradientScale: Custom autograd function for controlling gradient magnitude
2. FlowAnalyzer: Main analysis engine that tracks gradient flow through networks
3. Simulation runners for various network architectures

The key insight is treating neural network gradient flow as a fluid dynamics problem:
- Gradients are "pressure" flowing backward through the network
- Each layer is a "pipe segment" that can amplify, dampen, or block flow
- Training pathologies manifest as fluid dynamics problems

This metaphor enables intuitive diagnosis:
- Vanishing gradients = clogged pipes (no pressure reaches early layers)
- Exploding gradients = burst pipes (pressure too high, system unstable)
- Dead neurons = disconnected pipes (no flow at all)
- Unstable training = turbulent flow (pressure varies wildly)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager

from .metrics import FlowMetrics, LayerHealth
from .hooks import HookManager


# =============================================================================
# Gradient Scaling Utilities
# =============================================================================

class GradientScale(torch.autograd.Function):
    """
    Custom autograd function that scales gradients during backward pass.

    This is the key tool for controlling gradient flow through problematic layers.
    During forward pass, the tensor passes through unchanged. During backward,
    gradients are multiplied by a scale factor.

    Use cases:
    - Prevent gradient explosion in compression layers (scale < 1)
    - Amplify gradients in expansion layers (scale > 1)
    - Create "gradient barriers" to isolate network sections (scale ≈ 0)

    Example:
        >>> x = torch.randn(10, 256, requires_grad=True)
        >>> # Scale gradients by 0.1 during backward
        >>> y = GradientScale.apply(x, 0.1)
        >>> loss = y.sum()
        >>> loss.backward()
        >>> # x.grad is now 10x smaller than it would be without scaling
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        """Forward pass: return input unchanged, save scale for backward."""
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass: scale the gradient."""
        return grad_output * ctx.scale, None


def gradient_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Functional interface for gradient scaling.

    Args:
        x: Input tensor
        scale: Gradient scale factor (0.0 to inf, typically 0.001 to 10.0)

    Returns:
        Tensor identical to input, but with scaled gradients during backward

    Example:
        >>> # In a forward pass, scale gradients before a problematic layer
        >>> x = self.encoder(input)
        >>> x = gradient_scale(x, 0.01)  # Reduce gradient magnitude 100x
        >>> x = self.bottleneck(x)  # This layer won't explode now
    """
    return GradientScale.apply(x, scale)


# =============================================================================
# Flow Analyzer Engine
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for gradient flow analysis."""

    # Simulation parameters
    steps: int = 20
    batch_size: int = 4
    warmup_steps: int = 2

    # Thresholds for issue detection
    vanishing_threshold: float = 1e-6
    exploding_threshold: float = 10.0
    dead_layer_threshold: float = 0.9  # % of zero gradients
    turbulence_threshold: float = 3.0  # coefficient of variation

    # What to track
    track_weights: bool = True
    track_activations: bool = False
    track_temporal: bool = True

    # Device settings
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32


class FlowAnalyzer:
    """
    Main gradient flow analysis engine.

    The FlowAnalyzer attaches hooks to all layers in a neural network,
    runs a simulation by pumping data through, and collects comprehensive
    statistics about gradient flow at each layer.

    Basic Usage:
        >>> model = MyNeuralNetwork()
        >>> analyzer = FlowAnalyzer(model)
        >>>
        >>> # Run analysis with sample input
        >>> metrics = analyzer.analyze(
        ...     input_fn=lambda: torch.randn(4, 128),
        ...     loss_fn=lambda out: out.mean(),
        ...     steps=20
        ... )
        >>>
        >>> # Generate report
        >>> report = analyzer.generate_report(metrics)
        >>> report.print_summary()

    Advanced Usage (Custom Networks):
        >>> # For networks with complex forward signatures
        >>> def custom_forward(model, step):
        ...     obs = torch.randn(4, 128)
        ...     mem = torch.zeros(4, 16, 32)
        ...     return model.forward_with_memory(obs, mem)
        >>>
        >>> metrics = analyzer.analyze_custom(
        ...     forward_fn=custom_forward,
        ...     loss_fn=lambda out: out[0].mean(),
        ...     steps=20
        ... )

    Attributes:
        model: The PyTorch model being analyzed
        config: Analysis configuration
        hook_manager: Manages forward/backward hooks
        metrics: Collected flow metrics (populated after analyze())
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[AnalysisConfig] = None,
        name: Optional[str] = None
    ):
        """
        Initialize the flow analyzer.

        Args:
            model: PyTorch model to analyze
            config: Analysis configuration (uses defaults if None)
            name: Optional name for the model (for reports)
        """
        self.model = model
        self.config = config or AnalysisConfig()
        self.name = name or model.__class__.__name__
        self.hook_manager = HookManager()
        self.metrics: Dict[str, FlowMetrics] = {}
        self._is_attached = False

        # Infer device from model
        if self.config.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)

    def attach_hooks(self) -> None:
        """Attach gradient tracking hooks to all layers."""
        if self._is_attached:
            return

        self.metrics.clear()

        def make_backward_hook(name: str, layer_type: str, param_count: int):
            def hook(module, grad_input, grad_output):
                if name not in self.metrics:
                    self.metrics[name] = FlowMetrics(
                        name=name,
                        layer_type=layer_type,
                        param_count=param_count
                    )

                metric = self.metrics[name]

                # Track gradient from output
                if grad_output and grad_output[0] is not None:
                    grad = grad_output[0]

                    # Check for numerical issues
                    if torch.isnan(grad).any():
                        metric.nan_count += 1
                        metric.grad_norms.append(float('nan'))
                    elif torch.isinf(grad).any():
                        metric.inf_count += 1
                        metric.grad_norms.append(float('inf'))
                    else:
                        norm = grad.norm().item()
                        metric.grad_norms.append(norm)
                        if norm < 1e-12:
                            metric.zero_count += 1

                        # Track per-element statistics
                        metric.grad_means.append(grad.mean().item())
                        metric.grad_stds.append(grad.std().item())
                        metric.grad_maxs.append(grad.abs().max().item())
                else:
                    metric.zero_count += 1
                    metric.grad_norms.append(0.0)

            return hook

        def make_forward_hook(name: str):
            def hook(module, input, output):
                if name in self.metrics and self.config.track_activations:
                    metric = self.metrics[name]
                    if isinstance(output, torch.Tensor):
                        metric.activation_norms.append(output.norm().item())
                        metric.activation_means.append(output.mean().item())
            return hook

        # Attach to all leaf modules
        for name, module in self.model.named_modules():
            is_leaf = len(list(module.children())) == 0
            if is_leaf and name:
                param_count = sum(p.numel() for p in module.parameters(recurse=False))
                layer_type = type(module).__name__

                # Backward hook for gradient tracking
                handle = module.register_full_backward_hook(
                    make_backward_hook(name, layer_type, param_count)
                )
                self.hook_manager.add_hook(name, handle, "backward")

                # Forward hook for activation tracking
                if self.config.track_activations:
                    handle = module.register_forward_hook(make_forward_hook(name))
                    self.hook_manager.add_hook(name, handle, "forward")

        self._is_attached = True

    def detach_hooks(self) -> None:
        """Remove all hooks from the model."""
        self.hook_manager.remove_all()
        self._is_attached = False

    @contextmanager
    def tracking(self):
        """Context manager for temporary hook attachment."""
        self.attach_hooks()
        try:
            yield self
        finally:
            self.detach_hooks()

    def analyze(
        self,
        input_fn: Callable[[], torch.Tensor],
        loss_fn: Callable[[Any], torch.Tensor],
        steps: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, FlowMetrics]:
        """
        Run gradient flow analysis.

        Args:
            input_fn: Function that returns a batch of input tensors
            loss_fn: Function that takes model output and returns a scalar loss
            steps: Number of simulation steps (uses config default if None)
            optimizer: Optional optimizer for weight updates between steps
            progress_callback: Optional callback(step, total_steps) for progress

        Returns:
            Dictionary mapping layer names to FlowMetrics

        Example:
            >>> metrics = analyzer.analyze(
            ...     input_fn=lambda: torch.randn(4, 784),
            ...     loss_fn=lambda out: F.cross_entropy(out, torch.randint(0, 10, (4,))),
            ...     steps=20
            ... )
        """
        steps = steps or self.config.steps

        # Create dummy optimizer if none provided
        if optimizer is None:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        self.model.train()
        self.attach_hooks()

        try:
            for step in range(steps):
                if progress_callback:
                    progress_callback(step, steps)

                optimizer.zero_grad()

                # Forward pass
                inputs = input_fn()
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                output = self.model(inputs)

                # Compute loss
                loss = loss_fn(output)

                # Backward pass
                loss.backward()

                # Optional weight update
                optimizer.step()

                # Track weight norms if configured
                if self.config.track_weights:
                    self._capture_weight_norms()

        finally:
            self.detach_hooks()

        return self.metrics

    def analyze_custom(
        self,
        forward_fn: Callable[[nn.Module, int], Any],
        loss_fn: Callable[[Any], torch.Tensor],
        steps: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, FlowMetrics]:
        """
        Run analysis with custom forward function.

        Use this for models with complex forward signatures (e.g., RNNs,
        Transformers with memory, multi-input models).

        Args:
            forward_fn: Function(model, step) -> output
            loss_fn: Function(output) -> scalar loss
            steps: Number of simulation steps
            optimizer: Optional optimizer

        Returns:
            Dictionary mapping layer names to FlowMetrics
        """
        steps = steps or self.config.steps

        if optimizer is None:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        self.model.train()
        self.attach_hooks()

        try:
            for step in range(steps):
                optimizer.zero_grad()

                # Custom forward
                output = forward_fn(self.model, step)

                # Compute loss
                loss = loss_fn(output)

                # Backward
                loss.backward()
                optimizer.step()

                if self.config.track_weights:
                    self._capture_weight_norms()

        finally:
            self.detach_hooks()

        return self.metrics

    def _capture_weight_norms(self) -> None:
        """Capture current weight norms for tracked layers."""
        for name, module in self.model.named_modules():
            if name in self.metrics:
                weight_norm = sum(
                    p.norm().item()
                    for p in module.parameters(recurse=False)
                )
                self.metrics[name].weight_norms.append(weight_norm)

    def compute_health(self) -> Dict[str, LayerHealth]:
        """
        Compute health scores for all tracked layers.

        Returns:
            Dictionary mapping layer names to LayerHealth objects
        """
        health = {}
        for name, metric in self.metrics.items():
            health[name] = LayerHealth.from_metrics(metric, self.config)
        return health

    def find_issues(self) -> List[Dict[str, Any]]:
        """
        Identify all gradient flow issues.

        Returns:
            List of issue dictionaries with keys:
            - type: Issue type (VANISHING, EXPLODING, DEAD, UNSTABLE, NUMERICAL)
            - severity: CRITICAL, HIGH, MEDIUM, LOW
            - layer: Layer name
            - layer_type: Type of layer (Linear, Conv2d, etc.)
            - info: Human-readable description
        """
        issues = []

        for name, metric in self.metrics.items():
            # Check for vanishing gradients
            if metric.mean_pressure > 0 and metric.mean_pressure < self.config.vanishing_threshold:
                severity = "CRITICAL" if metric.mean_pressure < 1e-8 else "HIGH"
                issues.append({
                    "type": "VANISHING",
                    "severity": severity,
                    "layer": name,
                    "layer_type": metric.layer_type,
                    "info": f"Mean gradient norm: {metric.mean_pressure:.2e}"
                })

            # Check for exploding gradients
            if metric.max_pressure > self.config.exploding_threshold:
                severity = "CRITICAL" if metric.max_pressure > 100 else "HIGH"
                issues.append({
                    "type": "EXPLODING",
                    "severity": severity,
                    "layer": name,
                    "layer_type": metric.layer_type,
                    "info": f"Max gradient norm: {metric.max_pressure:.2f}"
                })

            # Check for dead layers
            if len(metric.grad_norms) > 0:
                zero_ratio = metric.zero_count / len(metric.grad_norms)
                if zero_ratio > self.config.dead_layer_threshold:
                    severity = "HIGH" if zero_ratio > 0.99 else "MEDIUM"
                    issues.append({
                        "type": "DEAD",
                        "severity": severity,
                        "layer": name,
                        "layer_type": metric.layer_type,
                        "info": f"Zero gradient ratio: {zero_ratio:.1%}"
                    })

            # Check for unstable layers
            if metric.mean_pressure > 0 and metric.turbulence > 0:
                cv = metric.turbulence / metric.mean_pressure
                if cv > self.config.turbulence_threshold:
                    severity = "HIGH" if cv > 5.0 else "MEDIUM"
                    issues.append({
                        "type": "UNSTABLE",
                        "severity": severity,
                        "layer": name,
                        "layer_type": metric.layer_type,
                        "info": f"Coefficient of variation: {cv:.2f}"
                    })

            # Check for numerical issues
            if metric.nan_count > 0 or metric.inf_count > 0:
                issues.append({
                    "type": "NUMERICAL",
                    "severity": "CRITICAL",
                    "layer": name,
                    "layer_type": metric.layer_type,
                    "info": f"NaN: {metric.nan_count}, Inf: {metric.inf_count}"
                })

        return issues

    def generate_report(self, metrics: Optional[Dict[str, FlowMetrics]] = None):
        """
        Generate a FlowReport from collected metrics.

        Args:
            metrics: Optional metrics dict (uses self.metrics if None)

        Returns:
            FlowReport object for visualization and export
        """
        from ..visualizers.report import FlowReport

        metrics = metrics or self.metrics
        issues = self.find_issues()
        health = self.compute_health()

        return FlowReport(
            model_name=self.name,
            metrics=metrics,
            issues=issues,
            health=health,
            config=self.config
        )
