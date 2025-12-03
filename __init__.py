# -*- coding: utf-8 -*-
"""
Gradient Analysis Toolkit
==========================

Advanced gradient propagation analysis for neural network debugging and optimization.

This toolkit provides comprehensive gradient analysis through neural networks,
enabling precise diagnosis of training pathologies:

- **Magnitude** (gradient L2 norm): Strength of gradient signals at each layer
- **Temporal Variance**: Gradient stability and consistency over time
- **Change Rate**: How quickly gradient patterns evolve during training
- **Divergence/Curl**: Vector field properties showing expansion, contraction, and rotation
- **Vanishing Gradients**: Weak or zero gradients blocking learning
- **Exploding Gradients**: Excessive gradient magnitudes causing instability

Quick Start:
    >>> from gradient_flow import GradientFlowAnalyzer
    >>>
    >>> analyzer = GradientFlowAnalyzer(model)
    >>> issues = analyzer.analyze(input_fn, loss_fn, steps=20)
    >>>
    >>> analyzer.print_summary(issues)
    >>> for issue in issues:
    >>>     print(issue)

The GradientFlowAnalyzer automatically runs fused analysis combining:
- Scalar gradient statistics (magnitude, variance, temporal stability)
- Vector field analysis (divergence, curl, flow coherence)
- Optional Jacobian spectral analysis

Each detected issue includes:
- Severity level (CRITICAL, HIGH, MEDIUM, LOW, INFO)
- Issue type (VANISHING, EXPLODING, DEAD, UNSTABLE, NUMERICAL, BOTTLENECK)
- Clear description and recommended fixes

For detailed documentation, visit: https://gradient-flow.dev
"""

__version__ = "1.0.0"
__author__ = "Gradient Flow Team"

from .core.engine import FlowAnalyzer, GradientScale, gradient_scale, AnalysisConfig
from .core.metrics import FlowMetrics, LayerHealth
from .core.hooks import HookManager, BackwardHook, ForwardHook, VectorBackwardHook, AdaptiveSampler
from .core.fluid_dynamics import (
    GradientField,
    VectorMetrics,
    FluidOperators,
    StreamlineTracer,
    PressureVelocityCoupling
)
from .analyzers import (
    GradientFlowAnalyzer,
    GradientIssue,
    IssueSeverity,
    IssueType,
    RecurrentAnalyzer,
    ReservoirDynamics,
    RLPolicyAnalyzer,
    PolicyDistributionStats,
    TransformerAnalyzer,
    AttentionStats,
)

__all__ = [
    # Core
    "FlowAnalyzer",
    "GradientScale",
    "gradient_scale",
    "AnalysisConfig",
    "FlowMetrics",
    "LayerHealth",
    "HookManager",
    "BackwardHook",
    "ForwardHook",
    # Fluid dynamics
    "VectorBackwardHook",
    "AdaptiveSampler",
    "GradientField",
    "VectorMetrics",
    "FluidOperators",
    "StreamlineTracer",
    "PressureVelocityCoupling",
    # Analyzers
    "GradientFlowAnalyzer",
    "GradientIssue",
    "IssueSeverity",
    "IssueType",
    "RecurrentAnalyzer",
    "ReservoirDynamics",
    "RLPolicyAnalyzer",
    "PolicyDistributionStats",
    "TransformerAnalyzer",
    "AttentionStats",
]
