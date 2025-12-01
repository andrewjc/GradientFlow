# -*- coding: utf-8 -*-
"""
Gradient Hydrodynamics Toolkit
==============================

A revolutionary approach to neural network debugging through fluid dynamics metaphors.

This toolkit treats gradient flow through neural networks as fluid flow through pipes,
enabling intuitive diagnosis of training pathologies:

- **Pressure** (gradient magnitude): How much "force" is flowing through each layer
- **Turbulence** (gradient variance): Flow stability over time
- **Velocity** (gradient change rate): How quickly flow patterns change
- **Blockages** (vanishing gradients): Clogged pipes stopping flow
- **Bursts** (exploding gradients): Over-pressurized pipes about to fail

Quick Start:
    >>> from gradient_flow import FlowAnalyzer
    >>> analyzer = FlowAnalyzer(model)
    >>> report = analyzer.analyze(sample_input, steps=20)
    >>> report.print_summary()
    >>> report.save_html("flow_report.html")

For detailed documentation, visit: https://gradient-flow.dev
"""

__version__ = "1.0.0"
__author__ = "Gradient Flow Team"

from .core.engine import FlowAnalyzer, GradientScale, gradient_scale, AnalysisConfig
from .core.metrics import FlowMetrics, LayerHealth
from .core.hooks import HookManager, BackwardHook, ForwardHook
from .analyzers import (
    RecurrentAnalyzer,
    RLPolicyAnalyzer,
    TransformerAnalyzer,
    TreeGPTAnalyzer
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
    # Analyzers
    "RecurrentAnalyzer",
    "RLPolicyAnalyzer",
    "TransformerAnalyzer",
    "TreeGPTAnalyzer",
]
