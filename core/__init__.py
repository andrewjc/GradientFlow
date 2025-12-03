"""Core gradient flow analysis components."""

from .engine import FlowAnalyzer, GradientScale, gradient_scale, AnalysisConfig
from .metrics import FlowMetrics, LayerHealth
from .hooks import HookManager, BackwardHook, ForwardHook, VectorBackwardHook, AdaptiveSampler
from .fluid_dynamics import (
    GradientField,
    VectorMetrics,
    FluidOperators,
    StreamlineTracer,
    PressureVelocityCoupling
)
from .jacobian_analyzer import (
    JacobianComputer,
    JacobianAnalyzer,
    JacobianMetrics,
    compute_network_jacobian_chain
)

__all__ = [
    # Basic components
    "FlowAnalyzer",
    "GradientScale",
    "gradient_scale",
    "AnalysisConfig",
    "FlowMetrics",
    "LayerHealth",
    "HookManager",
    "BackwardHook",
    "ForwardHook",
    # Fluid dynamics components
    "VectorBackwardHook",
    "AdaptiveSampler",
    "GradientField",
    "VectorMetrics",
    "FluidOperators",
    "StreamlineTracer",
    "PressureVelocityCoupling",
    # Jacobian analysis components
    "JacobianComputer",
    "JacobianAnalyzer",
    "JacobianMetrics",
    "compute_network_jacobian_chain",
]
