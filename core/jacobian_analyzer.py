# -*- coding: utf-8 -*-
"""
Jacobian-Based Flow Analysis
=============================

This module uses PyTorch's automatic differentiation to compute Jacobians
and analyze how gradient vector fields transform through each layer.

Key Concepts:
    - Jacobian Matrix J: ∂y/∂x for layer transformation y = f(x)
    - Singular Value Decomposition: J = UΣV^T
    - Eigenanalysis: reveals principal flow directions
    - Condition Number: measures numerical stability
    - Gradient Transformation: ∇L/∂x = J^T · ∇L/∂y

The Jacobian reveals how each layer transforms the gradient flow field,
enabling precise analysis of gradient dynamics through the network.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import warnings


# =============================================================================
# Jacobian Computation Utilities
# =============================================================================

class JacobianComputer:
    """
    Computes Jacobians for neural network layers using PyTorch's autodiff.

    PyTorch provides several functions for Jacobian computation:
    1. torch.autograd.functional.jacobian() - Full Jacobian matrix
    2. torch.autograd.functional.jvp() - Jacobian-vector product (forward-mode)
    3. torch.autograd.functional.vjp() - Vector-Jacobian product (backward-mode)

    For gradient flow analysis, we primarily use the Jacobian to understand
    how gradients transform: grad_x = J^T @ grad_y
    """

    @staticmethod
    def compute_full_jacobian(
        func: Callable[[torch.Tensor], torch.Tensor],
        input_tensor: torch.Tensor,
        vectorize: bool = False
    ) -> torch.Tensor:
        """
        Compute the full Jacobian matrix J = ∂y/∂x.

        Args:
            func: Function y = f(x) to analyze
            input_tensor: Input tensor x
            vectorize: Use vectorized computation (faster but more memory)

        Returns:
            Jacobian tensor of shape (output_size, input_size)

        Example:
            >>> layer = nn.Linear(10, 5)
            >>> x = torch.randn(10, requires_grad=True)
            >>> J = JacobianComputer.compute_full_jacobian(layer, x)
            >>> print(J.shape)  # torch.Size([5, 10])
        """
        try:
            if vectorize:
                # Faster but uses more memory
                jacobian = torch.autograd.functional.jacobian(
                    func, input_tensor, vectorize=True
                )
            else:
                # Slower but memory efficient
                jacobian = torch.autograd.functional.jacobian(
                    func, input_tensor, vectorize=False
                )

            # Flatten to 2D matrix if needed
            if jacobian.dim() > 2:
                output_size = jacobian.shape[0]
                input_size = input_tensor.numel()
                jacobian = jacobian.reshape(output_size, input_size)

            return jacobian

        except RuntimeError as e:
            warnings.warn(f"Jacobian computation failed: {e}")
            return None

    @staticmethod
    def compute_jacobian_vector_product(
        func: Callable[[torch.Tensor], torch.Tensor],
        input_tensor: torch.Tensor,
        vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jacobian-vector product: J @ v

        This is forward-mode autodiff and is efficient when output_dim < input_dim.

        Args:
            func: Function y = f(x)
            input_tensor: Input x
            vector: Vector v to multiply with J

        Returns:
            Result of J @ v
        """
        return torch.autograd.functional.jvp(func, input_tensor, vector)[1]

    @staticmethod
    def compute_vector_jacobian_product(
        func: Callable[[torch.Tensor], torch.Tensor],
        input_tensor: torch.Tensor,
        vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute vector-Jacobian product: v^T @ J

        This is backward-mode autodiff (standard backprop) and is efficient
        when output_dim > input_dim.

        Args:
            func: Function y = f(x)
            input_tensor: Input x
            vector: Vector v to left-multiply with J

        Returns:
            Result of v^T @ J
        """
        return torch.autograd.functional.vjp(func, input_tensor, vector)[1]

    @staticmethod
    def compute_gradient_transformation(
        func: Callable[[torch.Tensor], torch.Tensor],
        input_tensor: torch.Tensor,
        output_gradient: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute how output gradient transforms to input gradient: grad_x = J^T @ grad_y

        This is exactly what happens during backpropagation.

        Args:
            func: Layer transformation y = f(x)
            input_tensor: Input x
            output_gradient: Gradient w.r.t. output ∇L/∂y

        Returns:
            Gradient w.r.t. input ∇L/∂x = J^T @ ∇L/∂y
        """
        return JacobianComputer.compute_vector_jacobian_product(
            func, input_tensor, output_gradient
        )


# =============================================================================
# Jacobian Analysis Data Structures
# =============================================================================

@dataclass
class JacobianMetrics:
    """
    Metrics derived from Jacobian analysis of a layer.

    Attributes:
        layer_name: Name of the layer
        jacobian_norm: Frobenius norm of Jacobian (overall magnitude)
        singular_values: Singular values from SVD (sorted descending)
        condition_number: κ(J) = σ_max / σ_min (numerical stability)
        rank: Effective rank of Jacobian
        spectral_radius: Largest singular value (max amplification)
        spectral_gap: σ_max - σ_min (spread of singular values)
        isotropy: How uniform singular values are (1 = isotropic, 0 = anisotropic)
        gradient_amplification: Expected gradient amplification through layer
        flow_distortion: Measure of how much flow is distorted
        principal_directions: Top-k principal flow directions (from SVD)
    """

    layer_name: str
    layer_type: str

    # Jacobian properties
    jacobian_norm: float = 0.0
    singular_values: List[float] = field(default_factory=list)
    condition_number: float = 0.0
    rank: int = 0
    spectral_radius: float = 0.0
    spectral_gap: float = 0.0

    # Flow properties
    isotropy: float = 0.0
    gradient_amplification: float = 0.0
    flow_distortion: float = 0.0

    # Directional information
    principal_directions: List[torch.Tensor] = field(default_factory=list)
    max_amplification_direction: Optional[torch.Tensor] = None
    min_amplification_direction: Optional[torch.Tensor] = None

    @property
    def is_well_conditioned(self) -> bool:
        """Check if Jacobian is well-conditioned (κ < 1000)."""
        return self.condition_number < 1000.0

    @property
    def is_nearly_isotropic(self) -> bool:
        """Check if transformation is nearly isotropic (uniform in all directions)."""
        return self.isotropy > 0.8

    @property
    def is_gradient_stable(self) -> bool:
        """Check if gradient flow through layer is stable."""
        return 0.5 < self.gradient_amplification < 2.0

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "layer_name": self.layer_name,
            "layer_type": self.layer_type,
            "jacobian_norm": self.jacobian_norm,
            "condition_number": self.condition_number,
            "spectral_radius": self.spectral_radius,
            "spectral_gap": self.spectral_gap,
            "rank": self.rank,
            "isotropy": self.isotropy,
            "gradient_amplification": self.gradient_amplification,
            "flow_distortion": self.flow_distortion,
            "is_well_conditioned": self.is_well_conditioned,
            "is_nearly_isotropic": self.is_nearly_isotropic,
            "is_gradient_stable": self.is_gradient_stable,
        }


# =============================================================================
# Jacobian Analyzer
# =============================================================================

class JacobianAnalyzer:
    """
    Analyzes gradient flow using Jacobian matrices.

    This analyzer computes the Jacobian J = ∂y/∂x for each layer and
    performs spectral analysis to understand how gradients transform.

    Key insights from Jacobian analysis:
    1. Singular values reveal amplification/attenuation factors
    2. Condition number indicates numerical stability
    3. Principal directions (from SVD) show flow structure
    4. Isotropy measures uniformity of transformation

    Example:
        >>> analyzer = JacobianAnalyzer()
        >>> metrics = analyzer.analyze_layer(
        ...     layer=nn.Linear(128, 64),
        ...     sample_input=torch.randn(128)
        ... )
        >>> print(f"Condition number: {metrics.condition_number:.2f}")
        >>> print(f"Gradient amplification: {metrics.gradient_amplification:.2f}")
    """

    def __init__(
        self,
        compute_full_jacobian: bool = True,
        max_singular_values: int = 10,
        use_vectorized: bool = False
    ):
        """
        Initialize Jacobian analyzer.

        Args:
            compute_full_jacobian: Compute full Jacobian (memory intensive)
            max_singular_values: Number of singular values to keep
            use_vectorized: Use vectorized Jacobian computation (faster but more memory)
        """
        self.compute_full_jacobian = compute_full_jacobian
        self.max_singular_values = max_singular_values
        self.use_vectorized = use_vectorized

    def analyze_layer(
        self,
        layer: nn.Module,
        sample_input: torch.Tensor,
        layer_name: str = ""
    ) -> JacobianMetrics:
        """
        Analyze a single layer using its Jacobian.

        Args:
            layer: Neural network layer to analyze
            sample_input: Sample input tensor
            layer_name: Name of the layer

        Returns:
            JacobianMetrics with analysis results
        """
        layer_name = layer_name or layer.__class__.__name__
        layer_type = layer.__class__.__name__

        # Ensure input requires grad
        if not sample_input.requires_grad:
            sample_input = sample_input.detach().clone().requires_grad_(True)

        # Flatten input if needed
        original_shape = sample_input.shape
        if sample_input.dim() > 1:
            sample_input_flat = sample_input.flatten()
        else:
            sample_input_flat = sample_input

        # Define layer function
        def layer_func(x):
            x_shaped = x.reshape(original_shape)
            output = layer(x_shaped)
            return output.flatten()

        try:
            # Compute Jacobian
            jacobian = JacobianComputer.compute_full_jacobian(
                layer_func,
                sample_input_flat,
                vectorize=self.use_vectorized
            )

            if jacobian is None:
                return self._create_empty_metrics(layer_name, layer_type)

            # Perform SVD for spectral analysis
            metrics = self._analyze_jacobian_spectrum(
                jacobian, layer_name, layer_type
            )

            return metrics

        except Exception as e:
            warnings.warn(f"Failed to analyze layer {layer_name}: {e}")
            return self._create_empty_metrics(layer_name, layer_type)

    def _analyze_jacobian_spectrum(
        self,
        jacobian: torch.Tensor,
        layer_name: str,
        layer_type: str
    ) -> JacobianMetrics:
        """
        Perform spectral analysis on Jacobian matrix.

        Uses SVD: J = UΣV^T to extract singular values and principal directions.
        """
        # Compute SVD
        try:
            U, S, Vt = torch.linalg.svd(jacobian, full_matrices=False)
        except RuntimeError:
            # Fallback for numerical issues
            return self._create_empty_metrics(layer_name, layer_type)

        # Extract singular values
        singular_values = S.detach().cpu().numpy().tolist()

        # Keep only top-k
        if len(singular_values) > self.max_singular_values:
            singular_values = singular_values[:self.max_singular_values]

        # Compute metrics
        jacobian_norm = torch.norm(jacobian, p='fro').item()
        spectral_radius = S[0].item() if len(S) > 0 else 0.0
        min_singular = S[-1].item() if len(S) > 0 else 1e-10

        condition_number = spectral_radius / (min_singular + 1e-10)
        spectral_gap = spectral_radius - min_singular

        # Effective rank (number of significant singular values)
        threshold = spectral_radius * 1e-6
        rank = torch.sum(S > threshold).item()

        # Isotropy: how uniform are the singular values?
        # 1 = perfectly isotropic, 0 = highly anisotropic
        if len(S) > 1:
            s_normalized = S / (spectral_radius + 1e-10)
            isotropy = 1.0 - float(torch.std(s_normalized).item())
            isotropy = max(0.0, min(1.0, isotropy))
        else:
            isotropy = 1.0

        # Gradient amplification: expected norm amplification
        # E[||J^T v||] for random unit vector v
        gradient_amplification = float(torch.mean(S).item())

        # Flow distortion: how much does the transformation distort the flow?
        # Measured by ratio of max to min singular value
        flow_distortion = condition_number

        # Principal directions (from V^T)
        principal_directions = []
        for i in range(min(3, Vt.shape[0])):
            principal_directions.append(Vt[i, :].detach().cpu())

        # Max/min amplification directions
        max_amp_dir = Vt[0, :].detach().cpu() if Vt.shape[0] > 0 else None
        min_amp_dir = Vt[-1, :].detach().cpu() if Vt.shape[0] > 0 else None

        return JacobianMetrics(
            layer_name=layer_name,
            layer_type=layer_type,
            jacobian_norm=jacobian_norm,
            singular_values=singular_values,
            condition_number=condition_number,
            rank=rank,
            spectral_radius=spectral_radius,
            spectral_gap=spectral_gap,
            isotropy=isotropy,
            gradient_amplification=gradient_amplification,
            flow_distortion=flow_distortion,
            principal_directions=principal_directions,
            max_amplification_direction=max_amp_dir,
            min_amplification_direction=min_amp_dir,
        )

    def _create_empty_metrics(self, layer_name: str, layer_type: str) -> JacobianMetrics:
        """Create empty metrics for failed analysis."""
        return JacobianMetrics(
            layer_name=layer_name,
            layer_type=layer_type
        )

    def analyze_network(
        self,
        model: nn.Module,
        sample_input: torch.Tensor
    ) -> Dict[str, JacobianMetrics]:
        """
        Analyze all layers in a network.

        Args:
            model: Neural network to analyze
            sample_input: Sample input tensor

        Returns:
            Dictionary mapping layer names to JacobianMetrics
        """
        metrics = {}

        # Track intermediate outputs
        intermediate_outputs = {}

        def create_forward_hook(name):
            def hook(module, input, output):
                intermediate_outputs[name] = output.detach().clone()
            return hook

        # Attach hooks
        handles = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0 and name:  # Leaf modules
                handle = module.register_forward_hook(create_forward_hook(name))
                handles.append(handle)

        # Forward pass to get intermediate outputs
        with torch.no_grad():
            model(sample_input)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Analyze each layer
        current_input = sample_input
        for name, module in model.named_modules():
            if len(list(module.children())) == 0 and name:
                print(f"Analyzing layer: {name}")

                # Analyze this layer
                layer_metrics = self.analyze_layer(module, current_input, name)
                metrics[name] = layer_metrics

                # Update input for next layer
                if name in intermediate_outputs:
                    current_input = intermediate_outputs[name]

        return metrics


# =============================================================================
# Network-Wide Jacobian Analysis
# =============================================================================

def compute_network_jacobian_chain(
    metrics_dict: Dict[str, JacobianMetrics]
) -> Dict[str, float]:
    """
    Analyze the composition of Jacobians through the network.

    When gradients flow backward through multiple layers, the total
    transformation is J_total = J_n @ J_{n-1} @ ... @ J_1.

    This function estimates properties of this composed transformation.

    Args:
        metrics_dict: Dictionary of JacobianMetrics per layer

    Returns:
        Dictionary with network-wide metrics
    """
    layer_names = list(metrics_dict.keys())

    if not layer_names:
        return {}

    # Product of gradient amplifications
    total_amplification = 1.0
    for metrics in metrics_dict.values():
        total_amplification *= metrics.gradient_amplification

    # Product of condition numbers (upper bound)
    max_condition_number = 1.0
    for metrics in metrics_dict.values():
        max_condition_number *= metrics.condition_number

    # Average isotropy
    avg_isotropy = np.mean([m.isotropy for m in metrics_dict.values()])

    # Identify problematic layers
    ill_conditioned_layers = [
        name for name, m in metrics_dict.items()
        if not m.is_well_conditioned
    ]

    anisotropic_layers = [
        name for name, m in metrics_dict.items()
        if not m.is_nearly_isotropic
    ]

    unstable_layers = [
        name for name, m in metrics_dict.items()
        if not m.is_gradient_stable
    ]

    return {
        "total_gradient_amplification": total_amplification,
        "max_condition_number": max_condition_number,
        "average_isotropy": avg_isotropy,
        "ill_conditioned_layers": ill_conditioned_layers,
        "anisotropic_layers": anisotropic_layers,
        "unstable_gradient_layers": unstable_layers,
        "num_layers": len(metrics_dict),
    }
