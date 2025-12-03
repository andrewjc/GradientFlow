# -*- coding: utf-8 -*-
"""
Hook Management System
======================

Provides a clean interface for managing PyTorch hooks on neural networks.
Supports forward hooks, backward hooks, and full backward hooks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from .fluid_dynamics import (
    GradientField,
    VectorMetrics,
    FluidOperators,
    PressureVelocityCoupling
)


class HookType(Enum):
    """Types of PyTorch hooks."""
    FORWARD = "forward"
    BACKWARD = "backward"
    FORWARD_PRE = "forward_pre"


@dataclass
class HookInfo:
    """Information about a registered hook."""
    name: str
    handle: torch.utils.hooks.RemovableHandle
    hook_type: str
    module_type: str


class HookManager:
    """
    Manages PyTorch hooks on neural network modules.

    Provides a clean interface for:
    - Registering hooks on specific layers or all layers
    - Tracking registered hooks
    - Removing hooks cleanly
    - Preventing duplicate hook registration

    Example:
        >>> manager = HookManager()
        >>> for name, module in model.named_modules():
        ...     handle = module.register_backward_hook(my_hook)
        ...     manager.add_hook(name, handle, "backward")
        ...
        >>> # Later, remove all hooks
        >>> manager.remove_all()
    """

    def __init__(self):
        """Initialize the hook manager."""
        self.hooks: Dict[str, List[HookInfo]] = {}
        self._active = True

    def add_hook(
        self,
        name: str,
        handle: torch.utils.hooks.RemovableHandle,
        hook_type: str,
        module_type: str = ""
    ) -> None:
        """
        Register a hook handle for tracking.

        Args:
            name: Name of the module (from named_modules())
            handle: The RemovableHandle returned by register_*_hook
            hook_type: Type of hook ("forward", "backward", etc.)
            module_type: Optional type name of the module
        """
        if name not in self.hooks:
            self.hooks[name] = []

        self.hooks[name].append(HookInfo(
            name=name,
            handle=handle,
            hook_type=hook_type,
            module_type=module_type
        ))

    def remove_hook(self, name: str) -> None:
        """Remove all hooks for a specific module."""
        if name in self.hooks:
            for hook_info in self.hooks[name]:
                hook_info.handle.remove()
            del self.hooks[name]

    def remove_all(self) -> None:
        """Remove all registered hooks."""
        for name in list(self.hooks.keys()):
            self.remove_hook(name)
        self._active = False

    def get_hooks(self, name: str) -> List[HookInfo]:
        """Get all hooks for a specific module."""
        return self.hooks.get(name, [])

    def list_hooks(self) -> List[str]:
        """List all modules with registered hooks."""
        return list(self.hooks.keys())

    @property
    def count(self) -> int:
        """Total number of registered hooks."""
        return sum(len(hooks) for hooks in self.hooks.values())

    def __len__(self) -> int:
        return self.count

    def __contains__(self, name: str) -> bool:
        return name in self.hooks


# =============================================================================
# Pre-built Hook Functions
# =============================================================================

class BackwardHook:
    """
    Factory for creating backward hooks that collect gradient statistics.

    Example:
        >>> collector = {}
        >>> hook = BackwardHook.gradient_norm_collector(collector)
        >>> handle = module.register_full_backward_hook(hook)
    """

    @staticmethod
    def gradient_norm_collector(
        storage: Dict[str, List[float]],
        name: str
    ) -> Callable:
        """
        Create a hook that collects gradient norms.

        Args:
            storage: Dictionary to store results
            name: Key for this module in storage

        Returns:
            Hook function for register_full_backward_hook
        """
        if name not in storage:
            storage[name] = []

        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                norm = grad_output[0].norm().item()
                storage[name].append(norm)
            else:
                storage[name].append(0.0)

        return hook

    @staticmethod
    def gradient_stats_collector(
        storage: Dict[str, Dict[str, List[float]]],
        name: str
    ) -> Callable:
        """
        Create a hook that collects comprehensive gradient statistics.

        Args:
            storage: Dictionary to store results
            name: Key for this module in storage

        Returns:
            Hook function for register_full_backward_hook
        """
        if name not in storage:
            storage[name] = {
                "norms": [],
                "means": [],
                "stds": [],
                "maxs": [],
                "mins": []
            }

        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                grad = grad_output[0]
                storage[name]["norms"].append(grad.norm().item())
                storage[name]["means"].append(grad.mean().item())
                storage[name]["stds"].append(grad.std().item())
                storage[name]["maxs"].append(grad.abs().max().item())
                storage[name]["mins"].append(grad.abs().min().item())
            else:
                for key in storage[name]:
                    storage[name][key].append(0.0)

        return hook


class ForwardHook:
    """Factory for creating forward hooks that collect activation statistics."""

    @staticmethod
    def activation_norm_collector(
        storage: Dict[str, List[float]],
        name: str
    ) -> Callable:
        """
        Create a hook that collects activation norms.

        Args:
            storage: Dictionary to store results
            name: Key for this module in storage

        Returns:
            Hook function for register_forward_hook
        """
        if name not in storage:
            storage[name] = []

        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                storage[name].append(output.norm().item())
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    storage[name].append(output[0].norm().item())

        return hook

    @staticmethod
    def activation_stats_collector(
        storage: Dict[str, Dict[str, List[float]]],
        name: str
    ) -> Callable:
        """
        Create a hook that collects comprehensive activation statistics.

        Args:
            storage: Dictionary to store results
            name: Key for this module in storage

        Returns:
            Hook function for register_forward_hook
        """
        if name not in storage:
            storage[name] = {
                "norms": [],
                "means": [],
                "stds": [],
                "maxs": [],
                "sparsity": []
            }

        def hook(module, input, output):
            tensor = None
            if isinstance(output, torch.Tensor):
                tensor = output
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    tensor = output[0]

            if tensor is not None:
                storage[name]["norms"].append(tensor.norm().item())
                storage[name]["means"].append(tensor.mean().item())
                storage[name]["stds"].append(tensor.std().item())
                storage[name]["maxs"].append(tensor.abs().max().item())
                # Sparsity: fraction of values near zero
                sparsity = (tensor.abs() < 1e-6).float().mean().item()
                storage[name]["sparsity"].append(sparsity)

        return hook


# =============================================================================
# Vector-Aware Hooks (Fluid Dynamics)
# =============================================================================

class VectorBackwardHook:
    """
    Advanced backward hooks that capture full gradient vector fields
    and compute fluid dynamics metrics.

    These hooks go beyond scalar statistics to treat gradients as
    vector fields with both magnitude and direction.
    """

    @staticmethod
    def vector_field_collector(
        storage: Dict[str, List[GradientField]],
        name: str,
        keep_tensors: bool = False,
        sample_rate: float = 1.0
    ) -> Callable:
        """
        Create a hook that captures full gradient vector fields.

        This is the foundation for true fluid dynamics analysis.
        Instead of just storing grad.norm(), we store the full gradient
        with direction information.

        Args:
            storage: Dictionary to store GradientField objects
            name: Key for this module in storage
            keep_tensors: If True, keeps full gradient tensors (memory intensive)
            sample_rate: Fraction of gradients to keep (0.0-1.0) for memory efficiency

        Returns:
            Hook function for register_full_backward_hook

        Example:
            >>> fields = {}
            >>> hook = VectorBackwardHook.vector_field_collector(
            ...     fields, "layer1", keep_tensors=False, sample_rate=0.1
            ... )
            >>> handle = module.register_full_backward_hook(hook)
        """
        if name not in storage:
            storage[name] = []

        step_counter = [0]  # Mutable counter for closure

        def hook(module, grad_input, grad_output):
            step = step_counter[0]
            step_counter[0] += 1

            # Randomly sample based on sample_rate
            if sample_rate < 1.0:
                import random
                if random.random() > sample_rate:
                    return

            if grad_output[0] is not None:
                grad = grad_output[0]
                field = GradientField.from_gradient(
                    grad,
                    step=step,
                    keep_tensor=keep_tensors
                )
                storage[name].append(field)

        return hook

    @staticmethod
    def fluid_metrics_collector(
        storage: Dict[str, VectorMetrics],
        name: str,
        layer_type: str = "",
        compute_divergence: bool = True,
        compute_curl: bool = True
    ) -> Callable:
        """
        Create a hook that computes fluid dynamics metrics on the fly.

        This hook computes:
        - Pressure (gradient magnitude)
        - Direction vectors
        - Divergence (expansion/contraction)
        - Curl (rotation/vorticity)
        - Directional changes between steps

        Args:
            storage: Dictionary to store VectorMetrics objects
            name: Layer name
            layer_type: Type of layer (Linear, Conv2d, etc.)
            compute_divergence: Whether to compute divergence (expensive)
            compute_curl: Whether to compute curl (expensive)

        Returns:
            Hook function for register_full_backward_hook
        """
        if name not in storage:
            storage[name] = VectorMetrics(name=name, layer_type=layer_type)

        previous_field = [None]  # Store previous field for directional changes

        def hook(module, grad_input, grad_output):
            if grad_output[0] is None:
                return

            grad = grad_output[0]
            metrics = storage[name]

            # Create gradient field
            field = GradientField.from_gradient(grad, keep_tensor=compute_divergence or compute_curl)

            # Store pressure (magnitude)
            metrics.pressures.append(field.magnitude)

            # Store direction (if healthy)
            if field.direction is not None:
                metrics.directions.append(field.direction.cpu())

            # Compute divergence
            if compute_divergence and not field.is_zero:
                div = FluidOperators.compute_divergence(field)
                metrics.divergences.append(div)

            # Compute curl
            if compute_curl and not field.is_zero:
                curl = FluidOperators.compute_curl(field)
                metrics.curls.append(curl)

            # Compute directional change from previous step
            if previous_field[0] is not None:
                change = FluidOperators.compute_directional_change(
                    previous_field[0], field
                )
                metrics.directional_changes.append(change)

            # Compute strain rate
            if not field.is_zero:
                strain = FluidOperators.compute_strain_rate(field)
                metrics.strain_rates.append(strain)

            # Update previous field
            previous_field[0] = field

        return hook

    @staticmethod
    def sampled_vector_collector(
        storage: Dict[str, List[GradientField]],
        name: str,
        max_elements: int = 1000
    ) -> Callable:
        """
        Create a hook that samples gradient tensors for memory efficiency.

        Instead of keeping the full gradient tensor, this hook randomly
        samples a subset of elements. This preserves directional information
        while dramatically reducing memory usage.

        Args:
            storage: Dictionary to store sampled GradientField objects
            name: Layer name
            max_elements: Maximum number of gradient elements to keep

        Returns:
            Hook function for register_full_backward_hook
        """
        if name not in storage:
            storage[name] = []

        step_counter = [0]

        def hook(module, grad_input, grad_output):
            step = step_counter[0]
            step_counter[0] += 1

            if grad_output[0] is None:
                return

            grad = grad_output[0]

            # Check for numerical issues
            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()

            # Compute magnitude
            magnitude = grad.norm().item()
            is_zero = magnitude < 1e-12

            # Sample a subset of elements
            grad_flat = grad.flatten()
            total_elements = grad_flat.numel()

            if total_elements > max_elements:
                # Random sampling without replacement
                indices = torch.randperm(total_elements)[:max_elements]
                sampled_grad = grad_flat[indices]
            else:
                sampled_grad = grad_flat

            # Compute direction from sampled gradient
            direction = None
            if not is_zero and not has_nan and not has_inf:
                sampled_magnitude = sampled_grad.norm().item()
                direction = sampled_grad / (sampled_magnitude + 1e-12)

            # Create field with sampled data
            field = GradientField(
                tensor=sampled_grad.detach().clone(),
                magnitude=magnitude,
                direction=direction,
                shape=grad.shape,
                device=grad.device,
                step=step,
                has_nan=has_nan,
                has_inf=has_inf,
                is_zero=is_zero
            )

            storage[name].append(field)

        return hook


# =============================================================================
# Adaptive Sampling Strategy
# =============================================================================

class AdaptiveSampler:
    """
    Implements adaptive sampling strategies for memory-efficient gradient capture.

    The sampler automatically adjusts sampling rate based on:
    - Available memory
    - Gradient tensor size
    - Analysis requirements
    """

    def __init__(
        self,
        memory_budget_mb: float = 100.0,
        min_samples_per_layer: int = 10
    ):
        """
        Initialize adaptive sampler.

        Args:
            memory_budget_mb: Total memory budget in megabytes
            min_samples_per_layer: Minimum samples to keep per layer
        """
        self.memory_budget_mb = memory_budget_mb
        self.min_samples_per_layer = min_samples_per_layer
        self.current_usage_mb = 0.0

    def compute_sample_rate(
        self,
        tensor_size: Tuple[int, ...],
        num_layers: int,
        num_steps: int
    ) -> float:
        """
        Compute optimal sampling rate given constraints.

        Args:
            tensor_size: Size of gradient tensor
            num_layers: Number of layers to track
            num_steps: Number of simulation steps

        Returns:
            Sampling rate in [0, 1]
        """
        # Estimate memory per sample (assuming float32)
        elements = 1
        for dim in tensor_size:
            elements *= dim
        memory_per_sample_mb = elements * 4 / (1024 ** 2)

        # Total samples needed
        total_samples = num_layers * num_steps

        # Compute rate that fits in budget
        if memory_per_sample_mb * total_samples <= self.memory_budget_mb:
            return 1.0  # Can keep everything

        # Need to sample
        affordable_samples = self.memory_budget_mb / memory_per_sample_mb
        rate = affordable_samples / total_samples

        # Ensure minimum samples per layer
        min_total = self.min_samples_per_layer * num_layers
        min_rate = min_total / total_samples

        return max(rate, min_rate, 0.01)  # At least 1%

    def should_keep_tensor(
        self,
        field: GradientField,
        importance: float = 0.5
    ) -> bool:
        """
        Decide whether to keep a full tensor based on importance.

        Args:
            field: GradientField to evaluate
            importance: Importance score [0, 1] - higher = more likely to keep

        Returns:
            True if should keep, False otherwise
        """
        # Always drop if memory exhausted
        usage = field.memory_usage_mb()
        if self.current_usage_mb + usage > self.memory_budget_mb:
            return False

        # Probabilistic keeping based on importance
        import random
        if random.random() < importance:
            self.current_usage_mb += usage
            return True

        return False
