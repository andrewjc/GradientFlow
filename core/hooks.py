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
