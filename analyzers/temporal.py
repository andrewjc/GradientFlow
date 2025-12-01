# -*- coding: utf-8 -*-
"""
Temporal Network Analyzer
=========================

Analyzer for recurrent and temporal neural networks including RNNs,
LSTMs, GRUs, and custom recurrent architectures. Focuses on gradient
flow through time and common issues like vanishing/exploding gradients
in long sequences.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field

from ..core.engine import FlowAnalyzer
from ..core.metrics import FlowMetrics, LayerHealth
from ..core.hooks import HookManager
from ..visualizers.report import FlowReport


@dataclass
class TemporalMetrics:
    """Metrics specific to temporal gradient flow."""
    layer_name: str
    timestep_pressures: List[float] = field(default_factory=list)
    temporal_decay_rate: float = 0.0
    max_timestep: int = 0
    min_timestep: int = 0
    effective_memory: int = 0  # How many timesteps have meaningful gradients

    @property
    def has_vanishing(self) -> bool:
        """Check if gradients vanish over time."""
        return self.temporal_decay_rate < -0.3

    @property
    def has_exploding(self) -> bool:
        """Check if gradients explode over time."""
        return self.temporal_decay_rate > 0.3


class TemporalAnalyzer:
    """
    Analyzer for recurrent and temporal neural networks.

    This analyzer is designed for:
    - Vanilla RNNs
    - LSTMs
    - GRUs
    - Custom recurrent blocks
    - Temporal convolutional networks

    Key diagnostics:
    - Gradient flow through time steps
    - Temporal decay/explosion rates
    - Effective memory length
    - Gate-specific analysis (for LSTM/GRU)
    - Hidden state gradient accumulation

    Example:
        >>> analyzer = TemporalAnalyzer(sequence_length=100)
        >>> report = analyzer.analyze(model, sequences, targets, loss_fn)
        >>> report.print_summary()
        >>>
        >>> # Get temporal metrics
        >>> temporal = analyzer.get_temporal_metrics()
        >>> for tm in temporal:
        ...     print(f"{tm.layer_name}: decay={tm.temporal_decay_rate:.3f}")
    """

    # Common recurrent layer types
    RNN_LAYERS = (nn.RNN, nn.LSTM, nn.GRU, nn.RNNCell, nn.LSTMCell, nn.GRUCell)

    def __init__(
        self,
        sequence_length: Optional[int] = None,
        track_gates: bool = True,
        memory_threshold: float = 0.01,
    ):
        """
        Initialize temporal analyzer.

        Args:
            sequence_length: Expected sequence length (auto-detected if None)
            track_gates: Track individual gate gradients for LSTM/GRU
            memory_threshold: Threshold for effective memory calculation
        """
        self.sequence_length = sequence_length
        self.track_gates = track_gates
        self.memory_threshold = memory_threshold

        self._flow_analyzer: Optional[FlowAnalyzer] = None
        self._temporal_metrics: Dict[str, TemporalMetrics] = {}
        self._recurrent_layers: Dict[str, nn.Module] = {}
        self._timestep_hooks: Dict[str, List[float]] = {}

    def analyze(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        sample_target: torch.Tensor,
        loss_fn: Callable,
        model_name: str = "TemporalNetwork",
        num_samples: int = 1,
    ) -> FlowReport:
        """
        Analyze gradient flow in a temporal network.

        Args:
            model: The neural network to analyze
            sample_input: Sample input tensor (batch, seq_len, features)
            sample_target: Sample target tensor
            loss_fn: Loss function
            model_name: Name for the report
            num_samples: Number of analysis iterations

        Returns:
            FlowReport with analysis results
        """
        # Detect sequence length
        if self.sequence_length is None and len(sample_input.shape) >= 2:
            self.sequence_length = sample_input.shape[1]

        # Find recurrent layers
        self._find_recurrent_layers(model)

        # Run flow analysis
        self._flow_analyzer = FlowAnalyzer(model, model_name)
        report = self._flow_analyzer.analyze(
            sample_input,
            sample_target,
            loss_fn,
            num_samples=num_samples,
        )

        # Add temporal-specific analysis
        self._analyze_temporal_flow(model, sample_input, sample_target, loss_fn)
        self._add_temporal_issues(report)

        return report

    def analyze_unrolled(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        sample_target: torch.Tensor,
        loss_fn: Callable,
        forward_fn: Callable,
        model_name: str = "TemporalNetwork",
    ) -> FlowReport:
        """
        Analyze gradient flow with explicit unrolling.

        Use this for custom recurrent architectures where you need
        to manually unroll the computation.

        Args:
            model: The neural network to analyze
            sample_input: Sample input tensor
            sample_target: Sample target tensor
            loss_fn: Loss function
            forward_fn: Custom forward function that unrolls computation
            model_name: Name for the report

        Returns:
            FlowReport with analysis results
        """
        self._flow_analyzer = FlowAnalyzer(model, model_name)

        def custom_forward(x: torch.Tensor) -> torch.Tensor:
            return forward_fn(model, x)

        report = self._flow_analyzer.analyze_custom(
            sample_input,
            sample_target,
            loss_fn,
            custom_forward,
        )

        self._add_temporal_issues(report)
        return report

    def _find_recurrent_layers(self, model: nn.Module) -> None:
        """Identify recurrent layers in the model."""
        self._recurrent_layers.clear()

        for name, module in model.named_modules():
            if isinstance(module, self.RNN_LAYERS):
                self._recurrent_layers[name] = module

    def _analyze_temporal_flow(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        sample_target: torch.Tensor,
        loss_fn: Callable,
    ) -> None:
        """Analyze gradient flow through time steps."""
        self._temporal_metrics.clear()
        self._timestep_hooks.clear()

        if not self._recurrent_layers:
            return

        # For each recurrent layer, track gradients at each timestep
        for name, module in self._recurrent_layers.items():
            self._timestep_hooks[name] = []

            if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                self._analyze_builtin_rnn(name, module, model, sample_input, sample_target, loss_fn)

    def _analyze_builtin_rnn(
        self,
        name: str,
        module: nn.Module,
        model: nn.Module,
        sample_input: torch.Tensor,
        sample_target: torch.Tensor,
        loss_fn: Callable,
    ) -> None:
        """Analyze gradient flow in built-in RNN modules."""
        # PyTorch RNNs have weight_ih_l0, weight_hh_l0, etc.
        # We track the hidden-to-hidden weight gradient as proxy for temporal flow

        timestep_grads = []

        def hook_fn(grad):
            timestep_grads.append(grad.norm().item())
            return grad

        # Register hook on hidden-to-hidden weights
        weight_hh = getattr(module, 'weight_hh_l0', None)
        if weight_hh is not None and weight_hh.requires_grad:
            handle = weight_hh.register_hook(hook_fn)

            try:
                model.zero_grad()
                output = model(sample_input)

                if isinstance(output, tuple):
                    output = output[0]

                if output.shape != sample_target.shape:
                    if len(output.shape) == 3:
                        output = output[:, -1, :]

                loss = loss_fn(output, sample_target)
                loss.backward()
            finally:
                handle.remove()

            if timestep_grads:
                self._create_temporal_metrics(name, timestep_grads)

    def _create_temporal_metrics(self, name: str, grads: List[float]) -> None:
        """Create temporal metrics from gradient sequence."""
        if not grads:
            return

        pressures = grads
        self._timestep_hooks[name] = pressures

        # Calculate decay rate
        decay_rate = 0.0
        if len(pressures) >= 2 and pressures[0] > 0 and pressures[-1] > 0:
            decay_rate = np.log(pressures[-1] / pressures[0]) / len(pressures)

        # Find max/min timesteps
        max_idx = np.argmax(pressures)
        min_idx = np.argmin(pressures)

        # Calculate effective memory
        max_pressure = max(pressures)
        threshold = max_pressure * self.memory_threshold
        effective_memory = sum(1 for p in pressures if p > threshold)

        self._temporal_metrics[name] = TemporalMetrics(
            layer_name=name,
            timestep_pressures=pressures,
            temporal_decay_rate=decay_rate,
            max_timestep=int(max_idx),
            min_timestep=int(min_idx),
            effective_memory=effective_memory,
        )

    def _add_temporal_issues(self, report: FlowReport) -> None:
        """Add temporal-specific issues to report."""
        for name, tm in self._temporal_metrics.items():
            if tm.has_vanishing:
                severity = "CRITICAL" if tm.temporal_decay_rate < -0.5 else "HIGH"
                report.issues.append({
                    "type": "VANISHING_TEMPORAL",
                    "severity": severity,
                    "layer": name,
                    "layer_type": "RNN",
                    "info": f"Temporal decay rate: {tm.temporal_decay_rate:.3f}, effective memory: {tm.effective_memory} steps"
                })

            if tm.has_exploding:
                severity = "CRITICAL" if tm.temporal_decay_rate > 0.5 else "HIGH"
                report.issues.append({
                    "type": "EXPLODING_TEMPORAL",
                    "severity": severity,
                    "layer": name,
                    "layer_type": "RNN",
                    "info": f"Temporal growth rate: {tm.temporal_decay_rate:.3f}"
                })

            if tm.effective_memory < 5 and self.sequence_length and self.sequence_length > 10:
                report.issues.append({
                    "type": "SHORT_MEMORY",
                    "severity": "MEDIUM",
                    "layer": name,
                    "layer_type": "RNN",
                    "info": f"Effective memory only {tm.effective_memory} steps (seq_len={self.sequence_length})"
                })

    def get_temporal_metrics(self) -> List[TemporalMetrics]:
        """
        Get temporal metrics for all recurrent layers.

        Returns:
            List of TemporalMetrics for each recurrent layer
        """
        return list(self._temporal_metrics.values())

    def get_timestep_profile(self, layer_name: str) -> Optional[List[float]]:
        """
        Get gradient pressure profile across timesteps.

        Args:
            layer_name: Name of the recurrent layer

        Returns:
            List of pressures at each timestep, or None if not found
        """
        return self._timestep_hooks.get(layer_name)

    def get_effective_memory(self) -> Dict[str, int]:
        """
        Get effective memory length for each recurrent layer.

        Returns:
            Dictionary mapping layer name to effective memory length
        """
        return {
            name: tm.effective_memory
            for name, tm in self._temporal_metrics.items()
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._flow_analyzer:
            self._flow_analyzer.cleanup()
