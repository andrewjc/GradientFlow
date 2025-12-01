# -*- coding: utf-8 -*-
"""
Recurrent Network Analyzer
===========================

Analyzer for recurrent neural networks including RNNs, LSTMs, GRUs,
and reservoir computing architectures. Focuses on temporal dynamics,
spectral properties, and gradient flow through time.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass

from ..core.engine import FlowAnalyzer
from ..core.metrics import FlowMetrics, LayerHealth
from ..visualizers.report import FlowReport


@dataclass
class ReservoirDynamics:
    """Analysis results for reservoir computing dynamics."""
    spectral_radius: float
    criticality: str  # "SUBCRITICAL", "NEAR_CRITICAL", "CRITICAL", "SUPERCRITICAL"
    mixing_coefficient: float
    effective_rank: float
    low_rank_ratio: float
    eigenvalue_spread: float


class RecurrentAnalyzer:
    """
    Analyzer for recurrent neural networks and reservoir computing.

    This analyzer is designed for:
    - Vanilla RNNs
    - LSTMs / GRUs
    - Reservoir Computing Networks
    - Echo State Networks

    Key diagnostics:
    - Spectral radius (stability analysis)
    - Temporal gradient flow
    - Hidden state dynamics
    - Long-term dependency capture

    Example:
        >>> analyzer = RecurrentAnalyzer()
        >>> dynamics = analyzer.analyze_reservoir(model.recurrent_layer)
        >>> print(f"Spectral radius: {dynamics.spectral_radius:.3f}")
        >>> print(f"Criticality: {dynamics.criticality}")
    """

    def __init__(self, optimal_radius: float = 0.9, critical_band: float = 0.15):
        """
        Initialize recurrent analyzer.

        Args:
            optimal_radius: Target spectral radius for stability (default: 0.9)
            critical_band: Acceptable deviation from optimal (default: 0.15)
        """
        self.optimal_radius = optimal_radius
        self.critical_band = critical_band

    def analyze_reservoir(
        self,
        recurrent_weight: torch.Tensor,
        mixing_weight: Optional[torch.Tensor] = None,
        diagonal_weight: Optional[torch.Tensor] = None,
    ) -> ReservoirDynamics:
        """
        Analyze reservoir dynamics from weight matrices.

        This computes spectral properties and stability metrics for reservoir
        computing layers or recurrent layers.

        Args:
            recurrent_weight: Main recurrent weight matrix (e.g., W_hh)
            mixing_weight: Optional low-rank mixing matrix (U @ V^T form)
            diagonal_weight: Optional diagonal component

        Returns:
            ReservoirDynamics with spectral analysis results
        """
        # Construct full weight matrix
        W = recurrent_weight.clone()

        # Add low-rank component if present
        if mixing_weight is not None:
            W = W + mixing_weight

        # Add diagonal if present
        if diagonal_weight is not None:
            W = W + torch.diag(diagonal_weight)

        # Move to CPU for eigenvalue computation
        W_np = W.detach().cpu().numpy()

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(W_np)
        eigenvalue_mags = np.abs(eigenvalues)

        # Spectral radius (largest eigenvalue magnitude)
        spectral_radius = float(np.max(eigenvalue_mags))

        # Criticality assessment
        if spectral_radius < self.optimal_radius - self.critical_band:
            criticality = "SUBCRITICAL"
        elif spectral_radius > self.optimal_radius + self.critical_band:
            criticality = "SUPERCRITICAL"
        elif abs(spectral_radius - self.optimal_radius) < self.critical_band / 2:
            criticality = "NEAR_CRITICAL"
        else:
            criticality = "CRITICAL"

        # Effective rank (participation ratio of eigenvalues)
        eigenvalue_probs = eigenvalue_mags / (eigenvalue_mags.sum() + 1e-8)
        effective_rank = float(np.exp(-np.sum(eigenvalue_probs * np.log(eigenvalue_probs + 1e-8))))

        # Eigenvalue spread (max/min ratio)
        eigenvalue_spread = float(np.max(eigenvalue_mags) / (np.min(eigenvalue_mags) + 1e-8))

        # Mixing coefficient (if using low-rank component)
        if mixing_weight is not None:
            mixing_norm = torch.linalg.norm(mixing_weight).item()
            recurrent_norm = torch.linalg.norm(recurrent_weight).item()
            mixing_coefficient = mixing_norm / (recurrent_norm + mixing_norm + 1e-8)
        else:
            mixing_coefficient = 0.0

        # Low-rank approximation quality
        if mixing_weight is not None:
            low_rank_ratio = float(torch.linalg.matrix_rank(mixing_weight).item() / W.shape[0])
        else:
            low_rank_ratio = 1.0  # Full rank if no mixing

        return ReservoirDynamics(
            spectral_radius=spectral_radius,
            criticality=criticality,
            mixing_coefficient=mixing_coefficient,
            effective_rank=effective_rank,
            low_rank_ratio=low_rank_ratio,
            eigenvalue_spread=eigenvalue_spread,
        )

    def analyze_hidden_states(
        self,
        hidden_states: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Analyze temporal evolution of hidden states.

        Args:
            hidden_states: List of hidden state tensors over time [T x [B, H]]

        Returns:
            Dictionary with temporal statistics
        """
        # Stack hidden states
        h_tensor = torch.stack(hidden_states)  # [T, B, H]
        T, B, H = h_tensor.shape

        # Activity statistics
        mean_activity = h_tensor.mean(dim=(0, 1))  # [H]
        std_activity = h_tensor.std(dim=(0, 1))  # [H]

        # Temporal correlation
        if T > 1:
            h_diff = h_tensor[1:] - h_tensor[:-1]  # [T-1, B, H]
            temporal_change = torch.linalg.norm(h_diff, dim=-1).mean().item()
        else:
            temporal_change = 0.0

        # Effective dimensionality (participation ratio)
        h_flat = h_tensor.reshape(-1, H)  # [T*B, H]
        cov = torch.cov(h_flat.T)
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        eigenvalue_probs = eigenvalues / (eigenvalues.sum() + 1e-8)
        effective_dim = torch.exp(-torch.sum(eigenvalue_probs * torch.log(eigenvalue_probs + 1e-8))).item()

        return {
            "mean_activity": mean_activity.mean().item(),
            "std_activity": std_activity.mean().item(),
            "temporal_change_rate": temporal_change,
            "effective_dimensionality": effective_dim,
            "dimensionality_ratio": effective_dim / H,
        }

    def find_issues(self, dynamics: ReservoirDynamics) -> List[Dict[str, Any]]:
        """
        Identify issues with reservoir dynamics.

        Args:
            dynamics: ReservoirDynamics from analyze_reservoir()

        Returns:
            List of issue dictionaries with severity and description
        """
        issues = []

        # Check spectral radius
        if dynamics.criticality == "SUBCRITICAL":
            issues.append({
                "severity": "CRITICAL",
                "component": "reservoir",
                "issue": "spectral_radius_too_low",
                "description": f"Spectral radius {dynamics.spectral_radius:.3f} < {self.optimal_radius - self.critical_band:.3f}. "
                              f"Reservoir may forget information too quickly (fast decay).",
            })
        elif dynamics.criticality == "SUPERCRITICAL":
            issues.append({
                "severity": "CRITICAL",
                "component": "reservoir",
                "issue": "spectral_radius_too_high",
                "description": f"Spectral radius {dynamics.spectral_radius:.3f} > {self.optimal_radius + self.critical_band:.3f}. "
                              f"Reservoir may be unstable (exploding states).",
            })

        # Check effective rank
        if dynamics.effective_rank < 4.0:
            issues.append({
                "severity": "HIGH",
                "component": "reservoir",
                "issue": "low_effective_rank",
                "description": f"Effective rank {dynamics.effective_rank:.1f} is very low. "
                              f"Reservoir may have insufficient capacity.",
            })

        return issues
