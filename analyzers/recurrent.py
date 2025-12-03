"""
RNN Analyzer
===========================

Analyzer for recurrent neural networks including RNNs, LSTMs, GRUs,
and reservoir computing architectures. Focuses on temporal dynamics,
spectral properties, and gradient flow through time.

Extends GradientFlowAnalyzer with RNN-specific diagnostics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass

from .gradient_flow import GradientFlowAnalyzer, GradientIssue, IssueSeverity, IssueType


@dataclass
class ReservoirDynamics:
    """Analysis results for reservoir computing dynamics."""
    spectral_radius: float
    criticality: str  # "SUBCRITICAL", "NEAR_CRITICAL", "CRITICAL", "SUPERCRITICAL"
    mixing_coefficient: float
    effective_rank: float
    low_rank_ratio: float
    eigenvalue_spread: float

    # Temporal gradient metrics
    temporal_decay_rate: Optional[float] = None
    effective_memory_length: Optional[float] = None


class RecurrentAnalyzer:
    """
    Analyzer for recurrent neural networks and reservoir computing.

    This analyzer is designed for:
    - Vanilla RNNs
    - LSTMs / GRUs
    - Reservoir Computing Networks
    - Echo State Networks

    Key diagnostics:
    - Gradient flow analysis (via GradientFlowAnalyzer)
    - Spectral radius (stability analysis)
    - Temporal gradient flow
    - Hidden state dynamics
    - Long-term dependency capture

    Example:
        >>> analyzer = RecurrentAnalyzer(model)
        >>> issues, dynamics = analyzer.analyze(
        >>>     input_fn=lambda: torch.randn(32, 100, 64),
        >>>     loss_fn=lambda output: F.cross_entropy(output, targets),
        >>>     steps=10
        >>> )
        >>> analyzer.print_summary(issues, dynamics)
    """

    def __init__(
        self,
        model: nn.Module,
        enable_rnn_analyzer: bool = True,  # Enabled for RNN spectral analysis
        enable_circular_flow_analyser: bool = False,
        optimal_radius: float = 0.9,
        critical_band: float = 0.15,
        track_hidden_states: bool = True
    ):
        """
        Initialize recurrent analyzer.

        Args:
            model: RNN model to analyze
            enable_rnn_analyzer: Whether to compute Jacobian (expensive for RNNs)
            enable_circular_flow_analyser: Whether to compute divergence/curl
            optimal_radius: Target spectral radius for stability (default: 0.9)
            critical_band: Acceptable deviation from optimal (default: 0.15)
            track_hidden_states: Whether to capture hidden states during analysis
        """
        # Use GradientFlowAnalyzer as the underlying engine
        self.base_analyzer = GradientFlowAnalyzer(
            model,
            enable_rnn_analyzer=enable_rnn_analyzer,
            enable_circular_flow_analyser=enable_circular_flow_analyser
        )

        self.model = model
        self.optimal_radius = optimal_radius
        self.critical_band = critical_band
        self.track_hidden_states = track_hidden_states

        # Storage for RNN-specific metrics
        self._hidden_states: List[torch.Tensor] = []
        self._recurrent_weights: List[torch.Tensor] = []

    def analyze(
        self,
        input_fn: Callable[[], torch.Tensor],
        loss_fn: Callable[[Any], torch.Tensor],
        steps: int = 10,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Tuple[List[GradientIssue], Optional[ReservoirDynamics]]:
        """
        Analyze recurrent network gradient flow with RNN-specific diagnostics.

        Args:
            input_fn: Function that returns input tensors (should return sequences)
            loss_fn: Function that takes model output and returns scalar loss
            steps: Number of analysis steps
            optimizer: Optional optimizer

        Returns:
            Tuple of (gradient_issues, reservoir_dynamics)
        """
        # Reset storage
        self._hidden_states.clear()
        self._recurrent_weights.clear()

        # Setup hooks to capture RNN-specific data
        if self.track_hidden_states:
            self._setup_rnn_hooks()

        # Run base gradient flow analysis
        issues = self.base_analyzer.analyze(
            input_fn=input_fn,
            loss_fn=loss_fn,
            steps=steps,
            optimizer=optimizer
        )

        # Remove hooks
        if self.track_hidden_states:
            self._remove_rnn_hooks()

        # Analyze RNN-specific patterns
        dynamics = self._analyze_rnn_specific()

        # Add RNN-specific issues
        rnn_issues = self._detect_rnn_issues(dynamics)
        issues.extend(rnn_issues)

        return issues, dynamics

    def _setup_rnn_hooks(self):
        """Setup hooks to capture RNN weights and hidden states."""
        self._hooks = []

        for name, module in self.model.named_modules():
            # Detect RNN/LSTM/GRU modules
            if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
                # Extract recurrent weights
                for param_name, param in module.named_parameters():
                    if 'weight_hh' in param_name:
                        self._recurrent_weights.append(param.data.clone())

    def _remove_rnn_hooks(self):
        """Remove RNN hooks."""
        if hasattr(self, '_hooks'):
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()

    def _analyze_rnn_specific(self) -> Optional[ReservoirDynamics]:
        """Analyze RNN-specific patterns from collected data."""
        if not self._recurrent_weights:
            return None

        # Analyze the first recurrent weight matrix
        # (for multi-layer RNNs, this could be extended)
        return self.analyze_reservoir(self._recurrent_weights[0])

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

        # Temporal decay rate (based on spectral radius)
        # In RNNs, gradient magnitude decays by approximately spectral_radius per timestep
        temporal_decay_rate = spectral_radius if spectral_radius < 1.0 else 1.0

        # Effective memory length (timesteps until gradient < 1% of original)
        if spectral_radius > 0 and spectral_radius < 1.0:
            # Solve: spectral_radius^t = 0.01
            effective_memory_length = float(np.log(0.01) / np.log(spectral_radius))
        else:
            effective_memory_length = float('inf') if spectral_radius >= 1.0 else 1.0

        return ReservoirDynamics(
            spectral_radius=spectral_radius,
            criticality=criticality,
            mixing_coefficient=mixing_coefficient,
            effective_rank=effective_rank,
            low_rank_ratio=low_rank_ratio,
            eigenvalue_spread=eigenvalue_spread,
            temporal_decay_rate=temporal_decay_rate,
            effective_memory_length=effective_memory_length,
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

    def _detect_rnn_issues(
        self,
        dynamics: Optional[ReservoirDynamics]
    ) -> List[GradientIssue]:
        """Detect RNN-specific gradient issues."""
        issues = []

        if dynamics is None:
            return issues

        # Issue: Spectral radius too low (fast forgetting)
        if dynamics.criticality == "SUBCRITICAL":
            issues.append(GradientIssue(
                layer="recurrent_layer",
                layer_type="RNN/LSTM/GRU",
                issue_type=IssueType.VANISHING,
                severity=IssueSeverity.HIGH,
                description=f"Spectral radius {dynamics.spectral_radius:.3f} is too low (< {self.optimal_radius - self.critical_band:.3f}). "
                           f"Network may forget information too quickly. Effective memory: {dynamics.effective_memory_length:.1f} steps.",
                magnitude=dynamics.spectral_radius,
                recommended_actions=[
                    "Increase spectral radius by scaling recurrent weights",
                    "Use orthogonal initialization for recurrent weights",
                    "Consider using LSTM/GRU instead of vanilla RNN",
                    "Add skip connections or residual connections"
                ],
                metrics={
                    'spectral_radius': dynamics.spectral_radius,
                    'criticality': dynamics.criticality,
                    'effective_memory_length': dynamics.effective_memory_length
                }
            ))

        # Issue: Spectral radius too high (instability risk)
        if dynamics.criticality == "SUPERCRITICAL":
            issues.append(GradientIssue(
                layer="recurrent_layer",
                layer_type="RNN/LSTM/GRU",
                issue_type=IssueType.EXPLODING,
                severity=IssueSeverity.CRITICAL,
                description=f"Spectral radius {dynamics.spectral_radius:.3f} is too high (> {self.optimal_radius + self.critical_band:.3f}). "
                           f"Network may have exploding hidden states and gradients.",
                magnitude=dynamics.spectral_radius,
                recommended_actions=[
                    "Apply spectral normalization to recurrent weights",
                    "Use gradient clipping (clip_grad_norm_)",
                    "Reduce learning rate",
                    "Initialize recurrent weights with smaller scale"
                ],
                metrics={
                    'spectral_radius': dynamics.spectral_radius,
                    'criticality': dynamics.criticality,
                    'eigenvalue_spread': dynamics.eigenvalue_spread
                }
            ))

        # Issue: Low effective rank
        if dynamics.effective_rank < 4.0:
            issues.append(GradientIssue(
                layer="recurrent_layer",
                layer_type="RNN/LSTM/GRU",
                issue_type=IssueType.BOTTLENECK,
                severity=IssueSeverity.MEDIUM,
                description=f"Effective rank {dynamics.effective_rank:.1f} is very low. "
                           f"Recurrent layer may have insufficient capacity for complex temporal patterns.",
                magnitude=dynamics.effective_rank,
                recommended_actions=[
                    "Increase hidden state dimension",
                    "Use better initialization (orthogonal or identity + noise)",
                    "Check for rank collapse during training",
                    "Consider using multiple recurrent layers"
                ],
                metrics={
                    'effective_rank': dynamics.effective_rank,
                    'low_rank_ratio': dynamics.low_rank_ratio
                }
            ))

        # Issue: Very short effective memory
        if dynamics.effective_memory_length is not None and dynamics.effective_memory_length < 10.0:
            issues.append(GradientIssue(
                layer="recurrent_layer",
                layer_type="RNN/LSTM/GRU",
                issue_type=IssueType.VANISHING,
                severity=IssueSeverity.MEDIUM,
                description=f"Effective memory length is only {dynamics.effective_memory_length:.1f} timesteps. "
                           f"Network may struggle with long-term dependencies.",
                magnitude=dynamics.effective_memory_length,
                recommended_actions=[
                    "Use LSTM/GRU with forget gate bias initialized to 1.0",
                    "Add attention mechanism for long sequences",
                    "Consider Transformer architecture for long-range dependencies",
                    "Use layer normalization in recurrent connections"
                ],
                metrics={
                    'effective_memory_length': dynamics.effective_memory_length,
                    'spectral_radius': dynamics.spectral_radius
                }
            ))

        return issues

    def print_summary(self, issues: List[GradientIssue], dynamics: Optional[ReservoirDynamics]):
        """Print analysis summary with RNN-specific info."""
        # Use base analyzer's print_summary
        self.base_analyzer.print_summary(issues)

        # Add RNN-specific summary
        if dynamics:
            print("\n" + "=" * 80)
            print("RNN-SPECIFIC METRICS")
            print("=" * 80)
            print(f"\nSpectral Properties:")
            print(f"  Spectral radius:       {dynamics.spectral_radius:.3f}")
            print(f"  Criticality:           {dynamics.criticality}")
            print(f"  Effective rank:        {dynamics.effective_rank:.1f}")
            print(f"  Eigenvalue spread:     {dynamics.eigenvalue_spread:.1f}x")

            print(f"\nTemporal Dynamics:")
            if dynamics.temporal_decay_rate is not None:
                print(f"  Temporal decay rate:   {dynamics.temporal_decay_rate:.3f}")
            if dynamics.effective_memory_length is not None:
                if dynamics.effective_memory_length == float('inf'):
                    print(f"  Effective memory:      Infinite (unstable)")
                else:
                    print(f"  Effective memory:      {dynamics.effective_memory_length:.1f} timesteps")

            if dynamics.mixing_coefficient > 0:
                print(f"\nMixing Properties:")
                print(f"  Mixing coefficient:    {dynamics.mixing_coefficient:.3f}")
                print(f"  Low-rank ratio:        {dynamics.low_rank_ratio:.3f}")

    def get_healthy_layers(self, issues: List[GradientIssue]) -> List[str]:
        """Get list of healthy layers."""
        return self.base_analyzer.get_healthy_layers(issues)
