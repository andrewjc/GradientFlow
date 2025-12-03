"""
Reinforcement Learning Policy Analyzer
=======================================

Analyzer for RL policy networks including actor-critic architectures,
policy gradient methods, and value-based methods. Focuses on action
distribution properties and exploration behavior.

Extends GradientFlowAnalyzer with RL policy-specific diagnostics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass

from .gradient_flow import GradientFlowAnalyzer, GradientIssue, IssueSeverity, IssueType


@dataclass
class PolicyDistributionStats:
    """Statistics for policy action distributions."""
    discrete_entropy_mean: float
    discrete_entropy_normalized: float
    continuous_std_mean: float
    entropy_collapse_risk: str  # "LOW", "MEDIUM", "HIGH"
    exploration_adequacy: str  # "TOO_LOW", "ADEQUATE", "TOO_HIGH"

    # Additional gradient-related metrics
    policy_gradient_magnitude: Optional[float] = None
    value_gradient_magnitude: Optional[float] = None


class RLPolicyAnalyzer:
    """
    Analyzer for reinforcement learning policy networks.

    This analyzer is designed for:
    - Actor-Critic architectures
    - Policy Gradient methods (PPO, A2C, SAC)
    - Discrete and continuous action spaces
    - Mixed action spaces

    Key diagnostics:
    - Gradient flow analysis (via GradientFlowAnalyzer)
    - Action distribution entropy (exploration measure)
    - Continuous action std deviation
    - Policy collapse detection
    - Exploration adequacy
    - Policy vs Value gradient balance

    Example:
        >>> analyzer = RLPolicyAnalyzer(model)
        >>> issues, stats = analyzer.analyze(
        >>>     input_fn=lambda: torch.randn(32, 64),
        >>>     loss_fn=lambda output: compute_policy_loss(output),
        >>>     steps=10
        >>> )
        >>> analyzer.print_summary(issues, stats)
    """

    def __init__(
        self,
        model: nn.Module,
        enable_rnn_analyzer: bool = False,
        enable_circular_flow_analyser: bool = True,
        min_entropy_threshold: float = 0.5,
        min_std_threshold: float = 0.1,
        max_std_threshold: float = 2.0,
        track_distributions: bool = True
    ):
        """
        Initialize RL policy analyzer.

        Args:
            model: RL policy model to analyze
            enable_rnn_analyzer: Whether to compute Jacobian (expensive)
            enable_circular_flow_analyser: Whether to compute divergence/curl
            min_entropy_threshold: Minimum normalized entropy for healthy exploration (default: 0.5)
            min_std_threshold: Minimum std for continuous actions (default: 0.1)
            max_std_threshold: Maximum std before too random (default: 2.0)
            track_distributions: Whether to capture action distributions during analysis
        """
        # Use GradientFlowAnalyzer as the underlying engine
        self.base_analyzer = GradientFlowAnalyzer(
            model,
            enable_rnn_analyzer=enable_rnn_analyzer,
            enable_circular_flow_analyser=enable_circular_flow_analyser
        )

        self.model = model
        self.min_entropy_threshold = min_entropy_threshold
        self.min_std_threshold = min_std_threshold
        self.max_std_threshold = max_std_threshold
        self.track_distributions = track_distributions

        # Storage for RL-specific metrics
        self._logits_list: List[torch.Tensor] = []
        self._std_list: List[torch.Tensor] = []
        self._mean_list: List[torch.Tensor] = []
        self._value_list: List[torch.Tensor] = []

    def analyze(
        self,
        input_fn: Callable[[], torch.Tensor],
        loss_fn: Callable[[Any], torch.Tensor],
        steps: int = 10,
        optimizer: Optional[torch.optim.Optimizer] = None,
        action_extractor: Optional[Callable[[Any], Tuple]] = None
    ) -> Tuple[List[GradientIssue], Optional[PolicyDistributionStats]]:
        """
        Analyze RL policy gradient flow with policy-specific diagnostics.

        Args:
            input_fn: Function that returns input tensors (observations)
            loss_fn: Function that takes model output and returns scalar loss
            steps: Number of analysis steps
            optimizer: Optional optimizer
            action_extractor: Optional function to extract (logits, std, mean, value) from model output

        Returns:
            Tuple of (gradient_issues, policy_distribution_stats)
        """
        # Reset storage
        self._logits_list.clear()
        self._std_list.clear()
        self._mean_list.clear()
        self._value_list.clear()

        # Setup hooks to capture distributions if tracking enabled
        if self.track_distributions and action_extractor:
            # Wrap the loss function to capture distributions
            original_loss_fn = loss_fn

            def wrapped_loss_fn(output):
                # Extract action distributions
                extracted = action_extractor(output)
                if extracted:
                    if len(extracted) >= 1 and extracted[0] is not None:
                        self._logits_list.append(extracted[0].detach())
                    if len(extracted) >= 2 and extracted[1] is not None:
                        self._std_list.append(extracted[1].detach())
                    if len(extracted) >= 3 and extracted[2] is not None:
                        self._mean_list.append(extracted[2].detach())
                    if len(extracted) >= 4 and extracted[3] is not None:
                        self._value_list.append(extracted[3].detach())

                # Compute loss
                return original_loss_fn(output)

            loss_fn = wrapped_loss_fn

        # Run base gradient flow analysis
        issues = self.base_analyzer.analyze(
            input_fn=input_fn,
            loss_fn=loss_fn,
            steps=steps,
            optimizer=optimizer
        )

        # Analyze RL-specific patterns
        policy_stats = self._analyze_policy_specific()

        # Add RL-specific issues
        rl_issues = self._detect_policy_issues(policy_stats)
        issues.extend(rl_issues)

        return issues, policy_stats

    def _analyze_policy_specific(self) -> Optional[PolicyDistributionStats]:
        """Analyze RL policy-specific patterns from collected data."""
        if not self._logits_list and not self._std_list:
            return None

        return self.analyze_action_distributions(
            logits_list=self._logits_list if self._logits_list else None,
            std_list=self._std_list if self._std_list else None,
            mean_list=self._mean_list if self._mean_list else None
        )

    def analyze_discrete_distribution(
        self,
        logits_list: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Analyze discrete action distribution properties.

        Args:
            logits_list: List of logit tensors over time [T x [B, num_actions]]

        Returns:
            Dictionary with entropy and distribution statistics
        """
        if not logits_list:
            return {
                "mean_entropy": 0.0,
                "normalized_entropy": 0.0,
                "entropy_std": 0.0,
                "min_entropy": 0.0,
                "max_entropy": 0.0,
            }

        # Stack logits
        logits_tensor = torch.stack(logits_list)  # [T, B, A]
        T, B, A = logits_tensor.shape

        # Compute probabilities
        probs = F.softmax(logits_tensor, dim=-1)  # [T, B, A]

        # Compute entropy per timestep/batch
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [T, B]

        # Statistics
        mean_entropy = entropy.mean().item()
        std_entropy = entropy.std().item()
        min_entropy = entropy.min().item()
        max_entropy = entropy.max().item()

        # Normalized entropy (0 to 1, where 1 = uniform distribution)
        max_possible_entropy = np.log(A)
        normalized_entropy = mean_entropy / max_possible_entropy

        return {
            "mean_entropy": mean_entropy,
            "normalized_entropy": normalized_entropy,
            "entropy_std": std_entropy,
            "min_entropy": min_entropy,
            "max_entropy": max_entropy,
            "num_actions": A,
            "max_possible_entropy": max_possible_entropy,
        }

    def analyze_continuous_distribution(
        self,
        std_list: List[torch.Tensor],
        mean_list: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze continuous action distribution properties.

        Args:
            std_list: List of std tensors over time [T x [B, action_dim]]
            mean_list: Optional list of mean tensors [T x [B, action_dim]]

        Returns:
            Dictionary with std statistics
        """
        if not std_list:
            return {
                "mean_std": 0.0,
                "std_of_std": 0.0,
                "min_std": 0.0,
                "max_std": 0.0,
            }

        # Handle variable-sized tensors by concatenating instead of stacking
        # This handles cases where action dimensions might vary across timesteps
        std_tensor = torch.cat([s.flatten() for s in std_list])  # [T*B*D]

        # Statistics
        mean_std = std_tensor.mean().item()
        std_of_std = std_tensor.std().item() if std_tensor.numel() > 1 else 0.0
        min_std = std_tensor.min().item()
        max_std = std_tensor.max().item()

        result = {
            "mean_std": mean_std,
            "std_of_std": std_of_std,
            "min_std": min_std,
            "max_std": max_std,
        }

        # If means provided, analyze mean magnitude
        if mean_list:
            mean_tensor = torch.cat([m.flatten() for m in mean_list])  # [T*B*D]
            result["mean_action_magnitude"] = torch.abs(mean_tensor).mean().item()
            result["mean_action_std"] = mean_tensor.std().item() if mean_tensor.numel() > 1 else 0.0

        return result

    def analyze_action_distributions(
        self,
        logits_list: Optional[List[torch.Tensor]] = None,
        std_list: Optional[List[torch.Tensor]] = None,
        mean_list: Optional[List[torch.Tensor]] = None,
    ) -> PolicyDistributionStats:
        """
        Comprehensive analysis of action distributions.

        Args:
            logits_list: Discrete action logits [T x [B, num_actions]]
            std_list: Continuous action stds [T x [B, action_dim]]
            mean_list: Continuous action means [T x [B, action_dim]]

        Returns:
            PolicyDistributionStats with comprehensive diagnostics
        """
        # Analyze discrete if provided
        discrete_entropy_mean = 0.0
        discrete_entropy_normalized = 0.0

        if logits_list:
            discrete_stats = self.analyze_discrete_distribution(logits_list)
            discrete_entropy_mean = discrete_stats["mean_entropy"]
            discrete_entropy_normalized = discrete_stats["normalized_entropy"]

        # Analyze continuous if provided
        continuous_std_mean = 0.0

        if std_list:
            continuous_stats = self.analyze_continuous_distribution(std_list, mean_list)
            continuous_std_mean = continuous_stats["mean_std"]

        # Determine entropy collapse risk
        if discrete_entropy_normalized < 0.2:
            entropy_collapse_risk = "HIGH"
        elif discrete_entropy_normalized < 0.4:
            entropy_collapse_risk = "MEDIUM"
        else:
            entropy_collapse_risk = "LOW"

        # Determine exploration adequacy
        # Check both discrete and continuous
        discrete_adequate = discrete_entropy_normalized >= self.min_entropy_threshold if logits_list else True
        continuous_adequate = (
            self.min_std_threshold <= continuous_std_mean <= self.max_std_threshold
            if std_list else True
        )

        if discrete_adequate and continuous_adequate:
            exploration_adequacy = "ADEQUATE"
        elif (discrete_entropy_normalized < self.min_entropy_threshold) or (continuous_std_mean < self.min_std_threshold):
            exploration_adequacy = "TOO_LOW"
        else:
            exploration_adequacy = "TOO_HIGH"

        return PolicyDistributionStats(
            discrete_entropy_mean=discrete_entropy_mean,
            discrete_entropy_normalized=discrete_entropy_normalized,
            continuous_std_mean=continuous_std_mean,
            entropy_collapse_risk=entropy_collapse_risk,
            exploration_adequacy=exploration_adequacy,
        )

    def _detect_policy_issues(
        self,
        stats: Optional[PolicyDistributionStats]
    ) -> List[GradientIssue]:
        """Detect RL policy-specific gradient issues."""
        issues = []

        if stats is None:
            return issues

        # Issue: High entropy collapse risk
        if stats.entropy_collapse_risk == "HIGH":
            issues.append(GradientIssue(
                layer="policy_head",
                layer_type="Actor/Policy",
                issue_type=IssueType.DEAD,
                severity=IssueSeverity.CRITICAL,
                description=f"Policy entropy {stats.discrete_entropy_normalized:.1%} is critically low. "
                           f"Agent has likely stopped exploring and may be stuck in a local optimum.",
                magnitude=stats.discrete_entropy_normalized,
                recommended_actions=[
                    "Increase entropy regularization coefficient",
                    "Reset or add noise to policy network weights",
                    "Use epsilon-greedy exploration temporarily",
                    "Check if rewards are too sparse or shaped incorrectly",
                    "Ensure policy gradients are flowing (not saturated)"
                ],
                metrics={
                    'entropy_normalized': stats.discrete_entropy_normalized,
                    'entropy_collapse_risk': stats.entropy_collapse_risk
                }
            ))
        elif stats.entropy_collapse_risk == "MEDIUM":
            issues.append(GradientIssue(
                layer="policy_head",
                layer_type="Actor/Policy",
                issue_type=IssueType.UNSTABLE,
                severity=IssueSeverity.HIGH,
                description=f"Policy entropy {stats.discrete_entropy_normalized:.1%} is below healthy threshold. "
                           f"Exploration may be insufficient for continued learning.",
                magnitude=stats.discrete_entropy_normalized,
                recommended_actions=[
                    "Increase entropy bonus in loss function",
                    "Reduce policy learning rate temporarily",
                    "Add exploration noise to actions",
                    "Monitor entropy over time - may stabilize naturally"
                ],
                metrics={
                    'entropy_normalized': stats.discrete_entropy_normalized,
                    'entropy_collapse_risk': stats.entropy_collapse_risk
                }
            ))

        # Issue: Insufficient exploration
        if stats.exploration_adequacy == "TOO_LOW":
            issues.append(GradientIssue(
                layer="policy_head",
                layer_type="Actor/Policy",
                issue_type=IssueType.SATURATION,
                severity=IssueSeverity.HIGH,
                description=f"Action distribution is too deterministic (entropy: {stats.discrete_entropy_normalized:.1%}, "
                           f"std: {stats.continuous_std_mean:.3f}). Agent may converge prematurely.",
                magnitude=max(stats.discrete_entropy_normalized, stats.continuous_std_mean),
                recommended_actions=[
                    "Increase exploration: higher entropy coefficient or action noise",
                    "Check if policy network has saturating activations (sigmoid/tanh at output)",
                    "Reduce policy update frequency (lower policy/value ratio)",
                    "Use curiosity-driven exploration or intrinsic rewards"
                ],
                metrics={
                    'exploration_adequacy': stats.exploration_adequacy,
                    'entropy_normalized': stats.discrete_entropy_normalized,
                    'continuous_std_mean': stats.continuous_std_mean
                }
            ))

        # Issue: Excessive exploration
        elif stats.exploration_adequacy == "TOO_HIGH":
            issues.append(GradientIssue(
                layer="policy_head",
                layer_type="Actor/Policy",
                issue_type=IssueType.UNSTABLE,
                severity=IssueSeverity.MEDIUM,
                description=f"Action distribution is too random (std: {stats.continuous_std_mean:.3f}). "
                           f"Agent may not be learning effectively from experience.",
                magnitude=stats.continuous_std_mean,
                recommended_actions=[
                    "Decrease exploration: lower entropy coefficient or action noise",
                    "Increase policy learning rate to faster convergence",
                    "Check if value function is accurate (affects policy gradients)",
                    "Verify reward signal is not too noisy"
                ],
                metrics={
                    'exploration_adequacy': stats.exploration_adequacy,
                    'continuous_std_mean': stats.continuous_std_mean
                }
            ))

        # Issue: Very low continuous std
        if stats.continuous_std_mean > 0 and stats.continuous_std_mean < self.min_std_threshold:
            issues.append(GradientIssue(
                layer="policy_head",
                layer_type="Actor/Policy",
                issue_type=IssueType.SATURATION,
                severity=IssueSeverity.HIGH,
                description=f"Continuous action std {stats.continuous_std_mean:.3f} is very low. "
                           f"Limited exploration in continuous action space.",
                magnitude=stats.continuous_std_mean,
                recommended_actions=[
                    "Increase minimum std or add std regularization",
                    "Check if std network has vanishing gradients",
                    "Use log-std parameterization with proper initialization",
                    "Add action noise during training (parameter space noise)"
                ],
                metrics={
                    'continuous_std_mean': stats.continuous_std_mean,
                    'min_std_threshold': self.min_std_threshold
                }
            ))

        return issues

    def print_summary(self, issues: List[GradientIssue], stats: Optional[PolicyDistributionStats]):
        """Print analysis summary with RL policy-specific info."""
        # Use base analyzer's print_summary
        self.base_analyzer.print_summary(issues)

        # Add RL-specific summary
        if stats:
            print("\n" + "=" * 80)
            print("RL POLICY METRICS")
            print("=" * 80)
            print(f"\nAction Distribution:")
            if stats.discrete_entropy_mean > 0:
                print(f"  Discrete entropy:      {stats.discrete_entropy_mean:.3f}")
                print(f"  Normalized entropy:    {stats.discrete_entropy_normalized:.1%}")
            if stats.continuous_std_mean > 0:
                print(f"  Continuous std:        {stats.continuous_std_mean:.3f}")

            print(f"\nExploration Status:")
            print(f"  Collapse risk:         {stats.entropy_collapse_risk}")
            print(f"  Exploration:           {stats.exploration_adequacy}")

            if stats.policy_gradient_magnitude is not None:
                print(f"\nGradient Balance:")
                print(f"  Policy gradients:      {stats.policy_gradient_magnitude:.3e}")
                if stats.value_gradient_magnitude is not None:
                    print(f"  Value gradients:       {stats.value_gradient_magnitude:.3e}")
                    ratio = stats.policy_gradient_magnitude / (stats.value_gradient_magnitude + 1e-8)
                    print(f"  Policy/Value ratio:    {ratio:.2f}x")

    def get_healthy_layers(self, issues: List[GradientIssue]) -> List[str]:
        """Get list of healthy layers."""
        return self.base_analyzer.get_healthy_layers(issues)
