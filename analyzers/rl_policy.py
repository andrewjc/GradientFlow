# -*- coding: utf-8 -*-
"""
Reinforcement Learning Policy Analyzer
=======================================

Analyzer for RL policy networks including actor-critic architectures,
policy gradient methods, and value-based methods. Focuses on action
distribution properties and exploration behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PolicyDistributionStats:
    """Statistics for policy action distributions."""
    discrete_entropy_mean: float
    discrete_entropy_normalized: float
    continuous_std_mean: float
    entropy_collapse_risk: str  # "LOW", "MEDIUM", "HIGH"
    exploration_adequacy: str  # "TOO_LOW", "ADEQUATE", "TOO_HIGH"


class RLPolicyAnalyzer:
    """
    Analyzer for reinforcement learning policy networks.

    This analyzer is designed for:
    - Actor-Critic architectures
    - Policy Gradient methods (PPO, A2C, SAC)
    - Discrete and continuous action spaces
    - Mixed action spaces

    Key diagnostics:
    - Action distribution entropy (exploration measure)
    - Continuous action std deviation
    - Policy collapse detection
    - Exploration adequacy

    Example:
        >>> analyzer = RLPolicyAnalyzer()
        >>> stats = analyzer.analyze_distributions(logits_list, std_list)
        >>> print(f"Exploration: {stats.exploration_adequacy}")
        >>> print(f"Entropy: {stats.discrete_entropy_normalized:.2%}")
    """

    def __init__(
        self,
        min_entropy_threshold: float = 0.5,
        min_std_threshold: float = 0.1,
        max_std_threshold: float = 2.0,
    ):
        """
        Initialize RL policy analyzer.

        Args:
            min_entropy_threshold: Minimum normalized entropy for healthy exploration (default: 0.5)
            min_std_threshold: Minimum std for continuous actions (default: 0.1)
            max_std_threshold: Maximum std before too random (default: 2.0)
        """
        self.min_entropy_threshold = min_entropy_threshold
        self.min_std_threshold = min_std_threshold
        self.max_std_threshold = max_std_threshold

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

        # Stack stds
        std_tensor = torch.stack(std_list)  # [T, B, D]

        # Statistics
        mean_std = std_tensor.mean().item()
        std_of_std = std_tensor.std().item()
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
            mean_tensor = torch.stack(mean_list)  # [T, B, D]
            result["mean_action_magnitude"] = torch.abs(mean_tensor).mean().item()
            result["mean_action_std"] = mean_tensor.std(dim=(0, 1)).mean().item()

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

    def find_issues(self, stats: PolicyDistributionStats) -> List[Dict[str, Any]]:
        """
        Identify issues with policy distributions.

        Args:
            stats: PolicyDistributionStats from analyze_action_distributions()

        Returns:
            List of issue dictionaries with severity and description
        """
        issues = []

        # Check entropy collapse
        if stats.entropy_collapse_risk == "HIGH":
            issues.append({
                "severity": "CRITICAL",
                "component": "policy",
                "issue": "entropy_collapse",
                "description": f"Policy entropy {stats.discrete_entropy_normalized:.1%} is very low. "
                              f"Agent may have stopped exploring.",
            })
        elif stats.entropy_collapse_risk == "MEDIUM":
            issues.append({
                "severity": "HIGH",
                "component": "policy",
                "issue": "low_entropy",
                "description": f"Policy entropy {stats.discrete_entropy_normalized:.1%} is below recommended threshold. "
                              f"Exploration may be insufficient.",
            })

        # Check exploration adequacy
        if stats.exploration_adequacy == "TOO_LOW":
            issues.append({
                "severity": "HIGH",
                "component": "policy",
                "issue": "insufficient_exploration",
                "description": "Action distribution too deterministic. Agent may get stuck in local optima.",
            })
        elif stats.exploration_adequacy == "TOO_HIGH":
            issues.append({
                "severity": "MEDIUM",
                "component": "policy",
                "issue": "excessive_exploration",
                "description": "Action distribution too random. Agent may not be learning effectively.",
            })

        # Check continuous std
        if stats.continuous_std_mean > 0 and stats.continuous_std_mean < self.min_std_threshold:
            issues.append({
                "severity": "HIGH",
                "component": "policy",
                "issue": "low_continuous_std",
                "description": f"Continuous action std {stats.continuous_std_mean:.3f} is very low. "
                              f"Limited exploration in continuous space.",
            })

        return issues
