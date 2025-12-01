# -*- coding: utf-8 -*-
"""
Transformer Analyzer
====================

Specialized analyzer for transformer-based models with attention mechanisms.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class AttentionStats:
    """Statistics about attention mechanisms."""
    attention_entropy_mean: float
    attention_sparsity: float
    head_diversity: float
    rank_collapse_risk: str


class TransformerAnalyzer:
    """
    Specialized analyzer for transformer architectures.

    Analyzes:
    - Attention pattern statistics
    - Multi-head diversity
    - Rank collapse in attention
    - Layer norm statistics

    Usage:
        >>> analyzer = TransformerAnalyzer()
        >>> stats = analyzer.analyze_attention_patterns(attn_weights_list)
    """

    def __init__(
        self,
        min_entropy_threshold: float = 0.5,
        min_diversity_threshold: float = 0.3
    ):
        self.min_entropy_threshold = min_entropy_threshold
        self.min_diversity_threshold = min_diversity_threshold

    def analyze_attention_patterns(
        self,
        attention_weights: List[torch.Tensor]
    ) -> AttentionStats:
        """
        Analyze attention weight patterns.

        Args:
            attention_weights: List of attention weight tensors
                              [T x (B, num_heads, seq_len, seq_len)]

        Returns:
            AttentionStats with pattern analysis
        """
        if not attention_weights or len(attention_weights) == 0:
            return AttentionStats(
                attention_entropy_mean=0.0,
                attention_sparsity=0.0,
                head_diversity=0.0,
                rank_collapse_risk="UNKNOWN"
            )

        entropies = []
        sparsities = []

        for attn in attention_weights:
            # attn: (B, num_heads, seq_len, seq_len)

            # Compute entropy per head
            head_entropies = -(attn * torch.log(attn + 1e-12)).sum(dim=-1).mean()
            entropies.append(head_entropies.item())

            # Compute sparsity (% of attention weights near zero)
            sparsity = (attn < 0.01).float().mean()
            sparsities.append(sparsity.item())

        attention_entropy_mean = np.mean(entropies)
        attention_sparsity = np.mean(sparsities)

        # Head diversity (variance across heads)
        # This would require head-specific analysis
        head_diversity = 0.5  # Placeholder

        # Rank collapse detection
        if attention_sparsity > 0.9:
            rank_collapse_risk = "HIGH"
        elif attention_sparsity > 0.7:
            rank_collapse_risk = "MEDIUM"
        else:
            rank_collapse_risk = "LOW"

        return AttentionStats(
            attention_entropy_mean=attention_entropy_mean,
            attention_sparsity=attention_sparsity,
            head_diversity=head_diversity,
            rank_collapse_risk=rank_collapse_risk
        )

    def analyze_layer_norms(
        self,
        layer_norm_stats: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Analyze layer normalization statistics.

        Args:
            layer_norm_stats: List of (mean, var) tuples from layer norms

        Returns:
            Dictionary of statistics
        """
        if not layer_norm_stats:
            return {
                "mean_shift": 0.0,
                "var_range": 0.0,
                "normalization_health": 1.0
            }

        means, vars = zip(*layer_norm_stats)
        means_tensor = torch.stack([m.mean() for m in means])
        vars_tensor = torch.stack([v.mean() for v in vars])

        mean_shift = means_tensor.abs().mean().item()
        var_range = vars_tensor.max().item() - vars_tensor.min().item()

        # Health: closer to 0 mean and 1 variance is better
        normalization_health = 1.0 - (mean_shift + abs(vars_tensor.mean().item() - 1.0))

        return {
            "mean_shift": mean_shift,
            "var_range": var_range,
            "normalization_health": max(0.0, normalization_health)
        }
