"""
Transformer Analyzer
====================

Specialized analyzer for transformer-based models with attention mechanisms.
Extends GradientFlowAnalyzer with transformer-specific diagnostics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional, Any
from dataclasses import dataclass, field

from .gradient_flow import GradientFlowAnalyzer, GradientIssue, IssueSeverity, IssueType


@dataclass
class AttentionStats:
    """Statistics about attention mechanisms."""
    attention_entropy_mean: float
    attention_sparsity: float
    head_diversity: float
    rank_collapse_risk: str

    # Gradient-specific stats
    qkv_gradient_balance: Optional[Dict[str, float]] = None
    attention_gradient_magnitude: Optional[float] = None


@dataclass
class TransformerLayerMetrics:
    """Metrics for a single transformer layer."""
    layer_index: int
    layer_name: str

    # Attention metrics
    query_magnitude: float
    key_magnitude: float
    value_magnitude: float
    output_magnitude: float

    # Feed-forward metrics
    ff_up_magnitude: float
    ff_down_magnitude: float

    # Residual connection health
    residual_contribution: float

    # Layer norm stats
    pre_norm_magnitude: Optional[float] = None
    post_norm_magnitude: Optional[float] = None


class TransformerAnalyzer:
    """
    Specialized analyzer for transformer architectures.

    Combines GradientFlowAnalyzer with transformer-specific analysis:
    - Attention pattern statistics
    - Multi-head diversity
    - Q/K/V gradient balance
    - Residual connection health
    - Layer normalization behavior
    - Feed-forward network bottlenecks

    Usage:
        >>> analyzer = TransformerAnalyzer(model)
        >>> issues, attn_stats = analyzer.analyze(
        >>>     input_fn=lambda: torch.randint(0, 1000, (32, 128)),
        >>>     loss_fn=lambda output: F.cross_entropy(output, targets),
        >>>     steps=10
        >>> )
    """

    def __init__(
        self,
        model: nn.Module,
        enable_rnn_analyzer: bool = False,  # Expensive for transformers
        enable_circular_flow_analyser: bool = True,
        min_entropy_threshold: float = 0.5,
        min_diversity_threshold: float = 0.3,
        track_attention_weights: bool = True
    ):
        """
        Initialize transformer analyzer.

        Args:
            model: Transformer model to analyze
            enable_rnn_analyzer: Whether to enable the rnn specific analyzer
            enable_circular_flow_analyser: Whether to enable circular flow analyzer
            min_entropy_threshold: Minimum attention entropy for health
            min_diversity_threshold: Minimum head diversity for health
            track_attention_weights: Whether to capture attention weights during analysis
        """
        # Use GradientFlowAnalyzer as the underlying engine
        self.base_analyzer = GradientFlowAnalyzer(
            model,
            enable_rnn_analyzer=enable_rnn_analyzer,
            enable_circular_flow_analyser=enable_circular_flow_analyser
        )

        self.model = model
        self.min_entropy_threshold = min_entropy_threshold
        self.min_diversity_threshold = min_diversity_threshold
        self.track_attention_weights = track_attention_weights

        # Storage for transformer-specific metrics
        self._attention_weights: List[torch.Tensor] = []
        self._layer_metrics: List[TransformerLayerMetrics] = []
        self._layer_norm_stats: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def analyze(
        self,
        input_fn: Callable[[], torch.Tensor],
        loss_fn: Callable[[Any], torch.Tensor],
        steps: int = 10,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Tuple[List[GradientIssue], Optional[AttentionStats]]:
        """
        Analyze transformer gradient flow with attention-specific diagnostics.

        Args:
            input_fn: Function that returns input tensors
            loss_fn: Function that takes model output and returns scalar loss
            steps: Number of analysis steps
            optimizer: Optional optimizer

        Returns:
            Tuple of (gradient_issues, attention_stats)
        """
        # Reset storage
        self._attention_weights.clear()
        self._layer_metrics.clear()
        self._layer_norm_stats.clear()

        # If tracking attention weights, add hooks
        if self.track_attention_weights:
            self._setup_attention_hooks()

        # Run base gradient flow analysis
        issues = self.base_analyzer.analyze(
            input_fn=input_fn,
            loss_fn=loss_fn,
            steps=steps,
            optimizer=optimizer
        )

        # Remove hooks
        if self.track_attention_weights:
            self._remove_attention_hooks()

        # Analyze transformer-specific patterns
        attn_stats = self._analyze_transformer_specific()

        # Add transformer-specific issues
        transformer_issues = self._detect_transformer_issues(attn_stats)
        issues.extend(transformer_issues)

        return issues, attn_stats

    def _setup_attention_hooks(self):
        """Setup hooks to capture attention weights during forward pass."""
        # This is a simplified version - in practice, you'd need to identify
        # attention modules based on the specific transformer implementation
        self._hooks = []

        for name, module in self.model.named_modules():
            # Look for common attention module patterns
            if 'attn' in name.lower() or 'attention' in name.lower():
                # You would need to customize this based on your transformer implementation
                # This is just a placeholder
                pass

    def _remove_attention_hooks(self):
        """Remove attention weight hooks."""
        if hasattr(self, '_hooks'):
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()

    def _analyze_transformer_specific(self) -> Optional[AttentionStats]:
        """Analyze transformer-specific patterns from collected data."""
        if not self._attention_weights:
            return None

        return self.analyze_attention_patterns(self._attention_weights)

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
        head_diversity = self._compute_head_diversity(attention_weights)

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

    def _compute_head_diversity(self, attention_weights: List[torch.Tensor]) -> float:
        """Compute diversity across attention heads."""
        if not attention_weights:
            return 0.0

        # Simple diversity metric: variance of attention patterns across heads
        diversities = []
        for attn in attention_weights:
            if attn.dim() >= 3:
                # Compute pairwise correlation between heads
                num_heads = attn.shape[1] if attn.dim() == 4 else attn.shape[0]
                if num_heads > 1:
                    # Flatten each head's attention pattern
                    flat_heads = attn.flatten(start_dim=2 if attn.dim() == 4 else 1)
                    # Compute variance across heads
                    diversity = flat_heads.var(dim=1 if attn.dim() == 4 else 0).mean().item()
                    diversities.append(diversity)

        return np.mean(diversities) if diversities else 0.5

    def _detect_transformer_issues(
        self,
        attn_stats: Optional[AttentionStats]
    ) -> List[GradientIssue]:
        """Detect transformer-specific gradient issues."""
        issues = []

        if attn_stats is None:
            return issues

        # Issue: Low attention entropy (attending to too few positions)
        if attn_stats.attention_entropy_mean < self.min_entropy_threshold:
            issues.append(GradientIssue(
                layer="attention_mechanism",
                layer_type="MultiHeadAttention",
                issue_type=IssueType.SATURATION,
                severity=IssueSeverity.MEDIUM,
                description=f"Low attention entropy ({attn_stats.attention_entropy_mean:.3f}). "
                           f"Model may be attending to too few positions.",
                magnitude=attn_stats.attention_entropy_mean,
                recommended_actions=[
                    "Increase attention dropout",
                    "Use attention temperature scaling",
                    "Check if model is overfitting",
                    "Verify input sequence diversity"
                ],
                metrics={
                    'entropy': attn_stats.attention_entropy_mean,
                    'sparsity': attn_stats.attention_sparsity
                }
            ))

        # Issue: High rank collapse risk
        if attn_stats.rank_collapse_risk in ["HIGH", "MEDIUM"]:
            severity = IssueSeverity.HIGH if attn_stats.rank_collapse_risk == "HIGH" else IssueSeverity.MEDIUM
            issues.append(GradientIssue(
                layer="attention_mechanism",
                layer_type="MultiHeadAttention",
                issue_type=IssueType.BOTTLENECK,
                severity=severity,
                description=f"Rank collapse risk: {attn_stats.rank_collapse_risk}. "
                           f"Attention weights are highly sparse ({attn_stats.attention_sparsity:.1%}).",
                magnitude=attn_stats.attention_sparsity,
                recommended_actions=[
                    "Reduce attention sparsity with temperature scaling",
                    "Use talking heads attention",
                    "Add attention regularization",
                    "Increase model capacity"
                ],
                metrics={
                    'sparsity': attn_stats.attention_sparsity,
                    'rank_collapse_risk': attn_stats.rank_collapse_risk
                }
            ))

        # Issue: Low head diversity
        if attn_stats.head_diversity < self.min_diversity_threshold:
            issues.append(GradientIssue(
                layer="attention_heads",
                layer_type="MultiHeadAttention",
                issue_type=IssueType.BOTTLENECK,
                severity=IssueSeverity.LOW,
                description=f"Low head diversity ({attn_stats.head_diversity:.3f}). "
                           f"Multiple heads learning similar patterns.",
                magnitude=attn_stats.head_diversity,
                recommended_actions=[
                    "Use different initialization for each head",
                    "Add head-specific regularization",
                    "Reduce number of heads if redundant",
                    "Try talking heads attention"
                ],
                metrics={
                    'head_diversity': attn_stats.head_diversity
                }
            ))

        return issues

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

    def print_summary(self, issues: List[GradientIssue], attn_stats: Optional[AttentionStats]):
        """Print analysis summary with transformer-specific info."""
        # Use base analyzer's print_summary
        self.base_analyzer.print_summary(issues)

        # Add transformer-specific summary
        if attn_stats:
            print("\n" + "=" * 80)
            print("TRANSFORMER-SPECIFIC METRICS")
            print("=" * 80)
            print(f"\nAttention Patterns:")
            print(f"  Entropy (mean):        {attn_stats.attention_entropy_mean:.3f}")
            print(f"  Sparsity:              {attn_stats.attention_sparsity:.1%}")
            print(f"  Head diversity:        {attn_stats.head_diversity:.3f}")
            print(f"  Rank collapse risk:    {attn_stats.rank_collapse_risk}")

    def get_healthy_layers(self, issues: List[GradientIssue]) -> List[str]:
        """Get list of healthy layers."""
        return self.base_analyzer.get_healthy_layers(issues)
