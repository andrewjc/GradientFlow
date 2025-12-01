#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example 5: Transformer Analysis
===============================

Analyzing gradient flow in Transformer architectures.
Transformers have unique gradient patterns due to:
- Multi-head attention mechanisms
- Residual connections at every layer
- Layer normalization placement
- Feed-forward network bottlenecks

Key concepts introduced:
- AttentionAnalyzer for transformer architectures
- Query/Key/Value gradient patterns
- Residual connection health
- Layer-wise gradient distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
sys.path.insert(0, '..')
from gradient_flow import (
    FlowAnalyzer,
    AttentionAnalyzer,
    ComparativeAnalyzer,
)


# =============================================================================
# Step 1: Define Transformer Components
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)

        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """A single transformer encoder block."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ):
        super().__init__()

        self.pre_norm = pre_norm

        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if self.pre_norm:
            # Pre-norm: norm -> sublayer -> residual
            attn_out = self.attn(self.norm1(x), mask)
            x = x + self.dropout1(attn_out)

            ffn_out = self.ffn(self.norm2(x))
            x = x + self.dropout2(ffn_out)
        else:
            # Post-norm: sublayer -> residual -> norm
            attn_out = self.attn(x, mask)
            x = self.norm1(x + self.dropout1(attn_out))

            ffn_out = self.ffn(x)
            x = self.norm2(x + self.dropout2(ffn_out))

        return x


class Transformer(nn.Module):
    """A complete transformer encoder."""

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        num_classes: int = 10,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.pre_norm = pre_norm

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (learned)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, pre_norm)
            for _ in range(num_layers)
        ])

        # Final norm (for pre-norm)
        if pre_norm:
            self.final_norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = x.shape

        # Token + position embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(x) + self.pos_embedding(positions)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final norm
        if self.pre_norm:
            x = self.final_norm(x)

        # Classification (use [CLS] token position, i.e., first token)
        cls_token = x[:, 0, :]
        logits = self.classifier(cls_token)

        return logits


# =============================================================================
# Step 2: Create Sample Data
# =============================================================================

def create_transformer_data(
    batch_size: int = 16,
    seq_len: int = 128,
    vocab_size: int = 10000,
    num_classes: int = 10,
):
    """Create dummy transformer input data."""
    tokens = torch.randint(1, vocab_size, (batch_size, seq_len))  # 0 is padding
    labels = torch.randint(0, num_classes, (batch_size,))
    return tokens, labels


# =============================================================================
# Step 3: Analyze Transformer
# =============================================================================

def main():
    print("=" * 60)
    print("Example 5: Transformer Gradient Flow Analysis")
    print("=" * 60)

    # Create model
    model = Transformer(
        vocab_size=10000,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        pre_norm=True,
    )
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Layers: 6")
    print(f"Heads: 8")
    print(f"d_model: 256")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Sample data
    tokens, labels = create_transformer_data(seq_len=64)
    loss_fn = nn.CrossEntropyLoss()

    # =============================================================================
    # Basic Analysis
    # =============================================================================

    print("\n" + "-" * 60)
    print("Step 1: Basic Gradient Flow Analysis")
    print("-" * 60)

    analyzer = FlowAnalyzer(model, "Transformer-6L")
    report = analyzer.analyze(
        sample_input=tokens,
        sample_target=labels,
        loss_fn=loss_fn,
        num_samples=3,
    )

    report.print_summary()
    report.print_issues()
    analyzer.cleanup()

    # =============================================================================
    # Attention-Specific Analysis
    # =============================================================================

    print("\n" + "-" * 60)
    print("Step 2: Attention-Specific Analysis")
    print("-" * 60)

    attn_analyzer = AttentionAnalyzer(
        num_heads=8,
        track_heads=True,
        residual_analysis=True,
    )

    report = attn_analyzer.analyze(
        model=model,
        sample_input=tokens,
        sample_target=labels,
        loss_fn=loss_fn,
        model_name="Transformer-6L",
    )

    # Get transformer metrics
    tf_metrics = attn_analyzer.get_transformer_metrics()

    if tf_metrics:
        print(f"\nTransformer Overview:")
        print(f"  Num layers: {tf_metrics.num_layers}")
        print(f"  Residual decay: {tf_metrics.residual_decay:.3f}")
        print(f"  Avg attention pressure: {tf_metrics.attention_avg_pressure:.2e}")

        print(f"\n{'Layer':<10} {'Q Pressure':>12} {'K Pressure':>12} {'V Pressure':>12} {'Output':>12}")
        print("-" * 60)

        for lm in tf_metrics.layer_metrics:
            print(f"{lm.layer_index:<10} {lm.query_pressure:>12.2e} {lm.key_pressure:>12.2e} {lm.value_pressure:>12.2e} {lm.output_pressure:>12.2e}")

    # Attention profile
    profile = attn_analyzer.get_attention_profile()
    if profile:
        print(f"\nAttention Gradient Profile:")
        print(f"{'Layer':>6} {'Query':>12} {'Key':>12} {'Value':>12}")
        print("-" * 45)
        for layer_idx, q, k, v in profile:
            bar_q = "#" * min(int(q * 100), 20)
            print(f"{layer_idx:>6} {q:>12.2e} {k:>12.2e} {v:>12.2e}  {bar_q}")

    attn_analyzer.cleanup()

    # =============================================================================
    # Compare Pre-Norm vs Post-Norm
    # =============================================================================

    print("\n" + "-" * 60)
    print("Step 3: Pre-Norm vs Post-Norm Comparison")
    print("-" * 60)

    # Create both variants
    model_prenorm = Transformer(num_layers=6, pre_norm=True)
    model_postnorm = Transformer(num_layers=6, pre_norm=False)

    comparator = ComparativeAnalyzer()

    print("\nAnalyzing Pre-Norm transformer...")
    comparator.analyze_model(model_prenorm, tokens, labels, loss_fn, "PreNorm")

    print("Analyzing Post-Norm transformer...")
    comparator.analyze_model(model_postnorm, tokens, labels, loss_fn, "PostNorm")

    comparison = comparator.compare("PreNorm", "PostNorm")
    print(f"\nComparison Summary:")
    print(f"  Average health change: {comparison.avg_health_change:+.1f}")
    print(f"  Average pressure change: {comparison.avg_pressure_change:+.1%}")
    print(f"  Improved layers: {len(comparison.improved_layers)}")
    print(f"  Degraded layers: {len(comparison.degraded_layers)}")

    # Focus on norm layers
    print(f"\nNormalization Layer Comparison:")
    print(f"{'Layer':<40} {'PreNorm':>12} {'PostNorm':>12} {'Change':>12}")
    print("-" * 80)

    for c in comparison.layer_comparisons:
        if 'norm' in c.layer_name.lower():
            change = c.pressure_b / max(c.pressure_a, 1e-10)
            print(f"{c.layer_name:<40} {c.pressure_a:>12.2e} {c.pressure_b:>12.2e} {change:>12.2f}x")

    # =============================================================================
    # Key Insights
    # =============================================================================

    print("\n" + "=" * 60)
    print("Transformer Gradient Flow Insights")
    print("=" * 60)

    print("""
    ATTENTION MECHANISM GRADIENTS:

    1. Query/Key/Value Flow:
       - Q and K: Gradient magnitude depends on sequence length
       - V: Usually has the most stable gradients
       - Imbalanced Q/K/V can indicate attention head issues

    2. Softmax Saturation:
       - Very peaked attention -> near-zero gradients for non-attended tokens
       - Can lead to "dead" attention patterns
       - Temperature scaling or dropout can help

    3. Multi-Head Balance:
       - Heads should have similar gradient magnitudes
       - Imbalanced heads suggest some aren't learning
       - Consider head pruning or reinitialization

    RESIDUAL CONNECTIONS:

    1. Gradient Highway:
       - Residual connections allow gradients to skip layers
       - Critical for training deep transformers
       - Decay ratio should be close to 1.0

    2. Pre-Norm vs Post-Norm:
       - Pre-Norm: More stable gradients, easier to train
       - Post-Norm: Original design, can have gradient issues
       - Pre-Norm generally recommended for deep models

    FFN BOTTLENECKS:

    1. Expansion/Contraction:
       - FFN expands then contracts: d_model -> d_ff -> d_model
       - Can create gradient bottleneck at contraction
       - Monitor fc2 (down-projection) gradients

    2. Activation Function:
       - GELU preferred over ReLU for transformers
       - ReLU can cause dead neurons in FFN
       - Check for high sparsity in FFN gradients

    LAYER-WISE PATTERNS:

    1. Early Layers:
       - Often have smaller gradients
       - May benefit from higher learning rate

    2. Late Layers:
       - Usually have larger gradients
       - More task-specific

    3. Variance:
       - High variance across layers = unstable training
       - Consider layer-wise learning rates or gradient scaling
    """)

    # Export
    report.save_html("transformer_analysis.html")
    print("\nSaved report to: transformer_analysis.html")

    print("\nDone!")


if __name__ == "__main__":
    main()
