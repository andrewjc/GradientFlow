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
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from gradient_flow import GradientFlowAnalyzer


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
    # Basic Gradient Flow Analysis
    # =============================================================================

    print("\n" + "-" * 60)
    print("Step 1: Basic Gradient Flow Analysis")
    print("-" * 60)

    # Create analyzer with scientifically validated defaults
    #
    # Configuration rationale for Transformers:
    # - enable_jacobian=False: Not needed for standard gradient pathology detection
    # - enable_vector_analysis=False: Validation shows no detection benefit
    #
    # This configuration provides 80% detection at minimal cost (26x faster than full analysis)
    # See: gradient_flow/validation/VALIDATION_RESULTS.md
    #
    # ADVANCED: For attention-specific analysis (entropy, head diversity, rank collapse):
    # Use TransformerAnalyzer instead:
    #   from gradient_flow import TransformerAnalyzer
    #   analyzer = TransformerAnalyzer(model, min_entropy_threshold=0.5, min_diversity_threshold=0.3)
    #   issues, attn_stats = analyzer.analyze(input_fn, loss_fn, steps=20)
    #   print(f"Attention entropy: {attn_stats.attention_entropy_mean:.3f}")
    #   print(f"Head diversity: {attn_stats.head_diversity:.3f}")
    #
    # TransformerAnalyzer adds:
    # - Attention pattern analysis (entropy, sparsity)
    # - Head diversity metrics
    # - Rank collapse detection
    # - All on top of base gradient flow analysis
    analyzer = GradientFlowAnalyzer(
        model,
        enable_rnn_analyzer=False,      # Not needed for standard transformer analysis
        enable_circular_flow_analyser=False # Not beneficial for standard pathologies
    )

    # Parameters
    batch_size = 16
    seq_len = 64
    vocab_size = 10000
    num_classes = 10

    def input_fn():
        return torch.randint(1, vocab_size, (batch_size, seq_len))

    def loss_fn_wrapper(output):
        targets = torch.randint(0, num_classes, (batch_size,))
        return loss_fn(output, targets)

    print("\nAnalyzing gradient propagation...\n")
    issues = analyzer.analyze(
        input_fn=input_fn,
        loss_fn=loss_fn_wrapper,
        steps=5
    )

    analyzer.print_summary(issues)

    if issues:
        print("\nTop Issues:")
        for issue in issues[:5]:
            print(f"\n{issue}")

if __name__ == "__main__":
    main()
