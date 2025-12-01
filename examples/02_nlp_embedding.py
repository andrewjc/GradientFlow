#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example 2: NLP with Embeddings
==============================

Analyzing gradient flow in NLP models with embedding layers.
Embedding layers are notorious for gradient issues due to
sparse updates and high dimensionality.

Key concepts introduced:
- Embedding layer gradient analysis
- Sparse gradient patterns
- StandardAnalyzer for feedforward architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, '..')
from gradient_flow import FlowAnalyzer, StandardAnalyzer


# =============================================================================
# Step 1: Define an NLP Classification Model
# =============================================================================

class TextClassifier(nn.Module):
    """
    A simple text classifier with:
    - Embedding layer
    - Mean pooling
    - Classification head

    This is a common architecture for sentiment analysis, topic classification, etc.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 5,
    ):
        super().__init__()

        # Embedding layer - maps token IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Classification head
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len) - token IDs

        # Embed tokens
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Mean pooling over sequence
        pooled = embedded.mean(dim=1)  # (batch, embed_dim)

        # Classification layers
        h = F.relu(self.fc1(pooled))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        output = self.fc3(h)

        return output


# =============================================================================
# Step 2: Create Sample Data
# =============================================================================

def create_sample_data(batch_size: int = 32, seq_len: int = 50, vocab_size: int = 10000):
    """Create dummy text data."""
    # Random token sequences
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Random class labels
    targets = torch.randint(0, 5, (batch_size,))
    return inputs, targets


# =============================================================================
# Step 3: Analyze with StandardAnalyzer
# =============================================================================

def main():
    print("=" * 60)
    print("Example 2: NLP Text Classifier Gradient Flow Analysis")
    print("=" * 60)

    # Create model
    model = TextClassifier()
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Vocabulary: 10,000 tokens")
    print(f"Embedding dim: 128")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create sample data
    inputs, targets = create_sample_data()
    loss_fn = nn.CrossEntropyLoss()

    # =============================================================================
    # Using StandardAnalyzer for Depth Analysis
    # =============================================================================

    print("\n" + "-" * 60)
    print("Using StandardAnalyzer (includes depth analysis)")
    print("-" * 60)

    analyzer = StandardAnalyzer(
        track_activations=True,
        depth_analysis=True,
        bottleneck_threshold=0.25,  # Flag compression below 25%
    )

    report = analyzer.analyze(
        model=model,
        sample_input=inputs,
        sample_target=targets,
        loss_fn=loss_fn,
        model_name="TextClassifier",
        num_samples=5,
    )

    # =============================================================================
    # View Results
    # =============================================================================

    report.print_summary()
    report.print_issues()

    # =============================================================================
    # Layer Type Statistics
    # =============================================================================

    print("\n" + "-" * 60)
    print("Layer Type Statistics")
    print("-" * 60)

    stats = analyzer.get_layer_type_stats()
    print(f"\n{'Layer Type':<20} {'Count':>8} {'Avg Pressure':>15} {'Max Pressure':>15}")
    print("-" * 60)
    for s in stats:
        print(f"{s.layer_type:<20} {s.count:>8} {s.avg_pressure:>15.2e} {s.max_pressure:>15.2e}")

    # =============================================================================
    # Depth Profile
    # =============================================================================

    print("\n" + "-" * 60)
    print("Gradient Pressure by Depth")
    print("-" * 60)

    profile = analyzer.get_depth_profile()
    print(f"\n{'Depth':>8} {'Mean Pressure':>15} {'Std':>12}")
    print("-" * 40)
    for depth, mean_p, std_p in profile:
        bar = "#" * int(min(50, mean_p * 10))
        print(f"{depth:>8} {mean_p:>15.2e} {std_p:>12.2e}  {bar}")

    # =============================================================================
    # NLP-Specific Insights
    # =============================================================================

    print("\n" + "=" * 60)
    print("NLP-Specific Gradient Insights")
    print("=" * 60)

    print("""
    EMBEDDING LAYER CONSIDERATIONS:

    1. Sparse Updates:
       - Only tokens in the batch receive gradient updates
       - Most of the embedding matrix is untouched each step
       - Can lead to slow learning for rare words

    2. Gradient Magnitude:
       - Embedding gradients are typically smaller than dense layers
       - This is normal due to the averaging over sequence length

    3. Common Issues:
       - Very low pressure: Embeddings not receiving signal
       - Very high pressure: Learning rate too high for embeddings
       - Consider using separate learning rates for embeddings

    4. Recommendations:
       - Use pretrained embeddings if available
       - Consider freezing embeddings initially
       - Monitor gradient-to-weight ratio for embeddings
    """)

    # =============================================================================
    # Compare Embedding vs Dense Layers
    # =============================================================================

    print("\n" + "-" * 60)
    print("Embedding vs Dense Layer Comparison")
    print("-" * 60)

    embedding_metrics = report.metrics.get("embedding")
    fc1_metrics = report.metrics.get("fc1")

    if embedding_metrics and fc1_metrics:
        print(f"\n{'Metric':<20} {'Embedding':>15} {'FC1':>15} {'Ratio':>10}")
        print("-" * 60)
        print(f"{'Mean Pressure':<20} {embedding_metrics.mean_pressure:>15.2e} {fc1_metrics.mean_pressure:>15.2e} {embedding_metrics.mean_pressure/max(fc1_metrics.mean_pressure, 1e-10):>10.2f}")
        print(f"{'Max Pressure':<20} {embedding_metrics.max_pressure:>15.2e} {fc1_metrics.max_pressure:>15.2e} {embedding_metrics.max_pressure/max(fc1_metrics.max_pressure, 1e-10):>10.2f}")

    # Export
    report.save_html("nlp_text_classifier_report.html")
    print("\nSaved report to: nlp_text_classifier_report.html")

    analyzer.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    main()
