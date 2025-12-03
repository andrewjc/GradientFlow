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
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from gradient_flow import GradientFlowAnalyzer


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
    # Using GradientFlowAnalyzer with Scientifically Validated Defaults
    # =============================================================================

    print("\n" + "-" * 60)
    print("Using GradientFlowAnalyzer (Optimized Configuration)")
    print("-" * 60)

    # Create analyzer with scientifically validated defaults
    #
    # Configuration rationale for NLP models:
    # - enable_jacobian=False: Not needed for feedforward architectures (embeddings + linear layers)
    # - enable_vector_analysis=False: Validation shows no detection benefit for standard pathologies
    #
    # This configuration provides 80% detection at minimal cost (26x faster than full analysis)
    # See: gradient_flow/validation/VALIDATION_RESULTS.md
    #
    # For NLP models with embeddings:
    # - Embedding layers have sparse gradient patterns (only updated tokens receive gradients)
    # - Scalar metrics (magnitude, variance) are sufficient to detect vanishing/dead gradients
    # - Complex methods add no value for detecting embedding-specific issues
    analyzer = GradientFlowAnalyzer(
        model,
        enable_rnn_analyzer=False,      # Not needed for feedforward NLP models
        enable_circular_flow_analyser=False # Not beneficial for standard gradient pathologies
    )

    # Define input and loss functions
    def input_fn():
        return torch.randint(0, 10000, (32, 50))

    def loss_fn_wrapper(output):
        targets = torch.randint(0, 5, (32,))
        return loss_fn(output, targets)

    # Run analysis
    print("\nAnalyzing gradient propagation...\n")
    issues = analyzer.analyze(
        input_fn=input_fn,
        loss_fn=loss_fn_wrapper,
        steps=5
    )

    # =============================================================================
    # View Results
    # =============================================================================

    analyzer.print_summary(issues)

    # Print detailed issues
    if issues:
        print("\n" + "-" * 60)
        print("DETAILED ISSUES")
        print("-" * 60)
        for issue in issues[:5]:  # Show first 5
            print(f"\n{issue}")

    # Get healthy layers
    healthy = analyzer.get_healthy_layers(issues)
    if healthy:
        print("\n" + "-" * 60)
        print(f"HEALTHY LAYERS ({len(healthy)} total)")
        print("-" * 60)
        for layer in healthy[:10]:
            print(f"  - {layer}")

if __name__ == "__main__":
    main()
