#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example 1: Simple MLP Analysis
==============================

This is the simplest example of gradient flow analysis.
We'll create a basic multilayer perceptron and analyze its gradient flow.

Key concepts introduced:
- FlowAnalyzer: The main analysis engine
- FlowReport: The analysis results container
- Basic gradient "pressure" and "health" concepts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the gradient flow toolkit
import sys
sys.path.insert(0, '..')
from gradient_flow import FlowAnalyzer


# =============================================================================
# Step 1: Define a Simple MLP
# =============================================================================

class SimpleMLP(nn.Module):
    """
    A basic multilayer perceptron with 4 hidden layers.

    This is a classic architecture that can suffer from vanishing
    gradients in the early layers if not initialized properly.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, output_dim: int = 10):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# =============================================================================
# Step 2: Create Sample Data
# =============================================================================

def create_sample_data(batch_size: int = 32):
    """Create dummy MNIST-like data."""
    # Simulated flattened images
    inputs = torch.randn(batch_size, 784)
    # Random class labels
    targets = torch.randint(0, 10, (batch_size,))
    return inputs, targets


# =============================================================================
# Step 3: Analyze Gradient Flow
# =============================================================================

def main():
    print("=" * 60)
    print("Example 1: Simple MLP Gradient Flow Analysis")
    print("=" * 60)

    # Create model
    model = SimpleMLP()
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create sample data
    inputs, targets = create_sample_data()

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # =============================================================================
    # The Key Part: Gradient Flow Analysis
    # =============================================================================

    # Create the analyzer
    analyzer = FlowAnalyzer(model, model_name="SimpleMLP")

    # Run analysis
    # This performs a forward pass, computes loss, runs backward,
    # and collects gradient statistics at every layer
    print("\nAnalyzing gradient flow...")
    report = analyzer.analyze(
        sample_input=inputs,
        sample_target=targets,
        loss_fn=loss_fn,
        num_samples=5,  # Average over 5 forward/backward passes
    )

    # =============================================================================
    # Step 4: View Results
    # =============================================================================

    # Print summary
    print("\n" + "=" * 60)
    report.print_summary()

    # Print layer details
    report.print_layers(n=10)

    # Print any issues
    report.print_issues()

    # Print recommendations
    report.print_recommendations()

    # =============================================================================
    # Step 5: Understanding the Results
    # =============================================================================

    print("\n" + "=" * 60)
    print("Understanding the Results")
    print("=" * 60)

    print("""
    The analysis uses a "hydrodynamics" metaphor:

    PRESSURE (gradient magnitude):
        - Low pressure = vanishing gradients (layer not learning)
        - High pressure = exploding gradients (unstable training)
        - Healthy range: roughly 0.1 - 10

    HEALTH SCORE (0-100%):
        - Based on pressure, turbulence, and layer type
        - >75% = Healthy
        - 50-75% = Attention needed
        - <50% = Critical issues

    COMMON ISSUES:
        - VANISHING: Gradients too small to update weights
        - EXPLODING: Gradients too large, causing NaN
        - BOTTLENECK: Compression layer amplifying gradients
    """)

    # =============================================================================
    # Step 6: Export Report
    # =============================================================================

    # Save as HTML for visual exploration
    report.save_html("simple_mlp_report.html")
    print("\nSaved HTML report to: simple_mlp_report.html")

    # Save as JSON for programmatic access
    report.save_json("simple_mlp_report.json")
    print("Saved JSON report to: simple_mlp_report.json")

    # Cleanup
    analyzer.cleanup()

    print("\nDone!")


if __name__ == "__main__":
    main()
