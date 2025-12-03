"""
    Example 3: RNN Sequence Analysis
    ================================

    Analyzing gradient flow in recurrent neural networks.
    RNNs are particularly susceptible to vanishing/exploding gradients
    due to gradient flow through time.

    Key concepts introduced:
    - TemporalAnalyzer for recurrent architectures
    - Temporal decay analysis
    - Effective memory length
    - Gate-level analysis (LSTM/GRU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from gradient_flow import GradientFlowAnalyzer


# =============================================================================
# Step 1: Define RNN Models
# =============================================================================

class VanillaRNN(nn.Module):
    """
    A simple vanilla RNN for sequence classification.

    Vanilla RNNs are notorious for vanishing gradients in long sequences.
    This example demonstrates how to diagnose this issue.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
    ):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh',
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        output, _ = self.rnn(x)
        # Use final hidden state
        final_hidden = output[:, -1, :]
        return self.fc(final_hidden)


class LSTMClassifier(nn.Module):
    """
    An LSTM for sequence classification.

    LSTMs use gates to control gradient flow:
    - Forget gate: What to discard from cell state
    - Input gate: What new information to add
    - Output gate: What to output from cell state

    These gates help maintain gradients over long sequences.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hidden, cell) = self.lstm(x)
        final_hidden = output[:, -1, :]
        return self.fc(final_hidden)


class GRUClassifier(nn.Module):
    """
    A GRU for sequence classification.

    GRUs are a simplified version of LSTMs with:
    - Reset gate: Controls access to previous hidden state
    - Update gate: Controls blending of old and new state

    Often comparable performance to LSTMs with fewer parameters.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, hidden = self.gru(x)
        final_hidden = output[:, -1, :]
        return self.fc(final_hidden)


# =============================================================================
# Step 2: Create Sample Data
# =============================================================================

def create_sequence_data(
    batch_size: int = 32,
    seq_len: int = 100,
    input_dim: int = 32,
):
    """Create dummy sequence data."""
    inputs = torch.randn(batch_size, seq_len, input_dim)
    targets = torch.randint(0, 10, (batch_size,))
    return inputs, targets


# =============================================================================
# Step 3: Analyze Different RNN Types
# =============================================================================

def analyze_rnn(
    model: nn.Module,
    name: str,
    seq_len: int,
    input_dim: int,
    loss_fn: nn.Module,
):
    """Analyze a single RNN model."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")

    # Create analyzer optimized for RNN analysis
    #
    # Configuration rationale for RNNs:
    # - enable_jacobian=True: NEEDED for spectral radius analysis (gradient flow through time)
    # - enable_vector_analysis=False: Not needed - temporal patterns don't benefit from curl/divergence
    #
    # For RNNs, Jacobian analysis IS valuable for detecting:
    # - Exploding/vanishing through time (eigenvalue magnitude)
    # - Effective memory length (spectral radius)
    # - Temporal gradient decay patterns
    #
    # Use RecurrentAnalyzer for RNN-specific features (spectral radius, temporal dynamics)
    analyzer = GradientFlowAnalyzer(
        model,
        enable_rnn_analyzer=True,        # ESSENTIAL for RNN gradient flow analysis
        enable_circular_flow_analyser=False # Not beneficial for temporal sequences
    )

    # Define input and loss functions
    def input_fn():
        return torch.randn(32, seq_len, input_dim)

    def loss_fn_wrapper(output):
        targets = torch.randint(0, 10, (32,))
        return loss_fn(output, targets)

    # Run analysis
    print(f"\nAnalyzing gradient propagation (seq_len={seq_len})...\n")
    issues = analyzer.analyze(
        input_fn=input_fn,
        loss_fn=loss_fn_wrapper,
        steps=5
    )

    # Print summary
    analyzer.print_summary(issues)

    # Print detailed issues
    if issues:
        print("\n" + "-" * 60)
        print("DETAILED ISSUES")
        print("-" * 60)
        for issue in issues[:3]:  # Show first 3
            print(f"\n{issue}")

    # Get healthy layers
    healthy = analyzer.get_healthy_layers(issues)
    if healthy:
        print("\n" + "-" * 60)
        print(f"HEALTHY LAYERS ({len(healthy)} total)")
        print("-" * 60)
        for layer in healthy[:5]:
            print(f"  - {layer}")

    return issues


def main():
    print("=" * 60)
    print("Example 3: RNN Gradient Flow Analysis")
    print("=" * 60)

    # Parameters
    seq_len = 100
    input_dim = 32
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nSequence length: {seq_len} timesteps")
    print(f"Input dim: {input_dim}")
    print(f"Hidden dim: 128")
    print(f"Num layers: 2")

    # =============================================================================
    # Analyze Vanilla RNN
    # =============================================================================

    rnn = VanillaRNN(input_dim=input_dim)
    analyze_rnn(rnn, "VanillaRNN", seq_len, input_dim, loss_fn)

    # =============================================================================
    # Analyze LSTM
    # =============================================================================

    lstm = LSTMClassifier(input_dim=input_dim)
    analyze_rnn(lstm, "LSTM", seq_len, input_dim, loss_fn)

    # =============================================================================
    # Analyze GRU
    # =============================================================================

    gru = GRUClassifier(input_dim=input_dim)
    analyze_rnn(gru, "GRU", seq_len, input_dim, loss_fn)

if __name__ == "__main__":
    main()
