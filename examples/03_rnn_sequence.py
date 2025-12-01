#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
sys.path.insert(0, '..')
from gradient_flow import FlowAnalyzer, TemporalAnalyzer


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
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: nn.Module,
):
    """Analyze a single RNN model."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")

    # Use TemporalAnalyzer for RNN-specific analysis
    analyzer = TemporalAnalyzer(
        sequence_length=inputs.shape[1],
        track_gates=True,
        memory_threshold=0.01,
    )

    report = analyzer.analyze(
        model=model,
        sample_input=inputs,
        sample_target=targets,
        loss_fn=loss_fn,
        model_name=name,
        num_samples=3,
    )

    # Print summary
    report.print_summary()
    report.print_issues()

    # Temporal-specific metrics
    temporal_metrics = analyzer.get_temporal_metrics()

    if temporal_metrics:
        print(f"\n{'Temporal Analysis':^60}")
        print("-" * 60)
        print(f"{'Layer':<30} {'Decay Rate':>15} {'Eff. Memory':>12}")
        print("-" * 60)

        for tm in temporal_metrics:
            decay_str = f"{tm.temporal_decay_rate:+.3f}"
            if tm.has_vanishing:
                decay_str += " (VANISHING)"
            elif tm.has_exploding:
                decay_str += " (EXPLODING)"

            print(f"{tm.layer_name:<30} {decay_str:>15} {tm.effective_memory:>12}")

    # Effective memory
    memory_info = analyzer.get_effective_memory()
    if memory_info:
        print(f"\nEffective Memory (timesteps with meaningful gradients):")
        for layer, memory in memory_info.items():
            bar = "#" * min(memory, 50)
            print(f"  {layer}: {memory} steps {bar}")

    analyzer.cleanup()
    return report


def main():
    print("=" * 60)
    print("Example 3: RNN Gradient Flow Analysis")
    print("=" * 60)

    # Create sample data
    seq_len = 100
    inputs, targets = create_sequence_data(seq_len=seq_len)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nSequence length: {seq_len} timesteps")
    print(f"Input dim: 32")
    print(f"Hidden dim: 128")
    print(f"Num layers: 2")

    # =============================================================================
    # Analyze Vanilla RNN
    # =============================================================================

    rnn = VanillaRNN()
    analyze_rnn(rnn, "VanillaRNN", inputs, targets, loss_fn)

    # =============================================================================
    # Analyze LSTM
    # =============================================================================

    lstm = LSTMClassifier()
    analyze_rnn(lstm, "LSTM", inputs, targets, loss_fn)

    # =============================================================================
    # Analyze GRU
    # =============================================================================

    gru = GRUClassifier()
    analyze_rnn(gru, "GRU", inputs, targets, loss_fn)

    # =============================================================================
    # Insights
    # =============================================================================

    print("\n" + "=" * 60)
    print("RNN Gradient Flow Insights")
    print("=" * 60)

    print("""
    TEMPORAL GRADIENT PATTERNS:

    1. Vanilla RNN:
       - Gradients often decay exponentially through time
       - "Temporal decay rate" < -0.3 indicates vanishing gradients
       - Effective memory is typically short (5-20 timesteps)
       - Use gradient clipping to prevent explosion

    2. LSTM:
       - Cell state provides a "gradient highway"
       - Forget gate bias helps maintain long-term memory
       - Much better gradient flow for long sequences
       - Initialize forget gate bias to 1.0 for better memory

    3. GRU:
       - Simpler than LSTM, often similar performance
       - Update gate controls gradient flow
       - Can still struggle with very long sequences

    DIAGNOSING ISSUES:

    - Low effective memory: Model can't learn long-range dependencies
    - High temporal decay: Gradients vanish before reaching early timesteps
    - Temporal growth: Exploding gradients (rare with gated RNNs)

    SOLUTIONS:

    - For vanishing: Use LSTM/GRU, reduce sequence length, use attention
    - For exploding: Gradient clipping, reduce learning rate
    - For short memory: Add attention mechanism, use Transformer
    """)

    print("\nDone!")


if __name__ == "__main__":
    main()
