"""
    Example 4: Memory-Augmented RL Agent
    ====================================

    Analyzing gradient flow in memory-augmented reinforcement learning agents.
    These architectures combine:
    - Tree-structured routing blocks
    - Episodic memory modules
    - Actor-critic networks
    - Recurrent processing

    This example demonstrates real-world gradient debugging
    for complex, multi-component architectures.

    Key concepts introduced:
    - Analyzing custom architectures
    - Identifying compression bottlenecks
    - Using gradient scaling to fix issues
    - Comparative analysis during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from gradient_flow import GradientFlowAnalyzer, gradient_scale


# =============================================================================
# Step 1: Define Memory Agent Components
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms


class RoutingBlock(nn.Module):
    """
    A tree-routing block that compresses high-dimensional input
    to low-dimensional routing decisions.

    This is where gradient issues commonly occur due to
    the large fan-in compression ratio.
    """

    def __init__(self, input_dim: int = 256, routing_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.routing_dim = routing_dim

        # Direct compression (prone to gradient issues)
        self.compress = nn.Linear(input_dim, routing_dim, bias=False)

        # Routing weights
        self.route = nn.Linear(routing_dim, 4, bias=True)

        # Initialize carefully
        nn.init.orthogonal_(self.compress.weight, gain=0.1)
        nn.init.orthogonal_(self.route.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compress to routing space
        z = self.compress(x)
        z = F.gelu(z)

        # Compute routing probabilities
        logits = self.route(z)
        probs = F.softmax(logits, dim=-1)

        return probs


class ImprovedRoutingBlock(nn.Module):
    """
    An improved routing block with gradient-aware design.

    Fixes applied:
    1. Bottleneck architecture (reduces compression ratio per layer)
    2. Input normalization
    3. Gradient scaling at critical points
    """

    def __init__(self, input_dim: int = 256, routing_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.routing_dim = routing_dim

        # Bottleneck dimension (intermediate step)
        bottleneck_dim = max(routing_dim * 2, min(64, input_dim // 4))

        # Input normalization
        self.norm = RMSNorm(input_dim)

        # Two-stage compression (reduces gradient amplification)
        self.compress_down = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.compress_proj = nn.Linear(bottleneck_dim, routing_dim, bias=False)

        # Routing weights
        self.route = nn.Linear(routing_dim, 4, bias=True)

        # Gradient scaling factors
        self._input_grad_scale = 0.01
        self._bottleneck_grad_scale = 0.1

        # Initialize
        nn.init.orthogonal_(self.compress_down.weight, gain=0.1)
        nn.init.orthogonal_(self.compress_proj.weight, gain=0.1)
        nn.init.orthogonal_(self.route.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input
        x = self.norm(x)

        # Apply gradient scaling at input (reduces backprop magnitude)
        x = gradient_scale(x, self._input_grad_scale)

        # First compression stage
        h = self.compress_down(x)
        h = gradient_scale(h, self._bottleneck_grad_scale)
        h = F.gelu(h)

        # Second compression stage
        z = self.compress_proj(h)
        z = gradient_scale(z, self._bottleneck_grad_scale)

        # Routing
        logits = self.route(z)
        probs = F.softmax(logits, dim=-1)

        return probs


class MemoryModule(nn.Module):
    """A simple memory module with read/write operations."""

    def __init__(self, memory_size: int = 128, memory_dim: int = 64):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim

        # Memory banks (initialized empty)
        self.register_buffer('memory', torch.zeros(1, memory_size, memory_dim))

        # Read/write controllers
        self.read_key = nn.Linear(256, memory_dim)
        self.write_key = nn.Linear(256, memory_dim)
        self.write_value = nn.Linear(256, memory_dim)

    def forward(self, query: torch.Tensor, mode: str = 'read') -> torch.Tensor:
        batch_size = query.shape[0]

        # Expand memory for batch
        memory = self.memory.expand(batch_size, -1, -1)

        if mode == 'read':
            # Compute attention over memory
            key = self.read_key(query)  # (batch, memory_dim)
            scores = torch.bmm(memory, key.unsqueeze(-1)).squeeze(-1)  # (batch, memory_size)
            weights = F.softmax(scores / np.sqrt(self.memory_dim), dim=-1)
            read = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)  # (batch, memory_dim)
            return read
        else:
            # Write to memory (simplified - no address control)
            write_k = self.write_key(query)
            write_v = self.write_value(query)
            return write_v


class MemoryAgent(nn.Module):
    """
    A memory-augmented RL agent combining:
    - Feature encoder
    - Routing blocks
    - Memory module
    - Actor-critic heads
    """

    def __init__(
        self,
        obs_dim: int = 256,
        action_dim: int = 8,
        use_improved_routing: bool = False,
    ):
        super().__init__()

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            RMSNorm(256),
        )

        # Routing block (original or improved)
        if use_improved_routing:
            self.routing = ImprovedRoutingBlock(256, 16)
        else:
            self.routing = RoutingBlock(256, 16)

        # Memory module
        self.memory = MemoryModule()

        # Actor-critic heads
        self.actor = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple:
        # Encode observation
        features = self.encoder(obs)

        # Get routing decision (for logging, not used in forward)
        routing = self.routing(features)

        # Read from memory
        memory_read = self.memory(features, mode='read')

        # Combine features and memory
        combined = torch.cat([features, memory_read], dim=-1)

        # Actor-critic outputs
        action_logits = self.actor(combined)
        value = self.critic(combined)

        return action_logits, value, routing


# =============================================================================
# Step 2: Comparative Analysis
# =============================================================================

def compare_routing_blocks():
    """Compare original vs improved routing blocks."""
    print("\n" + "=" * 60)
    print("Comparing Routing Block Implementations")
    print("=" * 60)

    # Create sample data
    batch_size = 32
    obs = torch.randn(batch_size, 256)
    target_actions = torch.randint(0, 8, (batch_size,))
    target_values = torch.randn(batch_size, 1)

    def loss_fn(outputs, targets):
        action_logits, values, _ = outputs
        action_loss = F.cross_entropy(action_logits, targets)
        value_loss = F.mse_loss(values, target_values)
        return action_loss + value_loss

    # Create comparator
    comparator = ComparativeAnalyzer()

    # Analyze original routing
    print("\nAnalyzing ORIGINAL routing block...")
    model_orig = MemoryAgent(use_improved_routing=False)
    report_orig = comparator.analyze_model(
        model_orig, obs, target_actions, loss_fn, name="original"
    )

    # Analyze improved routing
    print("Analyzing IMPROVED routing block...")
    model_improved = MemoryAgent(use_improved_routing=True)
    report_improved = comparator.analyze_model(
        model_improved, obs, target_actions, loss_fn, name="improved"
    )

    # Compare
    comparison = comparator.compare("original", "improved")
    comparator.print_comparison(comparison)

    # Focus on routing block
    print("\n" + "-" * 60)
    print("Routing Block Comparison")
    print("-" * 60)

    for c in comparison.layer_comparisons:
        if "routing" in c.layer_name.lower() or "compress" in c.layer_name.lower():
            status = "IMPROVED" if c.improved else ("DEGRADED" if c.degraded else "STABLE")
            print(f"{c.layer_name:<40} {c.pressure_a:>10.1f} -> {c.pressure_b:>10.1f} ({status})")


def main():
    print("=" * 60)
    print("Example 4: Memory-Augmented RL Agent Analysis")
    print("=" * 60)

    # =============================================================================
    # Basic Analysis of Original Agent
    # =============================================================================

    print("\n" + "-" * 60)
    print("Step 1: Analyze Original Agent (with problematic routing)")
    print("-" * 60)

    model = MemoryAgent(use_improved_routing=False)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Sample data parameters
    batch_size = 32
    obs_dim = 256
    num_actions = 8

    # Create analyzer with scientifically validated defaults
    #
    # Configuration rationale for memory-augmented agents:
    # - enable_jacobian=False: Not needed for feedforward architectures
    # - enable_vector_analysis=False: Validation shows no detection benefit for standard pathologies
    #
    # This configuration provides 80% detection at minimal cost (26x faster than full analysis)
    # See: gradient_flow/validation/VALIDATION_RESULTS.md
    #
    # For memory agents:
    # - Compression bottlenecks (256 -> 16 dims) are detected via scalar magnitude analysis
    # - Gradient amplification through fan-in is captured by magnitude and variance metrics
    # - Complex methods add no additional detection value for these pathologies
    analyzer = GradientFlowAnalyzer(
        model,
        enable_rnn_analyzer=False,      # Not needed for feedforward memory architectures
        enable_circular_flow_analyser=False # Not beneficial for compression bottleneck detection
    )

    def input_fn():
        return torch.randn(batch_size, obs_dim)

    def loss_fn_wrapper(outputs):
        action_logits, values, _ = outputs
        targets = torch.randint(0, num_actions, (batch_size,))
        target_vals = torch.randn(batch_size, 1)
        action_loss = F.cross_entropy(action_logits, targets)
        value_loss = F.mse_loss(values, target_vals)
        return action_loss + value_loss

    print("\nAnalyzing gradient propagation (original model)...\n")
    issues = analyzer.analyze(
        input_fn=input_fn,
        loss_fn=loss_fn_wrapper,
        steps=5
    )

    analyzer.print_summary(issues)

    if issues:
        print("\nTop Issues:")
        for issue in issues[:3]:
            print(f"\n{issue}")

    # =============================================================================
    # The Fix: Improved Routing
    # =============================================================================

    print("\n" + "-" * 60)
    print("Step 2: Analyze Fixed Agent (with improved routing)")
    print("-" * 60)

    model_fixed = MemoryAgent(use_improved_routing=True)

    # Use same scientifically validated configuration for comparison
    analyzer2 = GradientFlowAnalyzer(
        model_fixed,
        enable_rnn_analyzer=False,      # Same defaults for fair comparison
        enable_circular_flow_analyser=False
    )

    print("\nAnalyzing gradient propagation (fixed model)...\n")
    issues_fixed = analyzer2.analyze(
        input_fn=input_fn,
        loss_fn=loss_fn_wrapper,
        steps=5
    )

    analyzer2.print_summary(issues_fixed)

    if issues_fixed:
        print("\nTop Issues:")
        for issue in issues_fixed[:3]:
            print(f"\n{issue}")

    # Compare results
    print("\n" + "-" * 60)
    print("Comparison:")
    print("-" * 60)
    print(f"Original model issues: {len(issues)}")
    print(f"Fixed model issues: {len(issues_fixed)}")

    if len(issues_fixed) < len(issues):
        print(f"\n[SUCCESS] Fixed model has {len(issues) - len(issues_fixed)} fewer issues!")

if __name__ == "__main__":
    main()
