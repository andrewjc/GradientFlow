"""
    Example 1: Simple MLP Analysis
    ==============================

    This is the simplest example of gradient analysis.
    We'll create a basic multilayer perceptron and analyze its gradient propagation.

    Key concepts introduced:
    - FlowAnalyzer: The main analysis engine
    - FlowReport: The analysis results container
    - Basic gradient magnitude and health concepts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the gradient flow toolkit
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from gradient_flow import GradientFlowAnalyzer


# =============================================================================
# Step 1: Define a Simple MLP and Buggy Variations
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


class VanishingGradientMLP(nn.Module):
    """
    Designed to exhibit VANISHING gradients.

    Issues:
    - Deep network (8 layers)
    - Sigmoid activations (saturate easily)
    - Small weight initialization
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, output_dim: int = 10):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, output_dim)

        # Initialize with very small weights
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6, self.fc7, self.fc8]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        x = torch.sigmoid(self.fc7(x))
        x = self.fc8(x)
        return x


class ExplodingGradientMLP(nn.Module):
    """
    Designed to exhibit EXPLODING gradients.

    Issues:
    - Very large weight initialization
    - No normalization
    - Deep network without skip connections
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, output_dim: int = 10):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        # Initialize with very large weights
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(layer.weight, mean=0.0, std=5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No activation functions to allow gradients to explode
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class DeadLayerMLP(nn.Module):
    """
    Designed to exhibit DEAD layers.

    Issues:
    - Zero-initialized weights in fc2
    - Causes fc1 to also have zero gradients
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, output_dim: int = 10):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Zero out fc2 weights to create dead layer
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Always outputs zero
        x = self.fc3(x)
        return x


class UnstableGradientMLP(nn.Module):
    """
    Designed to exhibit UNSTABLE gradients.

    Issues:
    - Mix of large and small weights
    - No batch normalization
    - Inconsistent layer sizes causing gradient variance
    """

    def __init__(self, input_dim: int = 784, output_dim: int = 10):
        super().__init__()

        # Varying layer sizes
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 64)   # Bottleneck
        self.fc3 = nn.Linear(64, 512)   # Expansion
        self.fc4 = nn.Linear(512, output_dim)

        # Mix of initialization scales
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=2.0)  # Large
        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.01) # Small
        nn.init.normal_(self.fc4.weight, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class NumericalIssueMLP(nn.Module):
    """
    Designed to exhibit NUMERICAL issues (NaN/Inf).

    Issues:
    - Extremely large weights causing overflow
    - exp() operation on large values
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, output_dim: int = 10):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Extremely large weights
        nn.init.normal_(self.fc2.weight, mean=0.0, std=10.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # This will cause Inf for large values
        x = torch.exp(x)
        # This will cause NaN (Inf/Inf)
        x = x / (x + 1e-8)
        x = self.fc3(x)
        return x


class BottleneckMLP(nn.Module):
    """
    Designed to exhibit BOTTLENECK (strong gradient convergence).

    Issues:
    - Severe bottleneck layer (784 -> 8 -> 784)
    - Forces gradients to converge through narrow layer
    """

    def __init__(self, input_dim: int = 784, output_dim: int = 10):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 8)      # Extreme bottleneck
        self.fc3 = nn.Linear(8, 512)
        self.fc4 = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Bottleneck
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
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
# Step 3: Test Functions for Each Issue Type
# =============================================================================

def run_analysis(model, model_name, description, loss_fn):
    """Helper function to run gradient analysis."""
    print("\n" + "=" * 80)
    print(f"Testing: {model_name}")
    print("=" * 80)
    print(f"Description: {description}\n")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create analyzer with scientifically validated defaults
    #
    # Configuration rationale:
    # - enable_jacobian=False: Validation shows no detection benefit for feedforward networks
    # - enable_vector_analysis=False: Validation shows no detection benefit for standard pathologies
    #
    # This configuration provides 80% detection at minimal cost (26x faster than full analysis)
    # See: gradient_flow/validation/VALIDATION_RESULTS.md
    analyzer = GradientFlowAnalyzer(
        model,
        enable_rnn_analyzer=False,      # Not needed for MLPs - scalar metrics sufficient
        enable_circular_flow_analyser=False # Not needed for standard pathologies
    )

    # Define input and loss functions
    def input_fn():
        return torch.randn(32, 784)

    def loss_fn_wrapper(output):
        targets = torch.randint(0, 10, (32,))
        return loss_fn(output, targets)

    # Run analysis
    print("\nAnalyzing gradient propagation...\n")
    issues = analyzer.analyze(
        input_fn=input_fn,
        loss_fn=loss_fn_wrapper,
        steps=5
    )

    # Print results
    analyzer.print_summary(issues)

    if issues:
        print("\nDetailed Issues:")
        print("-" * 80)
        for issue in issues[:3]:  # Show first 3 issues
            print(f"\n{issue}")

    return issues


def test_vanishing_gradients():
    """Test VANISHING gradient detection."""
    model = VanishingGradientMLP()
    loss_fn = nn.CrossEntropyLoss()
    return run_analysis(
        model,
        "VanishingGradientMLP",
        "Deep network with sigmoid activations and small weights",
        loss_fn
    )


def test_exploding_gradients():
    """Test EXPLODING gradient detection."""
    model = ExplodingGradientMLP()
    loss_fn = nn.CrossEntropyLoss()
    return run_analysis(
        model,
        "ExplodingGradientMLP",
        "Large weight initialization without normalization",
        loss_fn
    )


def test_dead_layer():
    """Test DEAD layer detection."""
    model = DeadLayerMLP()
    loss_fn = nn.CrossEntropyLoss()
    return run_analysis(
        model,
        "DeadLayerMLP",
        "Network with zero-initialized layer (fc2)",
        loss_fn
    )


def test_unstable_gradients():
    """Test UNSTABLE gradient detection."""
    model = UnstableGradientMLP()
    loss_fn = nn.CrossEntropyLoss()
    return run_analysis(
        model,
        "UnstableGradientMLP",
        "Mixed initialization scales and varying layer sizes",
        loss_fn
    )


def test_numerical_issues():
    """Test NUMERICAL issue detection (NaN/Inf)."""
    model = NumericalIssueMLP()
    loss_fn = nn.CrossEntropyLoss()
    return run_analysis(
        model,
        "NumericalIssueMLP",
        "Extremely large weights with exp() operations",
        loss_fn
    )


def test_bottleneck():
    """Test BOTTLENECK detection."""
    model = BottleneckMLP()
    loss_fn = nn.CrossEntropyLoss()
    return run_analysis(
        model,
        "BottleneckMLP",
        "Severe bottleneck layer (512 -> 8 -> 512)",
        loss_fn
    )


# =============================================================================
# Step 4: Main Function - Test Healthy and Buggy Models
# =============================================================================

def main():
    print("=" * 80)
    print(" " * 20 + "GRADIENT FLOW ANALYSIS EXAMPLES")
    print("=" * 80)
    print("\nThis example demonstrates gradient issue detection across different")
    print("model architectures, each designed to exhibit specific pathologies.\n")

    # =============================================================================
    # Test 1: Healthy Model (Baseline)
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 1: HEALTHY MODEL (BASELINE)")
    print("=" * 80)

    model = SimpleMLP()
    loss_fn = nn.CrossEntropyLoss()
    healthy_issues = run_analysis(
        model,
        "SimpleMLP",
        "Well-initialized MLP with proper activations (ReLU)",
        loss_fn
    )

    # =============================================================================
    # Test 2-7: Problematic Models
    # =============================================================================
    print("\n\n" + "=" * 80)
    print("GRADIENT PATHOLOGY TESTS")
    print("=" * 80)
    print("\nTesting various gradient flow issues...")

    # Track results
    all_tests = []

    # Test VANISHING
    print("\n" + "#" * 80)
    print("# Issue Type: VANISHING GRADIENTS")
    print("#" * 80)
    vanishing_issues = test_vanishing_gradients()
    all_tests.append(("VANISHING", vanishing_issues))

    # Test EXPLODING
    print("\n" + "#" * 80)
    print("# Issue Type: EXPLODING GRADIENTS")
    print("#" * 80)
    exploding_issues = test_exploding_gradients()
    all_tests.append(("EXPLODING", exploding_issues))

    # Test DEAD
    print("\n" + "#" * 80)
    print("# Issue Type: DEAD LAYERS")
    print("#" * 80)
    dead_issues = test_dead_layer()
    all_tests.append(("DEAD", dead_issues))

    # Test UNSTABLE
    print("\n" + "#" * 80)
    print("# Issue Type: UNSTABLE GRADIENTS")
    print("#" * 80)
    unstable_issues = test_unstable_gradients()
    all_tests.append(("UNSTABLE", unstable_issues))

    # Test NUMERICAL
    print("\n" + "#" * 80)
    print("# Issue Type: NUMERICAL ISSUES")
    print("#" * 80)
    numerical_issues = test_numerical_issues()
    all_tests.append(("NUMERICAL", numerical_issues))

    # Test BOTTLENECK
    print("\n" + "#" * 80)
    print("# Issue Type: BOTTLENECK")
    print("#" * 80)
    bottleneck_issues = test_bottleneck()
    all_tests.append(("BOTTLENECK", bottleneck_issues))

    # =============================================================================
    # Summary of All Tests
    # =============================================================================
    print("\n\n" + "=" * 80)
    print("SUMMARY OF ALL TESTS")
    print("=" * 80)

    print("\nDetection Results:")
    print("-" * 80)
    print(f"{'Issue Type':<20} {'Detected':<12} {'Count':<10} {'Status'}")
    print("-" * 80)

    for issue_type, issues in all_tests:
        detected_types = {issue.issue_type.value for issue in issues}
        was_detected = issue_type in detected_types
        status = "[OK] DETECTED" if was_detected else "[FAIL] NOT DETECTED"
        print(f"{issue_type:<20} {'Yes' if was_detected else 'No':<12} {len(issues):<10} {status}")

    print("-" * 80)
    total_issues = sum(len(issues) for _, issues in all_tests)
    print(f"Total issues detected across all tests: {total_issues}")

if __name__ == "__main__":
    main()
