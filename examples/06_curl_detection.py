"""
    Example 6: Detecting Circular Gradient Flow with Curl Analysis
    ===============================================================

    This example demonstrates how to detect rotational/circular gradient patterns
    that cannot be detected with standard scalar analysis alone.

    Curl (∇×v) measures the rotational component of gradient flow:
    - High curl: Circular, oscillating patterns (adversarial training, competing objectives)
    - Low curl: Direct, conservative flow (standard supervised learning)

    Key concepts:
    - Adversarial architectures with competing objectives
    - Circular gradient flow detection
    - Vector field analysis vs scalar magnitude analysis
    - When curl analysis is valuable

    Model Design:
    We create a "cyclic network" where gradients flow in circular patterns due to
    competing objectives, simulating adversarial training or multi-task conflicts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from gradient_flow import GradientFlowAnalyzer


# =============================================================================
# Models with Circular Gradient Flow
# =============================================================================

class CyclicEncoder(nn.Module):
    """
    Encoder that creates circular dependencies through competing objectives.

    This simulates a GAN-like scenario where the encoder tries to:
    1. Maximize discriminability (push representations apart)
    2. Minimize reconstruction error (pull representations together)

    These competing objectives create rotational gradient flow.
    """

    def __init__(self, input_dim: int = 256, latent_dim: int = 64):
        super().__init__()

        # Encoding path
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Decoding path (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

        # Discriminator path (competing objective)
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Classification path (another competing objective)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x: torch.Tensor) -> dict:
        # Encode
        z = self.encoder(x)

        # Decode (reconstruction objective)
        x_recon = self.decoder(z)

        # Discriminate (adversarial objective - wants to maximize)
        disc_score = self.discriminator(z)

        # Classify (classification objective)
        class_logits = self.classifier(z)

        return {
            'latent': z,
            'reconstruction': x_recon,
            'discriminator': disc_score,
            'classification': class_logits
        }


class AdversarialPair(nn.Module):
    """
    Generator-Discriminator pair that creates true adversarial gradient flow.

    The generator and discriminator have opposing objectives, which creates
    circular gradient patterns that cannot be detected with scalar analysis alone.
    """

    def __init__(self, noise_dim: int = 64, data_dim: int = 256):
        super().__init__()

        self.noise_dim = noise_dim

        # Generator: noise -> fake data
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, data_dim),
            nn.Tanh()
        )

        # Discriminator: data -> real/fake score
        self.discriminator = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, real_data: torch.Tensor, noise: torch.Tensor = None) -> dict:
        batch_size = real_data.size(0)

        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=real_data.device)

        # Generate fake data
        fake_data = self.generator(noise)

        # Discriminator on real and fake
        real_score = self.discriminator(real_data)
        fake_score = self.discriminator(fake_data)

        return {
            'fake_data': fake_data,
            'real_score': real_score,
            'fake_score': fake_score
        }


class MultiTaskNetwork(nn.Module):
    """
    Multi-task network with competing objectives that create gradient conflicts.

    Different tasks pull gradients in different directions, creating rotational
    patterns in the shared representations.
    """

    def __init__(self, input_dim: int = 256):
        super().__init__()

        # Shared encoder
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Task 1: Classification (wants sparse, discriminative features)
        self.task1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        # Task 2: Reconstruction (wants dense, complete features)
        self.task2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

        # Task 3: Contrastive (wants normalized, metric features)
        self.task3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x: torch.Tensor) -> dict:
        shared_features = self.shared(x)

        return {
            'classification': self.task1(shared_features),
            'reconstruction': self.task2(shared_features),
            'embedding': self.task3(shared_features)
        }


# =============================================================================
# Model Wrapper for GradientFlowAnalyzer
# =============================================================================

class CyclicNetworkWrapper(nn.Module):
    """Wrapper to provide standard forward() method for analyzer."""

    def __init__(self, model, model_type: str):
        super().__init__()
        self.model = model
        self.model_type = model_type

    def forward(self, x: torch.Tensor) -> dict:
        if self.model_type == 'cyclic':
            return self.model(x)
        elif self.model_type == 'adversarial':
            # For adversarial, x should be real data
            return self.model(x)
        elif self.model_type == 'multitask':
            return self.model(x)
        else:
            return self.model(x)


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_cyclic_encoder():
    """Analyze the cyclic encoder with competing objectives."""
    print("EXAMPLE 1: Cyclic Encoder with Competing Objectives")

    print("\nModel Architecture:")
    print("  - Encoder: input -> latent (64D)")
    print("  - Decoder: latent -> reconstruction (reconstruction loss)")
    print("  - Discriminator: latent -> real/fake (adversarial loss)")
    print("  - Classifier: latent -> classes (classification loss)")
    print("  - Competing objectives create circular gradient flow")

    model = CyclicEncoder(input_dim=256, latent_dim=64)
    wrapped_model = CyclicNetworkWrapper(model, 'cyclic')

    analyzer = GradientFlowAnalyzer(
        wrapped_model,
        enable_rnn_analyzer=False,              # Not needed for curl
        enable_circular_flow_analyser=True     # ESSENTIAL for detecting circular flow
    )

    batch_size = 32
    input_dim = 256

    def input_fn():
        return torch.randn(batch_size, input_dim)

    def loss_fn(outputs):
        """
        Competing loss terms that create circular gradients:
        1. Reconstruction loss (pulls latent to preserve info)
        2. Adversarial loss (pulls latent to fool discriminator)
        3. Classification loss (pulls latent to be discriminative)
        """
        x = torch.randn(batch_size, input_dim)

        # Reconstruction loss (minimize)
        recon_loss = F.mse_loss(outputs['reconstruction'], x)

        # Adversarial loss (maximize discriminator confusion)
        # Discriminator wants to output 0.5 (uncertain)
        target_score = torch.ones_like(outputs['discriminator']) * 0.5
        adv_loss = F.mse_loss(outputs['discriminator'], target_score)

        # Classification loss
        targets = torch.randint(0, 10, (batch_size,))
        class_loss = F.cross_entropy(outputs['classification'], targets)

        # Conflicting weights create circular flow
        total_loss = recon_loss + 0.5 * adv_loss + 0.3 * class_loss

        return total_loss

    issues = analyzer.analyze(
        input_fn=input_fn,
        loss_fn=loss_fn,
        steps=20
    )

    # Print results
    analyzer.print_summary(issues)

    if issues:
        print("Detected Issues:")
        for issue in issues[:]:
            print(f"\n{issue}")

    return issues, analyzer


def analyze_adversarial_network():
    """Analyze true adversarial network (GAN-like)."""
    print("EXAMPLE 2: Adversarial Network (GAN-like)")

    print("\nModel Architecture:")
    print("  - Generator: noise -> fake data")
    print("  - Discriminator: data -> real/fake score")
    print("  - True adversarial objective creates strongest circular gradients")

    model = AdversarialPair(noise_dim=64, data_dim=256)
    wrapped_model = CyclicNetworkWrapper(model, 'adversarial')

    analyzer = GradientFlowAnalyzer(
        wrapped_model,
        enable_rnn_analyzer=False,
        enable_circular_flow_analyser=True
    )

    batch_size = 32
    data_dim = 256

    def input_fn():
        return torch.randn(batch_size, data_dim)

    def loss_fn(outputs):
        """
        Adversarial loss that creates circular flow:
        - Generator wants to maximize fake_score
        - Discriminator wants to minimize fake_score and maximize real_score
        """
        # Discriminator loss: real should be 1, fake should be 0
        real_labels = torch.ones_like(outputs['real_score'])
        fake_labels = torch.zeros_like(outputs['fake_score'])

        disc_loss = F.binary_cross_entropy(outputs['real_score'], real_labels) + \
                    F.binary_cross_entropy(outputs['fake_score'], fake_labels)

        # Generator loss: wants discriminator to think fake is real
        gen_labels = torch.ones_like(outputs['fake_score'])
        gen_loss = F.binary_cross_entropy(outputs['fake_score'], gen_labels)

        # Combined (creates circular gradient flow)
        total_loss = disc_loss + gen_loss

        return total_loss

    issues = analyzer.analyze(
        input_fn=input_fn,
        loss_fn=loss_fn,
        steps=20
    )

    analyzer.print_summary(issues)

    if issues:
        print("Detected Issues")
        for issue in issues[:]:
            print(f"\n{issue}")

    return issues, analyzer


def analyze_multitask_network():
    """Analyze multi-task network with conflicting objectives."""
    print("EXAMPLE 3: Multi-Task Network with Conflicting Objectives")

    print("\nModel Architecture:")
    print("  - Shared encoder: input -> shared features")
    print("  - Task 1: Classification (wants sparse features)")
    print("  - Task 2: Reconstruction (wants dense features)")
    print("  - Task 3: Contrastive (wants normalized features)")
    print("  - Conflicts in shared layers create rotational gradients")

    model = MultiTaskNetwork(input_dim=256)
    wrapped_model = CyclicNetworkWrapper(model, 'multitask')

    analyzer = GradientFlowAnalyzer(
        wrapped_model,
        enable_rnn_analyzer=False,
        enable_circular_flow_analyser=True
    )

    batch_size = 32
    input_dim = 256

    def input_fn():
        return torch.randn(batch_size, input_dim)

    def loss_fn(outputs):
        """
        Multi-task loss with competing objectives:
        - Classification wants discriminative features
        - Reconstruction wants complete information
        - Embedding wants metric structure
        """
        x = torch.randn(batch_size, input_dim)

        # Classification loss
        class_targets = torch.randint(0, 10, (batch_size,))
        class_loss = F.cross_entropy(outputs['classification'], class_targets)

        # Reconstruction loss
        recon_loss = F.mse_loss(outputs['reconstruction'], x)

        # Contrastive loss (simplified)
        embeddings = outputs['embedding']
        # Want embeddings to be unit norm (conflicts with other tasks)
        norm_loss = ((embeddings.norm(dim=1) - 1.0) ** 2).mean()

        # Weighted combination (different weights create conflicts)
        total_loss = 1.0 * class_loss + 0.8 * recon_loss + 0.6 * norm_loss

        return total_loss

    issues = analyzer.analyze(
        input_fn=input_fn,
        loss_fn=loss_fn,
        steps=20
    )

    analyzer.print_summary(issues)

    if issues:
        print("Detected Issues:")
        for issue in issues[:]:
            print(f"\n{issue}")

    return issues, analyzer


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*80)
    print("Example 6: Detecting Circular Gradient Flow with Curl Analysis")
    print("="*80)

    print("\n" + "="*80)
    print("OVERVIEW: When Curl Analysis Matters")
    print("="*80)
    print("""
Curl (curl v) detects rotational/circular patterns in gradient flow that
scalar magnitude analysis cannot detect.

When to use vector analysis (curl + divergence):
1. Adversarial training (GANs, adversarial robustness)
2. Multi-task learning with competing objectives
3. Networks with cyclic dependencies
4. Any training exhibiting oscillating behavior

When scalar analysis is sufficient:
1. Standard supervised learning
2. Single-objective optimization
3. Feedforward networks with clear objectives

This example demonstrates three scenarios where curl analysis is essential.
""")

    # Run all examples
    print("\n" + "="*80)
    print("Running Examples...")
    print("="*80)

    # Example 1: Cyclic Encoder
    analyze_cyclic_encoder()

    # Example 2: Adversarial Network
    analyze_adversarial_network()

    # Example 3: Multi-task Network
    analyze_multitask_network()

    # Final insights
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. Curl Analysis Detects:
   - Circular/rotational gradient patterns
   - Adversarial conflicts (generator vs discriminator)
   - Multi-task conflicts in shared layers
   - Oscillating training dynamics

2. When to Enable Vector Analysis:
   - GANs and adversarial training
   - Multi-task learning
   - Meta-learning with inner/outer loops
   - Any oscillating or unstable training

3. Performance Consideration:
   - Vector analysis adds ~3x computational cost
   - Only enable when circular flow is expected
   - Standard supervised learning doesn't need it

4. Curl vs Divergence:
   - Curl: Measures rotation (circular patterns)
   - Divergence: Measures expansion/contraction (bottlenecks)
   - Both provide complementary information
""")

    print("\nDone!")


if __name__ == "__main__":
    main()
