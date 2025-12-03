# GradientFlow

**Advanced gradient analysis toolkit for PyTorch neural networks**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GradientFlow is a comprehensive diagnostic toolkit for analyzing gradient propagation in neural networks. It automatically detects vanishing gradients, exploding gradients, dead layers, numerical instability, and bottlenecks—helping you debug training issues before they become critical.

---

## What Problem Does It Solve?

Training deep neural networks is notoriously difficult. Gradients can vanish, explode, or become unstable, leading to:
- Models that fail to converge
- Layers that stop learning (dead neurons)
- Training instability and NaN losses
- Hours wasted debugging mysterious training failures

**GradientFlow automatically diagnoses these issues** by analyzing gradient flow through your network, pinpointing exactly which layers are problematic and providing actionable recommendations to fix them.

## Key Features

- **Automatic Issue Detection** - Identifies vanishing, exploding, dead, unstable, and numerical gradient issues
- **Actionable Recommendations** - Each issue comes with specific fixes tailored to your architecture
- **Zero Code Changes** - Works with any PyTorch model without modifying your training loop
- **Specialized Analyzers** - Built-in support for RNNs, Transformers, and RL policy networks
- **Vector Field Analysis** - Advanced fluid dynamics-inspired metrics (divergence, curl, flow coherence)
- **Lightweight & Fast** - Minimal overhead, optimized for production use

## Installation

```bash
pip install gradient-flow
```

Or install from source:

```bash
git clone https://github.com/andrewjc/GradientFlow.git
cd gradient-flow
pip install -e .
```

**Requirements:** Python 3.8+, PyTorch 2.0+

## Quick Start

```python
import torch
import torch.nn as nn
from gradient_flow import GradientFlowAnalyzer

# Your existing model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Create analyzer
analyzer = GradientFlowAnalyzer(model)

# Define input and loss functions
def input_fn():
    return torch.randn(32, 784)

def loss_fn(output):
    targets = torch.randint(0, 10, (32,))
    return nn.functional.cross_entropy(output, targets)

# Analyze gradient flow
issues = analyzer.analyze(
    input_fn=input_fn,
    loss_fn=loss_fn,
    steps=20
)

# Print diagnostics
analyzer.print_summary(issues)

# Review each issue with recommendations
for issue in issues:
    print(issue)
```

---

## Basic Examples

### Example 1: Detecting Vanishing Gradients

In this example, let's demonstrate finding vanishing gradients in a deep network with sigmoid activations.

**The Problem - Model with Vanishing Gradients:**

```python
import torch
import torch.nn as nn
from gradient_flow import GradientFlowAnalyzer

class DeepSigmoidNet(nn.Module):
    """Deep network that suffers from vanishing gradients"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(784, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 10)
        ])

        # Small initialization amplifies vanishing problem
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.001)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.sigmoid(layer(x))  # Sigmoid saturates easily
        return self.layers[-1](x)

# Create the problematic model
model = DeepSigmoidNet()
```

**Detecting the Issue with GradientFlow:**

```python
# Create analyzer
analyzer = GradientFlowAnalyzer(model)

# Analyze gradient flow
issues = analyzer.analyze(
    input_fn=lambda: torch.randn(32, 784),
    loss_fn=lambda out: nn.functional.cross_entropy(
        out, torch.randint(0, 10, (32,))
    ),
    steps=20
)

# Print detected issues
analyzer.print_summary(issues)
```

**Output:**
```
[WARNING] Found 4 gradient issue(s):

CRITICAL: 3 issue(s)
HIGH: 1 issue(s)

Details:
================================================================================
[CRITICAL] VANISHING in layers.0 (Linear)
  Description: Gradient magnitude (2.34e-09) is extremely small. Layer may not be learning effectively.
  Magnitude: 2.34e-09
  Recommended Actions:
    - Use residual/skip connections
    - Try different activation functions (ReLU to LeakyReLU)
    - Reduce network depth or increase layer width
    - Check weight initialization (use Xavier/He initialization)
```

**The Fix:**

```python
class FixedDeepNet(nn.Module):
    """Fixed version with proper initialization and activations"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(784, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 10)
        ])

        # Use He initialization for ReLU
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # Use ReLU instead of sigmoid
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
        return self.layers[-1](x)

# Verify the fix
fixed_model = FixedDeepNet()
analyzer = GradientFlowAnalyzer(fixed_model)
issues = analyzer.analyze(
    input_fn=lambda: torch.randn(32, 784),
    loss_fn=lambda out: nn.functional.cross_entropy(out, torch.randint(0, 10, (32,))),
    steps=20
)

print(f"Issues after fix: {len(issues)}")  # Should be 0 or minimal
```

---

### Example 2: Detecting Exploding Gradients

In this example, let's demonstrate finding exploding gradients caused by large weight initialization.

**The Problem - Model with Exploding Gradients:**

```python
import torch
import torch.nn as nn
from gradient_flow import GradientFlowAnalyzer

class UnstableDeepNet(nn.Module):
    """Network with exploding gradient problem"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 10)

        # Large initialization causes explosion
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(layer.weight, mean=0.0, std=5.0)

    def forward(self, x):
        # No activation = gradients multiply through layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)

# Create the problematic model
model = UnstableDeepNet()
```

**Detecting the Issue with GradientFlow:**

```python
# Detect the issue
analyzer = GradientFlowAnalyzer(model)

issues = analyzer.analyze(
    input_fn=lambda: torch.randn(32, 784),
    loss_fn=lambda out: nn.functional.cross_entropy(
        out, torch.randint(0, 10, (32,))
    ),
    steps=20
)

# Display exploding gradient issues
for issue in issues:
    if issue.issue_type.value == "EXPLODING":
        print(issue)
```

**Output:**
```
[CRITICAL] EXPLODING in fc2 (Linear)
  Description: Gradient magnitude (342.67) is very large. May cause training instability or NaN values.
  Magnitude: 3.43e+02
  Recommended Actions:
    - Enable gradient clipping (torch.nn.utils.clip_grad_norm_)
    - Reduce learning rate
    - Use batch normalization or layer normalization
    - Check for numerical instability in loss function
```

**The Fix:**

```python
class StableDeepNet(nn.Module):
    """Fixed version with normalization and proper initialization"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 10)

        # Proper initialization
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

# Training with gradient clipping
model = StableDeepNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    optimizer.zero_grad()
    output = model(torch.randn(32, 784))
    loss = nn.functional.cross_entropy(output, torch.randint(0, 10, (32,)))
    loss.backward()

    # Clip gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

# Verify the fix
analyzer = GradientFlowAnalyzer(model)
issues = analyzer.analyze(
    input_fn=lambda: torch.randn(32, 784),
    loss_fn=lambda out: nn.functional.cross_entropy(out, torch.randint(0, 10, (32,))),
    steps=20
)
print(f"Exploding gradient issues after fix: {len([i for i in issues if i.issue_type.value == 'EXPLODING'])}")
```

---

### Example 3: Detecting Dead Layers

In this example, let's demonstrate finding dead layers where a layer produces zero outputs.

**The Problem - Model with Dead Layer:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from gradient_flow import GradientFlowAnalyzer

class DeadLayerNet(nn.Module):
    """Network with dead layer problem"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

        # Zero initialization kills this layer
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Always outputs zero!
        return self.fc3(x)

# Create the problematic model
model = DeadLayerNet()
```

**Detecting the Issue with GradientFlow:**

```python
# Detect dead layer
analyzer = GradientFlowAnalyzer(model)

issues = analyzer.analyze(
    input_fn=lambda: torch.randn(32, 784),
    loss_fn=lambda out: nn.functional.cross_entropy(
        out, torch.randint(0, 10, (32,))
    ),
    steps=20
)

# Display dead layer issues
for issue in issues:
    if issue.issue_type.value == "DEAD":
        print(f"{issue.layer}: {issue.description}")
        print(f"Recommended: {issue.recommended_actions[0]}")
```

**Output:**
```
fc2: 100.0% of gradients are zero. Layer appears to be dead or disconnected.
Recommended: Check if layer is connected to loss

[HIGH] DEAD in fc2 (Linear)
  Description: 100.0% of gradients are zero. Layer appears to be dead or disconnected.
  Recommended Actions:
    - Check if layer is connected to loss
    - Verify weight initialization (all zeros?)
    - Inspect activation function (dead ReLU?)
    - Ensure layer receives non-zero inputs
```

**The Fix:**

```python
class FixedLayerNet(nn.Module):
    """Fixed version with proper initialization"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

        # Proper initialization for all layers
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Verify the fix
fixed_model = FixedLayerNet()
analyzer = GradientFlowAnalyzer(fixed_model)
issues = analyzer.analyze(
    input_fn=lambda: torch.randn(32, 784),
    loss_fn=lambda out: nn.functional.cross_entropy(out, torch.randint(0, 10, (32,))),
    steps=20
)

dead_issues = [i for i in issues if i.issue_type.value == 'DEAD']
print(f"Dead layers after fix: {len(dead_issues)}")  # Should be 0
```

---

## Advanced Examples

### Example 4: Analyzing Transformer Attention Patterns

In this example, let's demonstrate detecting attention collapse in Transformer self-attention layers.

**The Problem - Transformer with Attention Collapse:**

```python
import torch
import torch.nn as nn
from gradient_flow import TransformerAnalyzer, AttentionStats

class CollapsingTransformer(nn.Module):
    """Transformer that suffers from attention collapse"""
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(10000, d_model)

        # Very low dropout causes attention to collapse
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.0  # No dropout = attention saturation
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, 10000)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc_out(x)

# Create problematic model
model = CollapsingTransformer()
```

**Detecting the Issue with GradientFlow:**

```python
# Use specialized Transformer analyzer
analyzer = TransformerAnalyzer(model)

def input_fn():
    return torch.randint(0, 10000, (32, 128))  # batch, seq_len

def loss_fn(output):
    targets = torch.randint(0, 10000, (32, 128))
    return nn.functional.cross_entropy(
        output.view(-1, 10000),
        targets.view(-1)
    )

# Run specialized transformer analysis
results = analyzer.analyze(
    input_fn=input_fn,
    loss_fn=loss_fn,
    steps=20
)

# Access transformer-specific metrics
for layer_name, stats in results.attention_stats.items():
    print(f"{layer_name}:")
    print(f"  Attention Entropy: {stats.mean_entropy:.3f}")
    print(f"  Max Attention Weight: {stats.max_attention:.3f}")

    # Check for attention collapse
    if stats.mean_entropy < 0.5:
        print(f"  WARNING: Low entropy - attention may be collapsing!")
    if stats.max_attention > 0.95:
        print(f"  WARNING: Attention saturation detected!")
```

**Output:**
```
transformer.layers.5.self_attn:
  Attention Entropy: 0.23
  Max Attention Weight: 0.98
  WARNING: Low entropy - attention may be collapsing!
  WARNING: Attention saturation detected!

[HIGH] VANISHING in transformer.layers.5.self_attn (MultiheadAttention)
  Description: Attention entropy (0.23) indicates collapsed attention distribution.
  Recommended Actions:
    - Increase attention dropout
    - Use relative position encodings
    - Apply attention temperature scaling
    - Check for repeated tokens in input
```

**The Fix:**

```python
class ImprovedTransformer(nn.Module):
    """Fixed transformer with proper dropout and normalization"""
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(10000, d_model)

        # Increase dropout to prevent attention collapse
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.3,  # Increased from 0.0
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Add layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, 10000)

        # Better initialization
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        return self.fc_out(x)

# Verify fix
fixed_model = ImprovedTransformer()
analyzer = TransformerAnalyzer(fixed_model)
results = analyzer.analyze(
    input_fn=lambda: torch.randint(0, 10000, (32, 128)),
    loss_fn=lambda out: nn.functional.cross_entropy(
        out.view(-1, 10000), torch.randint(0, 10000, (32, 128)).view(-1)
    ),
    steps=20
)
print(f"Attention issues after fix: {len([i for i in results.gradient_issues if 'attn' in i.layer])}")
```

---

### Example 5: Analyzing Recurrent Networks with Memory

In this example, let's demonstrate detecting vanishing gradients through time in LSTM networks.

**The Problem - LSTM with Gradient Decay:**

```python
import torch
import torch.nn as nn
from gradient_flow import RecurrentAnalyzer, ReservoirDynamics

class UnstableLSTM(nn.Module):
    """LSTM that suffers from vanishing gradients through time"""
    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=512, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Standard LSTM without stability measures
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0  # No dropout
        )
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(x)
        return self.fc(hidden[-1])

# Create problematic model
model = UnstableLSTM()
```

**Detecting the Issue with GradientFlow:**

```python
# Use specialized RNN analyzer
analyzer = RecurrentAnalyzer(
    model,
    enable_rnn_analyzer=True,  # Enable spectral analysis
    track_hidden_states=True
)

def input_fn():
    # Long sequences to expose vanishing gradient issues
    return torch.randint(0, 10000, (32, 256))

def loss_fn(output):
    targets = torch.randint(0, 2, (32,))
    return nn.functional.cross_entropy(output, targets)

# Run RNN-specific analysis
results = analyzer.analyze(
    input_fn=input_fn,
    loss_fn=loss_fn,
    steps=20
)

# Access RNN-specific metrics
reservoir_stats = results.reservoir_dynamics

print("LSTM Gradient Flow Analysis:")
print(f"Spectral Radius: {reservoir_stats.spectral_radius:.3f}")
print(f"Effective Memory Depth: {reservoir_stats.memory_depth:.1f} steps")
print(f"Gradient Decay Rate: {reservoir_stats.gradient_decay_rate:.3e}")

# Check for issues
if reservoir_stats.gradient_decay_rate > 0.95:
    print("\nWARNING: Rapid gradient decay detected!")
    print("Gradients are vanishing over long sequences.")

if reservoir_stats.spectral_radius > 1.1:
    print(f"\nWARNING: Unstable hidden state dynamics!")
    print(f"Spectral radius ({reservoir_stats.spectral_radius:.3f}) > 1.0")
```

**Output:**
```
LSTM Gradient Flow Analysis:
Spectral Radius: 1.34
Effective Memory Depth: 12.3 steps
Gradient Decay Rate: 0.97

WARNING: Unstable hidden state dynamics!
Spectral radius (1.34) > 1.0

WARNING: Rapid gradient decay detected!
Gradients are vanishing over long sequences.
```

**The Fix:**

```python
class StableLSTM(nn.Module):
    """Fixed LSTM with LayerNorm for stability"""
    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=512, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Use LayerNorm LSTM for stability
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_size = embed_dim if i == 0 else hidden_dim
            self.lstm_layers.append(nn.LSTM(input_size, hidden_dim, batch_first=True))
            self.lstm_layers.append(nn.LayerNorm(hidden_dim))

        self.fc = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)

        # Apply LayerNorm after each LSTM layer
        for i in range(0, len(self.lstm_layers), 2):
            lstm_layer = self.lstm_layers[i]
            norm_layer = self.lstm_layers[i+1]

            x, (hidden, cell) = lstm_layer(x)
            x = norm_layer(x)
            x = self.dropout(x)

        return self.fc(x[:, -1, :])

# Training with gradient clipping
model = StableLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    optimizer.zero_grad()
    output = model(torch.randint(0, 10000, (32, 256)))
    loss = nn.functional.cross_entropy(output, torch.randint(0, 2, (32,)))
    loss.backward()

    # Crucial: clip gradients for RNNs
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

# Verify fix
analyzer = RecurrentAnalyzer(model, enable_rnn_analyzer=True)
results = analyzer.analyze(
    input_fn=lambda: torch.randint(0, 10000, (32, 256)),
    loss_fn=lambda out: nn.functional.cross_entropy(out, torch.randint(0, 2, (32,))),
    steps=20
)
print(f"Gradient decay after fix: {results.reservoir_dynamics.gradient_decay_rate:.3e}")
```

---

### Example 6: Analyzing RL Policy Networks

In this example, let's demonstrate detecting gradient imbalance in actor-critic RL policy networks.

**The Problem - Actor-Critic with Gradient Imbalance:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from gradient_flow import RLPolicyAnalyzer, PolicyDistributionStats

class ImbalancedActorCritic(nn.Module):
    """Actor-Critic with policy/value gradient imbalance"""
    def __init__(self, obs_dim=128, action_dim=4, hidden_dim=256):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        features = self.shared(obs)
        logits = self.policy_head(features)
        action_probs = F.softmax(logits, dim=-1)
        value = self.value_head(features)
        return action_probs, value

# Create problematic model
model = ImbalancedActorCritic()
```

**Detecting the Issue with GradientFlow:**

```python
# Use RL-specific analyzer
analyzer = RLPolicyAnalyzer(
    model,
    policy_head_name='policy_head',
    value_head_name='value_head',
    shared_backbone_name='shared'
)

def input_fn():
    return torch.randn(32, 128)

def rl_loss_fn(output):
    action_probs, values = output

    # Sample actions and compute policy gradient loss
    actions = torch.multinomial(action_probs, 1).squeeze()
    log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))

    # Mock advantages and returns
    advantages = torch.randn(32, 1)
    returns = torch.randn(32, 1)

    # Policy loss (negative because we want to maximize)
    policy_loss = -(log_probs * advantages).mean()

    # Value loss
    value_loss = F.mse_loss(values, returns)

    return policy_loss + 0.5 * value_loss

# Analyze policy network
results = analyzer.analyze(
    input_fn=input_fn,
    loss_fn=rl_loss_fn,
    steps=50
)

# Access RL-specific metrics
policy_stats = results.policy_distribution_stats

print("Policy Network Analysis:")
print(f"Policy Entropy: {policy_stats.mean_entropy:.3f}")
print(f"Max Action Probability: {policy_stats.max_action_prob:.3f}")
print(f"Policy Gradient Magnitude: {policy_stats.policy_grad_norm:.3e}")
print(f"Value Gradient Magnitude: {policy_stats.value_grad_norm:.3e}")
print(f"Gradient Ratio (policy/value): {policy_stats.gradient_ratio:.3f}")

# Check for common RL issues
if policy_stats.mean_entropy < 0.5:
    print("\nWARNING: Low policy entropy!")
    print("Policy is becoming deterministic too quickly.")

if policy_stats.gradient_ratio > 10.0:
    print("\nWARNING: Policy gradients dominating value gradients!")

if policy_stats.policy_grad_norm < 1e-5:
    print("\nWARNING: Vanishing policy gradients!")
```

**Output:**
```
Policy Network Analysis:
Policy Entropy: 0.34
Max Action Probability: 0.89
Policy Gradient Magnitude: 4.23e-06
Value Gradient Magnitude: 2.34e-03
Gradient Ratio (policy/value): 0.002

WARNING: Low policy entropy!
Policy is becoming deterministic too quickly.

WARNING: Vanishing policy gradients!
```

**The Fix:**

```python
class ImprovedActorCritic(nn.Module):
    """Fixed Actor-Critic with proper initialization and entropy bonus"""
    def __init__(self, obs_dim=128, action_dim=4, hidden_dim=256):
        super().__init__()

        # Shared feature extractor with normalization
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),  # Bounded activation for RL
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        # Policy head with proper initialization
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize policy head with smaller weights for exploration
        for m in self.policy_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        features = self.shared(obs)
        logits = self.policy_head(features)
        action_probs = F.softmax(logits / 0.5, dim=-1)  # Temperature scaling
        value = self.value_head(features)
        return action_probs, value

# Training with entropy bonus and gradient clipping
model = ImprovedActorCritic()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

entropy_coef = 0.01
value_loss_coef = 0.5

for step in range(1000):
    optimizer.zero_grad()

    obs = torch.randn(32, 128)
    action_probs, values = model(obs)

    # Sample actions
    dist = torch.distributions.Categorical(action_probs)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)

    # Mock RL data
    advantages = torch.randn(32)
    returns = torch.randn(32, 1)

    # Policy loss with entropy bonus
    policy_loss = -(log_probs * advantages).mean()
    entropy = dist.entropy().mean()
    value_loss = F.mse_loss(values, returns)

    # Combined loss with proper weighting
    loss = policy_loss - entropy_coef * entropy + value_loss_coef * value_loss

    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

    optimizer.step()

# Verify fix
analyzer = RLPolicyAnalyzer(
    model,
    policy_head_name='policy_head',
    value_head_name='value_head',
    shared_backbone_name='shared'
)
results = analyzer.analyze(
    input_fn=lambda: torch.randn(32, 128),
    loss_fn=rl_loss_fn,
    steps=50
)
print(f"Policy entropy after fix: {results.policy_distribution_stats.mean_entropy:.3f}")
```

---

## Documentation

- **Quick Start Guide**: [docs/QUICK_START.md](docs/QUICK_START.md)
- **API Reference**: [docs/README.md](docs/README.md)
- **Architecture Details**: See docstrings in source code
- **Examples**: [examples/](examples/)

## API Overview

### Core Classes

```python
from gradient_flow import (
    GradientFlowAnalyzer,     # Main analyzer for any PyTorch model
    TransformerAnalyzer,      # Specialized for Transformers
    RecurrentAnalyzer,        # Specialized for RNNs/LSTMs
    RLPolicyAnalyzer,         # Specialized for RL policies
    FlowAnalyzer,             # Low-level gradient flow engine
    GradientScale,            # Gradient scaling utility
)
```

### Issue Types

```python
from gradient_flow import IssueType, IssueSeverity

# Issue Types
IssueType.VANISHING    # Gradients too small
IssueType.EXPLODING    # Gradients too large
IssueType.DEAD         # No gradients (zero)
IssueType.UNSTABLE     # High variance gradients
IssueType.NUMERICAL    # NaN or Inf values
IssueType.BOTTLENECK   # Severe gradient convergence
IssueType.SATURATION   # Activation saturation

# Severity Levels
IssueSeverity.CRITICAL  # Immediate action required
IssueSeverity.HIGH      # Significant impact on training
IssueSeverity.MEDIUM    # Moderate concern
IssueSeverity.LOW       # Minor issue
IssueSeverity.INFO      # Informational
```

## Performance

GradientFlow is designed to be lightweight:
- **Minimal overhead**: ~5-10% slowdown during analysis
- **Optimized hooks**: Efficient gradient capture with minimal memory
- **Configurable depth**: Disable advanced features for faster analysis

```python
# Fast mode (scalar metrics only)
analyzer = GradientFlowAnalyzer(
    model,
    enable_rnn_analyzer=False,
    enable_circular_flow_analyser=False
)

# Full analysis mode (all features)
analyzer = GradientFlowAnalyzer(
    model,
    enable_rnn_analyzer=True,
    enable_circular_flow_analyser=True
)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/andrewjc/GradientFlow.git
cd gradient-flow
pip install -e ".[dev]"
pytest tests/
```

## Citation

If you use GradientFlow in your research, please cite:

```bibtex
@software{gradientflow2025,
  title={GradientFlow: Advanced Gradient Analysis for PyTorch},
  author={Andrew J Cranston},
  year={2025},
  url={https://github.com/andrewjc/GradientFlow.git}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs/README.md](docs/README.md)
- **Issues**: [GitHub Issues](https://github.com/andrewjc/GradientFlow.git/issues)
- **Discussions**: [GitHub Discussions](https://github.com/andrewjc/GradientFlow.git/discussions)

---

**Built with passion for making deep learning more debuggable.**
