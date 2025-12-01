# Gradient Flow Toolkit

> **The gradient debugging tool and static analysis framework.**

Treat gradient flow through neural networks as fluid flow through pipes, enabling intuitive diagnosis of training pathologies like vanishing gradients, exploding activations, and bottlenecks.

## 🌊 Gradients!


| Concept | Fluid Metaphor | Technical Meaning | What It Tells You |
|---------|---------------|-------------------|-------------------|
| 💧 **Pressure** | Water pressure in pipes | Gradient magnitude | How much "force" is flowing through each layer |
| 🌊 **Turbulence** | Chaotic, unstable flow | Gradient variance | How stable the training is |
| ⚡ **Velocity** | Flow speed changes | Gradient change rate | How quickly patterns evolve |
| 🚫 **Blockage** | Clogged pipes | Vanishing gradients | Layers that can't learn |
| 💥 **Burst** | Over-pressurized pipes | Exploding gradients | Layers about to fail |
| 🔧 **Valve** | Flow control | Gradient scaling | Pressure relief at critical points |

## 🚀 Quick Start

```python
from gradient_flow import FlowAnalyzer

# Create analyzer
analyzer = FlowAnalyzer(model, "MyModel")

# Analyze gradient flow
report = analyzer.analyze(sample_input, sample_target, loss_fn)

# View results
report.print_summary()      # Console output with colors
report.save_html("report.html")  # Interactive HTML report
```

**Output:**
```
====================================================================
 GRADIENT FLOW ANALYSIS - MyModel
====================================================================

Summary:
  Total layers analyzed: 15
  Average health: 78.5%
  Healthy layers (>=75%): 12/15
  Issues found: 3 (1 critical, 2 high)
  Overall status: WARNING

====================================================================
Issues Detected (3 total)
====================================================================

Type         Severity   Layer                Details
----------------------------------------------------------------------
BOTTLENECK   CRITICAL   encoder.compress     Compression 16.0x with pressure 342.5
VANISHING    HIGH       lstm.weight_hh_l0    Temporal decay rate: -0.423
IMBALANCE    HIGH       attn.q_proj          Q/K/V pressure imbalance: ratio=12.3x
```

## 📦 Installation

```bash
pip install gradient-flow
```

Or from source:
```bash
git clone https://github.com/andrewjc/GradientFlow.git
cd gradient-flow
pip install -e .
```

## 🎯 Core Features

### 1. Universal Gradient Analysis

Works with any PyTorch model:
- MLPs, CNNs (StandardAnalyzer)
- RNNs, LSTMs, GRUs (TemporalAnalyzer)
- Transformers, BERT, GPT (AttentionAnalyzer)
- Custom architectures (FlowAnalyzer)

### 2. Intelligent Diagnostics

Automatically detects:
- ✅ Vanishing gradients
- ✅ Exploding gradients
- ✅ Compression bottlenecks
- ✅ Dead neurons
- ✅ Residual degradation
- ✅ Attention saturation

### 3. Actionable Recommendations

Not just detection - **solutions**:
```python
report.print_recommendations()
```
```
Recommendations:
  1. Add gradient scaling (scale=0.01) before encoder.compress
  2. Consider bottleneck architecture: 256→64→16 instead of 256→16
  3. Initialize forget gate bias to 1.0 for better LSTM memory
  4. Use Pre-Norm instead of Post-Norm for transformer stability
```

### 4. Comparative Analysis

Track gradient health evolution:
```python
from gradient_flow import ComparativeAnalyzer

comparator = ComparativeAnalyzer()

# Compare checkpoints
comparison = comparator.compare_checkpoints(
    model_factory=lambda: MyModel(),
    checkpoint_a="epoch_10.pt",
    checkpoint_b="epoch_50.pt",
    sample_input=x,
    sample_target=y,
    loss_fn=loss_fn
)

print(f"Improved layers: {len(comparison.improved_layers)}")
print(f"Degraded layers: {len(comparison.degraded_layers)}")
```

## 🛠️ The Toolkit

### Core Components

```python
from gradient_flow import (
    FlowAnalyzer,         # Main analysis engine
    gradient_scale,       # Gradient scaling function
    FlowMetrics,          # Per-layer metrics
    LayerHealth,          # Health assessment
)
```

### Specialized Analyzers

```python
from gradient_flow import (
    StandardAnalyzer,     # MLPs, CNNs
    TemporalAnalyzer,     # RNNs, LSTMs, GRUs
    AttentionAnalyzer,    # Transformers
    ComparativeAnalyzer,  # Checkpoint comparison
)
```

### Visualization

```python
from gradient_flow import (
    FlowReport,           # Central report class
    ConsoleReporter,      # ANSI colored console output
    HTMLReportGenerator,  # Interactive HTML reports
)
```

## 📚 Examples

Progressive tutorials from simple to complex:

1. **[Simple MLP](examples/01_simple_mlp.py)** - Basic feedforward analysis
2. **[NLP Embeddings](examples/02_nlp_embedding.py)** - Text classifiers with embeddings
3. **[RNN Sequences](examples/03_rnn_sequence.py)** - Temporal gradient flow
4. **[Memory Agent](examples/04_memory_agent.py)** - Complex RL agent with memory
5. **[Transformer](examples/05_transformer.py)** - Full transformer analysis

Run any example:
```bash
python examples/01_simple_mlp.py
```

## 🔧 Common Solutions

### 1. Exploding Gradients in Compression Layers

**Problem:** Layer compresses 256→16 dimensions, gradients explode to 1000+

**Solution:** Gradient scaling
```python
from gradient_flow import gradient_scale

class ImprovedLayer(nn.Module):
    def forward(self, x):
        # Scale gradients before compression
        x = gradient_scale(x, 0.01)  # Reduce backward pressure 100x
        z = self.compress(x)
        return z
```

### 2. Bottleneck Architecture

**Problem:** Single-step compression amplifies gradients

**Solution:** Staged compression
```python
# BAD: 16x amplification in one layer
self.compress = nn.Linear(256, 16)

# GOOD: Split into 4x + 4x = 16x total
self.compress_down = nn.Linear(256, 64)
self.compress_proj = nn.Linear(64, 16)
```

### 3. Vanishing Gradients in Deep Networks

**Problem:** Early layers receive near-zero gradients

**Solution:** Residual connections + Pre-Norm
```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Pre-Norm: normalize before sublayer
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

## 🧪 Real-World Example

Analyzing a memory-augmented RL agent that had exploding gradients:

```python
from gradient_flow import StandardAnalyzer

analyzer = StandardAnalyzer(bottleneck_threshold=0.1)
report = analyzer.analyze(model, obs, actions, loss_fn)

report.print_issues()
```

**Before:**
```
BOTTLENECK   CRITICAL   routing.compress     Max pressure: 22,458
```

**After applying gradient scaling:**
```
routing.compress_down    Max pressure: 45
routing.compress_proj    Max pressure: 170
```

**Result:** 130x improvement, stable training ✅

## 📖 Documentation

- **[Main Website](docs/index.html)** - Methodology and overview
- **[API Reference](docs/api.html)** - Complete API documentation
- **[Examples](examples/)** - Progressive tutorials

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Additional analyzers for specialized architectures
- Visualization improvements
- Performance optimizations
- Documentation and examples

**Made with ❤️ for the deep learning community**
