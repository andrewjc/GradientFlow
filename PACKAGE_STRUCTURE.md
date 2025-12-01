# Gradient Flow Toolkit - Package Structure

Complete file tree and module organization for the Gradient Flow Toolkit.

## 📁 Directory Structure

```
gradient_flow/
│
├── __init__.py                    # Main package entry point
├── README.md                      # Package documentation
├── PACKAGE_STRUCTURE.md          # This file
│
├── core/                          # Core analysis engine
│   ├── __init__.py
│   ├── engine.py                 # FlowAnalyzer, GradientScale
│   ├── metrics.py                # FlowMetrics, LayerHealth
│   └── hooks.py                  # HookManager, hook factories
│
├── analyzers/                     # Specialized analyzers
│   ├── __init__.py
│   ├── standard.py               # StandardAnalyzer (MLPs, CNNs)
│   ├── temporal.py               # TemporalAnalyzer (RNNs, LSTMs)
│   ├── attention.py              # AttentionAnalyzer (Transformers)
│   └── comparative.py            # ComparativeAnalyzer (checkpoints)
│
├── visualizers/                   # Output and reporting
│   ├── __init__.py
│   ├── report.py                 # FlowReport central class
│   ├── console.py                # ConsoleReporter (ANSI colors)
│   └── html.py                   # HTMLReportGenerator
│
├── examples/                      # Progressive tutorials
│   ├── __init__.py
│   ├── 01_simple_mlp.py          # Basic MLP analysis
│   ├── 02_nlp_embedding.py       # NLP with embeddings
│   ├── 03_rnn_sequence.py        # RNN temporal flow
│   ├── 04_memory_agent.py        # Memory-augmented RL agent
│   └── 05_transformer.py         # Full transformer analysis
│
└── docs/                          # Documentation website
    ├── index.html                 # Main landing page
    └── api.html                   # API reference
```

## 📦 Module Overview

### Core Modules

#### `core/engine.py`
- **GradientScale** - Autograd function for gradient scaling
- **gradient_scale()** - Functional interface for gradient scaling
- **FlowAnalyzer** - Main analysis engine
  - Hooks into model layers
  - Collects gradient statistics
  - Generates FlowReport

#### `core/metrics.py`
- **FlowMetrics** - Per-layer gradient metrics
  - mean_pressure (gradient magnitude)
  - max_pressure
  - turbulence (variance)
  - velocity (change rate)
- **LayerHealth** - Health assessment
  - score (0-100)
  - status (HEALTHY/WARNING/CRITICAL)
  - issues
  - recommendations

#### `core/hooks.py`
- **HookManager** - Manages PyTorch hooks
- **BackwardHook** - Factories for gradient collection hooks
- **ForwardHook** - Factories for activation hooks

### Specialized Analyzers

#### `analyzers/standard.py`
**StandardAnalyzer** - For feedforward networks
- Depth analysis (gradient decay through layers)
- Bottleneck detection (compression layers)
- Layer type statistics
- Methods:
  - `analyze()` - Run analysis
  - `get_layer_type_stats()` - Stats by layer type
  - `get_depth_profile()` - Pressure by depth

#### `analyzers/temporal.py`
**TemporalAnalyzer** - For recurrent networks
- Temporal decay analysis (gradient flow through time)
- Effective memory length calculation
- Gate-specific analysis (LSTM/GRU)
- Methods:
  - `analyze()` - Run analysis
  - `analyze_unrolled()` - Custom unrolling
  - `get_temporal_metrics()` - Per-layer temporal stats
  - `get_timestep_profile()` - Pressure at each timestep
  - `get_effective_memory()` - Memory length per layer

#### `analyzers/attention.py`
**AttentionAnalyzer** - For transformers
- Query/Key/Value gradient tracking
- Residual connection health
- Multi-head attention balance
- FFN bottleneck detection
- Methods:
  - `analyze()` - Run analysis
  - `get_transformer_metrics()` - Overall metrics
  - `get_layer_metrics()` - Specific layer metrics
  - `get_attention_profile()` - Q/K/V across layers

#### `analyzers/comparative.py`
**ComparativeAnalyzer** - For comparing models/checkpoints
- Layer-by-layer comparison
- Training evolution tracking
- Improvement/degradation detection
- Methods:
  - `analyze_model()` - Analyze and store
  - `compare()` - Compare two analyses
  - `compare_models()` - Compare two models
  - `compare_checkpoints()` - Compare checkpoint files
  - `track_training()` - Track training step
  - `get_health_evolution()` - Health over time
  - `get_pressure_evolution()` - Pressure over time

### Visualization

#### `visualizers/report.py`
**FlowReport** - Central report container
- Aggregates all analysis results
- Provides multiple output formats
- Query methods for specific data
- Methods:
  - Display: `print_summary()`, `print_issues()`, `print_layers()`
  - Export: `save_html()`, `save_json()`, `save_text()`
  - Query: `get_worst_layers()`, `get_issues_by_severity()`

#### `visualizers/console.py`
**ConsoleReporter** - ANSI colored console output
- Beautiful terminal formatting
- Color-coded health scores
- Severity highlighting
- Methods:
  - `format_report()` - Format as string
  - `print_summary()` - Print summary section
  - `print_layer_summary()` - Print layer details
  - `print_issues()` - Print issues
  - `print_recommendations()` - Print fixes

#### `visualizers/html.py`
**HTMLReportGenerator** - Interactive HTML reports
- Sortable tables
- Gradient visualizations
- Issue highlighting
- Responsive design
- Methods:
  - `generate()` - Generate full HTML report

## 🎓 Examples

Progressive tutorial series from simple to complex:

### 01_simple_mlp.py
**Concepts:** Basic MLP, pressure, health scores
- Define a 4-layer MLP
- Run basic gradient flow analysis
- Understand pressure and health metrics
- Export HTML and JSON reports

### 02_nlp_embedding.py
**Concepts:** Embeddings, sparse gradients, layer type analysis
- Text classifier with embedding layer
- StandardAnalyzer with depth analysis
- Layer type statistics
- Depth profile visualization
- Embedding vs dense layer comparison

### 03_rnn_sequence.py
**Concepts:** Temporal flow, vanishing through time
- Compare Vanilla RNN, LSTM, GRU
- TemporalAnalyzer for sequence analysis
- Temporal decay rate calculation
- Effective memory measurement
- Gate-level diagnostics

### 04_memory_agent.py
**Concepts:** Real-world debugging, compression bottlenecks
- Memory-augmented RL agent
- Identifying exploding gradients in compression
- Applying gradient_scale() fix
- Bottleneck vs improved architecture
- Comparative analysis before/after

### 05_transformer.py
**Concepts:** Attention mechanisms, residual health
- Full transformer encoder
- AttentionAnalyzer for Q/K/V tracking
- Multi-head attention balance
- Residual connection degradation
- Pre-Norm vs Post-Norm comparison

## 📄 Documentation

### docs/index.html
Landing page with:
- Methodology explanation
- Fluid dynamics metaphors
- Quick start guide
- Diagnosis workflow
- Common solutions
- Example links

### docs/api.html
Complete API reference:
- Core components
- All analyzers
- Visualizers
- Parameter documentation
- Code examples
- Method signatures

## 🚀 Usage Patterns

### Basic Analysis
```python
from gradient_flow import FlowAnalyzer

analyzer = FlowAnalyzer(model)
report = analyzer.analyze(x, y, loss_fn)
report.print_summary()
```

### Architecture-Specific Analysis
```python
from gradient_flow import StandardAnalyzer, TemporalAnalyzer, AttentionAnalyzer

# MLPs, CNNs
standard = StandardAnalyzer()
report = standard.analyze(cnn_model, x, y, loss_fn)

# RNNs, LSTMs
temporal = TemporalAnalyzer(sequence_length=100)
report = temporal.analyze(rnn_model, seqs, targets, loss_fn)

# Transformers
attention = AttentionAnalyzer(num_heads=8)
report = attention.analyze(transformer, tokens, labels, loss_fn)
```

### Comparative Analysis
```python
from gradient_flow import ComparativeAnalyzer

comparator = ComparativeAnalyzer()

# Track training
comparator.track_training(model, x, y, loss_fn, step=100)
comparator.track_training(model, x, y, loss_fn, step=500)

# View evolution
health_over_time = comparator.get_health_evolution("fc1")
```

### Fixing Issues
```python
from gradient_flow import gradient_scale

class FixedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(256)
        self.compress_down = nn.Linear(256, 64)
        self.compress_proj = nn.Linear(64, 16)

    def forward(self, x):
        x = self.norm(x)
        x = gradient_scale(x, 0.01)
        x = self.compress_down(x)
        x = F.gelu(x)
        x = self.compress_proj(x)
        return x
```

## 🎯 Key Innovations

1. **Hydraulic Metaphor** - Intuitive understanding of gradients
2. **Automatic Diagnosis** - Detect issues without manual inspection
3. **Actionable Recommendations** - Not just detection, but solutions
4. **Architecture-Aware** - Specialized analyzers for different network types
5. **Gradient Scaling** - Simple, powerful fix for exploding gradients
6. **Comparative Tracking** - Monitor gradient health evolution

## 📊 Metrics Explained

| Metric | Range | Good | Warning | Critical |
|--------|-------|------|---------|----------|
| Pressure | 0-∞ | 0.1-10 | 10-100 | >100 or <0.01 |
| Health Score | 0-100 | >75 | 50-75 | <50 |
| Turbulence | 0-∞ | <1.0 | 1.0-5.0 | >5.0 |
| Temporal Decay | -∞-∞ | -0.1 to +0.1 | ±0.1-0.3 | >±0.3 |

## 🔍 Issue Types

- **VANISHING** - Gradients too small (<0.01)
- **EXPLODING** - Gradients too large (>100)
- **BOTTLENECK** - Compression layer amplifying gradients
- **DECAY** - Gradient decay through depth
- **VANISHING_TEMPORAL** - Gradients vanish through time (RNN)
- **EXPLODING_TEMPORAL** - Gradients explode through time
- **SHORT_MEMORY** - Effective memory too short
- **ATTN_BOTTLENECK** - Attention weight bottleneck
- **FFN_BOTTLENECK** - Feed-forward network bottleneck
- **QKV_IMBALANCE** - Query/Key/Value imbalance
- **RESIDUAL_DECAY** - Residual connection degradation
- **LAYER_VARIANCE** - High variance across layers

## 📈 Performance

- **Minimal overhead** - Only active during analysis
- **Memory efficient** - Streaming statistics collection
- **Scalable** - Works on large models (tested up to 1B params)
- **Fast** - Analysis completes in seconds for most models
